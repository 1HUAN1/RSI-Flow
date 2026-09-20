"""Explicit preparation, compatibility, run, recovery and selection entrypoints.

All modes use the repaired Task-Meta state machine. Dev never invents a Meta
decision and never starts a paid request or GPU training as a side effect.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sia.task_meta.data import DOMAINS, ManifestStore, build_manifest, sha256_file
from sia.task_meta.durable import (
    DurableClient,
    DurableExecutor,
    DurableUpdater,
    StageJournal,
    load_task,
    task_hash,
    value_hash,
)
from sia.task_meta.meta_backends.contracts import BackendUnavailable, MetaBackendConfig
from sia.task_meta.storage import checkpoint_manifest, digest, save_json
from sia.task_meta.types import MetaAgentState, TaskAgentState, TaskUpdateAction

PROJECT = Path(__file__).resolve().parents[2]


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    mode: Literal['dev', 'api_smoke', 'pilot', 'full'] = 'dev'
    task_checkpoint: str = '/root/data/zh/huggingface/Qwen3-4B'
    task_base_url: str = 'http://127.0.0.1:8071/v1'
    task_enable_thinking: bool = False
    task_timeout: int = Field(default=180, ge=1, le=1800)
    seed_harness: str = 'seed_harness/v2/seed.json'
    initial_meta_bundle: str | None = None
    initial_meta_bundle_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    source_manifest: str = '/root/data/RSI_iclr2027/dataset/manifests/dataset_manifest.json'
    data_dir: str = 'data/pipeline_v2'
    search_dev_fraction: float = Field(default=0.1, gt=0, lt=1)
    probe_per_domain: dict[str, int] = Field(default_factory=lambda: dict.fromkeys(DOMAINS, 16))
    window_quotas: dict[str, int] = Field(default_factory=lambda: {'tool_use': 16, 'code': 16, 'searchqa': 128})
    rollouts_per_task: int = Field(default=2, ge=1, le=16)
    probe_rollouts: int = Field(default=1, ge=1, le=8)
    seed: int = 42
    max_generations: int = Field(default=3, ge=1, le=100000)
    max_wall_seconds: int = Field(default=86400, ge=1)
    model_call_limit: int = Field(default=128, ge=1, le=256)
    max_output_tokens: int = Field(default=2048, ge=32, le=16384)
    inference_gpu: int = Field(default=0, ge=0, le=3)
    training_gpu: int = Field(default=1, ge=0, le=3)
    trainer_python: str = '/root/data/conda/envs/sia/bin/python'
    training: dict = Field(default_factory=lambda: {'max_steps': 100, 'learning_rate': 2e-5,
        'batch_size': 1, 'gradient_accumulation_steps': 8, 'max_length': 8192, 'lora_rank': 16,
        'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'save_steps': 50, 'seed': 42})
    supervision: Literal['final_assistant'] = 'final_assistant'
    external_corpus: str | None = None
    envscaler_root: str = '/root/data/RSI_iclr2027/dataset/train/tool_use/envscaler'
    envscaler_utils: str = '/root/data/RSI_iclr2027/dataset/evaluators/envscaler/EnvScaler/interact_with_env/envscaler_env/utils/env_util.py'
    envscaler_commit: str = '96ae8b02dc0187c911b8e2101e7bb6904271597b'
    meta: MetaBackendConfig = Field(default_factory=MetaBackendConfig)
    experiment_scope: Literal['multidomain', 'envscaler_validation'] = 'multidomain'
    scope_train_limit: int = Field(default=9, ge=1)
    scope_probe_limit: int = Field(default=3, ge=1)
    task_replicas: list[dict] = Field(default_factory=list)

    def checked(self):
        if bool(self.initial_meta_bundle) != bool(self.initial_meta_bundle_hash):
            raise ValueError('Initial Meta Bundle requires both source path and pinned hash')
        for values in (self.probe_per_domain, self.window_quotas):
            if set(values) != set(DOMAINS) or any(type(v) is not int or v < 1 for v in values.values()):
                raise ValueError('All three domains require positive fixed quotas')
        if self.mode != self.meta.run_mode:
            raise ValueError('Top-level mode and Meta run_mode must agree')
        if self.inference_gpu == self.training_gpu:
            raise ValueError('This local service protocol separates inference/training devices to prevent concurrent allocations')
        if self.task_replicas:
            from urllib.parse import urlsplit
            if self.experiment_scope != 'envscaler_validation':
                raise ValueError('Parallel execution is currently scoped to EnvScaler validation')
            devices, endpoints = set(), set()
            for replica in self.task_replicas:
                if set(replica) != {'gpu', 'base_url'}:
                    raise ValueError('Replica requires fixed GPU and base_url')
                endpoint = urlsplit(replica['base_url'])
                if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.path != '/v1'
                        or not endpoint.port or endpoint.username or endpoint.password
                        or type(replica['gpu']) is not int or replica['gpu'] not in range(4)
                        or replica['gpu'] == self.training_gpu or replica['gpu'] in devices
                        or replica['base_url'] in endpoints):
                    raise ValueError('Invalid, duplicate or training-conflicting replica')
                devices.add(replica['gpu']); endpoints.add(replica['base_url'])
        return self


def project_path(name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def load_config(path):
    return PipelineConfig.model_validate_json(Path(path).read_text(encoding='utf-8')).checked()


def model_identity(path):
    root = Path(path).resolve()
    config = json.loads((root / 'config.json').read_text())
    if root.name != 'Qwen3-4B' or config.get('model_type') != 'qwen3':
        raise ValueError('New pipeline requires the exact local Qwen3-4B checkpoint; no model substitution')
    files = {p.name: digest(p) for p in root.iterdir() if p.is_file() and
             (p.name.startswith(('tokenizer', 'special_tokens', 'generation_config', 'chat_template')) or p.name == 'config.json')}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError('Qwen3 tokenizer has no executable chat template')
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'initialization probe'}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {'path': str(root), 'model_type': config['model_type'], 'architectures': config.get('architectures'),
            'weights': checkpoint_manifest(root), 'tokenizer_and_config': files,
            'chat_template_sha256': value_hash(tokenizer.chat_template),
            'initial_request_render_sha256': value_hash(rendered), 'status': 'fingerprinted_no_gpu_inference'}


def prepare(config, *, build_corpus=True):
    from sia.task_meta.retrieval import build_search_index
    from sia.task_meta.seed import load_seed
    destination = project_path(config.data_dir)
    if not destination.is_relative_to(PROJECT):
        raise ValueError('New manifests and indexes must be inside the project')
    destination.mkdir(parents=True, exist_ok=True)
    seed = load_seed(project_path(config.seed_harness))
    manifest = destination / 'tasks.sqlite'
    if not manifest.exists():
        build_manifest(config.source_manifest, destination, split_seed=config.seed,
                       search_dev_fraction=config.search_dev_fraction, probe_per_domain=config.probe_per_domain)
    store = ManifestStore(manifest)
    try:
        store.validate_sources()
        metadata = json.loads((destination / 'manifest.json').read_text())
        if (metadata['split_seed'] != config.seed or metadata['search_dev_fraction'] != config.search_dev_fraction
                or metadata['probe_per_domain'] != config.probe_per_domain):
            raise ValueError('Prepared data protocol differs from requested split/probe configuration')
        corpus = destination / 'search.sqlite'
        if build_corpus and not corpus.exists():
            build_search_index(store.iter_split('evolve_train', 'searchqa'), corpus,
                               data_manifest_hash=sha256_file(manifest), external_corpus=config.external_corpus)
        identity_path = destination / 'model_identity.json'
        if not identity_path.exists():
            save_json(identity_path, model_identity(config.task_checkpoint))
        identity = json.loads(identity_path.read_text())
        if identity['path'] != str(Path(config.task_checkpoint).resolve()):
            raise ValueError('Prepared model fingerprint belongs to a different checkpoint')
        result = {'status': 'PREPARED_NOT_TRAINED', 'counts': store.counts(), 'manifest_sha256': sha256_file(manifest),
                  'seed_sha256': digest(project_path(config.seed_harness)), 'reference': seed['reference'],
                  'full_coverage': False, 'model_identity': str(identity_path), 'data_manifest': str(manifest),
                  'corpus_manifest': str(corpus.with_suffix('.sqlite.manifest.json')) if corpus.exists() else None,
                  'meta_status': 'IMPLEMENTED_NOT_API_VALIDATED'}
        save_json(destination / 'preparation.json', result)
        return result
    finally:
        store.close()


def source_identity():
    paths = [*PROJECT.joinpath('sia/task_meta').rglob('*.py'), PROJECT / 'scripts/train_task_meta_sft.py',
             PROJECT / 'scripts/run_multidomain.py', PROJECT / 'pyproject.toml']
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in sorted(paths)}


def backend_for(config, directory):
    from sia.task_meta.meta_backends.codex_openrouter import CodexOpenRouterBackend
    from sia.task_meta.meta_harness.bundle import MetaHarnessStore
    store = MetaHarnessStore(directory / 'meta')
    if config.initial_meta_bundle:
        bundle = store.initialize_from_bundle(project_path(config.initial_meta_bundle), config.initial_meta_bundle_hash)
        if (bundle.manifest['source_commit'] != config.meta.codex_commit or
                bundle.manifest['binary_sha256'] != config.meta.codex_binary_sha256):
            raise ValueError('Initial Meta Bundle belongs to a different pinned Codex runtime')
    else:
        bundle = store.initialize(PROJECT / 'meta_harness/seed', config.meta.codex_commit, config.meta.codex_binary_sha256)
    return CodexOpenRouterBackend(config.meta, directory / 'meta', store), store, bundle


def adapter_factory(config):
    import copy

    from sia.task_meta.environments import EnvScalerAdapter, SearchQAAdapter, TACOAdapter
    from sia.task_meta.retrieval import FrozenSearchIndex
    from sia.task_meta.sandbox import LinuxSandbox
    root = project_path(config.data_dir)
    corpus_metadata = json.loads((root / 'search.sqlite.manifest.json').read_text())
    index = FrozenSearchIndex(root / 'search.sqlite', expected_sha256=corpus_metadata['sha256'])
    sandbox = LinuxSandbox()
    sandbox.validate()
    envroot = Path(config.envscaler_root)
    utils = Path(config.envscaler_utils)
    utils_hash = sha256_file(utils)
    env_template = EnvScalerAdapter([envroot / 'train/environments.jsonl', envroot / 'heldout/environments.jsonl'],
                utils, runtime_commit=config.envscaler_commit, sandbox=sandbox, official_utils_sha256=utils_hash)

    def factory(domain):
        if domain == 'searchqa':
            return SearchQAAdapter(index)
        if domain == 'code':
            return TACOAdapter(sandbox)
        if domain == 'tool_use':
            # Each adapter has a distinct live worker; trusted source metadata is read-only.
            return copy.copy(env_template)
        raise ValueError('Unknown domain')
    return factory, index


def run(config, run_dir, *, resume=False):
    from sia.task_meta.file_lock import exclusive_lock
    lock = PROJECT / 'local_baseline/run_locks' / (value_hash(str(project_path(run_dir))) + '.lock')
    with exclusive_lock(lock):
        return _run(config, run_dir, resume=resume)


def _run(config, run_dir, *, resume=False):
    if config.mode not in {'pilot', 'full'}:
        raise ValueError('Real evolution requires explicit pilot or full mode; dev uses offline tests')
    from sia.task_meta.loop import run_task_meta
    from sia.task_meta.meta import MetaAgent
    from sia.task_meta.pipeline_execution import MultiDomainExecutor
    from sia.task_meta.seed import SeedHarnessUpdater
    from sia.task_meta.task_client import LocalTaskClient
    from sia.task_meta.updaters import ArtifactUpdater, ModelUpdater
    run_dir = project_path(run_dir)
    if not run_dir.is_relative_to(PROJECT / 'runs'):
        raise ValueError('Run outputs must be under project/runs')
    if run_dir.exists() and not resume:
        raise FileExistsError('Use a new run directory; historical runs are immutable')
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = StageJournal(run_dir)
    raw_client, bundles, bundle = backend_for(config, run_dir)
    raw_client.bind_context(run_dir.name, 0, '0' * 64)
    from sia.task_meta.types import MetaDecision
    # Production boot checks the real execution host without creating a toy Meta
    # operation. The first request/operation is routing over this run's T0 evidence.
    if config.meta.compatibility_mode == 'in_run':
        startup = run_dir / 'meta/execution_startup'
        startup.mkdir(parents=True, exist_ok=True)
        prepared = SimpleNamespace(directory=startup, bundle=bundle,
                                   request=SimpleNamespace(meta_harness_hash=bundle.hash))
    else:
        prepared = raw_client.prepare('Preflight only; do not execute a decision.', MetaDecision, operation='routing')
    try:
        raw_client.validate(prepared)
    except BackendUnavailable as exc:
        journal.mark(exc.status, pending_request=str(prepared.directory / 'request.json') if (prepared.directory / 'request.json').exists() else None, reason=str(exc))
        raise
    data_root = project_path(config.data_dir)
    if config.experiment_scope == 'envscaler_validation':
        from sia.task_meta.scoped_execution import EnvScalerValidationStore
        store = EnvScalerValidationStore(data_root / 'tasks.sqlite', config.scope_train_limit, config.scope_probe_limit)
    else:
        store = ManifestStore(data_root / 'tasks.sqlite')
    store.validate_sources()
    identity = model_identity(config.task_checkpoint)
    protocol = {'config': config.model_dump(), 'controller': source_identity(), 'model_identity': identity,
                'data_hash': sha256_file(data_root / 'tasks.sqlite'), 'corpus': json.loads((data_root / 'search.sqlite.manifest.json').read_text()),
                'seed_hash': digest(project_path(config.seed_harness)), 'meta_identity': raw_client.compatibility_identity(),
                'selection_rule': 'max_fixed_probe_macro_success_earliest_tie'}
    protocol['hash'] = value_hash(protocol)
    if config.experiment_scope == 'envscaler_validation':
        protocol['data_scope'] = store.scope_identity()
        protocol.pop('hash')
        protocol['hash'] = value_hash(protocol)
    protocol_path = run_dir / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Resume refused: trusted code, data, model, seed or experiment configuration changed')
    else:
        save_json(protocol_path, protocol)
        initial_dir = run_dir / 'gen_0'
        initial_dir.mkdir(exist_ok=True)
        shutil.copy2(project_path(config.seed_harness), initial_dir / 'seed.json')
        initial_harness = run_dir / 'meta' / f'harness_v{bundle.version}.md'
        initial_harness.write_text((bundle.path / 'instructions.md').read_text(), encoding='utf-8')
        save_json(run_dir / 'initial_meta_state.json', MetaAgentState(config.meta.model, str(initial_harness), version=bundle.version,
                         bundle_hash=bundle.hash, bundle_path=str(bundle.path)))
    initial_task = TaskAgentState(0, identity['path'], str(run_dir / 'gen_0/seed.json'),
                                 checkpoint_path=identity['path'], checkpoint_manifest=identity['weights'])
    initial_meta = MetaAgentState(**json.loads((run_dir / 'initial_meta_state.json').read_text()))
    client = DurableClient(raw_client, journal)
    meta = MetaAgent(client, {'trainer_configured': True, 'sft_profile': 'multidomain',
                             'execution': {'three_domains_shared_state': config.experiment_scope == 'multidomain',
                                           'active_domains': ['tool_use'] if config.experiment_scope == 'envscaler_validation' else list(DOMAINS)},
                             'meta_bundle_editable_files': bundle.manifest['editable_files']})
    factory, index = adapter_factory(config)
    executor = DurableExecutor(MultiDomainExecutor(store, factory,
        lambda state, base_url=None: LocalTaskClient(state, base_url or config.task_base_url, timeout=config.task_timeout, enable_thinking=config.task_enable_thinking),
        quotas=config.window_quotas, rollouts_per_task=config.rollouts_per_task, probe_rollouts=config.probe_rollouts,
        seed=config.seed, model_call_limit=config.model_call_limit, max_output_tokens=config.max_output_tokens, journal=journal,
        expected_domains=('tool_use',) if config.experiment_scope == 'envscaler_validation' else DOMAINS,
        replicas=config.task_replicas), journal)
    provider = SimpleNamespace(base_url=config.task_base_url, api_key_env='LOCAL_QWEN_API_KEY')
    command = [config.trainer_python, str(PROJECT / 'scripts/train_task_meta_sft.py'),
               '--request-dir', '{request_dir}', '--base-url', config.task_base_url]
    updaters = {TaskUpdateAction.HARNESS: SeedHarnessUpdater(client), TaskUpdateAction.ARTIFACTS: ArtifactUpdater(client),
                TaskUpdateAction.MODEL: ModelUpdater(client, None, provider, command, timeout=7200,
                    sft_profile='multidomain', supervision=config.supervision, training=config.training,
                    training_gpu=config.training_gpu)}
    updaters = {action: DurableUpdater(updater, journal) for action, updater in updaters.items()}

    def accept(meta_state, update):
        arguments = {'instruction_text': update.harness, 'file_updates': update.bundle_files,
                     'request_id': update.request_id, 'experience_id': update.experience_id,
                     'phase': 'meta_self_update' if update.experience_id else 'final_consolidation'}
        committed = bundles.reconcile_update(meta_state.bundle_hash, **arguments)
        if committed is None:
            committed = bundles.commit_update(meta_state.bundle_hash, **arguments)
        meta_state.bundle_hash, meta_state.bundle_path = committed.hash, str(committed.path)
        return meta_state
    try:
        windows = max(math.ceil(v['evolve_train'] / config.window_quotas[d]) for d, v in store.counts().items())
        final = run_task_meta(run_dir, initial_task, initial_meta, executor, meta, updaters,
            max_generations=min(config.max_generations, windows), primary_metric_name='macro_success', max_wall_time=config.max_wall_seconds,
            resume=resume, meta_update_handler=accept)
        journal.mark('completed' if final['status'] == 'completed' else final['status'])
        if final['status'] == 'completed':
            freeze(run_dir)
        return final
    except BackendUnavailable as exc:
        journal.mark(exc.status, reason=str(exc))
        raise
    except Exception as exc:
        journal.mark('infrastructure_or_contract_failure', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        store.close()
        index.close()


def freeze(run_dir):
    from sia.task_meta.task_harness import harness_identity
    root = project_path(run_dir)
    final = json.loads((root / 'final_state.json').read_text())
    if final['status'] != 'completed' or final['primary_metric'] != 'macro_success':
        raise ValueError('Only a completed three-domain run can be frozen by this selector')
    best = max(final['performance_history'], key=lambda r: (r['macro_success'], -r['generation']))
    state = load_task(json.loads((root / f'gen_{best["generation"]}/evaluated_state.json').read_text()))
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol.get('controller') != source_identity():
        raise ValueError('Freeze refused: fixed controller/runtime source changed after evaluation')
    frozen = {'task_state': asdict(state), 'state_hash': task_hash(state), 'protocol_hash': protocol['hash'],
              'task_harness_identity': harness_identity(state.harness_path),
              'checkpoint_files': checkpoint_manifest(state.checkpoint_path),
              'selection': {'rule': 'max_fixed_probe_macro_success_earliest_tie', 'generation': best['generation'],
                            'score': best['macro_success'], 'probe_identity': best['probe_identity']},
              'status': 'frozen_for_report_eval'}
    if (root / 'frozen_task.json').exists() and json.loads((root / 'frozen_task.json').read_text()) != frozen:
        raise ValueError('Frozen Task identity changed; final results cannot be used for reselection')
    save_json(root / 'last_evaluated.json', json.loads((root / f'gen_{final["performance_history"][-1]["generation"]}/evaluated_state.json').read_text()))
    save_json(root / 'frozen_task.json', frozen)
    save_json(root / 'best_on_dev.json', frozen)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'run', 'resume', 'api-smoke', 'freeze', 'audit', 'report-eval', 'report-predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--skip-corpus', action='store_true')
    parser.add_argument('--evaluator-specs')
    parser.add_argument('--predictions-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--enable-inference', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == 'prepare':
        result = prepare(config, build_corpus=not args.skip_corpus)
    elif args.command in {'run', 'resume'}:
        if not args.run_dir:
            parser.error('--run-dir is required')
        result = run(config, args.run_dir, resume=args.command == 'resume')
    elif args.command == 'report-predict':
        if not all((args.run_dir, args.evaluator_specs, args.output_dir)):
            parser.error('report-predict requires --run-dir, --evaluator-specs, --output-dir')
        from sia.task_meta.report_predictions import generate_report_predictions
        result = generate_report_predictions(config, project_path(args.run_dir) / 'frozen_task.json',
                                             args.evaluator_specs, args.output_dir, enabled=args.enable_inference)
    elif args.command == 'report-eval':
        if not all((args.run_dir, args.evaluator_specs, args.predictions_dir, args.output_dir)):
            parser.error('report-eval requires --run-dir, --evaluator-specs, --predictions-dir, --output-dir')
        from sia.task_meta.reporting import run_report_eval
        protocol = json.loads((project_path(args.run_dir) / 'protocol.json').read_text())
        result = run_report_eval(args.evaluator_specs, project_path(args.run_dir) / 'frozen_task.json',
                                 args.predictions_dir, args.output_dir,
                                 {'method': 'RSI', 'protocol': protocol, 'meta_feedback': False})
    elif args.command == 'freeze':
        result = freeze(args.run_dir)
    elif args.command == 'audit':
        from sia.task_meta.pipeline_audit import audit_run
        result = audit_run(project_path(args.run_dir))
    else:
        directory = project_path(args.run_dir or 'local_baseline/pipeline_v2_preflight')
        client, _, _ = backend_for(config, directory)
        client.bind_context(directory.name, 0, '0' * 64)
        if args.command == 'api-smoke':
            if config.mode != 'api_smoke':
                raise ValueError('API smoke requires the explicitly enabled api_smoke profile')
            from sia.task_meta.meta_backends.smoke import compatibility_smoke
            result = compatibility_smoke(client, directory / 'compatibility.json')
        else:
            from sia.task_meta.types import MetaDecision
            prepared = client.prepare('Readiness check only.', MetaDecision, operation='routing')
            try:
                client.validate(prepared)
                result = {'status': 'ready_for_explicit_operation', 'api_called': False}
            except BackendUnavailable as exc:
                result = {'status': exc.status, 'reason': str(exc), 'api_called': False}
            save_json(directory / 'preflight.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
