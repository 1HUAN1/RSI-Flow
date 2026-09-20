"""Bounded three-domain windows plus a fixed read-only progress probe."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from sia.task_meta.data import DOMAINS
from sia.task_meta.durable import task_hash, value_hash
from sia.task_meta.storage import artifact_manifest, digest, manifest_diff, save_json
from sia.task_meta.task_client import TaskInfrastructureError, TaskContextBudgetExceeded
from sia.task_meta.types import ArtifactState, EvaluationResult, UpdatePending


def aggregate(rows, *, require_domains=True, expected_domains=DOMAINS):
    if any(r.get('infrastructure_error') for r in rows):
        raise TaskInfrastructureError('Unresolved infrastructure failure: score is invalid, with no dropped denominator')
    domains = {}
    for domain in DOMAINS:
        items = [r for r in rows if r['domain'] == domain]
        correct = sum(r['verification'].get('success') is True for r in items)
        domains[domain] = {'correct': correct, 'denominator': len(items),
                           'success_rate': correct / len(items) if items else None,
                           'mean_native_reward': sum(r['terminal_reward'] for r in items) / len(items) if items else None,
                           'errors': dict(Counter(r['error_type'] for r in items if r.get('error_type')))}
        if domain == 'tool_use' and items:
            domains[domain]['mean_native_partial_score'] = sum(r['metrics']['native_partial_score'] for r in items) / len(items)
        if domain == 'searchqa' and items:
            domains[domain]['f1'] = sum(r['metrics'].get('f1', 0) for r in items) / len(items)
    if require_domains and any(not domains[d]['denominator'] for d in expected_domains):
        raise ValueError('Fixed progress probe must represent all three domains')
    available = [v['success_rate'] for v in domains.values() if v['success_rate'] is not None]
    return {'domains': domains, 'macro_success': sum(available) / len(available) if available else None,
            'domain_weights': dict.fromkeys(expected_domains, 1 / len(expected_domains)), 'status': 'complete',
            'total_rollouts': len(rows), 'units': 'fraction_0_1'}


class MultiDomainExecutor:
    def __init__(self, store, adapter_factory, model_factory, *, quotas, rollouts_per_task=1,
                 probe_rollouts=1, seed=42, model_call_limit=128, max_output_tokens=2048,
                 max_artifact_chars=12000, journal=None, expected_domains=DOMAINS, replicas=()):
        self.store, self.adapter_factory, self.model_factory = store, adapter_factory, model_factory
        self.quotas, self.rollouts_per_task, self.probe_rollouts = quotas, rollouts_per_task, probe_rollouts
        self.seed, self.model_call_limit, self.max_output_tokens = seed, model_call_limit, max_output_tokens
        self.max_artifact_chars, self.journal = max_artifact_chars, journal
        self.expected_domains, self.replicas = expected_domains, replicas

    def _batch(self, state, spec, jobs, directory, assets, *, probe, sources):
        if self.replicas:
            from sia.task_meta.scoped_execution import run_parallel
            return run_parallel(self, state, spec, jobs, directory, assets, probe, sources)
        return [self._one(state, spec, task, number, directory, assets, probe=probe, artifact_sources=sources)
                for task, number in jobs]

    def _one(self, state, seed_spec, task, rollout, directory, artifacts_text, *, probe, artifact_sources=None):
        from sia.task_meta.sandbox import SandboxUnavailable
        from sia.task_meta.seed import _BudgetExhausted, run_seed
        identifier = hashlib.sha256(f'{task.task_id}:{rollout}'.encode()).hexdigest()
        path = directory / f'{identifier}.json'
        call_directory = directory / f'{identifier}.calls'
        random_seed = int(value_hash([self.seed, task.task_id, rollout, 'probe' if probe else 'train'])[:8], 16) % (2**31)
        binding = {'task': task.task_id, 'task_source_hash': task.source_hash,
                   'state_hash': task_hash(state), 'rollout': rollout, 'probe': probe}
        from sia.task_meta.round_evolution import scoped_rollout
        directory, binding = scoped_rollout(self, state, task, rollout, directory, binding)
        path = directory / f'{identifier}.json'
        call_directory = directory / f'{identifier}.calls'
        if path.exists():
            cached = json.loads(path.read_text(encoding='utf-8'))
            if cached['binding'] != binding:
                imports = directory.parent / 'rollout_import.json'
                item = json.loads(imports.read_text()).get('files', {}).get(path.name) if imports.is_file() else None
                prior = Path(item['source']) if item else None
                row = cached['row']
                expected = {key: value for key, value in binding.items() if key != 'state_hash'}
                observed = {key: value for key, value in cached['binding'].items() if key != 'state_hash'}
                # Explicit engineering import: original evidence bytes remain intact.
                if (not item or not prior.is_file() or prior.is_symlink() or path.is_symlink()
                        or digest(prior) != item['source_sha256'] or digest(path) != item['imported_sha256']
                        or expected != observed or row['harness_sha256'] != digest(Path(state.harness_path))
                        or row['artifact_input_manifest'] != state.artifacts.manifest or row['model_ref'] != state.model_ref
                        or row['external_task_budget'] != {'max_calls': self.model_call_limit,
                            'max_output_tokens': self.max_output_tokens, 'task_sampling': {'seed': random_seed}}
                        or any(call.get('response', {}).get('binding', {}).get('weights') != state.checkpoint_manifest
                               for call in row.get('transport_calls', []) if call.get('status') == 'completed')):
                    raise ValueError('Rollout cache input identity changed without verified engineering import')
            if not cached['row'].get('infrastructure_error'):
                from sia.task_meta.runtime_extensions import compact_rollout
                return compact_rollout(cached['row'], path) if getattr(self.store, 'compact_rollouts', False) and not probe else cached['row']
            if cached['row'].get('model_call_count', 0) > 0 or (call_directory.exists() and any(call_directory.iterdir())):
                raise UpdatePending('Incomplete Task rollout has model dispatch evidence; reconcile before retrying')
            attempt = 1
            while path.with_suffix(f'.attempt_{attempt}.json').exists():
                attempt += 1
            path.replace(path.with_suffix(f'.attempt_{attempt}.json'))
        if call_directory.exists() and any(call_directory.iterdir()):
            raise UpdatePending('Task call receipts exist without a completed rollout; refusing duplicate inference or tool effects')
        calls = []
        started = time.monotonic()
        environment = self.adapter_factory(task.domain)
        row = {'task_id': task.task_id, 'question_id': task.task_id, 'source': task.source,
               'domain': task.domain, 'split': 'search_dev' if probe else 'evolve_train',
               'rollout_id': rollout, 'seed': random_seed, 'state_hash': binding['state_hash'],
               'model_ref': state.model_ref, 'harness_sha256': digest(Path(state.harness_path)),
               'artifact_input_manifest': state.artifacts.manifest, 'infrastructure_error': False}
        try:
            public = environment.reset(task, str(rollout), random_seed)
            row.update(task_source_hash=task.source_hash, reset_hash=value_hash(public),
                       environment_type=type(environment).__module__ + "." + type(environment).__name__,
                       external_task_budget={"max_calls": self.model_call_limit, "max_output_tokens": self.max_output_tokens,
                                             "task_sampling": {"seed": random_seed}})
            client = (self.model_factory(state, base_url=self._replica_endpoint) if self.replicas
                      else self.model_factory(state))

            def model(messages, *, tools=None, seed, max_tokens, temperature):
                if max_tokens > self.max_output_tokens:
                    raise ValueError('Task model call exceeds immutable experiment budget')
                if len(calls) >= self.model_call_limit:
                    raise _BudgetExhausted('Maximum controller Task model calls reached')
                entry = {'call_id': len(calls), 'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(tools),
                         'seed': seed, 'max_tokens': max_tokens, 'temperature': temperature, 'status': 'started',
                         'usage_complete': False, 'unknown_usage_calls': 1,
                         'chat_template_kwargs': {'enable_thinking': getattr(client, 'enable_thinking', False)}}
                calls.append(entry)
                receipt_path = call_directory / f'{entry["call_id"]:04d}.json'
                receipt = {'binding': copy.deepcopy(binding), 'call_id': entry['call_id'],
                           'request_sha256': value_hash(entry), 'request': copy.deepcopy(entry), 'status': 'dispatching'}
                save_json(receipt_path, receipt)
                try:
                    response = copy.deepcopy(client(messages, tools=tools, seed=seed, max_tokens=max_tokens, temperature=temperature))
                    response.setdefault('chat_template_kwargs', copy.deepcopy(entry['chat_template_kwargs']))
                    entry.update(status='completed', response=copy.deepcopy(response))
                    usage = response.get('usage') or {}
                    entry['usage_complete'] = all(type(usage.get(key)) is int and usage[key] >= 0
                                                  for key in ('prompt_tokens', 'completion_tokens'))
                    entry['unknown_usage_calls'] = int(not entry['usage_complete'])
                    save_json(receipt_path, {**receipt, 'status': 'completed', 'response': copy.deepcopy(response)})
                    return response
                except TaskContextBudgetExceeded as exc:
                    entry.update(status='rejected', error_type=type(exc).__name__, usage_complete=True,
                                 unknown_usage_calls=0, generation_dispatched=False)
                    save_json(receipt_path, {**receipt, 'status': 'rejected', 'error_type': type(exc).__name__,
                                             'generation_dispatched': False})
                    raise
                except Exception as exc:
                    entry.update(status='failed', error_type=type(exc).__name__)
                    save_json(receipt_path, {**receipt, 'status': 'failed', 'error_type': type(exc).__name__,
                                             'response': copy.deepcopy(entry.get('response'))})
                    raise

            result = run_seed(seed_spec, model, environment, json.dumps(public, ensure_ascii=False),
                              artifacts_text, random_seed, artifact_sources=artifact_sources)
            for conversation in result.get('sft_conversations', []):
                call = calls[conversation['call_id']]
                conversation.update(tools=copy.deepcopy(call['tools']),
                    chat_template_kwargs=copy.deepcopy(call['chat_template_kwargs']),
                    binding=copy.deepcopy(call.get('response', {}).get('binding')),
                    seed=call['seed'], max_tokens=call['max_tokens'], temperature=call['temperature'])
            row.update(result)
            if result.get('infrastructure_failure'):
                raise TaskInfrastructureError(result.get('error') or 'Task infrastructure failure')
            answer = result['final_answer']
            if answer is None:
                answer = ''
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            result['final_answer'] = answer
            score = environment.evaluate(answer)
            if score.infrastructure_error:
                raise TaskInfrastructureError(score.error_type)
            row.update(result)
            row.update({'terminal_reward': score.reward, 'metrics': score.metrics,
                        'verification': score.verification, 'error_type': score.error_type or result.get('error_type'),
                        'execution_error_type': result.get('error_type'),
                        'verifier_details': score.details, 'valid_answer': bool(result['final_answer']),
                        'model_answer': result['final_answer']})
            # Notes and evaluation outputs never become one another's input.
            if probe:
                row['notes'] = []
        except (TaskInfrastructureError, SandboxUnavailable, OSError) as exc:
            row.update({'infrastructure_error': True, 'error_type': str(exc), 'terminal_reward': None,
                        'verification': {'status': 'infrastructure_error', 'success': False}, 'messages': []})
        finally:
            environment.close()
        row['transport_calls'] = calls
        row.setdefault('model_calls', [])
        row['chat_template_kwargs'] = {'enable_thinking': getattr(client, 'enable_thinking', False)} if 'client' in locals() else {}
        row['model_call_count'] = len(calls)
        row['wall_time_seconds'] = time.monotonic() - started
        usage = [c.get('response', {}).get('usage') or {} for c in calls]
        row['input_tokens'] = sum(item['prompt_tokens'] for item in usage
                                  if type(item.get('prompt_tokens')) is int and item['prompt_tokens'] >= 0)
        row['output_tokens'] = sum(item['completion_tokens'] for item in usage
                                   if type(item.get('completion_tokens')) is int and item['completion_tokens'] >= 0)
        row['unknown_usage_calls'] = sum(call.get('generation_dispatched') is not False and not all(type(item.get(key)) is int and item[key] >= 0
                                                 for key in ('prompt_tokens', 'completion_tokens'))
                                         for call, item in zip(calls, usage))
        row['usage_complete'] = row['unknown_usage_calls'] == 0
        row['output_truncated'] = any(c.get('response', {}).get('finish_reason') == 'length' for c in calls)
        from sia.task_meta.round_evolution import annotate_row
        annotate_row(self,state,row)
        save_json(path, {'binding': binding, 'row': row})
        from sia.task_meta.runtime_extensions import compact_rollout
        return compact_rollout(row, path) if getattr(self.store, 'compact_rollouts', False) and not probe else row

    def execute(self, state, directory):
        from sia.task_meta.seed import load_seed
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        seed_spec = load_seed(state.harness_path)
        original = task_hash(state)
        frozen = copy.deepcopy(state)
        snapshot = directory / 'artifacts_input'
        if state.artifacts.directory:
            if not snapshot.exists():
                shutil.copytree(state.artifacts.directory, snapshot)
            if artifact_manifest(snapshot) != state.artifacts.manifest:
                raise ValueError('Frozen input assets differ from the registered state')
            frozen.artifacts = ArtifactState(str(snapshot), artifact_manifest(snapshot))
        else:
            snapshot.mkdir(exist_ok=True)
            frozen.artifacts = ArtifactState(str(snapshot), [])
        inherited_path = directory / 'inherited_input_snapshot.json'
        if inherited_path.exists():
            import hashlib
            inherited = json.loads(inherited_path.read_text())
            continuation = json.loads((directory.parent / 'continuation.json').read_text())
            inherited_root = Path(continuation['source_run'])
            if hashlib.sha256((inherited_root / 'protocol.json').read_bytes()).hexdigest() != continuation['source_protocol_sha256']:
                raise ValueError('Inherited snapshot protocol changed')
            original_snapshot = inherited_root / f'gen_{state.generation}' / 'artifacts_input'
            if str(original_snapshot) != inherited['directory'] or artifact_manifest(original_snapshot) != frozen.artifacts.manifest:
                raise ValueError('Inherited snapshot content changed')
            source_receipt = inherited_root / f'gen_{state.generation}' / 'train_rollouts' / inherited['receipt_name']
            if hashlib.sha256(source_receipt.read_bytes()).hexdigest() != inherited['receipt_sha256']:
                raise ValueError('Inherited snapshot receipt changed')
            frozen.artifacts.directory = str(original_snapshot)
            if task_hash(frozen) != json.loads(source_receipt.read_text())['binding']['state_hash']:
                raise ValueError('Inherited frozen Task identity changed')
        artifact_sources = []
        for item in frozen.artifacts.manifest:
            if item['path'].startswith('submissions/'):
                # Task-scoped revised outputs (including internal probe outputs) are
                # evaluated directly, never injected into new Task prompts or SFT.
                continue
            artifact_sources.append({**item, 'content': (snapshot / item['path']).read_text(encoding='utf-8')})
        artifacts_text = '\n'.join(item['content'] for item in artifact_sources)[:self.max_artifact_chars]
        if seed_spec['schema_version'] == 2:
            if seed_spec['parts']['memory']['context']['artifact_char_limit'] > self.max_artifact_chars:
                raise ValueError('Task H artifact prompt budget exceeds the immutable experiment limit')
        else:
            artifact_sources = None
        cursor = dict.fromkeys(DOMAINS, 0)
        if state.generation and not getattr(self, 'round_protocol', None):
            previous = json.loads((directory.parent / f'gen_{state.generation - 1}' / 'window.json').read_text())
            cursor = previous['next_cursor']
        tasks, following = self.store.window(cursor, self.quotas)
        window = {'cursor': cursor, 'next_cursor': following, 'task_ids': [t.task_id for t in tasks],
                  'state_hash': original, 'quotas': self.quotas}
        if (directory / 'window.json').exists() and json.loads((directory / 'window.json').read_text()) != window:
            raise ValueError('Window schedule changed during recovery')
        save_json(directory / 'window.json', window)
        if self.replicas:
            from sia.task_meta.scoped_execution import sync_replicas
            save_json(directory / 'replica_bindings.json', sync_replicas(frozen, self.replicas))
        train = self._batch(frozen, seed_spec, ((t, n) for t in tasks for n in range(self.rollouts_per_task)),
                            directory / 'train_rollouts', artifacts_text, probe=False, sources=artifact_sources)
        training_scores = aggregate(train, require_domains=False, expected_domains=self.expected_domains)
        generated = directory / 'artifacts_generated'
        generated.mkdir(exist_ok=True)
        provenance = []
        for row in train:
            notes = [] if getattr(self, 'round_protocol', None) else row.get('notes') or []
            if isinstance(notes, (str, dict)):
                notes = [notes]
            for index, note in enumerate(notes):
                body = note if isinstance(note, str) else json.dumps(note, ensure_ascii=False)
                if not body.strip():
                    continue
                task_dir = generated / hashlib.sha256(row['task_id'].encode()).hexdigest() / f'rollout_{row["rollout_id"]}'
                task_dir.mkdir(parents=True, exist_ok=True)
                target = task_dir / f'note_{index}.md'
                target.write_text(body[:self.max_artifact_chars], encoding='utf-8')
                provenance.append({'path': target.relative_to(generated).as_posix(), 'sha256': digest(target),
                                   'task_id': row['task_id'], 'rollout_id': row['rollout_id'],
                                   'terminal_reward': row['terminal_reward'], 'knowledge_verified': False,
                                   'production_method': 'natural_task_rollout', 'split': 'evolve_train'})
        if self.journal:
            self.journal.mark('evaluating', generation=state.generation)
        probe = self._batch(frozen, seed_spec, [(t, n) for t in self.store.probe() for n in range(self.probe_rollouts)],
                            directory / 'probe_rollouts', artifacts_text, probe=not bool(getattr(self, 'round_protocol', None)), sources=artifact_sources)
        progress = aggregate(probe, expected_domains=self.expected_domains)
        progress['probe_identity'] = value_hash([t.task_id for t in self.store.probe()])
        progress['training_rewards'] = training_scores
        progress['evaluation_protocol'] = ('fixed_in_training_monitor_not_heldout_v1' if getattr(self.store, 'compact_rollouts', False) else 'fixed_search_dev_readonly_T_in_v1')
        if task_hash(state) != original or artifact_manifest(snapshot) != frozen.artifacts.manifest:
            raise ValueError('Task execution or probe mutated the frozen input')
        for name, rows in [('train', train), ('probe', probe)]:
            with (directory / f'{name}_trajectories.jsonl').open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        coverage = self.store.coverage(following)
        coverage.update({'unique_tasks_completed': sum(following.values()),
                         'training_rollouts_completed': sum(following.values()) * self.rollouts_per_task,
                         'full_coverage': coverage['all_tasks_scheduled'], 'status': 'complete_window',
                         'probe_rollouts_this_generation': len(probe), 'training_means_weights_updated': False})
        save_json(directory / 'coverage.json', coverage)
        save_json(directory.parent / 'coverage.json', coverage)
        costs = {'wall_time_seconds': sum(r['wall_time_seconds'] for r in train + probe),
                 'model_calls': sum(r['model_call_count'] for r in train + probe),
                 'input_tokens': sum(r['input_tokens'] for r in train + probe),
                 'output_tokens': sum(r['output_tokens'] for r in train + probe),
                 'unknown_usage_calls': sum(r.get('unknown_usage_calls', 0) for r in train + probe),
                 'usage_complete': all(r.get('usage_complete', False) for r in train + probe),
                 'api_cost_usd': None, 'gpu_hours': None}
        imports = directory / 'rollout_import.json'
        if imports.is_file():
            # Failed engineering attempts retain their cost, never their invalid score.
            discarded = json.loads(imports.read_text()).get('discarded_attempts', [])
            for item in discarded:
                prior = Path(item['source'])
                if prior.is_symlink() or digest(prior) != item['source_sha256']:
                    raise ValueError('Discarded engineering evidence changed')
                row = json.loads(prior.read_text())['row']
                if not row.get('infrastructure_error'):
                    raise ValueError('Cannot discard a valid scored rollout')
                for target, source in [('wall_time_seconds', 'wall_time_seconds'), ('model_calls', 'model_call_count'),
                                       ('input_tokens', 'input_tokens'), ('output_tokens', 'output_tokens'),
                                       ('unknown_usage_calls', 'unknown_usage_calls')]:
                    costs[target] += row[source]
                costs['usage_complete'] = costs['usage_complete'] and row['usage_complete']
            costs['discarded_engineering_attempts'] = len(discarded)
        output = ArtifactState(str(generated), artifact_manifest(generated))
        result = EvaluationResult(progress, train, costs, frozen, output, provenance,
                                manifest_diff(frozen.artifacts.manifest, output.manifest))
        if getattr(self, 'round_protocol', None):
            result = self.round_protocol.observed(result, directory)
        return result
