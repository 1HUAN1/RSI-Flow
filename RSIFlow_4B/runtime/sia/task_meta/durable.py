"""Stage receipts for replaying the existing loop without repeating interventions.

Completed stages are content-bound; uncertain external side effects fail closed.
This is transaction recovery, never performance-based rollback or candidate search.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from sia.task_meta.storage import artifact_manifest, digest, save_json
from sia.task_meta.types import (
    ArtifactState,
    EvaluationResult,
    TaskAgentState,
    TaskUpdate,
    TaskUpdateAction,
    UpdatePending,
)


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def load_task(value):
    value = dict(value)
    value['artifacts'] = ArtifactState(**value.get('artifacts', {}))
    return TaskAgentState(**value)


def task_hash(state):
    return value_hash({'state': asdict(state), 'harness': digest(Path(state.harness_path)),
                       'artifacts': artifact_manifest(state.artifacts.directory)})


class StageJournal:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        self.path = self.root / 'phase.json'

    def mark(self, phase, **details):
        save_json(self.path, {'phase': phase, **details})


class DurableUpdater:
    def __init__(self, updater, journal):
        self.updater, self.journal = updater, journal

    def apply(self, task, decision, context):
        base = self.journal.root / f'gen_{task.generation}'
        receipt = base / 'intervention_receipt.json'
        binding = {'task_hash': task_hash(task), 'decision': decision.model_dump(mode='json'),
                   'meta': asdict(context.meta_state) if context.meta_state else None}
        fingerprint = value_hash(binding)
        if receipt.exists():
            cached = json.loads(receipt.read_text(encoding='utf-8'))
            if cached['input_hash'] != fingerprint:
                raise ValueError('Intervention receipt belongs to different inputs')
            if cached['status'] != 'committed':
                raise UpdatePending('Uncertain intervention must be reconciled from its training/checkpoint evidence; refusing duplicate execution')
            successor = load_task(cached['successor'])
            if task_hash(successor) != cached['successor_hash']:
                raise ValueError('Committed successor was modified after receipt')
            update = dict(cached['update'])
            update['action'] = TaskUpdateAction(update['action'])
            return successor, TaskUpdate(**update)
        self.journal.mark('updating', generation=task.generation, action=decision.action.value)
        save_json(receipt, {'status': 'started', 'input_hash': fingerprint, 'binding': binding})
        try:
            successor, update = self.updater.apply(task, decision, context)
        except Exception as exc:
            from sia.task_meta.types import DecisionConstraintError
            if isinstance(exc, DecisionConstraintError):
                # The updater contract guarantees constraint rejection precedes commitment.
                receipt.unlink()
            raise
        save_json(receipt, {'status': 'committed', 'input_hash': fingerprint, 'binding': binding,
                            'successor': asdict(successor), 'successor_hash': task_hash(successor),
                            'update': asdict(update)})
        self.journal.mark('committed', generation=task.generation, action=decision.action.value)
        return successor, update


class DurableExecutor:
    durable_recovery = True

    def __init__(self, executor, journal):
        self.executor, self.journal = executor, journal

    def execute(self, task, directory):
        receipt = Path(directory) / 'execution_receipt.json'
        fingerprint = task_hash(task)
        scope = Path(directory)/'execution_scope.json'
        if scope.exists(): fingerprint = value_hash([fingerprint, digest(scope)])
        if receipt.exists():
            cached = json.loads(receipt.read_text(encoding='utf-8'))
            if cached['input_hash'] != fingerprint:
                raise ValueError('Execution receipt belongs to different Task input')
            value = cached['result']
            if cached.get('result_hash') != value_hash(value):
                raise ValueError('Execution receipt result integrity check failed')
            value['evaluated_state'] = load_task(value['evaluated_state']) if value['evaluated_state'] else None
            value['output_artifacts'] = ArtifactState(**value['output_artifacts']) if value['output_artifacts'] else None
            if _result_state_hashes(EvaluationResult(**value)) != cached.get('state_hashes'):
                raise ValueError('Execution receipt frozen input or output artifacts were modified')
            for item in cached.get('evidence_files', []):
                path = Path(directory) / item['path']
                if (not path.is_file() or path.is_symlink()
                        or not path.resolve().is_relative_to(Path(directory).resolve())
                        or digest(path) != item['sha256']):
                    raise ValueError('Execution receipt evidence file integrity check failed')
            return EvaluationResult(**value)
        self.journal.mark('executing', generation=task.generation)
        result = self.executor.execute(task, directory)
        value = asdict(result)
        evidence = []
        # These are executor-owned records, never mutable controller receipts or
        # trainer checkpoint directories from the preceding intervention.
        for name in ('window.json', 'coverage.json', 'train_trajectories.jsonl', 'probe_trajectories.jsonl'):
            path = Path(directory) / name
            if path.is_file():
                evidence.append({'path': name, 'sha256': digest(path)})
        for name in ('train_rollouts', 'probe_rollouts'):
            for path in sorted((Path(directory) / name).glob('*.json')):
                evidence.append({'path': path.relative_to(directory).as_posix(), 'sha256': digest(path)})
        save_json(receipt, {'input_hash': fingerprint, 'result': value, 'result_hash': value_hash(value),
                            'state_hashes': _result_state_hashes(result), 'evidence_files': evidence})
        return result


def _result_state_hashes(result):
    output = result.output_artifacts
    return {'evaluated_state': task_hash(result.evaluated_state) if result.evaluated_state else None,
            'output_artifacts': artifact_manifest(output.directory) if output else None}


class DurableClient:
    """Reuse accepted structured responses during deterministic controller replay."""
    def __init__(self, client, journal):
        self.client, self.journal = client, journal

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, prompt, schema, **kwargs):
        meta = kwargs.get('meta_state')
        validator = kwargs.get('validate_candidate')
        binding = {'prompt': prompt, 'schema': schema.model_json_schema(),
                   'meta': asdict(meta) if meta else None,
                   'kwargs': {k: v for k, v in kwargs.items() if k not in {'meta_state', 'validate_candidate'}},
                   'candidate_validator': getattr(validator, '__qualname__', None)}
        key = value_hash(binding)
        receipt = self.journal.root / 'meta' / 'response_receipts' / f'{key}.json'
        if receipt.exists():
            value = json.loads(receipt.read_text(encoding='utf-8'))
            if value.get('input_hash') != key or value.get('output_hash') != value_hash(value['output']):
                raise ValueError('Meta response receipt integrity check failed')
            result = schema.model_validate(value['output'])
            if validator is not None:
                checked = validator(result)
                if checked is False or (isinstance(checked, dict) and checked.get('passed') is False):
                    raise ValueError('Cached Meta result no longer passes the fixed candidate interface')
            return result
        self.journal.mark('meta_running', operation=kwargs.get('operation'), request_hash=key)
        result = self.client.complete(prompt, schema, **kwargs)
        output = result.model_dump(mode='json')
        save_json(receipt, {'input_hash': key, 'output': output, 'output_hash': value_hash(output)})
        return result


def record_experience(path, experience):
    """Append once; refuse altered history rather than silently replacing it."""
    value = asdict(experience)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            existing = json.loads(line)
            if existing['experience_id'] == value['experience_id']:
                if value_hash(existing) != value_hash(value):
                    raise ValueError('Recovered experience differs from immutable prior history')
                return
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')
