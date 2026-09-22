"""Audit and authorize the four-file Meta delivery repair for an interrupted run.

This never changes Task evidence, scores, weights, or a candidate. By default it
only checks readiness; --apply archives the old controller authorization and
rebinds the protocol so the existing resume launcher can continue.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from common import ROOT, read, write

sys.path.insert(0, str(ROOT / 'runtime'))
from sia.task_meta.completed_rollout_reuse import completed_request_reusable
from sia.task_meta.durable import load_task, task_hash, value_hash
from sia.task_meta.evolution_protocol import fingerprint
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.pipeline import PipelineConfig, source_identity
from sia.task_meta.round_validation import validate_round
from sia.task_meta.sequential_loop import content_identity
from sia.task_meta.storage import digest
from sia.task_meta.types import MetaDecision
from sia.task_meta.updaters import ModelUpdater


CHANGED_SOURCES = {
    'sia/task_meta/durable.py',
    'sia/task_meta/harnessforge_production.py',
    'sia/task_meta/meta_backends/codex_openrouter.py',
    'sia/task_meta/sequential_loop.py',
    'sia/task_meta/updaters.py',
}


def prepare(root, *, apply=False):
    root = Path(root).resolve(strict=True)
    current_protocol_path = root / 'protocol.json'
    current_proof_path = root / 'recovery/completed_rollout_reuse.json'
    old = read(current_protocol_path)
    old_proof = read(current_proof_path)
    config = PipelineConfig.model_validate(old['config']).checked()
    output = Path(config.output_root).resolve(strict=True)
    if not root.is_relative_to(output / 'runs'):
        raise ValueError('Not a registered experiment run')

    with exclusive_lock(output / 'locks/launch.lock'):
        with exclusive_lock(output / 'locks/run' / (value_hash(str(root)) + '.lock')):
            archive = root / 'recovery/meta_delivery_hotfix'
            prior_path = archive / 'protocol_before.json'
            proof_before_path = archive / 'completed_rollout_reuse_before.json'
            already_applied = prior_path.is_file() and proof_before_path.is_file()
            if prior_path.exists() != proof_before_path.exists():
                raise ValueError('Incomplete prior controller archive')
            prior = read(prior_path) if already_applied else old
            prior_proof = read(proof_before_path) if already_applied else old_proof
            current_controller = source_identity()
            changed = {name for name in set(prior['controller']) | set(current_controller)
                       if prior['controller'].get(name) != current_controller.get(name)}
            if changed != CHANGED_SOURCES:
                raise ValueError('Unexpected controller changes: ' + str(sorted(changed)))
            if prior_proof['controller'] != prior['controller']:
                raise ValueError('Original rollout reuse proof and protocol disagree')
            updated = copy.deepcopy(prior)
            updated['controller'] = current_controller
            updated.pop('hash', None)
            updated['hash'] = value_hash(updated)
            updated_proof = copy.deepcopy(prior_proof)
            updated_proof['controller'] = current_controller
            if ({k: v for k, v in old.items() if k not in {'controller', 'hash'}}
                    != {k: v for k, v in prior.items() if k not in {'controller', 'hash'}}
                    or {k: v for k, v in old_proof.items() if k != 'controller'}
                    != {k: v for k, v in prior_proof.items() if k != 'controller'}
                    or old_proof['controller'] != old['controller']
                    or old['hash'] != value_hash({k: v for k, v in old.items() if k != 'hash'})):
                raise ValueError('Unrelated protocol or rollout proof revision')
            revisions = [archive / 'revision.json', archive / 'revision_final.json']
            revisions += sorted(p for p in archive.glob('revision_*.json')
                                if p.stem.removeprefix('revision_').isdigit())
            saved_revisions = [p for p in revisions if p.exists()]
            followup = already_applied and old != updated
            if followup:
                prior_revision = read(saved_revisions[-1])
                if (prior_revision['new_protocol_sha256'] != digest(current_protocol_path)
                        or prior_revision['new_rollout_proof_sha256'] != digest(current_proof_path)):
                    raise ValueError('Current controller revision is not the archived migration')

            round0 = read(root / 'round_0/complete.json')
            validate_round(config, root, 0, round0)
            status = read(root / 'round_1/early_rollout/status.json')
            request = read(root / 'round_1/early_rollout/request.json')
            if status.get('status') != 'completed' or (root / 'round_1/complete.json').exists():
                raise ValueError('Expected completed Task2 baseline and unfinished Meta2')
            task = load_task(round0['task_after'])
            if request['task_hash'] != task_hash(task):
                raise ValueError('Task2 baseline is not bound to deployed Task1')
            for entry in prior_proof['entries'].values():
                for name, expected in entry['files'].items():
                    path = root / name
                    if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path) != expected:
                        raise ValueError('Original rollout evidence changed: ' + name)
                old_request = read(root / entry['early_request'])
                if old_request['protocol_sha256'] != digest(root / entry['prior_protocol']):
                    raise ValueError('Early rollout does not bind its archived protocol')
            baseline = root / 'round_1/before/gen_1'
            relative = baseline.relative_to(root).as_posix()
            entry = prior_proof['entries'].get(relative)
            if entry is None or entry['task_hash'] != task_hash(task):
                raise ValueError('Task2 baseline is not in the completed-rollout authorization')
            scope = read(baseline / 'execution_scope.json')['scope']
            execution = read(baseline / 'execution_receipt.json')
            if (scope['collection_stage'] != 'parent_pre_update'
                    or scope['protocol_hash'] != digest(root / entry['prior_protocol'])
                    or scope['implementation'] != fingerprint(read(root / entry['prior_protocol'])['controller'])
                    or execution['input_hash'] != value_hash([entry['task_hash'], digest(baseline / 'execution_scope.json')])
                    or status['execution_receipt_sha256'] != digest(baseline / 'execution_receipt.json')):
                raise ValueError('Task2 completed baseline is not reusable')

            attempt = read(root / 'round_1/candidates/MODEL/attempt.json')
            candidate = root / 'round_1/candidates/MODEL/gen_1'
            receipt = read(candidate / 'intervention_receipt.json')
            decision = MetaDecision.model_validate(attempt['decision'])
            if (attempt['status'] != 'prepared' or attempt['action'] != 'MODEL'
                    or attempt['parent_content_hash'] != content_identity(task)
                    or receipt['status'] != 'started'
                    or receipt['binding']['task_hash'] != task_hash(task)
                    or receipt['binding']['decision'] != decision.model_dump(mode='json')
                    or receipt['input_hash'] != value_hash(receipt['binding'])
                    or not ModelUpdater(None, None, None).retry_safe_before_effect(
                        task, decision, SimpleNamespace(directory=candidate))):
                raise ValueError('MODEL intervention is not demonstrably before its first Task effect')
            failed_audit = root / 'meta/operations/d3936b775129a9dec4b7139e/harness_runtime/policy_runtime.json'
            if not failed_audit.is_file():
                raise ValueError('Original Meta delivery failure audit is missing')

            report = {
                'status': 'ready_for_meta_delivery_resume',
                'run': str(root),
                'changed_controller_sources': sorted(changed),
                'task2_parent_rollout_reused': True,
                'model_training_started': False,
                'old_protocol_sha256': digest(prior_path) if already_applied else digest(current_protocol_path),
                'old_rollout_proof_sha256': digest(proof_before_path) if already_applied else digest(current_proof_path),
                'failed_meta_audit_sha256': digest(failed_audit),
            }
            if not apply:
                return report
            if not already_applied:
                archive.mkdir(parents=True, exist_ok=False)
                shutil.copy2(current_protocol_path, prior_path)
                shutil.copy2(current_proof_path, proof_before_path)
                shutil.copy2(failed_audit, archive / 'failed_model_request_policy_runtime.json')
            elif digest(archive / 'failed_model_request_policy_runtime.json') != digest(failed_audit):
                raise ValueError('Original failed Meta audit changed')
            if followup:
                suffix = 'intermediate' if len(saved_revisions) == 1 else f'followup_{len(saved_revisions) + 1:04d}'
                shutil.copy2(current_protocol_path, archive / f'protocol_{suffix}.json')
                shutil.copy2(current_proof_path, archive / f'completed_rollout_reuse_{suffix}.json')
            if read(current_proof_path) != updated_proof:
                write(current_proof_path, updated_proof)
            if read(current_protocol_path) != updated:
                write(current_protocol_path, updated)
            if not completed_request_reusable(root, root / 'round_1/before/gen_1', task_hash(task)):
                raise ValueError('Completed Task2 baseline cannot be reused after migration')
            record = {**report, 'status': 'authorized_meta_delivery_controller_revision',
                      'new_protocol_sha256': digest(current_protocol_path),
                      'new_rollout_proof_sha256': digest(current_proof_path)}
            record_path = (archive / 'revision.json' if not already_applied else
                           archive / 'revision_final.json' if followup and len(saved_revisions) == 1 else
                           archive / f'revision_{len(saved_revisions) + 1:04d}.json' if followup else
                           saved_revisions[-1])
            if record_path.exists() and read(record_path) != record:
                raise ValueError('Meta delivery migration receipt changed')
            write(record_path, record)
            return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--apply', action='store_true', help='Archive and authorize the verified controller revision')
    args = parser.parse_args()
    print(json.dumps(prepare(args.run_dir, apply=args.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
