"""Offline replay of the failed real input; no API, training, or run mutation.

Run with the sia interpreter and PYTHONPATH=runtime. The temporary delivery
workspace is removed on exit; original rollout/score/checkpoint files are read only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from sia.task_meta.meta_harness.evidence_delivery import prepare_delivery, canonical
from sia.task_meta.meta_harness.read_evidence import expand
from sia.task_meta.meta_backends.input_budget import (
    MetaInputBudget, attach_archives, archive_descriptors, inventory)
from sia.task_meta.meta_backends.local_execution import _stage_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--call', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    old = run / 'meta/calls' / args.call / 'workspace.input'
    outcome_path = old / 'meta_input/internal_dev_comparison.json'
    original_bytes = outcome_path.read_bytes()
    old_hash = hashlib.sha256(original_bytes).hexdigest()
    outcome = json.loads(original_bytes)
    payload = json.loads((old / 'meta_input/operation.json').read_text())
    library = json.loads((old / 'meta_input/principles.json').read_text())
    files = {p.relative_to(old).as_posix(): p.read_text() for p in old.rglob('*') if p.is_file()}
    files['meta_input/trajectory_archives.json'] = canonical(
        archive_descriptors(run, run / 'round_0/feedback', 0))
    payload = prepare_delivery(payload, files, library)
    with tempfile.TemporaryDirectory(prefix='rsiflow_evidence_replay_') as tmp:
        call = Path(tmp); work = call / 'workspace'; work.mkdir()
        for name, content in files.items():
            path = work / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        (call / 'codex_home').mkdir()
        (call / 'codex_home/config.toml').write_text('model="offline-fixture"')
        (call / 'schema.json').write_text('{}')
        budget = MetaInputBudget()
        attach_archives(work, run, budget)
        counts, paths = inventory(work, budget, 16000000)
        staged = _stage_files(call, budget)
        previous = Path.cwd()
        try:
            os.chdir(work)
            restored = expand(json.loads((work/'meta_input/internal_dev_comparison.json').read_text()))
        finally:
            os.chdir(previous)
        assert restored == outcome, 'Evidence content or hashes changed'
        archive_index = json.loads((work/'meta_input/archives/index.json').read_text())
        assert len(archive_index) == 360, 'Expected both complete 180-task passes'
        for record in archive_index:
            ref = record['content_reference']; raw = (work/ref['file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref['sha256']
            row = json.loads(raw)
            assert row['task_id'] == record['task_id']
            assert row['source_role'] == 'train_evolution'
        assert hashlib.sha256(outcome_path.read_bytes()).hexdigest() == old_hash
        print(json.dumps({'status': 'OFFLINE_REPLAY_PASSED',
            'original_comparison_bytes': len(original_bytes),
            'summary_comparison_bytes': (work/'meta_input/internal_dev_comparison.json').stat().st_size,
            'input_bytes_by_category': counts, 'files': len(paths),
            'staged_files': len(staged), 'complete_raw_records': len(archive_index),
            'comparison_lossless': True, 'original_run_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
