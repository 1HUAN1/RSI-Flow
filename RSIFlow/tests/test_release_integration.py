"""Exercise release wiring with synthetic records, without models or API calls."""
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]

from evolution_protocol import file_hash, fingerprint, freeze, make_manifest, read
from prepare_600_744 import prepare
from rsiflow.task_meta.round_evolution import RoundStore


def _source_manifest(tmp_path, counts, role, round_id):
    rows = []
    for source, count in counts.items():
        domain = ('tool_use' if source in {'envscaler', 'bfcl_v3', 'acebench'} else
                  'code' if source in {'deepcoder_taco', 'humaneval_plus', 'mbpp_plus'} else 'searchqa')
        record = tmp_path / f'{source}.jsonl'
        source_rows = []
        with record.open('wb') as stream:
            for i in range(count):
                identity = f'{role}-{source}-{i}'
                payload = (json.dumps({'id': identity, 'task_id': identity,
                                      'question': identity, 'problem': identity, 'task': identity}) + '\n').encode()
                offset = stream.tell()
                stream.write(payload)
                source_rows.append(dict(source=source, domain=domain, version='synthetic-v1',
                    original_split='synthetic', task_id=identity, native_id=identity,
                    group_id=identity, content_hash=fingerprint(identity), role=role, round_id=round_id,
                    purpose='evolution_train' if round_id else 'independent_validation',
                    record_file=str(record), record_offset=offset, record_length=len(payload),
                    strata={'interface': 'stdio' if i < count * 3 // 5 else 'function'}))
        rows.extend({**row, 'record_sha256': file_hash(record)} for row in source_rows)
    path = tmp_path / f'{role}.json'
    freeze(path, make_manifest(role, round_id, rows))
    return str(path)


@pytest.fixture
def profiles(tmp_path):
    profiles = [read(ROOT / f'configs/train_{size}.json') for size in ('4b', '8b')]
    training = _source_manifest(tmp_path,
        {source: count * 5 for source, count in profiles[0]['train_quotas_per_round'].items()},
        'train_evolution', 1)
    validation = _source_manifest(tmp_path, profiles[0]['validation_limits'], 'independent_validation', None)
    retrieval = tmp_path / 'retrieval.sqlite'
    sqlite3.connect(retrieval).close()
    freeze(retrieval.with_suffix('.sqlite.manifest.json'), {'role': 'synthetic-test-corpus'})
    paths = []
    for size, config in zip(('4b', '8b'), profiles):
        config.update(source_training_manifest=training, source_validation_manifest=validation,
            frozen_data_dir=str(tmp_path / 'frozen'), validation_manifest=str(tmp_path / 'validation/manifest.json'),
            retrieval_index=str(retrieval))
        path = tmp_path / f'train_{size}.json'
        freeze(path, config)
        paths.append(path)
    return paths


@pytest.mark.parametrize('first', [0, 1])
def test_backbones_reuse_same_complete_frozen_release(profiles, first):
    result = prepare(profiles[first])
    assert result['training_tasks'] == 3000
    assert result['tasks_per_round'] == [600] * 5
    assert result['validation_tasks'] == 744
    config = read(profiles[first])
    root = Path(config['frozen_data_dir'])
    hashes = {p: file_hash(p) for p in root.rglob('*') if p.is_file()}
    assert prepare(profiles[1-first])['status'] == 'already_prepared'
    assert hashes == {p: file_hash(p) for p in root.rglob('*') if p.is_file()}
    store = RoundStore(root / 'tasks.sqlite')
    ids = set()
    for number in range(1, 6):
        store.select_round(number)
        tasks = store.current['tasks']
        assert len(tasks) == 600
        assert not ids.intersection(t['task_id'] for t in tasks)
        ids.update(t['task_id'] for t in tasks)
        assert all(store._task(t).task_id == t['task_id'] for t in tasks)
    assert len(ids) == 3000


def test_backbone_reuse_still_rejects_changed_data_protocol(profiles):
    prepare(profiles[0])
    config = read(profiles[1])
    config['split_seed'] += 1
    profiles[1].write_text(json.dumps(config))
    with pytest.raises(FileExistsError):
        prepare(profiles[1])


def test_backbone_reuse_rejects_modified_saved_protocol(profiles):
    prepare(profiles[0])
    config = read(profiles[0])
    protocol = Path(config['frozen_data_dir']) / 'protocol.json'
    protocol.write_text(json.dumps(read(profiles[1])))
    with pytest.raises(FileExistsError):
        prepare(profiles[1])


def test_controller_subprocess_bootstraps_renamed_runtime(tmp_path):
    from launch_meta import SubprocessToolExecutor
    from rsiflow.task_meta.rsiflow_manifest import load_manifest
    checkpoint = tmp_path / 'synthetic-checkpoint'
    checkpoint.mkdir()
    # A hashable stand-in verifies transport/manifest wiring, not model inference.
    (checkpoint / 'model.safetensors').write_bytes(b'not-real-model-weights')
    state = tmp_path / 'run/active_task.json'
    executor = SubprocessToolExecutor(project_root=ROOT, run_dir=state.parent)
    result = executor.execute({'operation': 'bootstrap',
        'config_path': str(ROOT / 'configs/train_4b.json'),
        'checkpoint_path': str(checkpoint), 'state_path': str(state)})
    assert result['status'] == 'completed', result
    saved = read(state)
    assert saved['checkpoint_manifest'][0]['sha256'] == file_hash(checkpoint / 'model.safetensors')
    assert Path(saved['harness_path']).is_file()
    assert load_manifest(saved['harness_path']).bundle_sha256


def test_codex_executable_environment_override(tmp_path, monkeypatch):
    from persistent_meta import CodexAppServer
    monkeypatch.setenv('RSIFLOW_CODEX_EXECUTABLE', '/custom/bin/codex')
    server = CodexAppServer(codex_home=tmp_path, workspace=tmp_path, journal=tmp_path)
    assert server.executable == '/custom/bin/codex'
    explicit = CodexAppServer(codex_home=tmp_path, workspace=tmp_path, journal=tmp_path, executable='/explicit/codex')
    assert explicit.executable == '/explicit/codex'
