import json
from pathlib import Path

import pytest

from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.meta_harness.bundle import MetaHarnessStore


def test_exclusive_controller_lock_prevents_parallel_resume_and_releases(tmp_path):
    lock = tmp_path / 'run.lock'
    with exclusive_lock(lock), pytest.raises(OSError), exclusive_lock(lock):
        pytest.fail('A second controller acquired the active run')
    with exclusive_lock(lock):
        pass


def test_meta_bundle_recovers_rename_before_pointer_publication(tmp_path, monkeypatch):
    import sia.task_meta.meta_harness.bundle as module
    store = MetaHarnessStore(tmp_path / 'meta')
    initial = store.initialize(Path(__file__).resolve().parents[1] / 'meta_harness/seed')
    original = module.atomic_json
    def crash(path, value):
        if Path(path).name == 'active_harness.json':
            raise RuntimeError('injected pointer-publication crash')
        return original(path, value)
    monkeypatch.setattr(module, 'atomic_json', crash)
    with pytest.raises(RuntimeError, match='publication'):
        store.commit_update(initial.hash, instruction_text='Recovered bounded Meta instructions')
    assert store.active().hash == initial.hash
    monkeypatch.setattr(module, 'atomic_json', original)
    updated = store.commit_update(initial.hash, instruction_text='Recovered bounded Meta instructions')
    assert updated.version == 1 and updated.manifest['parent_hash'] == initial.hash
    assert len(list((tmp_path / 'meta/harness_versions').iterdir())) == 2
    assert json.loads((tmp_path / 'meta/active_harness.json').read_text())['bundle_hash'] == updated.hash
