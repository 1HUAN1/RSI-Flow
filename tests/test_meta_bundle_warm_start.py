from pathlib import Path

import pytest

from sia.task_meta.meta_harness import MetaHarnessStore


def test_import_keeps_g_identity_and_next_update_advances_from_it(tmp_path):
    old = MetaHarnessStore(tmp_path / 'old')
    g0 = old.initialize(Path(__file__).resolve().parents[1] / 'meta_harness/seed')
    g1 = old.commit_update(g0.hash, instruction_text=g0.read_files()['instructions.md'] + '\nUse observed outcomes.\n', request_id='source-update')
    new = MetaHarnessStore(tmp_path / 'new')
    with pytest.raises(ValueError, match='pinned hash'):
        new.initialize_from_bundle(g1.path, '0' * 64)
    imported = new.initialize_from_bundle(g1.path, g1.hash)
    assert imported.hash == g1.hash and imported.version == 1
    assert imported.read_files() == g1.read_files()
    with pytest.raises(ValueError, match='Duplicate Meta request'):
        new.commit_update(imported.hash, instruction_text='Duplicate must not commit', request_id='source-update')
    g2 = new.commit_update(imported.hash, instruction_text=imported.read_files()['instructions.md'] + '\nInspect dependencies.\n', request_id='next-update')
    assert g2.version == 2 and g2.manifest['parent_hash'] == g1.hash
    assert new.initialize_from_bundle(g1.path, g1.hash).hash == g2.hash
    assert old.active().hash == g1.hash


def test_missing_source_ancestor_is_rejected_before_activation(tmp_path):
    import shutil
    old = MetaHarnessStore(tmp_path / 'old')
    g0 = old.initialize(Path(__file__).resolve().parents[1] / 'meta_harness/seed')
    g1 = old.commit_update(g0.hash, instruction_text=g0.read_files()['instructions.md'] + '\nChanged.\n', request_id='source-update')
    orphan = tmp_path / 'orphan' / g1.path.name
    shutil.copytree(g1.path, orphan)
    new = MetaHarnessStore(tmp_path / 'new')
    with pytest.raises(FileNotFoundError):
        new.initialize_from_bundle(orphan, g1.hash)
    assert not (new.root / 'active_harness.json').exists()
