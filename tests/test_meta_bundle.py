import json
from pathlib import Path

import pytest

from sia.task_meta.meta_harness import LEGACY_EDITABLE_FILES, MetaHarnessStore

SEED = Path(__file__).resolve().parents[1] / "meta_harness/seed"


@pytest.fixture
def store(tmp_path):
    result = MetaHarnessStore(tmp_path / "meta")
    legacy_seed = tmp_path / "legacy_seed"
    legacy_seed.mkdir()
    for name in LEGACY_EDITABLE_FILES:
        (legacy_seed / name).write_bytes((SEED / name).read_bytes())
    result.initialize(legacy_seed, "a" * 40, "b" * 64)
    return result


def test_bundle_changes_actual_context_and_workflow(store):
    old = store.active()
    first, _ = old.render("A" * 5000 + "Z" * 5000, "schema")
    new = store.commit_update(old.hash, instruction_text="Updated Meta instructions", file_updates={
        "context.json": json.dumps({"max_evidence_chars": 4000, "selection": "tail"}),
        "workflow.json": json.dumps({"section_order": ["evidence", "response_contract", "instructions"], "checklist": ["Inspect negative outcomes"]})})
    second, load = store.active().render("A" * 5000 + "Z" * 5000, "schema")
    assert new.hash != old.hash and new.manifest["parent_hash"] == old.hash
    assert "AAAA" in first and "AAAA" not in second
    assert second.index("[evidence]") < second.index("[instructions]")
    assert load["selected_chars"] == 4000 and load["runtime_verified"] is False
    assert old.verify().version == 0


@pytest.mark.parametrize("filename", ["../controller.py", "config.toml", "runtime.py", "AGENTS.md", "manifest.json"])
def test_bundle_rejects_protected_changes(store, filename):
    old = store.active()
    with pytest.raises(ValueError):
        store.commit_update(old.hash, file_updates={filename: "bad"})
    assert store.active().hash == old.hash


def test_bundle_rejects_stale_commit(store):
    old = store.active()
    store.commit_update(old.hash, instruction_text="Updated instructions")
    with pytest.raises(ValueError, match="Stale"):
        store.commit_update(old.hash, instruction_text="replayed response")


@pytest.mark.parametrize("files", [{"context.json": '{"max_evidence_chars":999999}'},
                                  {"context.json": '{"selection":"execute_python"}'},
                                  {"workflow.json": '{"section_order":["evidence"]}'},
                                  {"workflow.json": '{"permissions":"root"}'}])
def test_bundle_invalid_update_is_atomic(store, files):
    old = store.active()
    with pytest.raises(ValueError):
        store.commit_update(old.hash, file_updates=files)
    assert store.active().hash == old.hash


def test_bundle_detects_mutation_and_unsupported_source(store):
    bundle = store.active()
    assert bundle.manifest["capabilities"]["runtime_source_edit"] is False
    (bundle.path / "instructions.md").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        store.active()


def test_bundle_initialization_is_idempotent(store):
    assert store.initialize(SEED).hash == store.active().hash
