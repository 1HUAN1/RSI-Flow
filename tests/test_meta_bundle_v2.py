"""Offline controller tests. All G edits below are engineer-written test_override.

These tests make no API/model calls and are not evidence of autonomous evolution.
"""
import json
from pathlib import Path

import pytest

from sia.task_meta.meta_harness import MetaHarnessStore
from sia.task_meta.meta_harness.bundle import (
    CONTRACT_VERSION,
    ENTRYPOINTS,
    LEGACY_EDITABLE_FILES,
    MECHANISM_LAYERS,
    MetaHarnessBundle,
    atomic_json,
    canonical,
    sha256,
)
from sia.task_meta.meta_harness.policies import default_evolution

SEED = Path(__file__).resolve().parents[1] / "meta_harness/seed"


@pytest.fixture
def store(tmp_path):
    store = MetaHarnessStore(tmp_path / "meta")
    store.initialize(SEED, "a" * 40, "b" * 64)
    return store


def legacy_seed(tmp_path):
    path = tmp_path / "legacy-seed"
    path.mkdir()
    for name in LEGACY_EDITABLE_FILES:
        (path / name).write_bytes((SEED / name).read_bytes())
    return path


def test_v2_manifest_declares_real_entrypoints_contract_and_source(store):
    current = store.active()
    assert current.schema_version == "meta-bundle-v2"
    assert current.execution_spec() == default_evolution()
    assert current.manifest["contract_version"] == CONTRACT_VERSION
    assert current.manifest["entrypoints"] == ENTRYPOINTS
    assert set(current.manifest["mechanism_layers"]) == set(MECHANISM_LAYERS)
    assert current.manifest["source_commit"] == "a" * 40
    assert current.manifest["binary_sha256"] == "b" * 64
    assert current.manifest["content_hash"] == current.content_hash
    assert current.manifest["provenance"]["origin"] == "seed"


def test_semantic_json_no_change_does_not_create_or_rewrite_version(store):
    old = store.active()
    before = {p.name: p.read_bytes() for p in old.path.iterdir()}
    updates = {name: json.dumps(json.loads(text), indent=4, sort_keys=False)
               for name, text in old.read_files().items() if name.endswith(".json")}
    result = store.commit_update(old.hash, file_updates=updates, request_id="test_override-nochange",
                                 experience_id="test_override-exp-1", phase="meta_self_update")
    assert result.hash == old.hash and result.path == old.path and result.version == 0
    assert before == {p.name: p.read_bytes() for p in old.path.iterdir()}
    assert len(list((store.root / "harness_versions").iterdir())) == 1
    event = json.loads(next((store.root / "commit_events").iterdir()).read_text())
    assert event["status"] == "NO_CHANGE" and event["input_bundle_hash"] == event["output_bundle_hash"]
    with pytest.raises(ValueError, match="Duplicate"):
        store.commit_update(old.hash, file_updates=updates, request_id="test_override-nochange")


def test_substantive_update_records_provenance_and_rejects_duplicate_request(store):
    old = store.active()
    new = store.commit_update(old.hash, instruction_text="test_override: inspect real failure evidence.",
                              request_id="test_override-request-1", experience_id="test_override-exp-1",
                              phase="meta_self_update", changed_mechanisms=["diagnosis"])
    assert new.version == 1 and new.hash != old.hash
    assert new.manifest["parent_hash"] == old.hash
    assert new.manifest["change_scope"] == ["instructions.md"]
    assert new.manifest["provenance"]["experience_id"] == "test_override-exp-1"
    assert new.manifest["provenance"]["claimed_changed_mechanisms"] == ["diagnosis"]
    assert old.verify().version == 0
    with pytest.raises(ValueError, match="Duplicate"):
        store.commit_update(new.hash, instruction_text="different candidate", request_id="test_override-request-1")
    with pytest.raises(ValueError, match="Stale"):
        store.commit_update(old.hash, instruction_text="stale candidate", request_id="test_override-request-2")


def test_duplicate_rejected_from_manifest_if_event_publication_crashed(store, monkeypatch):
    old = store.active()
    original = store._record_commit
    def crash(*args):
        raise RuntimeError("test_override: crash before event")
    monkeypatch.setattr(store, "_record_commit", crash)
    with pytest.raises(RuntimeError, match="before event"):
        store.commit_update(old.hash, instruction_text="Changed before event", request_id="test_override-crash")
    current = store.active()
    assert current.version == 1
    monkeypatch.setattr(store, "_record_commit", original)
    with pytest.raises(ValueError, match="Duplicate"):
        store.commit_update(current.hash, instruction_text="Replayed", request_id="test_override-crash")


def test_nochange_outer_receipt_gap_reconciles_without_new_commit(store):
    old = store.active()
    identity = {"request_id": "test_override-nochange-gap", "experience_id": "test_override-exp",
                "phase": "meta_self_update"}
    candidate = {"instruction_text": old.read_files()["instructions.md"] + "\n", **identity}
    assert store.reconcile_update(old.hash, **candidate) is None
    store.commit_update(old.hash, **candidate)
    # Simulated crash boundary: store event exists; outer acceptance is absent.
    before = {p.relative_to(store.root).as_posix(): p.read_bytes()
              for p in store.root.rglob("*") if p.is_file()}
    recovered = store.reconcile_update(old.hash, **candidate)
    assert recovered.hash == old.hash and recovered.version == old.version
    assert before == {p.relative_to(store.root).as_posix(): p.read_bytes()
                      for p in store.root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Duplicate"):
        store.commit_update(old.hash, **candidate)
    with pytest.raises(ValueError, match="event"):
        store.reconcile_update(old.hash, **{**candidate, "experience_id": "forged"})
    with pytest.raises(ValueError, match="content"):
        store.reconcile_update(old.hash, **{**candidate, "instruction_text": "Different instructions"})


@pytest.mark.parametrize("mismatch", ["request_id", "experience_id", "phase", "instruction_text"])
def test_same_child_content_cannot_be_adopted_for_another_request(store, mismatch):
    old = store.active()
    candidate = {"instruction_text": "test_override substantive change", "request_id": "test_override-child",
                 "experience_id": "test_override-exp", "phase": "meta_self_update"}
    new = store.commit_update(old.hash, **candidate)
    assert store.reconcile_update(old.hash, **candidate).hash == new.hash
    changed = {**candidate, mismatch: "another-request-or-content"}
    with pytest.raises(ValueError):
        store.reconcile_update(old.hash, **changed)
    assert store.active().hash == new.hash


def test_changed_child_reconciles_after_pointer_before_event_gap(store, monkeypatch):
    old = store.active()
    candidate = {"instruction_text": "test_override published child", "request_id": "test_override-before-event",
                 "experience_id": "test_override-exp", "phase": "meta_self_update"}
    def crash(*_):
        raise RuntimeError("test_override pointer published; event not yet written")
    monkeypatch.setattr(store, "_record_commit", crash)
    with pytest.raises(RuntimeError, match="pointer published"):
        store.commit_update(old.hash, **candidate)
    current = store.active()
    assert current.version == old.version + 1
    assert not (store.root / "commit_events").exists()
    assert store.reconcile_update(old.hash, **candidate).hash == current.hash
    assert not (store.root / "commit_events").exists()  # Reconciliation is read-only.
    with pytest.raises(ValueError, match="request/experience/phase"):
        store.reconcile_update(old.hash, **{**candidate, "request_id": "different"})


def test_v1_compatibility_and_explicit_migration_leave_old_snapshot_untouched(tmp_path):
    store = MetaHarnessStore(tmp_path / "legacy")
    old = store.initialize(legacy_seed(tmp_path))
    old_bytes = {p.name: p.read_bytes() for p in old.path.iterdir()}
    assert old.schema_version == "meta-bundle-v1" and old.execution_spec() is None
    assert store.initialize(SEED).hash == old.hash
    assert store.active().compatibility_mode == "legacy_v1_prompt_assembly"
    with pytest.raises(ValueError, match="explicit migration"):
        store.commit_update(old.hash, file_updates={"evolution.json": canonical(default_evolution()).decode()})
    new = store.migrate_to_v2(old.hash, request_id="test_override-explicit-migration")
    assert new.schema_version == "meta-bundle-v2" and new.manifest["parent_hash"] == old.hash
    assert new.manifest["provenance"]["source_hash"] == old.hash
    assert old_bytes == {p.name: p.read_bytes() for p in old.path.iterdir()}
    assert old.verify().execution_spec() is None


def test_markdown_and_v1_import_create_new_store_without_touching_history(tmp_path):
    markdown = tmp_path / "historical.md"
    markdown.write_text("Historical instructions", encoding="utf-8")
    new = MetaHarnessStore(tmp_path / "markdown-import").initialize_from_markdown(markdown)
    assert markdown.read_text() == "Historical instructions"
    assert new.manifest["provenance"]["source_hash"] == sha256(markdown.read_bytes())
    historical = MetaHarnessStore(tmp_path / "old")
    old = historical.initialize(legacy_seed(tmp_path))
    pointer_bytes = (historical.root / "active_harness.json").read_bytes()
    copied = MetaHarnessStore(tmp_path / "imported").initialize_from_v1(old)
    assert copied.version == 0 and copied.manifest["parent_hash"] is None
    assert copied.manifest["provenance"]["source_hash"] == old.hash
    assert pointer_bytes == (historical.root / "active_harness.json").read_bytes()
    assert historical.active().hash == old.hash


def test_manifest_file_tampering_and_extra_directories_rejected(store):
    current = store.active()
    manifest = current.path / "manifest.json"
    old_bytes = manifest.read_bytes()
    changed = json.loads(old_bytes)
    changed["runtime_validation"] = "forged"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="modified"):
        current.verify()
    manifest.write_bytes(old_bytes)
    (current.path / "undeclared").mkdir()
    with pytest.raises(ValueError, match="undeclared"):
        current.verify()


@pytest.mark.parametrize("directory", ["../outside", "/absolute", "v000_aaaaaaaaaaaa/../other", "v000_aaaaaaaaaaaa\\other"])
def test_active_pointer_rejects_escape_before_file_access(store, directory):
    atomic_json(store.root / "active_harness.json", {"directory": directory, "bundle_hash": store.active().hash})
    with pytest.raises(ValueError, match="Invalid active"):
        store.active()


@pytest.mark.parametrize("location", ["root", "ancestor", "snapshot"])
def test_symlink_roots_ancestors_and_snapshots_fail_closed(store, tmp_path, location):
    target = store.root if location != "snapshot" else store.active().path
    link = tmp_path / ("link-" + location)
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not permit this test user to create symlinks")
    with pytest.raises(ValueError, match="Symlink"):
        if location == "snapshot":
            MetaHarnessBundle(link, store.active().manifest).verify()
        else:
            MetaHarnessStore(link if location == "root" else link / "child")


@pytest.mark.parametrize("file,content", [
    ("context.json", '{"max_evidence_chars":4000,"max_evidence_chars":5000}'),
    ("context.json", '{"max_evidence_chars":true}'),
    ("evolution.json", '{"schema_version":1,"execute_python":"print(1)"}'),
    ("evolution.json", '{"schema_version":NaN}'),
])
def test_ambiguous_json_and_arbitrary_code_fail_atomically(store, file, content):
    old = store.active()
    with pytest.raises(ValueError):
        store.commit_update(old.hash, file_updates={file: content})
    assert store.active().hash == old.hash
    assert len(list((store.root / "harness_versions").iterdir())) == 1


@pytest.mark.parametrize("violation", ["loop_budget", "deep_condition", "external_tool"])
def test_invalid_executable_program_never_commits(store, violation):
    old = store.active()
    policy = old.execution_spec()
    if violation == "loop_budget":
        policy["workflows"]["routing"][-1]["max_attempts"] = 1000000
    elif violation == "deep_condition":
        condition = True
        for _ in range(10):
            condition = {"not": condition}
        policy["evidence"]["filter"] = condition
    else:
        policy["workflows"]["routing"].insert(0, {"id": "escape", "kind": "shell", "command": "read_credentials"})
    with pytest.raises(ValueError):
        store.commit_update(old.hash, file_updates={"evolution.json": json.dumps(policy)})
    assert store.active().hash == old.hash
    assert len(list((store.root / "harness_versions").iterdir())) == 1


def test_repeated_identical_compatibility_calls_return_same_bundle(store):
    old = store.active()
    assert store.commit_update(old.hash).hash == old.hash
    assert store.commit_update(old.hash, instruction_text=old.read_files()["instructions.md"]).hash == old.hash
    assert store.commit_update(old.hash, instruction_text=old.read_files()["instructions.md"] + "\n\n \t").hash == old.hash


def test_migration_cli_keeps_source_and_requires_new_destination(tmp_path, capsys):
    from scripts.migrate_meta_bundle import main
    source = tmp_path / "old.md"
    source.write_text("Historical Markdown G", encoding="utf-8")
    destination = tmp_path / "new-store"
    assert main(["markdown", "--source", str(source), "--destination", str(destination)]) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["status"] == "EXPLICIT_MIGRATION"
    assert source.read_text() == "Historical Markdown G"
    with pytest.raises(SystemExit):
        main(["markdown", "--source", str(source), "--destination", str(destination)])
