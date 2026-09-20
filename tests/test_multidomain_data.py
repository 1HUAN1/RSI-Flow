import json

import pytest

from sia.task_meta.data import DOMAINS, ManifestStore, audit_code_tool_leakage, build_manifest


def _catalog(tmp_path):
    entries = []

    def add(name, domain, role, rows):
        path = tmp_path / f"{len(entries)}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        entries.append({"dataset_name": name, "domain": domain, "role": role,
                        "local_path": str(path), "HF_revision_or_git_commit": "fixture-v1"})

    add("EnvScaler scenarios", "tool_use", "train", [{"task_id": f"t{i}", "canonical_env_id": "env0", "env_id": "env0_rl", "task": f"train tool {i}"} for i in range(5)])
    add("EnvScaler scenarios", "tool_use", "validation", [{"task_id": "v0", "canonical_env_id": "env1", "env_id": "env1_rl", "task": "validation tool"}])
    add("DeepCoder TACO", "code", "train", [{"problem": f"code {i}", "tests": "hidden", "solutions": ["secret"]} for i in range(7)])
    add("DeepCoder TACO", "code", "validation", [{"problem": "validation code", "tests": "hidden"}])
    add("NQ Open", "searchqa", "train", [{"question": f"question {i}", "answer": [f"answer {i}"]} for i in range(100)])
    add("HotpotQA train", "searchqa", "train", [{"id": "dupe", "question": " QUESTION   4 ", "answer": "hidden"}])
    add("HotpotQA dev", "searchqa", "final_test", [{"question": "question 0", "answer": "FINAL_SECRET"}])
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps({"datasets": entries}), encoding="utf-8")
    return path, entries


def test_preserved_splits_dedup_and_exhaustive_windows(tmp_path):
    catalog, _ = _catalog(tmp_path)
    path = build_manifest(catalog, tmp_path / "prepared", probe_per_domain=dict.fromkeys(DOMAINS, 1))
    store = ManifestStore(path)
    counts = store.counts()
    assert counts["tool_use"] == {"evolve_train": 5, "search_dev": 1, "probe": 1}
    assert counts["code"] == {"evolve_train": 7, "search_dev": 1, "probe": 1}
    assert sum(counts["searchqa"][key] for key in ("evolve_train", "search_dev")) == 99
    cursor, seen = {}, set()
    while True:
        records, following = store.window(cursor, dict.fromkeys(DOMAINS, 3))
        if not records:
            break
        assert not seen.intersection(task.task_id for task in records)
        assert all(task.split == "evolve_train" for task in records)
        seen.update(task.task_id for task in records)
        cursor = following
    assert store.coverage(cursor)["all_tasks_scheduled"]
    probe = store.probe()
    assert len(probe) == 3 and not seen.intersection(task.task_id for task in probe)
    assert {row["reason"] for row in store.omissions()} == {"duplicate_question_content", "final_question_overlap"}
    assert all("answer" not in task.public_payload() and "tests" not in task.public_payload() for task in probe)
    store.validate_sources()
    store.close()


def test_manifest_resume_same_window_and_frozen_source(tmp_path):
    catalog, entries = _catalog(tmp_path)
    path = build_manifest(catalog, tmp_path / "prepared", probe_per_domain=dict.fromkeys(DOMAINS, 1))
    first = ManifestStore(path)
    initial, cursor = first.window({}, dict.fromkeys(DOMAINS, 2))
    expected, following = first.window(cursor, dict.fromkeys(DOMAINS, 2))
    first.close()
    resumed = ManifestStore(path)
    actual, actual_cursor = resumed.window(cursor, dict.fromkeys(DOMAINS, 2))
    assert [t.task_id for t in actual] == [t.task_id for t in expected]
    assert following == actual_cursor
    assert not set(t.task_id for t in initial).intersection(t.task_id for t in actual)
    from pathlib import Path
    Path(entries[0]["local_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen data source changed"):
        resumed.validate_sources()
    resumed.close()


def test_refuses_tool_split_leakage_and_no_silent_resplit(tmp_path):
    catalog, entries = _catalog(tmp_path)
    from pathlib import Path
    heldout = Path(entries[1]["local_path"])
    row = json.loads(heldout.read_text())
    row["canonical_env_id"] = "env0"
    heldout.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split leakage"):
        build_manifest(catalog, tmp_path / "prepared", probe_per_domain=dict.fromkeys(DOMAINS, 1))


def test_immutable_manifest_and_bounds(tmp_path):
    catalog, _ = _catalog(tmp_path)
    path = build_manifest(catalog, tmp_path / "prepared", probe_per_domain=dict.fromkeys(DOMAINS, 1))
    with pytest.raises(FileExistsError):
        build_manifest(catalog, tmp_path / "prepared")
    store = ManifestStore(path)
    with pytest.raises(ValueError):
        store.window({}, {"code": 0})
    with pytest.raises(ValueError):
        list(store.iter_split("report_eval"))
    store.close()


def test_independent_final_audit_reads_public_prompts_and_exports_no_labels(tmp_path):
    catalog, entries = _catalog(tmp_path)
    for name, domain, row in [
        ("HumanEval+", "code", {"prompt": "code 1", "canonical_solution": "DO_NOT_EXPORT"}),
        ("BFCL V3", "tool_use", {"question": [[{"role": "user", "content": "train tool 1"}]], "answer": "DO_NOT_EXPORT"}),
    ]:
        path = tmp_path / f"final_{domain}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        entries.append({"dataset_name": name, "domain": domain, "role": "final_test", "local_path": str(path)})
    catalog.write_text(json.dumps({"datasets": entries}), encoding="utf-8")
    result = audit_code_tool_leakage(catalog)
    assert result["status"] == "failed_overlap"
    assert len(result["exact_cross_final_matches"]) == 2
    assert "DO_NOT_EXPORT" not in json.dumps(result)
    assert result["final_answers_exported"] is False
