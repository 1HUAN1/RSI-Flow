import json
import sys
from pathlib import Path

import pytest

from sia.task_meta.reporting import (
    BENCHMARKS,
    OfficialEvaluatorSpec,
    ReportBlocked,
    evaluate_official,
    export_predictions,
    freeze_best_on_dev,
    official_command,
    parse_official_metrics,
    task_fingerprint,
    write_report_tables,
)
from sia.task_meta.storage import digest


@pytest.fixture
def frozen(tmp_path):
    harness = tmp_path / "task.py"
    harness.write_text("# immutable Task")
    return task_fingerprint({"generation": 1, "model_ref": "Qwen3-4B", "harness_path": str(harness),
                             "artifacts": {"directory": None, "manifest": []}, "checkpoint_path": None})


def row(task_id, frozen_hash):
    return {"task_id": task_id, "state_hash": frozen_hash, "split": "report_eval", "final_submission_count": 1,
            "final_answer": "answer"}


def spec(tmp_path, benchmark="HotpotQA-dev"):
    script = tmp_path / "official_fixture.py"
    script.write_text("print({'em': 0.5, 'f1': 0.75})")
    data = tmp_path / "gold.json"
    data.write_text('[{"_id":"a"},{"_id":"b"}]')
    ids = tmp_path / "ids.json"
    ids.write_text('["a","b"]')
    return OfficialEvaluatorSpec(benchmark, "test-contract-fixture", "a" * 40, str(script), digest(script), sys.executable,
        str(data), digest(data), str(ids), digest(ids), "answer_only_open_retrieval", release_version="v0.1.10")


def test_fixed_dev_selection_cannot_use_final_scores(tmp_path, frozen):
    records = [{"split": "search_dev", "probe_manifest_hash": "p", "status": "completed", "metric": "macro",
                "score": .5, "generation": 0, "state_id": "g0"}]
    output = freeze_best_on_dev(records, {"g0": frozen["state"]}, probe_manifest_hash="p", selection_metric="macro",
                               mode="max", destination=tmp_path / "frozen.json", protocol_hash="protocol")
    assert output["selection"]["selected"]["state_id"] == "g0"
    records[0]["split"] = "report_eval"
    with pytest.raises(ValueError, match="search_dev"):
        freeze_best_on_dev(records, {"g0": frozen["state"]}, probe_manifest_hash="p", selection_metric="macro",
                           mode="max", destination=tmp_path / "frozen.json", protocol_hash="protocol")


def test_partial_prediction_keeps_fixed_denominator(tmp_path):
    with pytest.raises(ReportBlocked, match="1/2"):
        export_predictions("HumanEval+", [row("a", "frozen")], ["a", "b"], tmp_path, "frozen")


def test_duplicate_and_best_of_k_predictions_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        export_predictions("HumanEval+", [row("a", "frozen")] * 2, ["a"], tmp_path, "frozen")
    invalid = row("a", "frozen")
    invalid["final_submission_count"] = 8
    with pytest.raises(ValueError, match="pass@1"):
        export_predictions("HumanEval+", [invalid], ["a"], tmp_path, "frozen")


def test_lcb_export_has_one_submission_per_question(tmp_path):
    output = export_predictions("LiveCodeBench", [row("a", "frozen")], ["a"], tmp_path, "frozen")
    assert json.loads(output.read_text()) == [{"question_id": "a", "code_list": ["answer"]}]


def test_code_evaluator_does_not_execute_without_worker_isolation(tmp_path, frozen):
    result = evaluate_official(spec(tmp_path, "HumanEval+"), [row("a", frozen["state_hash"]), row("b", frozen["state_hash"])],
                               frozen, tmp_path / "eval")
    assert result["status"] == "blocked" and result["metrics"] is None
    assert "worker" in result["reason"]
    assert not (tmp_path / "eval/stdout.txt").exists()


def test_official_search_script_invoked_with_no_meta_feedback(tmp_path, frozen):
    result = evaluate_official(spec(tmp_path), [row("a", frozen["state_hash"]), row("b", frozen["state_hash"])], frozen, tmp_path / "eval")
    assert result["status"] == "completed" and result["metrics"] == {"EM": .5, "F1": .75}
    assert result["feedback_to_meta"] is False


def test_frozen_harness_mutation_blocks_report(tmp_path, frozen):
    Path(frozen["state"]["harness_path"]).write_text("mutated")
    with pytest.raises(ReportBlocked, match="modified"):
        evaluate_official(spec(tmp_path), [], frozen, tmp_path / "eval")


def test_evalplus_parse_requires_base_and_plus_pass(tmp_path):
    (tmp_path / "official_results.json").write_text(json.dumps({"eval": {
        "a": [{"base_status": "pass", "plus_status": "fail"}], "b": [{"base_status": "pass", "plus_status": "pass"}]}}))
    metrics, _ = parse_official_metrics(spec(tmp_path, "HumanEval+"), tmp_path, "")
    assert metrics == {"system_pass@1": .5}


def test_all_seven_table_slots_are_na_and_no_fake_overall(tmp_path):
    csv_file, latex = write_report_tables(tmp_path, [], {"method": "RSI_Codex"})
    assert csv_file.read_text().count("N/A") == 7
    assert latex.read_text().count("N/A") == 7
    manifest = json.loads((tmp_path / "report_manifest.json").read_text())
    assert tuple(manifest["benchmarks"]) == BENCHMARKS and manifest["overall"] is None


def test_native_cli_uses_official_evaluator_and_explicit_version(tmp_path):
    command = official_command(spec(tmp_path, "livecodebench"), tmp_path / "samples.json", tmp_path)
    assert "lcb_runner.runner.custom_evaluator" in command
    assert command[command.index("--release_version") + 1] == "v0.1.10"
