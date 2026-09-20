"""Characterization: lock the run_evaluation 'warning' outcome (results.json not
created). The skipped/success/error/timeout outcomes are covered in
test_run_evaluation.py.
"""

from unittest.mock import MagicMock, patch

from sia.orchestrator import run_evaluation


def _make_task_with_eval(task_dir):
    pub = task_dir / "data" / "public"
    pub.mkdir(parents=True)
    (pub / "evaluate.py").write_text("pass")


@patch("sia.orchestrator.subprocess.run")
def test_warning_when_results_json_missing(mock_run, tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    _make_task_with_eval(tmp_path / "task")

    # exit 0 but evaluate.py never wrote results.json
    mock_run.return_value = MagicMock(returncode=0, stdout="done, no results written", stderr="")

    result = run_evaluation(str(gen_dir), str(tmp_path / "task"), "/fake/venv")
    assert result["status"] == "warning"
    assert result["reason"] == "results.json not created by evaluate.py"
