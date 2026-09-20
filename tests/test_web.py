"""Tests for the runs-visualizer data layer and HTTP API."""

import json
from pathlib import Path

import pytest

from sia.web import runs as rd


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """A minimal but realistic runs/ tree: one run, two generations."""
    root = tmp_path / "runs"
    gen1 = root / "run_7" / "gen_1"
    gen2 = root / "run_7" / "gen_2"
    (gen1 / "agent_execution").mkdir(parents=True)
    gen2.mkdir(parents=True)

    (root / "run_7" / "context.md").write_text(
        "# Run Context: run_7\n\n"
        "**Task**: /tasks/gpqa\n"
        "**Meta Model**: kimi\n"
        "**Task Model**: haiku\n"
        "**Agent impl**: openhands\n"
        "**Started**: 2026-06-05 13:31:32\n"
        "**Max Generations**: 3\n\n"
        "---\n\n## Generation 1\n**Status**: ok\n",
        encoding="utf-8",
    )

    (gen1 / "target_agent.py").write_text("print('hello')\n", encoding="utf-8")
    (gen1 / "meta_agent_prompt.txt").write_text("meta prompt body", encoding="utf-8")
    (gen1 / "evaluation_results.json").write_text(
        json.dumps(
            {
                "total_questions": 4,
                "correct": 2,
                "incorrect": 2,
                "accuracy": 0.5,
                "accuracy_percent": 50.0,
                "details": [
                    {"question_id": 1, "domain": "Physics", "is_correct": True},
                    {"question_id": 2, "domain": "Physics", "is_correct": False},
                    {"question_id": 3, "domain": "Biology", "is_correct": True},
                    {"question_id": 4, "domain": "Biology", "is_correct": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    (gen1 / "agent_execution" / "execution_q1.json").write_text(
        json.dumps(
            [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert."}]},
                {"role": "user", "content": "Question 1?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Answer: A"}]},
            ]
        ),
        encoding="utf-8",
    )

    (gen2 / "improvement.md").write_text("# Plan\n- do better\n", encoding="utf-8")
    return root


def test_list_runs_summary(runs_root):
    runs = rd.list_runs(runs_root)
    assert len(runs) == 1
    r = runs[0]
    assert r.name == "run_7"
    assert r.agent_impl == "openhands"
    assert r.task_model == "haiku"
    assert r.max_generations == 3
    assert r.num_generations == 2
    assert r.best_accuracy_percent == 50.0


def test_get_run_detail_and_domains(runs_root):
    detail = rd.get_run(runs_root, "run_7")
    assert detail is not None
    assert detail.context_md is not None
    assert detail.context_md.startswith("# Run Context")
    gen1 = next(g for g in detail.generations if g.name == "gen_1")
    assert gen1.eval is not None
    assert gen1.eval.accuracy_percent == 50.0
    assert "target_agent" in gen1.artifacts
    assert "meta_prompt" in gen1.artifacts
    assert gen1.trajectories == [1]
    domains = {d.domain: d for d in gen1.domains}
    assert domains["Physics"].correct == 1 and domains["Physics"].total == 2
    assert domains["Biology"].accuracy_percent == 50.0


def test_eval_details_and_artifacts(runs_root):
    details = rd.get_eval_details(runs_root, "run_7", "gen_1")
    assert details is not None and len(details) == 4
    assert rd.get_artifact_text(runs_root, "run_7", "gen_1", "target_agent") == "print('hello')\n"
    improvement = rd.get_artifact_text(runs_root, "run_7", "gen_2", "improvement")
    assert improvement is not None and improvement.startswith("# Plan")


def test_trajectory_normalization(runs_root):
    turns = rd.get_trajectory(runs_root, "run_7", "gen_1", 1)
    assert turns is not None
    assert [t["role"] for t in turns] == ["system", "user", "assistant"]
    assert turns[0]["text"] == "You are an expert."
    assert turns[1]["text"] == "Question 1?"
    assert turns[2]["text"] == "Answer: A"


def test_missing_lookups_return_none(runs_root):
    assert rd.get_run(runs_root, "run_999") is None
    assert rd.get_trajectory(runs_root, "run_7", "gen_1", 999) is None
    assert rd.get_artifact_text(runs_root, "run_7", "gen_1", "nope") is None


@pytest.mark.parametrize("evil", ["..", "../etc", "run_7/../run_7", "foo/bar", ".", "/abs"])
def test_path_traversal_is_blocked(runs_root, evil):
    assert rd.get_run(runs_root, evil) is None
    assert rd._resolve_gen(runs_root, evil, "gen_1") is None
    assert rd._resolve_gen(runs_root, "run_7", evil) is None


def test_api_endpoints(runs_root):
    from fastapi.testclient import TestClient

    from sia.web import create_app

    client = TestClient(create_app(runs_root))

    assert client.get("/api/runs").json()[0]["name"] == "run_7"
    assert client.get("/api/runs/run_7").json()["agent_impl"] == "openhands"
    assert len(client.get("/api/runs/run_7/gens/gen_1/eval").json()) == 4
    assert "hello" in client.get("/api/runs/run_7/gens/gen_1/artifact/target_agent").text
    assert client.get("/api/runs/run_7/gens/gen_1/trajectory/1").json()[0]["role"] == "system"
    assert client.get("/api/runs/run_404").status_code == 404
    assert client.get("/").status_code == 200
