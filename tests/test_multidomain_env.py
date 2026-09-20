import json

import pytest

from sia.task_meta.data import TaskRecord
from sia.task_meta.environments import SearchQAAdapter, TACOAdapter, _stdin_match, _structured_match, answer_scores
from sia.task_meta.retrieval import FrozenSearchIndex, build_search_index, context_passages
from sia.task_meta.sandbox import SandboxResult


def task(domain, payload, split="evolve_train", source="fixture"):
    return TaskRecord("fixture:1", domain, source, split, "Find the answer", payload)


def test_real_fts_evidence_excludes_label_fields_and_freezes(tmp_path):
    record = task("searchqa", {"answer": "UNIQUE_LABEL_SECRET", "supporting_facts": "UNIQUE_SUPPORT_SECRET",
                  "question": "UNIQUE_QUESTION_SECRET", "context": {"title": ["Paris"], "sentences": [["Paris is the capital of France."]]}})
    path = build_search_index([record], tmp_path / "corpus.sqlite", data_manifest_hash="fixture")
    index = FrozenSearchIndex(path)
    adapter = SearchQAAdapter(index)
    prompt = adapter.reset(record, "r0", 42)
    assert prompt["retrieval_protocol"] == "open_retrieval"
    results = adapter.step("search", {"query": "capital France"})["results"]
    assert results and results[0]["text"] == "Paris is the capital of France."
    assert results[0]["corpus_hash"] == index.sha256
    assert not index.search("UNIQUE_LABEL_SECRET")
    assert not index.search("UNIQUE_SUPPORT_SECRET")
    assert not index.search("UNIQUE_QUESTION_SECRET")
    assert adapter.evaluate("wrong").reward == 0.0
    assert adapter.evaluate("UNIQUE_LABEL_SECRET").verification["exact_match"] is True
    with pytest.raises(ValueError):
        adapter.step("search", {"query": "France", "top_k": 100})
    with pytest.raises(ValueError, match="hash"):
        FrozenSearchIndex(path, expected_sha256="bad")
    index.close()


def test_corpus_rejects_dev_and_parses_2wiki(tmp_path):
    payload = {"context": json.dumps([["Title", ["First.", "Second."]]]), "answer": "hidden"}
    assert list(context_passages(payload)) == [("Title", "First. Second.")]
    with pytest.raises(ValueError, match="evolve_train"):
        build_search_index([task("searchqa", payload, "search_dev")], tmp_path / "bad.sqlite", data_manifest_hash="fixture")


def test_search_em_f1_success_is_full_answer_not_partial_reward():
    assert answer_scores("The Eiffel Tower", ["Eiffel Tower"])[0]
    exact, f1 = answer_scores("Tower", ["Eiffel Tower"])
    assert not exact and f1 > 0
    assert answer_scores("yes indeed", ["yes"]) == (False, 0.0)


def test_integer_verifier_does_not_accept_relative_tolerance_off_by_one():
    assert not _stdin_match("1000000000", "1000000001")
    assert _stdin_match("1.00000001", "1.0")
    assert _structured_match(10**1000, 10**1000)
    assert not _structured_match(10**1000, 10**1000 + 1)


class ExplicitFakeSandbox:
    """Offline protocol replay only; actual isolation is tested separately on Linux."""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def validate(self):
        return {"decision_source": "mock"}

    def run(self, code, stdin="", **kwargs):
        self.calls.append((code, stdin, kwargs))
        return SandboxResult(next(self.outputs), "", 0, False, 0.01, backend="test_override")


def test_code_hidden_expected_stays_in_trusted_controller_and_all_cases_run():
    sandbox = ExplicitFakeSandbox(["3\n", "WRONG\n", "8\n"])
    adapter = TACOAdapter(sandbox)
    problem = task("code", {"tests": json.dumps({"inputs": ["1 2\n", "2 3\n", "4 4\n"], "outputs": ["3\n", "5\n", "8\n"]}), "solutions": ["SECRET_SOLUTION"]})
    public = adapter.reset(problem, "rollout0", 42)
    assert "tests" not in public and "solutions" not in public
    result = adapter.evaluate("print(sum(map(int,input().split())))")
    assert result.reward == 0 and result.verification["full_verifier"]
    assert result.verification["tests_completed"] == 3
    assert result.metrics["test_pass_fraction"] == pytest.approx(2 / 3)
    assert len(sandbox.calls) == 3
    assert all("SECRET_SOLUTION" not in repr(call) for call in sandbox.calls)
    assert sandbox.calls[0][1] == "1 2\n"
    assert "outputs" not in repr(sandbox.calls)


def test_code_function_entry_and_invalid_fixture_are_explicit():
    sandbox = ExplicitFakeSandbox(["[3, 4]\n"])
    adapter = TACOAdapter(sandbox)
    public = adapter.reset(task("code", {"tests": json.dumps({"fn_name": "increment", "inputs": ["[2, 3]"], "outputs": ["[3, 4]"]})}), "r0", 42)
    assert public["program_entry"] == {"mode": "function", "function_name": "increment"}
    assert "[2, 3]" not in json.dumps(public)
    result = adapter.evaluate("def increment(xs): return [x+1 for x in xs]")
    assert result.verification["success"]
    spec = json.loads(sandbox.calls[0][1])
    assert spec == {"fn_name": "increment", "args": [[2, 3]]}
    adapter.reset(task("code", {"tests": "{}"}), "r1", 42)
    assert adapter.evaluate("print(1)").infrastructure_error


def test_taco_stdio_list_of_lines_is_not_misread_as_function_arguments():
    sandbox = ExplicitFakeSandbox(["2\n2\n0\n8\n"])
    adapter = TACOAdapter(sandbox)
    inputs = ["5", "4 1 2 3 4", "4", "3", "4", "6", "1", "", ""]
    adapter.reset(task("code", {"tests": json.dumps({"inputs": [inputs], "outputs": [["2", "2", "0", "8"]]})}), "r0", 42)
    result = adapter.evaluate("print('fixture-only program')")
    assert result.verification["full_verifier"] and result.verification["success"]
    assert sandbox.calls[0][1] == "\n".join(inputs) + "\n"
    assert "extra_files" not in sandbox.calls[0][2]
