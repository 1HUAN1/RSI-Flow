"""Cross-round context reaches the next Meta prompt and native workspace."""
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sia.task_meta.meta import MetaAgent
from sia.task_meta.meta_harness.bundle import MetaHarnessStore
from sia.task_meta.meta_harness.runtime import execute
from sia.task_meta.observations import build_observation, operation_input, round_context
from sia.task_meta.types import (
    EvaluationResult, ImprovementExperience, MetaAgentState, MetaDecision, TaskAgentState,
)


ROOT = Path(__file__).resolve().parents[1]


def experience(generation=0, *, accepted=False, executed=True):
    gain = 0.1 if accepted else -0.1 if executed else None
    return ImprovementExperience(
        generation=generation, state_before={}, state_after={},
        decision={"action": "HARNESS", "diagnosis": "History may hide tool results",
                  "diagnosis_kind": "hypothesis", "rationale": "Test memory retention first",
                  "expected_effect": "More complete tool observations", "used_principle_ids": ["skill.HARNESS.1"]},
        modification={"summary": "Change history retention", "semantic_status": "unverified"},
        performance_before={"macro_success": 0.5}, performance_after={"macro_success": 0.5 + (gain or 0)},
        performance_delta=gain, cost_before={}, cost_after={}, update_cost={},
        trajectory_before="/recorded/parent.json", trajectory_after="/recorded/child.json" if executed else None,
        experience_id=f"experience_{generation}_{generation + 1}", attempted_component="HARNESS",
        attempted_performance_delta=gain,
        attempted_outcome={"paired_evidence_complete": executed},
        deployment_status="deployed" if accepted else "parent_retained",
        candidate_attempts=[{"candidate_executed": executed,
                             "outcome_class": "accepted_positive_gain" if accepted else "negative_gain" if executed else "candidate_unavailable",
                             "reason": None if executed else "Candidate did not build"}],
    )


class MetaRoundContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        harness = self.root / "harness.json"
        harness.write_text("{}")
        self.task = TaskAgentState(1, "task-model", str(harness))
        self.meta = MetaAgentState("meta-model", str(harness), version=1)

    def observation(self, history):
        with patch("sia.task_meta.observations.harness_evidence", return_value=({}, {})):
            return build_observation(self.task, self.task, self.meta, EvaluationResult({}, []),
                                     [], history, [], {}, retain_raw=True)

    def test_first_round_has_no_invented_context(self):
        self.assertEqual(self.observation([]).previous_round_context, {})
        self.assertEqual(round_context(None), {})

    def test_rejected_and_unexecuted_candidates_are_not_presented_as_deployed(self):
        rejected = round_context(experience())
        self.assertEqual(rejected["result"]["deployment_status"], "parent_retained")
        self.assertEqual(rejected["result"]["candidate_gain"], -0.1)
        self.assertEqual(rejected["prior_reasoning"]["diagnosis_kind"], "hypothesis")
        unavailable = round_context(experience(executed=False))
        self.assertFalse(unavailable["result"]["candidate_executed"])
        self.assertIsNone(unavailable["result"]["candidate_gain"])
        self.assertEqual(unavailable["result"]["failure_reason"], "Candidate did not build")

    def test_next_route_receives_previous_context_and_complete_history(self):
        history = [experience(), experience(1, accepted=True)]
        obs = self.observation(history)
        captured = []
        client = SimpleNamespace(supports_evolution=True,
                                 complete=lambda *args, **kwargs: captured.append(kwargs))
        MetaAgent(client, {}).diagnose_and_route(self.meta, obs)
        envelope = captured[0]["operation_input"]
        context = envelope["trusted_facts"]["previous_round_context"]
        self.assertEqual(context["source_round"], 2)
        self.assertEqual(context["result"]["deployment_status"], "deployed")
        self.assertEqual(context["prior_reasoning"]["rationale"], history[-1].decision["rationale"])
        self.assertEqual(len(envelope["experiences"]), 2)
        envelope["trusted_facts"]["previous_round_context"]["prior_reasoning"]["rationale"] = "changed"
        self.assertEqual(obs.previous_round_context["prior_reasoning"]["rationale"], history[-1].decision["rationale"])

    def test_resume_reconstructs_same_context_from_saved_experience(self):
        source = experience(1)
        recovered = ImprovementExperience(**json.loads(json.dumps(asdict(source))))
        self.assertEqual(round_context(source), self.observation([recovered]).previous_round_context)

    def test_native_stage_contains_readable_full_previous_experience(self):
        history = [experience()]
        envelope = operation_input(self.observation(history))
        original = copy.deepcopy(envelope)
        store = MetaHarnessStore(self.root / "store")
        bundle = store.initialize(ROOT / "runtime/meta_harness/seed", "pinned", "binary")
        calls = []

        class StageCaptured(Exception):
            pass

        def capture(stage_id, prompt, schema, evidence_files=None, allowed_paths=None):
            calls.append((prompt, evidence_files))
            raise StageCaptured()

        with self.assertRaises(StageCaptured):
            execute(bundle, "routing", envelope, MetaDecision, capture, self.root / "audit")
        prompt, files = calls[0]
        prior = json.loads(files["meta_input/previous_round_experience.json"])
        self.assertEqual(prior, asdict(history[-1]))
        self.assertIn("previous_round_context", prompt)
        payload = json.loads(files["meta_input/operation.json"])
        self.assertEqual(payload["previous_round_context_file"], "meta_input/previous_round_experience.json")
        self.assertEqual(envelope, original)


if __name__ == "__main__":
    unittest.main()
