"""Offline regressions for native Codex candidate-file delivery."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from sia.task_meta.meta_backends import CodexOpenRouterBackend, MetaBackendConfig
from sia.task_meta.meta_backends.codex_openrouter import candidate_file_delivery
from sia.task_meta.meta_harness import MetaHarnessStore


SEED = Path(__file__).resolve().parents[1] / "runtime/meta_harness/seed"


class Answer(BaseModel):
    answer: int


class CandidateAttestationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        store = MetaHarnessStore(root / "meta")
        store.initialize(SEED)
        self.backend = CodexOpenRouterBackend(
            MetaBackendConfig(g_output_delivery="candidate_file_all", compatibility_mode="in_run"),
            root / "meta", store)
        self.backend.bind_context("offline_attestation", 1, "a" * 64)
        self.prepared = self.backend.prepare("Offline candidate fixture", Answer, operation="model_request")
        self.work = self.prepared.directory / "workspace"

    def deliver(self, *, response="valid", candidate=None, tool_ok=True, add_extra=False):
        request = self.prepared.request
        envelope = candidate or {**request.identity(), "result": {"answer": 7}}
        encoded = json.dumps(envelope, sort_keys=True).encode()
        (self.work / ".meta_candidate.json").write_bytes(encoded)
        receipt = {"request_id": request.request_id, "candidate_file": ".meta_candidate.json",
                   "candidate_sha256": hashlib.sha256(encoded).hexdigest()}
        if response == "valid":
            (self.work / ".meta_response.json").write_text(json.dumps(receipt))
        elif response == "truncated":
            (self.work / ".meta_response.json").write_text(json.dumps(receipt)[9:])
        elif response == "wrong_hash":
            (self.work / ".meta_response.json").write_text(json.dumps({**receipt, "candidate_sha256": "0" * 64}))
        elif response == "missing":
            pass
        else:
            raise AssertionError(response)
        if add_extra:
            (self.work / "unexpected.txt").write_text("outside allowlist")
        events = [{"type": "item.completed", "item": {"type": "command_execution",
                    "status": "completed" if tool_ok else "failed",
                    "exit_code": 0 if tool_ok else 1}},
                  {"type": "turn.completed"}]
        (self.prepared.directory / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events))
        return {"returncode": 0, "transport": [{"ordinal": 0, "returned_model":
                self.backend.config.model, "completed": True}]}

    def test_candidate_file_is_authoritative(self):
        result = self.backend.collect(self.prepared, self.deliver())
        self.assertEqual(result.answer, 7)
        report = json.loads((self.prepared.directory / "candidate_delivery.json").read_text())
        self.assertFalse(report["final_message_used"])
        self.assertEqual(report["source"], "isolated_codex_workspace_file")

    def test_provider_prefix_loss_does_not_discard_valid_candidate(self):
        result = self.backend.collect(self.prepared, self.deliver(response="truncated"))
        self.assertEqual(result.answer, 7)
        report = json.loads((self.prepared.directory / "candidate_delivery.json").read_text())
        self.assertFalse(report["final_message_used"])

    def test_missing_or_wrong_final_receipt_is_advisory(self):
        for response in ("missing", "wrong_hash"):
            with self.subTest(response=response):
                with self.make_fresh() as pair:
                    backend, prepared, work = pair
                    envelope = {**prepared.request.identity(), "result": {"answer": 7}}
                    data = json.dumps(envelope).encode()
                    (work / ".meta_candidate.json").write_bytes(data)
                    if response == "wrong_hash":
                        (work / ".meta_response.json").write_text(json.dumps({
                            "request_id": prepared.request.request_id,
                            "candidate_file": ".meta_candidate.json",
                            "candidate_sha256": "0" * 64}))
                    (prepared.directory / "events.jsonl").write_text(json.dumps({
                        "type": "item.completed", "item": {"type": "command_execution",
                        "status": "completed", "exit_code": 0}}) + "\n" +
                        json.dumps({"type": "turn.completed"}))
                    result = backend.collect(prepared, {"returncode": 0, "transport": [{
                        "ordinal": 0, "returned_model": backend.config.model, "completed": True}]})
                    self.assertEqual(result.answer, 7)

    def make_fresh(self):
        from contextlib import contextmanager

        @contextmanager
        def fixture():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                store = MetaHarnessStore(root / "meta")
                store.initialize(SEED)
                backend = CodexOpenRouterBackend(
                    MetaBackendConfig(g_output_delivery="candidate_file_all",
                                      compatibility_mode="in_run"), root / "meta", store)
                backend.bind_context("offline_attestation", 1, "a" * 64)
                prepared = backend.prepare("Offline candidate fixture", Answer, operation="model_request")
                yield backend, prepared, prepared.directory / "workspace"
        return fixture()

    def test_model_echoed_identity_is_advisory(self):
        candidate = {**self.prepared.request.identity(), "result": {"answer": 7}}
        candidate["request_id"] = "b" * 32
        candidate["generation"] = -1
        self.assertEqual(self.backend.collect(self.prepared, self.deliver(candidate=candidate)).answer, 7)

    def test_tool_event_is_not_an_extra_candidate_gate(self):
        result = self.backend.collect(self.prepared, self.deliver(tool_ok=False))
        self.assertEqual(result.answer, 7)

    def test_native_file_change_event_is_accepted(self):
        result = self.deliver(response="truncated")
        (self.prepared.directory / "events.jsonl").write_text("\n".join(json.dumps(event) for event in [
            {"type": "item.completed", "item": {"type": "file_change", "status": "completed"}},
            {"type": "turn.completed"}]))
        self.assertEqual(self.backend.collect(self.prepared, result).answer, 7)

    def test_invalid_candidate_schema_remains_rejected(self):
        candidate = {**self.prepared.request.identity(), "result": {"wrong_field": 7}}
        with self.assertRaisesRegex(ValueError, "answer"):
            self.backend.collect(self.prepared, self.deliver(candidate=candidate))

    def test_provider_model_identity_remains_required(self):
        result = self.deliver(response="truncated")
        result["transport"][0]["returned_model"] = "different-model"
        with self.assertRaisesRegex(ValueError, "model/completion"):
            self.backend.collect(self.prepared, result)

    def test_other_workspace_changes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.backend.collect(self.prepared, self.deliver(add_extra=True))

    def test_candidate_must_be_regular_and_has_recorded_digest(self):
        path = self.work / ".meta_candidate.json"
        with self.assertRaisesRegex(ValueError, "absent"):
            candidate_file_delivery(path)
        path.write_text("{}")
        content, report = candidate_file_delivery(path)
        self.assertEqual(content, b"{}")
        self.assertEqual(report["candidate_sha256"], hashlib.sha256(content).hexdigest())
        path.unlink()
        path.symlink_to(self.prepared.directory / "request.json")
        with self.assertRaisesRegex(ValueError, "regular"):
            candidate_file_delivery(path)


if __name__ == "__main__":
    unittest.main()
