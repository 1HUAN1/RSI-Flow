"""CPU-only contract tests for the real SFT adapter; these do not claim training."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "task_meta_sft", Path(__file__).resolve().parents[1] / "scripts/train_task_meta_sft.py"
)
sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sft)


class Tokenizer:
    def __init__(self, prefix=(11, 12, 13, 14), full=(11, 12, 13, 14, 21, 22)):
        self.prefix, self.full = list(prefix), list(full)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize
        assert return_dict is False
        return self.prefix if add_generation_prompt else self.full


class SFTContractTests(unittest.TestCase):
    def test_only_final_assistant_tokens_receive_loss(self):
        messages = [
            {"role": "user", "content": "first prompt"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "final prompt"},
            {"role": "assistant", "content": "recorded answer"},
        ]
        encoded = sft.encode_positive(Tokenizer(), messages, 128)
        self.assertEqual(encoded["input_ids"], [11, 12, 13, 14, 21, 22])
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, 21, 22])
        self.assertEqual(encoded["attention_mask"], [1] * 6)

    def test_ambiguous_assistant_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            sft.encode_positive(Tokenizer(prefix=(11, 99)), [{}, {}], 128)

    def test_overlong_rewarded_response_is_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            sft.encode_positive(Tokenizer(), [{}, {}], 5)

    def test_zero_positive_data_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sft_positive.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No positive"):
                sft.read_positive_rows(root, {})

    def test_nonpositive_or_nonfinite_rewards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for reward in (0, -1, float("nan"), True):
                with self.subTest(reward=reward):
                    (root / "sft_positive.jsonl").write_text(
                        json.dumps({"terminal_reward": reward}) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "finite positive"):
                        sft.read_positive_rows(root, {})

    def test_input_file_cannot_escape_request_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside.jsonl").write_text("", encoding="utf-8")
            request_dir = root / "request"
            request_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "inside"):
                sft.read_positive_rows(request_dir, {"sft_positive": "../outside.jsonl"})

    def test_checkpoint_must_be_ready_and_listed(self):
        checkpoint = Path(__file__).resolve().parent / "checkpoint"
        model_ref = str(checkpoint)
        calls = []

        def request(base_url, suffix, **kwargs):
            calls.append(suffix)
            if suffix == "local/checkpoints":
                self.assertEqual(kwargs["payload"], {"model_ref": model_ref, "checkpoint_path": model_ref})
                return {"ready": True, "model_ref": model_ref}
            return {"data": [{"id": model_ref}]}

        manifest = sft.register_checkpoint("http://127.0.0.1:8001/v1", checkpoint, request)
        self.assertEqual(calls, ["local/checkpoints", "models"])
        self.assertEqual(manifest["model_ref"], model_ref)

    def test_missing_served_checkpoint_is_not_completed(self):
        checkpoint = Path(__file__).resolve().parent / "checkpoint"

        def request(base_url, suffix, **kwargs):
            if suffix == "local/checkpoints":
                return {"ready": True, "model_ref": str(checkpoint)}
            return {"data": []}

        with self.assertRaisesRegex(RuntimeError, "absent"):
            sft.register_checkpoint("http://127.0.0.1:8001/v1", checkpoint, request)

    def test_remote_registration_url_is_rejected(self):
        for url in ("https://example.com/v1", "http://name:token@localhost/v1", "file:///tmp/v1"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "loopback"):
                sft.local_api_url(url, "models")

    def test_training_starts_from_current_checkpoint_not_initial_model_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            current.mkdir()
            (current / "config.json").write_text("{}")
            request = {"model_ref": "serving-id", "checkpoint_path": str(current)}
            self.assertEqual(sft.resolve_training_base(request), current.resolve())
            with self.assertRaisesRegex(ValueError, "cannot replace"):
                sft.resolve_training_base(request, root / "initial")

    def test_eight_steps_do_not_claim_all_sixteen_training_samples_seen(self):
        rows = [{"question_id": index // 8, "rollout_id": index % 8} for index in range(16)]
        encoded = [{"labels": [-100, index, index]} for index in range(16)]
        sampled = [2, 5, 1, 7, 13, 12, 0, 9]
        report = sft.sampling_report(rows, encoded, sampled)
        self.assertEqual(report["available_positive_samples"], 16)
        self.assertEqual(report["sampled_training_examples"], 8)
        self.assertEqual(report["unique_sampled_training_examples"], 8)
        self.assertEqual(report["sampled_supervised_tokens"], 16)
        self.assertEqual(report["available_supervised_tokens"], 32)
        self.assertFalse(report["all_available_samples_seen"])
        self.assertEqual(report["sampled_trajectory_ids"][4], {"question_id": 1, "rollout_id": 5})

    def test_repeated_training_examples_are_recorded_separately(self):
        rows = [{"question_id": 1, "rollout_id": 0}, {"question_id": 1, "rollout_id": 1}]
        report = sft.sampling_report(rows, [{"labels": [-100, 1]}, {"labels": [-100, 2, 3]}], [0, 1, 0])
        self.assertEqual(report["sampled_training_examples"], 3)
        self.assertEqual(report["unique_sampled_training_examples"], 2)
        self.assertEqual(report["sampled_supervised_tokens"], 4)
        self.assertEqual(report["unique_sampled_supervised_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
