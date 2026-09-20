"""CPU contract tests; synthetic tokenization is not a Qwen/GPU training claim."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from sia.task_meta.sft import (
    deduplicate_rows,
    encode_messages,
    pad_training_batch,
    select_positive_rows,
    validate_training_row,
)

SPEC = importlib.util.spec_from_file_location(
    "multidomain_trainer", Path(__file__).resolve().parents[1] / "scripts/train_task_meta_sft.py"
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def tool_dialogue():
    return [
        {"role": "system", "content": "Use the recorded tools."},
        {"role": "user", "content": "Find the capital."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query":"France"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Paris is the capital."},
        {"role": "assistant", "content": "Paris"},
    ]


def valid_row(domain="code"):
    return {"task_id": "task-1", "question_id": "task-1", "rollout_id": 0, "split": "evolve_train",
            "domain": domain, "terminal_reward": 1.0, "messages": tool_dialogue(),
            "verification": {"status": "completed", "verifier_id": "official-v1", "success": True,
                             "full_verifier": True, "task_success": True, "exact_match": 1.0}}


class NativeTokenizer:
    """Expose role/header/body boundaries and native tool calls for mask assertions."""

    headers: ClassVar[dict] = {"system": 100, "user": 200, "assistant": 300, "tool": 400}

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict, **kwargs):
        self.seen.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        result = [99]
        for message in messages:
            body = 31 if message.get("tool_calls") else {"system": 11, "user": 21, "assistant": 32, "tool": 41}[message["role"]]
            result.extend([self.headers[message["role"]], body, 9])
        if add_generation_prompt:
            result.append(300)
        return result


class MultidomainSelectionTests(unittest.TestCase):
    def test_each_domain_requires_complete_success_not_partial_positive_reward(self):
        for domain, field in (("code", "full_verifier"), ("tool_use", "task_success"), ("searchqa", "exact_match")):
            with self.subTest(domain=domain):
                row = valid_row(domain)
                self.assertEqual(select_positive_rows([row], profile="multidomain"), [row])
                row["verification"][field] = 0.5
                self.assertEqual(select_positive_rows([row], profile="multidomain"), [])

    def test_dev_and_report_samples_cannot_become_training_rows(self):
        for split in ("search_dev", "report_eval", "train", None):
            row = valid_row()
            row["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "evolve_train"):
                validate_training_row(row, "multidomain")

    def test_missing_failed_or_unfinished_verifiers_fail_closed(self):
        for record in (None, {}, {"success": True}, {"status": "incomplete", "success": True, "verifier_id": "v1"}):
            row = valid_row()
            row["verification"] = record
            self.assertEqual(select_positive_rows([row], profile="multidomain"), [])

    def test_native_calls_and_original_observations_are_preserved(self):
        row = valid_row()
        original = copy.deepcopy(row)
        result = select_positive_rows([row], profile="multidomain")
        self.assertEqual(result, [original])
        result[0]["messages"][3]["content"] = "changed copy"
        self.assertEqual(row, original)
        self.assertEqual(select_positive_rows([row]), [])  # Legacy remains text-only.

    def test_tool_id_must_bind_to_the_recorded_call(self):
        row = valid_row()
        row["messages"][3]["tool_call_id"] = "unrecorded"
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_training_row(row, "multidomain")

    def test_successful_rollout_expands_actual_calls_with_final_only_supervision(self):
        row = valid_row()
        row["sft_conversations"] = [
            {"call_id": 0, "operation": "planning", "messages": [
                {"role": "user", "content": "Plan the task"}, {"role": "assistant", "content": "Use search."}]},
            {"call_id": 1, "operation": "action", "messages": tool_dialogue()[:3]},
        ]
        selected = select_positive_rows([row], profile="multidomain")
        self.assertEqual(len(selected), 2)
        self.assertEqual([item["call_id"] for item in selected], [0, 1])
        self.assertTrue(all(item["sft_supervision"] == "final_assistant" for item in selected))
        self.assertEqual(selected[1]["messages"][-1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(selected[1]["original_task_id"], "task-1")
        row["split"] = "search_dev"
        self.assertEqual(select_positive_rows([row], profile="multidomain"), [])

    def test_successful_terminal_tool_observation_does_not_require_invented_assistant(self):
        row = valid_row("tool_use")
        row["messages"].pop()
        validate_training_row(row, "multidomain")
        encoded = encode_messages(NativeTokenizer(), row["messages"], 64, supervision="assistant_all")
        self.assertEqual([token for token in encoded["labels"] if token != -100], [31, 9])

    def test_dedup_has_source_evidence_and_keeps_raw_rows(self):
        row = valid_row()
        duplicate = copy.deepcopy(row)
        duplicate["rollout_id"] = 4
        unique, report = deduplicate_rows([row, duplicate])
        self.assertEqual(unique, [row])
        self.assertEqual(report["duplicates"][0]["rollout_id"], 4)
        self.assertEqual(report["duplicates"][0]["duplicate_of_row_index"], 0)
        self.assertEqual(duplicate["messages"], row["messages"])

    def test_trainer_revalidates_rows_and_limits_window_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = valid_row()
            (root / "sft_positive.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_samples"):
                trainer.read_positive_rows(root, {"sft_profile": "multidomain", "training": {"max_samples": 1}})
            row["split"] = "search_dev"
            (root / "sft_positive.jsonl").write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evolve_train"):
                trainer.read_positive_rows(root, {"sft_profile": "multidomain"})


class MultidomainMaskTests(unittest.TestCase):
    def test_all_assistant_calls_and_answers_receive_loss_but_no_tool_observations(self):
        tokenizer = NativeTokenizer()
        tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
        encoded = encode_messages(tokenizer, tool_dialogue(), 64, supervision="assistant_all", tools=tools,
                                   chat_template_kwargs={"enable_thinking": False})
        self.assertEqual([token for token in encoded["labels"] if token != -100], [31, 9, 32, 9])
        self.assertNotIn(41, encoded["labels"])
        self.assertNotIn(300, encoded["labels"])
        self.assertTrue(all(kwargs == {"tools": tools, "enable_thinking": False} for _, kwargs in tokenizer.seen))
        self.assertIsNone(tokenizer.seen[0][0][2]["content"])
        self.assertIn("tool_calls", tokenizer.seen[0][0][2])

    def test_final_assistant_mode_keeps_earlier_calls_masked(self):
        encoded = encode_messages(NativeTokenizer(), tool_dialogue(), 64)
        self.assertEqual([token for token in encoded["labels"] if token != -100], [32, 9])

    def test_historical_template_rewriting_is_rejected(self):
        class RewritingTokenizer(NativeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                if len(messages) == 3:
                    tokens[0] = 98
                return tokens

        with self.assertRaisesRegex(ValueError, "boundary"):
            encode_messages(RewritingTokenizer(), tool_dialogue(), 64, supervision="assistant_all")

    def test_template_override_and_truncation_cannot_bypass_mask_contract(self):
        with self.assertRaisesRegex(ValueError, "override"):
            encode_messages(NativeTokenizer(), tool_dialogue(), 64, chat_template_kwargs={"truncation": True})
        with self.assertRaisesRegex(ValueError, "exceeding"):
            encode_messages(NativeTokenizer(), tool_dialogue(), 4, supervision="assistant_all")

    def test_variable_length_batches_mask_padding_and_preserve_sampling_indices(self):
        batch = pad_training_batch([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3], "sample_index": 7},
            {"input_ids": [1, 4], "attention_mask": [1, 1], "labels": [-100, 4], "sample_index": 2},
        ], 0)
        self.assertEqual(batch["input_ids"][1], [1, 4, 0])
        self.assertEqual(batch["attention_mask"][1], [1, 1, 0])
        self.assertEqual(batch["labels"][1], [-100, 4, -100])
        self.assertEqual(batch["sample_index"], [7, 2])


class MultidomainTrainingOptionsTests(unittest.TestCase):
    def test_legacy_defaults_stay_legacy(self):
        options = trainer.resolve_training_options({}, SimpleNamespace())
        self.assertEqual(options["max_steps"], 8)
        self.assertEqual(options["learning_rate"], 1e-3)
        self.assertEqual(options["supervision"], "final_assistant")

    def test_new_profile_requires_explicit_budget_and_supervision(self):
        request = {"sft_profile": "multidomain", "supervision": "assistant_all"}
        with self.assertRaisesRegex(ValueError, "max_steps"):
            trainer.resolve_training_options(request, SimpleNamespace())
        request["training"] = {"max_steps": 100}
        request.pop("supervision")
        with self.assertRaisesRegex(ValueError, "supervision"):
            trainer.resolve_training_options(request, SimpleNamespace())

    def test_new_defaults_are_bounded_and_explicit_values_take_effect(self):
        options = trainer.resolve_training_options({"sft_profile": "multidomain", "supervision": "assistant_all",
            "training": {"max_steps": 30, "batch_size": 2, "gradient_accumulation_steps": 4}}, SimpleNamespace())
        self.assertEqual(options["max_steps"], 30)
        self.assertEqual(options["batch_size"], 2)
        self.assertEqual(options["gradient_accumulation_steps"], 4)
        self.assertEqual(options["learning_rate"], 2e-5)

    def test_conflicting_cli_and_request_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            trainer.resolve_training_options({"training": {"max_steps": 10}}, SimpleNamespace(steps=8))

    def test_nonfinite_or_unbounded_settings_are_rejected(self):
        for settings in ({"max_steps": 0}, {"max_steps": 10001}, {"learning_rate": float("nan")},
                         {"batch_size": True}, {"lora_target_modules": ["v_proj"]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                trainer.resolve_training_options({"training": settings}, SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
