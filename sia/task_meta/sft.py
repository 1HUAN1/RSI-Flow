"""Trusted SFT selection and native-chat loss masks, with no model/API dependency."""

from __future__ import annotations

import copy
import hashlib
import json
import math

PROFILES = {"legacy_gpqa", "multidomain"}
SUPERVISION_MODES = {"final_assistant", "assistant_all"}


def validate_messages(messages, *, allow_tools=False):
    """Keep recorded context intact; validate native calls without inventing text."""
    if (not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[-1], dict)
            or messages[-1].get("role") not in ({"assistant", "tool"} if allow_tools else {"assistant"})
            or not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages)):
        raise ValueError("Each positive rollout must end with a recorded assistant response")
    roles = {"system", "user", "assistant", "tool"} if allow_tools else {"system", "user", "assistant"}
    pending_calls = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in roles:
            raise ValueError("Unsupported recorded training message role")
        calls = message.get("tool_calls")
        if calls is not None:
            if not allow_tools or message["role"] != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("Native tool_calls require an assistant message and the multidomain profile")
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if (not isinstance(function, dict) or call.get("type", "function") != "function"
                        or not isinstance(function.get("name"), str) or not function["name"].strip()
                        or not isinstance(function.get("arguments"), (str, dict))):
                    raise ValueError("Malformed native assistant tool call")
                if isinstance(function["arguments"], str):
                    try:
                        arguments = json.loads(function["arguments"])
                    except json.JSONDecodeError as exc:
                        raise ValueError("Native tool arguments must be a JSON object") from exc
                    if not isinstance(arguments, dict):
                        raise ValueError("Native tool arguments must be a JSON object")
                if call.get("id") is not None:
                    if not isinstance(call["id"], str) or not call["id"] or call["id"] in pending_calls:
                        raise ValueError("Native tool call IDs must be nonempty and unique while pending")
                    pending_calls.add(call["id"])
        content = message.get("content")
        if (not (isinstance(content, str) and content.strip())
                and not (calls and (content is None or content == ""))
                and not (allow_tools and message["role"] == "tool" and content == "")):
            raise ValueError("Recorded training message content must be nonempty text or native assistant tool calls")
        if message["role"] == "tool" and message.get("tool_call_id") is not None:
            call_id = message["tool_call_id"]
            if call_id not in pending_calls:
                raise ValueError("Tool observation references an unknown or completed tool call")
            pending_calls.remove(call_id)
    final_calls = messages[-1].get("tool_calls") or []
    final_call_ids = {call.get("id") for call in final_calls if isinstance(call, dict)}
    if pending_calls - final_call_ids:
        raise ValueError("Training trajectory has unresolved native tool call IDs")


def validate_training_row(row, profile="legacy_gpqa"):
    """Only controller/evaluator records may supply verification, never Task claims.

    The multidomain schema deliberately does not infer full success from partial
    reward. The caller owns the trusted evaluator boundary and evidence references.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown SFT profile: {profile}")
    if not isinstance(row, dict):
        raise ValueError("SFT trajectory must be an object")
    reward = row.get("terminal_reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or reward <= 0:
        raise ValueError("SFT input must contain only finite positive terminal rewards")
    if profile == "multidomain":
        if row.get("split") != "evolve_train":
            raise ValueError("SFT requires evolve_train evidence; evaluation trajectories are excluded")
        domain = row.get("domain")
        if domain not in {"code", "tool_use", "searchqa"}:
            raise ValueError("SFT trajectory has no supported domain")
        verification = row.get("verification")
        if (not isinstance(verification, dict) or verification.get("status") != "completed"
                or verification.get("success") is not True
                or not isinstance(verification.get("verifier_id"), str)
                or not verification["verifier_id"].strip()):
            raise ValueError("SFT requires a completed successful trusted verifier record")
        if domain == "code" and verification.get("full_verifier") is not True:
            raise ValueError("Code SFT requires the complete training verifier, not public examples")
        if domain == "tool_use" and verification.get("task_success") is not True:
            raise ValueError("Tool SFT requires official task_success, not partial completion")
        exact_match = verification.get("exact_match")
        if domain == "searchqa" and not (exact_match is True or (type(exact_match) in (int, float) and exact_match == 1)):
            raise ValueError("Search SFT requires exact answer correctness, not positive F1")
        if row.get("error_type") in {"infrastructure", "model_service_unavailable", "evaluator_unavailable"}:
            raise ValueError("Infrastructure failures cannot supply SFT successes")
    validate_messages(row.get("messages"), allow_tools=profile == "multidomain")
    if row.get("tools") is not None and not isinstance(row["tools"], list):
        raise ValueError("Recorded tool definitions must be a list")


def select_positive_rows(trajectories, *, profile="legacy_gpqa"):
    """Select eligible recorded rows, retaining source IDs, verification and tools."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown SFT profile: {profile}")
    selected = []
    for row in trajectories:
        try:
            validate_training_row(row, profile)
        except ValueError:
            continue
        conversations = row.get("sft_conversations") if profile == "multidomain" else None
        if conversations is None:
            selected.append(copy.deepcopy(row))
            continue
        if not isinstance(conversations, list):
            continue
        # HarnessForge reconstructs earlier context. Only each real call's final
        # assistant is a recorded model target; reconstructed history is context.
        for conversation in conversations:
            if not isinstance(conversation, dict) or not isinstance(conversation.get("messages"), list):
                continue
            sample = copy.deepcopy({key: value for key, value in row.items()
                                    if key not in {"sft_conversations", "model_calls"}})
            sample.update({"messages": copy.deepcopy(conversation["messages"]),
                           "call_id": conversation.get("call_id"), "operation": conversation.get("operation"),
                           "original_task_id": row.get("task_id", row.get("question_id")),
                           "original_rollout_id": row.get("rollout_id"), "sft_supervision": "final_assistant"})
            # Per-call context belongs to the actual Qwen request, including
            # auxiliary roles. Never inherit a different call's tool/template
            # settings or let conversation metadata replace trusted rewards.
            for key in ("tools", "chat_template_kwargs", "binding", "role", "node", "seed",
                        "max_tokens", "temperature", "request_id"):
                if key in conversation:
                    sample[key] = copy.deepcopy(conversation[key])
            try:
                validate_training_row(sample, profile)
            except ValueError:
                continue
            selected.append(sample)
    return selected


def deduplicate_rows(rows):
    """Deduplicate identical model supervision while recording every source row."""
    unique, indices, duplicates = [], {}, []
    for index, row in enumerate(rows):
        supervised = {"messages": row["messages"], "tools": row.get("tools"), "sft_supervision": row.get("sft_supervision"),
                      "chat_template_kwargs": row.get("chat_template_kwargs", {})}
        fingerprint = hashlib.sha256(json.dumps(supervised, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
        if fingerprint in indices:
            duplicates.append({"row_index": index, "duplicate_of_row_index": indices[fingerprint],
                               "task_id": row.get("task_id", row.get("question_id")),
                               "rollout_id": row.get("rollout_id"), "call_id": row.get("call_id"), "sha256": fingerprint})
        else:
            indices[fingerprint] = index
            unique.append(row)
    return unique, {"input_rows": len(rows), "unique_rows": len(unique), "duplicates": duplicates}


def encode_messages(tokenizer, messages, max_length, *, supervision="final_assistant", tools=None,
                    chat_template_kwargs=None):
    """Use exact native-template prefixes to supervise assistant bodies/calls only.

    Some templates rewrite prior turns when later turns are appended. Those are
    rejected instead of guessing token boundaries or training tool observations.
    A compatible template must preserve each supervised assistant prefix.
    """
    if supervision not in SUPERVISION_MODES:
        raise ValueError(f"Unknown supervision mode: {supervision}")
    kwargs = dict(chat_template_kwargs or {})
    protected = {"tokenize", "add_generation_prompt", "return_dict", "tools", "chat_template", "return_tensors",
                 "truncation", "max_length", "padding"}
    if protected.intersection(kwargs):
        raise ValueError("Chat-template settings cannot override trusted tokenization or template selection")
    if tools is not None:
        kwargs["tools"] = tools

    def render(turns, generation=False):
        ids = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=generation,
                                            return_dict=False, **kwargs)
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("The tokenizer must return token ID lists")
        return ids

    full = render(messages)
    if len(full) > max_length:
        raise ValueError(f"Positive trajectory has {len(full)} tokens, exceeding --max-length={max_length}")
    labels = [-100] * len(full)
    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if supervision == "final_assistant":
        # Legacy tests use a token-only fixture; normal callers validate messages first.
        assistant_indices = assistant_indices[-1:] or [len(messages) - 1]
    for index in assistant_indices:
        prefix = render(messages[:index], generation=True)
        through = full if index == len(messages) - 1 else render(messages[:index + 1])
        if (len(through) <= len(prefix) or through[:len(prefix)] != prefix
                or full[:len(through)] != through):
            raise ValueError("Chat template does not preserve the assistant token boundary")
        labels[len(prefix):len(through)] = full[len(prefix):len(through)]
    if not any(label != -100 for label in labels):
        raise ValueError("No assistant tokens remain for supervision")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def pad_training_batch(features, pad_token_id):
    """Right pad variable-length examples and exclude padding from the loss."""
    longest = max(len(item["input_ids"]) for item in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": [], "sample_index": []}
    for item in features:
        padding = longest - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
        batch["labels"].append(item["labels"] + [-100] * padding)
        batch["sample_index"].append(item["sample_index"])
    return batch
