"""GPQA seed: editable strategy hooks with protected model/evaluator/artifact I/O.

Each repetition reads a frozen input snapshot and writes isolated new assets.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


def harness_config():
    return {
        "max_attempts": 1,
        "use_artifacts": True,
        "generate_artifacts": True,
        "artifact_char_limit": 6000,
        "retry_on_parse_error": True,
        "retry_on_api_error": True,
    }


def select_artifact_context(resources, question):
    return "\n".join(f"Resource: {item['path']}\n{item['content']}" for item in resources)


def retry_prompt(question, previous_output, error):
    return (
        "Review your previous response and solve the same question. "
        "Return a JSON object with answer (A/B/C/D) first. "
        f"The previous response could not be used: {error}"
    )


def format_question(question, artifact_context, max_tokens=128):
    options = "\n".join(f"{letter}) {text}" for letter, text in question["options"].items())
    instruction = (
        "Solve the multiple-choice problem. Return a JSON object with answer (A, B, C or D). "
        "Optionally add reusable_note: a brief general method useful for future problems, "
        "without a question ID or a question-answer mapping. Put answer first.\n"
    )
    if max_tokens <= 16:
        instruction = 'Solve the multiple-choice problem. Return only JSON: {"answer":"A"} (A/B/C/D).\n'
    return (
        instruction + f"Question: {question['Question']}\n{options}\n"
        + (f"Reusable resources from the current artifact snapshot:\n{artifact_context}" if artifact_context else "")
    )


def parse_answer(text):
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        payload = json.loads(text[start:end])
        answer = str(payload.get("answer", "")).strip().upper()
        note = payload.get("reusable_note", "")
        return answer if answer in {"A", "B", "C", "D"} else "", note if isinstance(note, str) else ""
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        match = re.search(r'"answer"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), ""
        match = re.fullmatch(r"\s*([ABCD])[.)]?\s*", text)
        return (match.group(1), "") if match else ("", "")


def safe_task_id(question_id):
    """A stable collision-resistant path component, never raw task-controlled paths."""
    encoded = json.dumps(question_id, ensure_ascii=False, sort_keys=True)
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", str(question_id))[:40] or "task"
    return label + "_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def validated_harness_config():
    config = harness_config()
    boolean_keys = {"use_artifacts", "generate_artifacts", "retry_on_parse_error", "retry_on_api_error"}
    expected = boolean_keys | {"max_attempts", "artifact_char_limit"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Harness config must contain exactly the six declared settings")
    if any(type(config[key]) is not bool for key in boolean_keys):
        raise ValueError("Harness flag settings must be boolean")
    if type(config["max_attempts"]) is not int or not 1 <= config["max_attempts"] <= 3:
        raise ValueError("Harness max_attempts must be an integer between 1 and 3")
    if type(config["artifact_char_limit"]) is not int or not 0 <= config["artifact_char_limit"] <= 12000:
        raise ValueError("Harness artifact_char_limit must be an integer between 0 and 12000")
    return config


def require_text(value, description, limit=24000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{description} must be a string of at most {limit} characters")
    return value


def run_question(question, resources, config, strategy, client, output_dir):
    """Protected execution: hooks influence strategy, never bindings or evidence."""
    context = ""
    if strategy["use_artifacts"]:
        context = require_text(select_artifact_context(copy.deepcopy(resources), copy.deepcopy(question)),
                               "Selected artifact context", limit=2000000)[:strategy["artifact_char_limit"]]
    prompt = require_text(format_question(copy.deepcopy(question), context, config["max_tokens"]), "Question prompt")
    if not strategy["generate_artifacts"]:
        prompt += "\nDo not generate a reusable_note or other artifact in this response."
    messages = [{"role": "user", "content": prompt}]
    record = {
        "question_id": question["id"], "messages": messages,
        "model_ref_requested": config["model_ref"], "service_binding": config.get("service_binding"),
        "model_answer": "", "model_answer_raw": "", "valid_answer": False,
        "parse_failure": False, "output_truncated": False, "api_error": False,
        "call_attempts": [], "generated_artifacts": [],
        "harness_config": strategy, "artifact_context_chars": len(context),
    }
    started = time.monotonic()
    for attempt in range(strategy["max_attempts"]):
        attempt_started = time.monotonic()
        call = {"attempt": attempt, "seed": config["seed"] + attempt * 100000,
                "model_ref_requested": config["model_ref"], "api_error": False,
                "parse_failure": False, "output_truncated": False}
        raw, answer, note = "", "", ""
        try:
            response = client.chat.completions.create(
                model=config["model_ref"], messages=copy.deepcopy(messages), max_tokens=config["max_tokens"],
                temperature=config["temperature"], seed=call["seed"],
            )
        except Exception as exc:
            # Exception bodies may include request headers or provider secrets; retain only a type/code.
            call.update({"api_error": True, "error": type(exc).__name__, "status_code": getattr(exc, "status_code", None)})
        else:
            if response.model != config["model_ref"]:
                raise ValueError("Task response model does not match the immutable requested model binding")
            expected_binding = config.get("service_binding")
            actual_binding = getattr(response, "local_checkpoint_binding", None)
            if (expected_binding and "weights" in expected_binding
                    and (not isinstance(actual_binding, dict)
                         or actual_binding.get("checkpoint_path") != expected_binding["checkpoint_path"]
                         or actual_binding.get("weights") != expected_binding["weights"])):
                raise ValueError("Task response checkpoint binding does not match the recorded loaded weights")
            call["response_checkpoint_binding"] = actual_binding
            choice = response.choices[0]
            raw = choice.message.content or ""
            call.update({"model_ref_response": response.model, "finish_reason": choice.finish_reason,
                         "output_truncated": choice.finish_reason == "length",
                         "model_answer_raw": raw,
                         "input_tokens": response.usage.prompt_tokens if response.usage else None,
                         "output_tokens": response.usage.completion_tokens if response.usage else None})
            parsed = parse_answer(raw)
            if (not isinstance(parsed, (list, tuple)) or len(parsed) != 2
                    or not all(isinstance(value, str) for value in parsed)):
                raise ValueError("parse_answer must return (answer_string, note_string)")
            answer, note = parsed
            if answer not in {"A", "B", "C", "D", ""}:
                raise ValueError("parse_answer returned a value outside A/B/C/D/empty")
            messages.append({"role": "assistant", "content": raw})
            call.update({"model_answer": answer, "parse_failure": not bool(answer)})
            if not answer:
                call["error"] = "No valid answer in model output"
            if strategy["generate_artifacts"] and note.strip():
                task_key = safe_task_id(question["id"])
                filename = "strategy.md" if attempt == 0 else f"strategy_attempt_{attempt}.md"
                relative = Path(task_key) / f"rollout_{config['rollout_id']}" / filename
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(note.strip() + "\n")
                record["generated_artifacts"].append({"path": relative.as_posix(), "attempt": attempt,
                                                      "production_method": "task_response.reusable_note"})
        call["wall_time_seconds"] = time.monotonic() - attempt_started
        record["call_attempts"].append(call)
        record.update({"model_answer": answer, "model_answer_raw": raw, "valid_answer": bool(answer),
                       "parse_failure": call["parse_failure"], "api_error": call["api_error"],
                       "output_truncated": call["output_truncated"],
                       "model_ref_response": call.get("model_ref_response"), "finish_reason": call.get("finish_reason"),
                       "response_checkpoint_binding": call.get("response_checkpoint_binding")})
        if "error" in call:
            record["error"] = call["error"]
        else:
            record.pop("error", None)
        should_retry = ((call["api_error"] and strategy["retry_on_api_error"])
                        or (call["parse_failure"] and strategy["retry_on_parse_error"]))
        if answer or not should_retry or attempt + 1 == strategy["max_attempts"]:
            break
        messages.append({"role": "user", "content": require_text(
            retry_prompt(copy.deepcopy(question), raw, call.get("error", "")), "Retry prompt")})
    for field in ("input_tokens", "output_tokens"):
        values = [call.get(field) for call in record["call_attempts"]]
        record[field] = sum(values) if all(isinstance(value, int) for value in values) else None
    record["parse_failure_count"] = sum(call["parse_failure"] for call in record["call_attempts"])
    record["api_error_count"] = sum(call["api_error"] for call in record["call_attempts"])
    record["output_truncation_count"] = sum(call["output_truncated"] for call in record["call_attempts"])
    record["wall_time_seconds"] = time.monotonic() - started
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--working_dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.working_dir / "target_config.json").read_text(encoding="utf-8"))
    strategy = validated_harness_config()
    resources = []
    artifact_dir = args.working_dir / "artifacts_input"
    if artifact_dir.exists():
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("Artifact input must not contain symlinks")
            if path.is_file():
                resources.append({"path": path.relative_to(artifact_dir).as_posix(),
                                  "content": path.read_text(encoding="utf-8")})
    questions = json.loads((args.dataset_dir / "diamond_questions.json").read_text(encoding="utf-8"))
    client = OpenAI(base_url=config["base_url"], api_key=os.environ[config["api_key_env"]],
                    timeout=config["timeout"], max_retries=0)
    details, trajectories = [], []
    for question in questions:
        record = run_question(question, resources, config, strategy, client, args.working_dir / "artifacts_generated")
        trajectories.append(record)
        details.append({"question_id": question["id"], "model_answer": record["model_answer"]})
        print(f"question={question['id']} completed elapsed={record['wall_time_seconds']:.2f}s", flush=True)
    (args.working_dir / "results").mkdir(exist_ok=True)
    (args.working_dir / "results" / "submission.json").write_text(json.dumps({"details": details}), encoding="utf-8")
    (args.working_dir / "agent_execution.json").write_text(json.dumps(trajectories, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
