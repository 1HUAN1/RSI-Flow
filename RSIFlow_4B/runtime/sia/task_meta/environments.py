"""Task-facing three-domain tools and controller-only terminal verification."""

from __future__ import annotations

import collections
import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .data import TaskRecord, content_hash, jsonl_rows, sha256_file
from .retrieval import FrozenSearchIndex
from .sandbox import LinuxSandbox, SandboxUnavailable


@dataclass(frozen=True)
class AdapterResult:
    reward: float | None
    metrics: dict
    verification: dict
    error_type: str | None = None
    infrastructure_error: bool = False
    details: dict = field(default_factory=dict)

    @classmethod
    def infrastructure_failure(cls, verifier_id: str, reason: str) -> AdapterResult:
        return cls(None, {}, {"status": "infrastructure_error", "verifier_id": verifier_id, "success": False},
                   reason, True)


class EnvironmentAdapter(Protocol):
    tools: list[dict]
    def reset(self, task: TaskRecord, rollout_id: str, seed: int) -> dict: ...
    def step(self, name: str, arguments: dict) -> dict: ...
    def evaluate(self, final_answer: str) -> AdapterResult: ...
    def close(self) -> None: ...


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, references: list[str]) -> tuple[bool, float]:
    prediction = normalize_answer(prediction)
    best_em, best_f1 = False, 0.0
    for reference in references:
        gold = normalize_answer(reference)
        em = prediction == gold
        pred_tokens, gold_tokens = prediction.split(), gold.split()
        if prediction in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}:
            f1 = float(em)
        else:
            shared = sum((collections.Counter(pred_tokens) & collections.Counter(gold_tokens)).values())
            f1 = 2 * shared / (len(pred_tokens) + len(gold_tokens)) if pred_tokens or gold_tokens else float(em)
        best_em, best_f1 = best_em or em, max(best_f1, f1)
    return best_em, best_f1


class SearchQAAdapter:
    def __init__(self, index: FrozenSearchIndex):
        self.index = index
        self.tools = [_tool("search", "Search the frozen offline passage corpus. Returns real retrieved evidence with document IDs.",
                    {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": index.max_results}}, ["query"])]

    def reset(self, task: TaskRecord, rollout_id: str, seed: int) -> dict:
        if task.domain != "searchqa":
            raise ValueError("SearchQA adapter received another domain")
        self.task = task
        return {**task.public_payload(), "retrieval_protocol": "open_retrieval", "corpus_hash": self.index.sha256,
                "instructions": "Use search for evidence. Submit only the short answer as your final answer."}

    def step(self, name: str, arguments: dict) -> dict:
        if name != "search" or set(arguments) - {"query", "top_k"}:
            raise ValueError("Undeclared search tool/arguments")
        return {"results": self.index.search(**arguments)}

    def evaluate(self, final_answer: str) -> AdapterResult:
        references = self.task.payload.get("answer")
        if isinstance(references, str):
            references = [references]
        verifier = "searchqa_normalized_em_f1_v1"
        if not isinstance(references, list) or not references or any(not isinstance(x, str) for x in references):
            return AdapterResult.infrastructure_failure(verifier, "invalid_reference_answers")
        exact_match, f1 = answer_scores(final_answer, references)
        return AdapterResult(float(exact_match), {"exact_match": float(exact_match), "f1": f1},
            {"status": "completed", "verifier_id": verifier, "success": exact_match, "exact_match": exact_match},
            None if exact_match else "incorrect_answer")

    def close(self):
        # Index is a shared read-only controller resource; owner closes it.
        pass


def extract_python(final_answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", final_answer, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("Expected one final Python program")
    code = blocks[0] if blocks else final_answer
    if not code.strip():
        raise ValueError("Empty final program")
    return code


def _stdin_match(actual: str, expected: str) -> bool:
    actual_tokens, expected_tokens = actual.split(), expected.split()
    if actual_tokens == expected_tokens:
        return True
    if len(actual_tokens) != len(expected_tokens):
        return False
    for observed, gold in zip(actual_tokens, expected_tokens, strict=True):
        if observed == gold:
            continue
        if re.fullmatch(r"[+-]?\d+", observed) and re.fullmatch(r"[+-]?\d+", gold):
            if int(observed) != int(gold):
                return False
            continue
        try:
            a, b = float(observed), float(gold)
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
                return False
        except (ValueError, OverflowError):
            return False
    return True


def _structured_match(actual, expected) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        try:
            return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        except OverflowError:
            return False
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_structured_match(a, b) for a, b in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_structured_match(actual[k], expected[k]) for k in actual)
    return actual == expected


def audit_taco_formats(paths: list[str | Path]) -> dict:
    """Inspect every declared case without executing solutions or exposing tests."""
    sources, unsupported = [], []
    for path_value in paths:
        path = Path(path_value)
        tasks, cases = 0, 0
        protocols = collections.Counter()
        for _, _, row in jsonl_rows(path):
            tasks += 1
            try:
                tests = row["tests"]
                tests = json.loads(tests) if isinstance(tests, str) else tests
                inputs, outputs, fn_name = tests["inputs"], tests["outputs"], tests.get("fn_name")
                if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
                    raise ValueError("missing_or_misaligned_cases")
                if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.isidentifier() or fn_name.startswith("__")):
                    raise ValueError("unsupported_function_entry")
                for input_value, expected in zip(inputs, outputs, strict=True):
                    cases += 1
                    if fn_name:
                        if isinstance(input_value, str):
                            [json.loads(line) for line in input_value.splitlines() if line.strip()]
                        elif not isinstance(input_value, list):
                            raise ValueError("unsupported_function_input")
                        if isinstance(expected, str):
                            json.loads(expected)
                        protocols["function"] += 1
                    else:
                        for value in (input_value, expected):
                            if not isinstance(value, str) and not (isinstance(value, list) and all(isinstance(line, str) for line in value)):
                                raise ValueError("unsupported_stdio_representation")
                        protocols["stdio_line_arrays" if isinstance(input_value, list) or isinstance(expected, list) else "stdio_strings"] += 1
            except (KeyError, TypeError, ValueError) as exc:
                unsupported.append({"source": str(path), "record_index": tasks - 1,
                                    "problem_hash": content_hash(row.get("problem", "")), "reason": str(exc)[:160]})
        sources.append({"path": str(path), "sha256": sha256_file(path), "tasks_inspected": tasks,
                        "cases_inspected": cases, "case_protocols": dict(protocols)})
    return {"schema_version": 1, "verifier_id": TACOAdapter.verifier_id, "sources": sources,
            "unsupported": unsupported, "status": "all_cases_supported" if not unsupported else "unsupported_cases",
            "solutions_executed": False, "labels_exported": False}


_FUNCTION_DRIVER = '''
import contextlib, io, json, pathlib, sys
spec = json.loads(sys.stdin.read())
scope = {"__name__": "candidate_solution"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(pathlib.Path("solution.py").read_text(), "solution.py", "exec"), scope)
    if "Solution" in scope:
        function = getattr(scope["Solution"](), spec["fn_name"])
    else:
        function = scope[spec["fn_name"]]
    result = function(*spec["args"])
print(json.dumps(result, allow_nan=False))
'''


class TACOAdapter:
    """Declared training verifier; final code benchmarks use official evaluators."""
    verifier_id = "deepcoder_taco_all_tests_stdio_or_function_v1_tol1e-6"

    def __init__(self, sandbox: LinuxSandbox):
        self.sandbox = sandbox
        self.tools = [_tool("run_code", "Run your Python program on input you supply in an isolated sandbox. Hidden evaluator tests are unavailable.",
                    {"code": {"type": "string"}, "stdin": {"type": "string"}}, ["code"])]

    def reset(self, task: TaskRecord, rollout_id: str, seed: int) -> dict:
        if task.domain != "code":
            raise ValueError("TACO adapter received another domain")
        self.task, self.seed = task, seed
        self.sandbox.validate()
        tests = task.payload.get("tests", {})
        tests = json.loads(tests) if isinstance(tests, str) else tests
        fn_name = tests.get("fn_name")
        # Entry point is part of the public executable interface; test inputs,
        # expected outputs and solutions remain in the trusted controller.
        entry = {"mode": "function", "function_name": fn_name} if fn_name else {"mode": "stdio"}
        return {**task.public_payload(), "program_entry": entry,
                "instructions": "Submit one complete Python program matching program_entry. For function mode, define the named function or a Solution method. You may test with input you supply."}

    def step(self, name: str, arguments: dict) -> dict:
        if name != "run_code" or set(arguments) - {"code", "stdin"}:
            raise ValueError("Undeclared code tool/arguments")
        if not isinstance(arguments.get("code"), str) or not isinstance(arguments.get("stdin", ""), str):
            raise ValueError("run_code requires string code/stdin")
        result = self.sandbox.run(arguments["code"], arguments.get("stdin", ""), seed=self.seed)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode,
                "timed_out": result.timed_out, "output_truncated": result.output_truncated}

    def evaluate(self, final_answer: str) -> AdapterResult:
        try:
            code = extract_python(final_answer)
        except ValueError:
            return AdapterResult(0.0, {"pass": 0.0}, {"status": "completed", "verifier_id": self.verifier_id,
                                 "success": False, "full_verifier": False}, "parse_failure")
        try:
            tests = self.task.payload["tests"]
            tests = json.loads(tests) if isinstance(tests, str) else tests
            inputs, outputs = tests["inputs"], tests["outputs"]
            if not isinstance(inputs, list) or not inputs or len(inputs) != len(outputs):
                raise ValueError("Missing/misaligned verifier cases")
        except (KeyError, TypeError, ValueError):
            return AdapterResult.infrastructure_failure(self.verifier_id, "invalid_taco_tests")
        fn_name = tests.get("fn_name")
        if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.isidentifier() or fn_name.startswith("__")):
            return AdapterResult.infrastructure_failure(self.verifier_id, "unsupported_function_entry")
        passed = 0
        outcomes = []
        for index, (input_value, expected) in enumerate(zip(inputs, outputs, strict=True)):
            try:
                if fn_name:
                    if isinstance(input_value, str):
                        args = [json.loads(line) for line in input_value.splitlines() if line.strip()]
                    elif isinstance(input_value, list):
                        args = input_value
                    else:
                        raise ValueError("Unsupported function input format")
                    expected_value = json.loads(expected) if isinstance(expected, str) else expected
                    result = self.sandbox.run(_FUNCTION_DRIVER, json.dumps({"fn_name": fn_name, "args": args}), extra_files={"solution.py": code}, seed=self.seed)
                    try:
                        match = _structured_match(json.loads(result.stdout), expected_value)
                    except (ValueError, TypeError):
                        match = False
                else:
                    # Prepared TACO also retains APPS-style arrays of stdin/stdout
                    # lines when fn_name is absent. These are not function args.
                    if isinstance(input_value, list) and all(isinstance(line, str) for line in input_value):
                        input_value = "\n".join(input_value) + "\n"
                    if isinstance(expected, list) and all(isinstance(line, str) for line in expected):
                        expected = "\n".join(expected)
                    if not isinstance(input_value, str) or not isinstance(expected, str):
                        raise ValueError("STDIO tests require strings or lists of string lines")
                    result = self.sandbox.run(code, input_value, seed=self.seed)
                    match = _stdin_match(result.stdout, expected)
            except SandboxUnavailable:
                return AdapterResult.infrastructure_failure(self.verifier_id, "code_sandbox_unavailable")
            except (ValueError, TypeError):
                return AdapterResult.infrastructure_failure(self.verifier_id, "unsupported_taco_test_format")
            success = match and result.returncode == 0 and not result.timed_out and not result.output_truncated
            passed += int(success)
            error = None if success else "timeout" if result.timed_out else "runtime_error" if result.returncode else "incorrect_output"
            # No hidden input, expected output or candidate echo is released as feedback.
            outcomes.append({"case_index": index, "passed": success, "error_type": error, "wall_seconds": result.wall_seconds})
        success = passed == len(inputs)
        failure_counts = dict(collections.Counter(item["error_type"] for item in outcomes if item["error_type"]))
        primary_error = next((kind for kind in ("timeout", "runtime_error", "incorrect_output") if kind in failure_counts), None)
        return AdapterResult(float(success), {"pass": float(success), "test_pass_fraction": passed / len(inputs)},
            {"status": "completed", "verifier_id": self.verifier_id, "success": success, "full_verifier": True,
             "tests_completed": len(inputs), "tests_total": len(inputs)}, primary_error,
            details={"test_outcomes": outcomes, "error_counts": failure_counts,
                     "comparator": "whitespace-token/structured; integers exact; float abs+relative tolerance 1e-6"})

    def close(self):
        pass


_ENV_WORKER = '''
import contextlib, io, json, pathlib, random, sys
official = {}
exec(compile(pathlib.Path("env_util.py").read_text(), "official_env_util.py", "exec"), official)
env = None
for line in sys.stdin:
    request = json.loads(line)
    with contextlib.redirect_stdout(io.StringIO()):
        operation = request["operation"]
        if operation == "reset" and env is None:
            random.seed(request["seed"])
            cls = official["init_env_class"](request["env_code"], request["class_name"])
            env = official["init_env_instance"](cls, request["init_config"])
            initial = official["get_state_info"](env)
            allowed_tools = request["allowed_tools"]
            response = {"initialized": True}
        elif operation == "step" and env is not None:
            action = request["action"]
            if action["name"] not in allowed_tools or action["name"].startswith("_"):
                raise ValueError("Undeclared environment action")
            try:
                observation = getattr(env, action["name"])(**action["arguments"])
                response = {"observation": str(observation), "runtime_error": False}
            except Exception as error:
                response = {"observation": type(error).__name__ + ": " + str(error), "runtime_error": True}
        elif operation == "evaluate" and env is not None:
            final = official["get_state_info"](env)
            checks = []
            for check in request["checklist"]:
                valid, result, error = official["run_check_function"](check["check_func"], initial, final)
                checks.append({"valid": valid, "result": result, "error": error})
            response = {"checks": checks}
        else:
            raise ValueError("Invalid environment protocol transition")
    print(json.dumps(response, allow_nan=False), flush=True)
'''


class EnvScalerAdapter:
    """Pinned official initialization/check functions with an isolated live worker.

    Native reward is the rounded fraction of official checks. Full success
    requires every valid check to return True; checker exceptions are infra
    failures, unlike ordinary invalid actions/incorrect environment state.
    """
    def __init__(self, environments_paths: list[str | Path], official_utils_path: str | Path, *,
                 runtime_commit: str, sandbox: LinuxSandbox, official_utils_sha256: str | None = None):
        if not re.fullmatch(r"[0-9a-f]{40}", runtime_commit):
            raise ValueError("EnvScaler runtime requires a pinned git commit")
        utils_path = Path(official_utils_path)
        self.utils_hash = sha256_file(utils_path)
        if official_utils_sha256 and self.utils_hash != official_utils_sha256:
            raise ValueError("Official EnvScaler verifier source hash mismatch")
        self.official_utils = utils_path.read_text(encoding="utf-8")
        self.verifier_id = f"envscaler:{runtime_commit}:official-checklist:{self.utils_hash}"
        self.sandbox = sandbox
        self.environments = {}
        for path in environments_paths:
            for _, _, row in jsonl_rows(Path(path)):
                key = str(row.get("canonical_env_id", row["env_id"]))
                if key in self.environments and self.environments[key] != row:
                    raise ValueError("Conflicting canonical environment metadata")
                self.environments[key] = row
        self.tools = []
        self.worker = None

    def reset(self, task: TaskRecord, rollout_id: str, seed: int) -> dict:
        if task.domain != "tool_use":
            raise ValueError("EnvScaler adapter received another domain")
        self.close()
        self.task, self.seed = task, seed
        self.sandbox.validate()
        key = str(task.payload.get("canonical_env_id", task.payload["env_id"]))
        self.env = self.environments[key]
        self.tools = self.env["tools"]
        self.tools = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
        self.tools = [tool for tool in self.tools if not tool["function"]["name"].startswith("_") and tool["function"]["name"] != "chat_with_user"]
        self.terminated = False
        init_config = self.task.payload["init_config"]
        init_config = json.loads(init_config) if isinstance(init_config, str) else init_config
        self.worker = self.sandbox.session(_ENV_WORKER, extra_files={"env_util.py": self.official_utils}, seed=seed)
        env_code = self.env["env_class_code"]
        if key == "env_178":
            import hashlib
            if hashlib.sha256(env_code.encode()).hexdigest() != "a9af811b70c593275234d6feb599c1ebaf83eaa16ef04a3d23b199516c163ad9":
                raise ValueError("Pinned EnvScaler datetime compatibility source changed")
            # The generated module imports the class after the module. Preserve
            # UTC timestamp semantics while addressing the actual bound class.
            if env_code.count("datetime.datetime.utcnow()") != 1:
                raise ValueError("Unexpected EnvScaler datetime repair target")
            env_code = env_code.replace("datetime.datetime.utcnow()", "datetime.utcnow()")
        self.worker.request({"operation": "reset", "seed": seed, "env_code": env_code,
            "class_name": task.payload["env_class_name"], "init_config": init_config,
            "allowed_tools": [tool["function"]["name"] for tool in self.tools]})
        return {**task.public_payload(), "environment_introduction": self.env.get("environment_introduction", ""),
                "constraints_rules": self.env.get("constraints_rules", []), "environment_tools": self.tools}

    def step(self, name: str, arguments: dict) -> dict:
        if self.terminated:
            return {"error": "environment_already_terminated", "terminated": True}
        if name not in {tool["function"]["name"] for tool in self.tools} or not isinstance(arguments, dict):
            return {"error": "invalid_tool_action", "terminated": False}
        observation = self.worker.request({"operation": "step", "action": {"name": name, "arguments": arguments}})
        self.terminated = observation["runtime_error"]
        return {**observation, "terminated": self.terminated}

    def evaluate(self, final_answer: str) -> AdapterResult:
        # Official BaseEnv.step terminates an environment method exception with
        # reward zero; it does not reinterpret a broken state as checker downtime.
        if self.terminated:
            return AdapterResult(0.0, {"task_success": 0.0, "native_partial_score": 0.0},
                {"status": "completed", "verifier_id": self.verifier_id, "success": False,
                 "task_success": False, "checks_completed": 0}, "environment_runtime_error")
        try:
            checklist = self.task.payload["checklist_with_func"]
            if self.task.task_id == "envscaler:env_142:env_142_rl-task_36":
                import hashlib
                if hashlib.sha256(checklist[6]["check_func"].encode()).hexdigest() != 'df71de4dc481ac892e1c4e2f42c61736405d013c1da72a75b7b146b64bc2ee4e':
                    raise ValueError("Pinned EnvScaler minute arithmetic checker changed")
                checklist = [dict(item) for item in checklist]
                checklist[6]["check_func"] = "def check_func(final_state):\n    replacement = None\n    for appt in final_state.get('appointments', {}).values():\n        if appt.get('patient_id') == 'PAT1' and appt.get('provider_id') == 'PROV1' and (appt.get('appointment_type') == 'video') and (appt.get('appointment_status') == 'scheduled'):\n            replacement = appt\n            break\n    if replacement is None:\n        return False\n    from datetime import datetime\n    repl_start = datetime.fromisoformat(replacement['scheduled_time'])\n    repl_end = repl_start + __import__('datetime').timedelta(minutes=30)\n    for appt in final_state.get('appointments', {}).values():\n        if appt.get('appointment_id') == replacement.get('appointment_id'):\n            continue\n        if appt.get('patient_id') != 'PAT1':\n            continue\n        other_start = datetime.fromisoformat(appt['scheduled_time'])\n        other_end = other_start + __import__('datetime').timedelta(minutes=30)\n        if not (repl_end <= other_start or repl_start >= other_end):\n            return False\n    return True"
            result = self.worker.request({"operation": "evaluate", "checklist": checklist})
        except SandboxUnavailable:
            return AdapterResult.infrastructure_failure(self.verifier_id, "envscaler_worker_unavailable")
        checks = result["checks"]
        if not checks or any(not check["valid"] or not isinstance(check["result"], bool) for check in checks):
            return AdapterResult.infrastructure_failure(self.verifier_id, "official_checker_failure")
        native_score = round(sum(check["result"] for check in checks) / len(checks), 4)
        success = all(check["result"] for check in checks) and not self.terminated
        return AdapterResult(float(success), {"task_success": float(success), "native_partial_score": native_score},
            {"status": "completed", "verifier_id": self.verifier_id, "success": success, "task_success": success,
             "checks_completed": len(checks)}, None if success else "environment_runtime_error" if self.terminated else "task_incomplete",
            details={"official_check_results": checks, "native_partial_score": native_score})

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None
