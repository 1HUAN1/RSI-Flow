"""Pinned EvalPlus comparisons with candidate execution moved behind safe IPC.

Native special-oracle/comparison statements are compiled from the pinned source,
never copied/reimplemented. Only candidate exec and resource/time wrappers change.
The official test arrays, reference outputs, and evaluator code remain trusted.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .evalplus_codec import UnsupportedWireValue, decode, encode
from .meta_harness.bundle import atomic_json
from .sandbox import LinuxSandbox, SandboxLimits, SandboxUnavailable
from .storage import digest

PINNED_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


class EvalPlusInfrastructureError(RuntimeError):
    pass


def _sync_mutation(original, returned):
    if type(original) is list and type(returned) is list:
        for left, right in zip(original, returned):
            _sync_mutation(left, right)
        original[:] = returned
    elif (type(original) is dict and type(returned) is dict) or (type(original) is set and type(returned) is set):
        original.clear()
        original.update(returned)
    elif type(original) is tuple and type(returned) is tuple and len(original) == len(returned):
        for left, right in zip(original, returned):
            _sync_mutation(left, right)


class CandidateProxy:
    def __init__(self, sandbox, initialize_timeout=10.0, *, total_seconds=63, output_not_none=False):
        self.sandbox = sandbox
        self.timeout = initialize_timeout
        self.session = None
        self.infrastructure_error = None
        self.calls = 0
        self.candidate_seconds = 0.0
        self.deadline = time.monotonic() + total_seconds
        self.global_timeout = False
        self.output_not_none = output_not_none

    def _request(self, payload):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.global_timeout = True
            raise TimeoutError("official suite wall timeout")
        self.sandbox.limits = replace(self.sandbox.limits, wall_seconds=min(remaining, payload["timeout"] + 3))
        try:
            return self.session.request(payload)
        except SandboxUnavailable as exc:
            if "timeout" in str(exc).lower():
                self.global_timeout = True
                raise TimeoutError("candidate exceeded bounded worker/suite timeout") from exc
            raise

    def initialize(self, code, entry_point):
        worker = Path(__file__).with_name("evalplus_worker.py").read_text(encoding="utf-8")
        codec = Path(__file__).with_name("evalplus_codec.py").read_text(encoding="utf-8")
        try:
            self.session = self.sandbox.session(worker, extra_files={"codec.py": codec})
            response = self._request({"operation": "initialize", "code": code, "entry_point": entry_point, "timeout": self.timeout})
        except SandboxUnavailable as exc:
            self.infrastructure_error = f"candidate session could not initialize: {exc}"
            raise EvalPlusInfrastructureError(self.infrastructure_error) from exc
        if response.get("status") == "timeout":
            self.global_timeout = True
        if response.get("status") != "ready":
            raise RuntimeError("candidate initialization failed")
        return self

    @contextlib.contextmanager
    def time_limit(self, seconds):
        previous, self.timeout = self.timeout, seconds
        try:
            yield
        finally:
            self.timeout = previous

    def __call__(self, *arguments):
        self.calls += 1
        try:
            request = {"operation": "call", "arguments": encode(arguments), "timeout": self.timeout,
                       "output_not_none": self.output_not_none}
            response = self._request(request)
            status = response.get("status")
            if status == "unsupported_wire_value":
                raise UnsupportedWireValue("candidate output cannot be represented faithfully")
            if status == "timeout":
                raise TimeoutError("candidate exceeded the official per-input timeout")
            if status != "ok":
                raise RuntimeError("candidate raised an exception")
            result = decode(response["result"])
            after = decode(response["arguments_after"])
            if type(after) is not tuple or len(after) != len(arguments):
                raise UnsupportedWireValue("invalid argument mutation payload")
            for before, changed in zip(arguments, after):
                _sync_mutation(before, changed)
            self.candidate_seconds += float(response.get("seconds", 0))
            return result
        except (UnsupportedWireValue, SandboxUnavailable, KeyError, TypeError) as exc:
            self.infrastructure_error = "unsupported codec or interrupted candidate transport"
            raise EvalPlusInfrastructureError(self.infrastructure_error) from exc

    def close(self):
        if self.session:
            self.session.close()
            self.session = None


def comparator_with_proxy(native, proxy):
    """Reject upstream shape changes instead of guessing patch equivalence."""
    tree = ast.parse(inspect.getsource(native.unsafe_execute))

    class Transform(ast.NodeTransformer):
        exec_count = 0
        guard_count = 0

        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "exec":
                    if ast.unparse(call) != "exec(code, exec_globals)":
                        raise EvalPlusInfrastructureError("Unrecognized native candidate execution entry")
                    self.exec_count += 1
                    return ast.parse("exec_globals[entry_point] = _candidate_proxy.initialize(code, entry_point)").body[0]
                if call.func.id == "reliability_guard":
                    self.guard_count += 1
                    return ast.Pass()
            return self.generic_visit(node)

    transform = Transform()
    transformed = transform.visit(tree)
    if (transform.exec_count, transform.guard_count) != (1, 1):
        raise EvalPlusInfrastructureError("Pinned EvalPlus execution boundary changed")
    ast.fix_missing_locations(transformed)
    namespace = dict(native.unsafe_execute.__globals__)
    namespace.update({"_candidate_proxy": proxy, "time_limit": proxy.time_limit})
    exec(compile(transformed, "<pinned-evalplus-ipc-comparator>", "exec"), namespace)
    return namespace["unsafe_execute"]


def isolated_check(native, *, dataset, code, inputs, entry_point, expected, atol, ref_time,
                   sandbox_factory=LinuxSandbox, min_time_limit=None, gt_time_limit_factor=None, fast_check=False):
    if len(inputs) != len(expected) or len(inputs) != len(ref_time) or not inputs:
        raise EvalPlusInfrastructureError("Invalid official test/reference contract")
    min_time_limit = native.DEFAULT_MIN_TIME_LIMIT if min_time_limit is None else min_time_limit
    gt_time_limit_factor = native.DEFAULT_GT_TIME_LIMIT_FACTOR if gt_time_limit_factor is None else gt_time_limit_factor
    time_limits = [max(min_time_limit, gt_time_limit_factor * float(t)) for t in ref_time]
    if any(not math.isfinite(t) or not 0 < t <= 120 for t in time_limits):
        raise EvalPlusInfrastructureError("Official per-input budget is unsupported")
    # Mirrors native untrusted_check's default global join timeout. No ambient
    # EVALPLUS_TIMEOUT_PER_TASK override can silently change the report protocol.
    total_seconds = min(60, sum(time_limits)) + (2 if fast_check else 3)
    limits = SandboxLimits(wall_seconds=total_seconds, cpu_seconds=math.ceil(total_seconds) + 1,
                           memory_bytes=4 * 1024**3, output_bytes=4000000, file_bytes=4000000)
    proxy = CandidateProxy(sandbox_factory(limits=limits), initialize_timeout=total_seconds, total_seconds=total_seconds,
        output_not_none=dataset == "mbpp" and entry_point in native.MBPP_OUTPUT_NOT_NONE_TASKS)
    status, progress, details = SimpleNamespace(value=native._UNKNOWN), SimpleNamespace(value=0), [False] * len(inputs)
    try:
        comparator = comparator_with_proxy(native, proxy)
        comparator(dataset, entry_point, code, copy.deepcopy(inputs), expected, time_limits, atol, fast_check,
                   status, details, progress)
        if proxy.infrastructure_error:
            raise EvalPlusInfrastructureError(proxy.infrastructure_error)
        state = native.TIMEOUT if proxy.global_timeout else native._mapping[status.value]
        details = details[:progress.value]
        if state == native.PASS and (len(details) != len(inputs) or not all(details)):
            state = native.FAIL
        evidence = {"candidate_calls": proxy.calls, "candidate_seconds": proxy.candidate_seconds,
                    "transport": "bounded_tagged_json", "persistent_state": True,
                    "timing": "native per-input timer inside worker; IPC counts against native default global join timeout",
                    "unsupported_value_policy": "infrastructure/incomplete", "suite_cpu_budget": limits.cpu_seconds}
        return state or native.TIMEOUT, details, evidence
    finally:
        proxy.close()


def load_pinned_evalplus(root: Path, commit: str):
    root = Path(root).resolve()
    if commit != PINNED_COMMIT:
        raise EvalPlusInfrastructureError("EvalPlus IPC adapter supports only the validated source commit")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
    if actual != commit or dirty.strip():
        raise EvalPlusInfrastructureError("Official EvalPlus source changed")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        native = importlib.import_module("evalplus.eval")
    finally:
        sys.dont_write_bytecode = previous_bytecode
    if not Path(native.__file__).resolve().is_relative_to(root):
        raise EvalPlusInfrastructureError("A different installed EvalPlus package was imported")
    return native


def evaluate_evalplus_isolated(spec, predictions_path: Path, output_dir: Path):
    """Run pinned official oracles/comparators on fixed local data, one candidate."""
    root = Path(spec.entrypoint).resolve().parents[1]
    native = load_pinned_evalplus(root, spec.commit)
    LinuxSandbox().validate()
    dataset = "humaneval" if spec.benchmark == "HumanEval+" else "mbpp"
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        mbpp = importlib.import_module("evalplus.data.mbpp")
        special = importlib.import_module("evalplus.eval._special_oracle")
        trusted_exec = importlib.import_module("evalplus.gen.util").trusted_exec
    finally:
        sys.dont_write_bytecode = previous_bytecode
    rows = [json.loads(line) for line in Path(spec.data_path).read_text().splitlines() if line.strip()]
    problems = {row["task_id"]: row for row in rows}
    samples = [json.loads(line) for line in Path(predictions_path).read_text().splitlines() if line.strip()]
    if len(samples) != len(problems) or {r["task_id"] for r in samples} != set(problems):
        raise EvalPlusInfrastructureError("Official data and predictions have different denominators")
    results = {"hash": hashlib.md5(Path(spec.data_path).read_bytes()).hexdigest(), "eval": {},
               "adapter": "pinned_evalplus_comparator_isolated_candidate_v1", "source_commit": spec.commit,
               "source_sha256": digest(Path(native.__file__)), "protocol_change": "candidate IPC isolation with native default time budgets",
               "parent_python": sys.executable, "candidate_python": str(LinuxSandbox().python)}
    started = time.monotonic()
    for sample in samples:
        if time.monotonic() - started > spec.timeout_seconds:
            raise EvalPlusInfrastructureError("Final official evaluation wall budget exhausted")
        task = problems[sample["task_id"]]
        item = {"task_id": task["task_id"], "solution": sample["solution"]}
        for kind in ("base", "plus"):
            inputs = task[kind + "_input"]
            if dataset == "mbpp":
                inputs = mbpp.mbpp_deserialize_inputs(task["task_id"], inputs)
            expected, ref_time = trusted_exec(task["prompt"] + task["canonical_solution"], inputs, task["entry_point"],
                record_time=True, output_not_none=task["entry_point"] in special.MBPP_OUTPUT_NOT_NONE_TASKS if dataset == "mbpp" else False)
            status, details, evidence = isolated_check(native, dataset=dataset, code=sample["solution"], inputs=inputs,
                entry_point=task["entry_point"], expected=expected, atol=task["atol"], ref_time=ref_time)
            item[kind + "_status"] = status
            item[kind + "_details"] = details
            item[kind + "_isolation"] = evidence
        results["eval"][task["task_id"]] = [item]
        atomic_json(output_dir / "official_results.partial.json", results)
    results["pass_at_k"] = {"plus": {"pass@1": sum(v[0]["base_status"] == v[0]["plus_status"] == native.PASS for v in results["eval"].values()) / len(results["eval"])}}
    atomic_json(output_dir / "official_results.json", results)
    (output_dir / "stdout.txt").write_text(json.dumps(results["pass_at_k"]) + "\n")
    (output_dir / "stderr.txt").write_text("")
    return {"returncode": 0, "worker_isolation": "linux_landlock_seccomp", "source_commit": spec.commit}
