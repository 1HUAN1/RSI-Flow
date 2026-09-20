"""Native comparison contracts, plus real Linux worker security when available."""

import ast
import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from sia.task_meta.evalplus_codec import UnsupportedWireValue, decode, encode
from sia.task_meta.evalplus_isolated import EvalPlusInfrastructureError, isolated_check
from sia.task_meta.sandbox import LinuxSandbox, probe_isolation


@pytest.fixture
def native():
    project = Path(__file__).resolve().parents[1]
    candidates = [project.parent / "references/evaluators/evalplus/evalplus",
                  Path("/root/data/RSI_iclr2027/dataset/evaluators/evalplus/evalplus")]
    root = next((p for p in candidates if (p / "evalplus/eval/__init__.py").is_file()), None)
    if root is None:
        pytest.skip("Pinned official EvalPlus reference source unavailable")
    path = root / "evalplus/eval/__init__.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"unsafe_execute", "is_floats"}]
    namespace = {"np": np, "List": list, "create_tempdir": contextlib.nullcontext,
                 "swallow_io": contextlib.nullcontext, "reliability_guard": lambda **kwargs: None,
                 "query_maximum_memory_bytes": lambda: 4 * 1024**3,
                 "_SUCCESS": 0, "_FAILED": 1, "_TIMEOUT": 2, "_UNKNOWN": 3,
                 "_mapping": {0: "pass", 1: "fail", 2: "timeout", 3: None},
                 "PASS": "pass", "FAIL": "fail", "TIMEOUT": "timeout",
                 "DEFAULT_MIN_TIME_LIMIT": 4.0, "DEFAULT_GT_TIME_LIMIT_FACTOR": 4.0}
    exec(compile((root / "evalplus/eval/_special_oracle.py").read_text(), "official-special-oracle", "exec"), namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class OfflineSession:
    """Explicit test fixture: toy code only, never used for model-generated code."""
    def __init__(self):
        self.scope = {}
        self.function = None
        self.payloads = []

    def request(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if payload["operation"] == "initialize":
            exec(payload["code"], self.scope)
            self.function = self.scope[payload["entry_point"]]
            return {"status": "ready"}
        args = decode(payload["arguments"])
        try:
            out = self.function(*args)
            if payload.get("output_not_none"):
                out = out if isinstance(out, bool) else out is not None
            return {"status": "ok", "result": encode(out), "arguments_after": encode(args), "seconds": 0.001}
        except UnsupportedWireValue:
            return {"status": "unsupported_wire_value"}
        except Exception:
            return {"status": "error", "error_type": "fixture_candidate_error"}

    def close(self):
        pass


class OfflineSandbox:
    sessions: ClassVar[list] = []

    def __init__(self, limits):
        self.limits = limits

    def session(self, *args, **kwargs):
        session = OfflineSession()
        self.sessions.append(session)
        return session


def check(native, code, inputs, expected, entry="f", dataset="humaneval", sandbox_factory=OfflineSandbox):
    return isolated_check(native, dataset=dataset, code=code, inputs=inputs, entry_point=entry,
        expected=expected, atol=0, ref_time=[.001] * len(inputs), sandbox_factory=sandbox_factory)


@pytest.mark.parametrize("value", [None, True, -123456789123456789, -0.0, 1.25, "text", b"\x00\xff", (1, "x"),
                                  {"x": [1, 2]}, {1, 2}, frozenset({"a"}), {(1, 2): b"data"}, complex(1.5, -2)])
def test_codec_roundtrips_builtin_types(value):
    restored = decode(json.loads(json.dumps(encode(value))))
    assert type(restored) is type(value) and restored == value


@pytest.mark.parametrize("value", [object(), 1 << 9000])
def test_codec_rejects_unrepresentable_values(value):
    with pytest.raises(UnsupportedWireValue):
        encode(value)


def test_codec_rejects_executable_tags_and_shared_alias():
    with pytest.raises(UnsupportedWireValue):
        decode({"t": "pickle", "v": "payload"})
    shared = []
    with pytest.raises(UnsupportedWireValue, match="aliases"):
        encode([shared, shared])


def test_official_mbpp_nonfinite_inputs_and_empty_tuples():
    import math
    for value in (float('inf'), -float('inf'), float('nan')):
        restored = decode(json.loads(json.dumps(encode(value), allow_nan=False)))
        assert math.isnan(restored) if math.isnan(value) else restored == value
    restored = decode(encode([(), ()]))
    assert restored[0] is restored[1]


def test_pinned_official_comparator_distinguishes_known_and_wrong(native):
    good = check(native, "def f(a,b): return a+b", [(1, 2), (-3, 7)], [3, 4])
    wrong = check(native, "def f(a,b): return a-b", [(1, 2), (-3, 7)], [3, 4])
    assert good[0] == "pass" and good[1] == [True, True]
    assert wrong[0] == "fail" and wrong[1] == [False, False]


def test_worker_receives_current_input_only_and_preserves_state(native):
    OfflineSandbox.sessions.clear()
    state, details, _ = check(native, "counter=0\ndef f(x):\n global counter\n counter+=1\n return x+counter", [(3,), (7,)], [4, 9])
    assert state == "pass" and details == [True, True]
    payloads = OfflineSandbox.sessions[-1].payloads
    assert len(payloads) == 3
    assert all("expected" not in payload and "inputs" not in payload for payload in payloads)
    assert decode(payloads[1]["arguments"]) == (3,) and decode(payloads[2]["arguments"]) == (7,)


def test_native_mbpp_special_set_and_not_none_oracles(native):
    assert check(native, "def similar_elements(a,b): return list(set(a)&set(b))[::-1]", [([1,2], [2,1])], [[1,2]],
                 entry="similar_elements", dataset="mbpp")[0] == "pass"
    assert check(native, "def check_str(x): return object()", [("anything",)], [True], entry="check_str", dataset="mbpp")[0] == "pass"
    assert check(native, "def check_str(x): return False", [("anything",)], [True], entry="check_str", dataset="mbpp")[0] == "fail"


def test_unrepresentable_candidate_output_is_infrastructure_not_wrong_answer(native):
    with pytest.raises(EvalPlusInfrastructureError, match="unsupported codec"):
        check(native, "def f(x): return object()", [(1,)], [1])


@pytest.mark.skipif(not probe_isolation().get("available"), reason="Real Linux isolation only")
def test_real_isolated_native_comparator_blocks_host_reads_and_expected_introspection(native, tmp_path):
    secret = tmp_path / "hidden_expected.json"
    secret.write_text("TOP_SECRET_EXPECTED_9842")
    code = """def f(x):
 import inspect, os, socket
 try:
  os.open(PATH, os.O_RDONLY)
  return 'LEAK'
 except PermissionError: pass
 try:
  socket.socket()
  return 'NETWORK'
 except PermissionError: pass
 frame=inspect.currentframe()
 while frame:
  if 'expected' in frame.f_locals or 'inputs' in frame.f_locals:
   return 'GOLD_IN_WORKER'
  frame=frame.f_back
 return x+1
""".replace("PATH", repr(str(secret)))
    assert check(native, code, [(2,)], [3], sandbox_factory=LinuxSandbox)[0] == "pass"
    assert check(native, "def f(x): return x+1", [(2,), (6,)], [3,7], sandbox_factory=LinuxSandbox)[0] == "pass"
    assert check(native, "def f(x): return x-1", [(2,)], [3], sandbox_factory=LinuxSandbox)[0] == "fail"
