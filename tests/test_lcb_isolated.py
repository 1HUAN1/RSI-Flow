import json
from pathlib import Path

import pytest
from test_evalplus_isolated import OfflineSandbox

from sia.task_meta.lcb_isolated import check_lcb, load_native
from sia.task_meta.sandbox import LinuxSandbox, probe_isolation


@pytest.fixture
def native_lcb():
    candidates = [Path(__file__).resolve().parents[2] / 'references/evaluators/livecodebench/LiveCodeBench',
                  Path('/root/data/RSI_iclr2027/dataset/evaluators/livecodebench/LiveCodeBench')]
    source = next((r / 'lcb_runner/evaluation/testing_util.py' for r in candidates if r.exists()), None)
    if source is None:
        pytest.skip('Pinned LCB source is not present')
    return load_native(source), source.read_text()


@pytest.mark.parametrize('code,fn,inputs,outputs,passed', [
    ('print(int(input())+1)', None, ['2', '4'], ['3', '5'], True),
    ('print(50000000000000000)', None, [''], ['50000000000000001'], False),
    ('class Solution:\n def f(self,x): return x+1', 'f', ['2', '4'], ['3', '5'], True),
    ('def f(x): return x-1', 'f', ['2'], ['3'], False),
])
def test_native_stdio_and_call_comparison(native_lcb, code, fn, inputs, outputs, passed):
    native, source = native_lcb
    result, _, evidence = check_lcb(native, source, code, {'input_output': json.dumps({'fn_name': fn,
                                    'inputs': inputs, 'outputs': outputs})}, sandbox_factory=OfflineSandbox)
    assert all(r is True for r in result) == passed
    assert evidence['expected_outputs_sent_to_candidate'] is False


@pytest.mark.skipif(not probe_isolation().get('available'), reason='Real Linux isolation')
def test_real_lcb_candidate_uses_native_stdio_wrapper(native_lcb):
    native, source = native_lcb
    sample = {'input_output': json.dumps({'inputs': ['1 2', '4 5'], 'outputs': ['3', '9']})}
    result, _, _ = check_lcb(native, source, 'print(sum(map(int,input().split())))', sample, sandbox_factory=LinuxSandbox)
    assert result == [True, True]
