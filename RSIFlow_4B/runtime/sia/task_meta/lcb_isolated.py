"""Pinned LiveCodeBench comparator with persistent isolated candidate IPC.

The official source handles input decoding, wrapping, comparisons and outcomes.
Only execution/OS guards are replaced. Expected outputs never enter the worker.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

from .evalplus_isolated import CandidateProxy, EvalPlusInfrastructureError
from .sandbox import LinuxSandbox, SandboxLimits
from .storage import digest, save_json

PIN = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'


def load_native(path):
    loader = importlib.util.spec_from_file_location('_rsi_pinned_lcb', path)
    native = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(native)
    return native


def check_lcb(native, source_text, code, sample, *, timeout=6, sandbox_factory=LinuxSandbox):
    tests = json.loads(sample['input_output'])
    if not tests.get('inputs') or len(tests['inputs']) != len(tests.get('outputs', [])):
        raise EvalPlusInfrastructureError('Invalid complete LCB test contract')
    total = (timeout + 1) * len(tests['inputs']) + 5
    proxy = CandidateProxy(sandbox_factory(limits=SandboxLimits(wall_seconds=total,
                           cpu_seconds=math.ceil(total), memory_bytes=4 * 1024**3,
                           output_bytes=4000000, file_bytes=4000000)), total_seconds=total,
                           initialize_timeout=timeout)
    proxy.timeout = timeout
    helper_names = {'MockStdinWithBuffer', 'MockBuffer', 'call_method'}
    helpers = [n for n in ast.parse(source_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in helper_names]
    if {n.name for n in helpers} != helper_names:
        raise EvalPlusInfrastructureError('Pinned LCB stdin boundary changed')
    helper_code = 'import sys\nfrom io import StringIO\nfrom unittest.mock import patch, mock_open\n' + '\n'.join(ast.unparse(n) for n in helpers)

    def get_function(compiled, name):
        candidate = compiled.code
        if name == 'wrapped_function':
            # This helper contains input adaptation only, no oracle or test suite.
            candidate += '\n' + helper_code + '''
def _rsi_lcb_entry(value):
    import contextlib, io
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        call_method(wrapped_function, value)
    return capture.getvalue()
'''
        else:
            if not isinstance(name, str) or not name.isidentifier():
                raise EvalPlusInfrastructureError('Invalid LCB function entry')
            candidate += f'\n_rsi_lcb_entry = Solution().{name}\n' if 'class Solution' in candidate else f'\n_rsi_lcb_entry = {name}\n'
        proxy.initialize(candidate, '_rsi_lcb_entry')

        def call(*args):
            try:
                return proxy(*args)
            except TimeoutError as exc:
                raise native.TimeoutException() from exc
        return call

    replacements = {'compile_code': lambda code, timeout: SimpleNamespace(code=code),
                    'get_function': get_function,
                    'call_method': lambda method, inputs: print(method(inputs), end=''),
                    'reliability_guard': lambda: None,
                    'signal': SimpleNamespace(SIGALRM=14, signal=lambda *args: None, alarm=lambda *args: None),
                    'faulthandler': SimpleNamespace(enable=lambda: None, disable=lambda: None)}
    old = {key: getattr(native, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        outcomes, metadata = native.run_test(sample, test=code, timeout=timeout)
        if proxy.infrastructure_error:
            raise EvalPlusInfrastructureError(proxy.infrastructure_error)
        return outcomes, metadata, {'candidate_calls': proxy.calls, 'candidate_seconds': proxy.candidate_seconds,
                'worker_isolation': 'landlock_seccomp', 'comparator': 'unmodified_pinned_source',
                'timing': 'native per-case timer inside worker; IPC included in native global budget',
                'expected_outputs_sent_to_candidate': False}
    finally:
        proxy.close()
        for key, value in old.items():
            setattr(native, key, value)


def evaluate_lcb_isolated(spec, predictions_path, output_dir):
    source = Path(spec.entrypoint)
    root = source.parents[2]
    actual = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True, capture_output=True, check=True).stdout.strip()
    if spec.commit != PIN or actual != PIN or dirty:
        raise EvalPlusInfrastructureError('LCB comparator source does not match its supported pin')
    native = load_native(source)
    # The pinned benchmark class performs native public/private case decoding.
    # Its pickle input is only the authenticated official dataset file, never a candidate response.
    problem_source = root / 'lcb_runner/benchmarks/code_generation.py'
    tree = ast.parse(problem_source.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'datasets')]
    problem_module = ModuleType('_trusted_lcb_problems')
    problem_module.__file__ = str(problem_source)
    sys.modules[problem_module.__name__] = problem_module
    namespace = vars(problem_module)
    exec(compile(tree, str(problem_source), 'exec'), namespace)
    problems = {}
    with Path(spec.data_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            problems[str(row['question_id'])] = row
    samples = json.loads(Path(predictions_path).read_text())
    if {str(s['question_id']) for s in samples} != set(problems) or len(samples) != len(problems):
        raise EvalPlusInfrastructureError('LCB denominator mismatch')
    results, started = [], time.monotonic()
    dates = []
    for sample in samples:
        if time.monotonic() - started > spec.timeout_seconds:
            raise EvalPlusInfrastructureError('LCB evaluation wall budget exhausted')
        problem = namespace['CodeGenerationProblem'](**problems[str(sample['question_id'])])
        dates.append(problem.contest_date.isoformat())
        if len(sample['code_list']) != 1:
            raise ValueError('One final submission is required')
        outcomes, metadata, evidence = check_lcb(native, source.read_text(), sample['code_list'][0], problem.get_evaluation_sample())
        passed = bool(outcomes) and all(item is True or (type(item) is int and item == 1) for item in outcomes)
        results.append({'question_id': str(sample['question_id']), 'passed': passed,
                        'outcomes': outcomes, 'official_metadata': metadata, 'isolation': evidence})
        save_json(Path(output_dir) / 'official_results.partial.json', results)
    summary = {'pass@1': sum(r['passed'] for r in results) / len(problems),
               'release_version': spec.release_version, 'date_min': min(dates), 'date_max': max(dates),
               'comparator_sha256': digest(source), 'candidate_python': str(LinuxSandbox().python)}
    save_json(Path(output_dir) / 'official_samples_codegeneration_output_eval.json', [summary, results])
    (Path(output_dir) / 'stdout.txt').write_text(json.dumps(summary))
    (Path(output_dir) / 'stderr.txt').write_text('')
    return {'returncode': 0}
