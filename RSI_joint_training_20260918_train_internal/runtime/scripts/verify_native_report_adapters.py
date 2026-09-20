#!/usr/bin/env python3
"""CPU fixtures for official evaluators, never formal predictions or SFT rows."""
import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.evalplus_codec import encode
from sia.task_meta.evalplus_isolated import PINNED_COMMIT, evaluate_evalplus_isolated, load_pinned_evalplus
from sia.task_meta.reporting import OfficialEvaluatorSpec, official_command, parse_official_metrics
from sia.task_meta.storage import digest, save_json


def verify(specs_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    configured = json.loads(Path(specs_path).read_text())['evaluators']
    report = {'decision_source': 'test_override', 'is_formal_benchmark': False, 'eligible_for_sft': False,
              'model_calls': 0, 'tests': {}, 'failures': {}}
    for identifier, value in configured.items():
        spec = OfficialEvaluatorSpec(**{**value, 'benchmark': identifier})
        spec.validate()
        if identifier == 'livecodebench':
            continue  # Native comparator+real isolation covered by test_lcb_isolated.
        directory = destination / identifier
        directory.mkdir()
        try:
            if identifier in {'humaneval_plus', 'mbpp_plus'}:
                load_pinned_evalplus(Path(spec.entrypoint).parents[1], PINNED_COMMIT)
                mbpp = importlib.import_module('evalplus.data.mbpp')
                tasks = [json.loads(line) for line in Path(spec.data_path).read_text().splitlines() if line.strip()]
                failures = []
                cases = 0
                for task in tasks:
                    for kind in ('base_input', 'plus_input'):
                        inputs = mbpp.mbpp_deserialize_inputs(task['task_id'], task[kind]) if identifier == 'mbpp_plus' else task[kind]
                        for number, args in enumerate(inputs):
                            cases += 1
                            try:
                                encode(tuple(args))
                            except ValueError as exc:
                                failures.append({'task_id': task['task_id'], 'kind': kind, 'index': number, 'reason': str(exc)})
                save_json(directory / 'all_inputs_codec_audit.json', {'cases': cases, 'unsupported': failures})
                if failures:
                    raise ValueError(f'{len(failures)} unsupported native input values')
                task = tasks[0]
                data = directory / 'fixture_gold.jsonl'
                data.write_text(json.dumps(task) + '\n')
                predictions = directory / 'fixture_submission.jsonl'
                predictions.write_text(json.dumps({'task_id': task['task_id'], 'solution': task['prompt'] + task['canonical_solution']}) + '\n')
                result = evaluate_evalplus_isolated(replace(spec, data_path=str(data), data_sha256=digest(data),
                                                           subset='first_source_reference_infrastructure_fixture'), predictions, directory)
                official = json.loads((directory / 'official_results.json').read_text())['eval'][task['task_id']][0]
                if not (result['returncode'] == 0 and official['base_status'] == official['plus_status'] == 'pass'):
                    raise ValueError('First reference did not pass native base+plus comparator')
                report['tests'][identifier] = {'native_tasks': len(tasks), 'input_cases_checked': cases,
                    'fixture': task['task_id'], 'base_and_plus_fixture_passed': True,
                    'candidate_source': 'official first canonical solution, test_override only'}
            else:
                task = json.loads(Path(spec.data_path).read_text())[0]
                data = directory / 'fixture_gold.json'
                save_json(data, [task])
                predictions = directory / 'fixture_submission.json'
                answer = {'answer': {task['_id']: ''}, 'sp': {task['_id']: []}, 'evidence': {task['_id']: []}}
                save_json(predictions, answer)
                command = official_command(replace(spec, data_path=str(data)), predictions, directory)
                result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                        env={'PATH': str(Path(sys.executable).parent), 'LANG': 'C.UTF-8'})
                if result.returncode:
                    raise RuntimeError(result.stderr)
                metrics, _ = parse_official_metrics(spec, directory, result.stdout)
                report['tests'][identifier] = {'native_script_executed': True, 'fixture_empty_answer_metrics': metrics}
            report['tests'][identifier]['status'] = 'passed'
        except Exception as exc:
            report['failures'][identifier] = f'{type(exc).__name__}: {exc}'
        save_json(destination / 'verification.json', report)
    report['status'] = 'CPU_NATIVE_ADAPTERS_PASSED' if not report['failures'] else 'CPU_NATIVE_ADAPTERS_FAILED'
    save_json(destination / 'verification.json', report)
    print(json.dumps(report, indent=2))
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs', default='configs/report-evaluators.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(verify(args.specs, args.output))
