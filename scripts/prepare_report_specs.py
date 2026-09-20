#!/usr/bin/env python3
"""CPU-only official evaluation registration; does not generate predictions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.storage import digest, save_json


def prepare(root, data_root):
    target = root / 'data/report_eval'
    target.mkdir(parents=True, exist_ok=True)
    specifications, blocked = {}, {}

    def register(identifier, source, entry, repo, commit, protocol, version, metadata=None):
        if not source.exists():
            blocked[identifier] = {'status': 'BLOCKED_REPORT_DATA', 'reason': 'Missing required native dataset', 'path': str(source)}
            return
        rows = json.loads(source.read_text()) if source.suffix == '.json' else [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if identifier in {'humaneval_plus', 'mbpp_plus'} and any(not {'base_input', 'plus_input', 'canonical_solution', 'entry_point', 'atol'} <= set(row) for row in rows):
            raise ValueError('EvalPlus enhanced native inputs are absent; base tests cannot substitute')
        key = '_id' if identifier in {'hotpotqa_dev', '2wiki_dev'} else 'question_id' if identifier == 'livecodebench' else 'task_id'
        ids = [str(row[key]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate official IDs')
        ids_path = target / (identifier + '_ids.json')
        save_json(ids_path, ids)
        specifications[identifier] = {'repository': repo, 'commit': commit, 'entrypoint': str(entry),
             'entrypoint_sha256': digest(entry), 'python_executable': sys.executable,
             'data_path': str(source), 'data_sha256': digest(source), 'task_ids_path': str(ids_path),
             'task_ids_sha256': digest(ids_path), 'protocol': protocol, 'release_version': version,
             'timeout_seconds': 86400, 'metadata': metadata or {}}

    for identifier, folder, repository, commit, filename, protocol in [
        ('hotpotqa_dev', 'hotpot', 'https://github.com/hotpotqa/hotpot', '3635853403a8735609ee997664e1528f4480762a', 'hotpot_evaluate_v1.py', 'official_hotpot_fullwiki_v1'),
        ('2wiki_dev', '2wiki', 'https://github.com/Alab-NII/2wikimultihop', '13800e5be57df1b4040b9b1588c6c811779e69e9', '2wikimultihop_evaluate.py', 'official_2wiki_original_v1')]:
        checkout = root / 'third_party/report_evaluators' / folder
        actual = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        if actual != commit:
            raise ValueError('Official Search evaluator source changed')
        source = data_root / 'test/searchqa' / identifier / 'dev.jsonl'
        converted = []
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                row['_id'] = row.get('_id', row.get('id'))
                for key in ('context', 'supporting_facts', 'evidences'):
                    value = row.get(key)
                    if isinstance(value, str):
                        row[key] = json.loads(value)
                if isinstance(row['supporting_facts'], dict):
                    sf = row['supporting_facts']
                    row['supporting_facts'] = list(zip(sf['title'], sf['sent_id'], strict=True))
                converted.append(row)
        gold = target / (identifier + '_official.json')
        save_json(gold, converted)
        register(identifier, gold, checkout / filename, repository, commit, protocol, 'v1',
                 {'native_source': str(source), 'source_sha256': digest(source),
                  'answer_aliases': 'original source has no answer_id/evidences_id; use corresponding official v1 evaluator'})
    for identifier, name, version in [('humaneval_plus', 'HumanEvalPlus', 'v0.1.10'), ('mbpp_plus', 'MbppPlus', 'v0.2.0')]:
        register(identifier, target / 'native' / f'{name}-{version}.jsonl',
                 data_root / 'evaluators/evalplus/evalplus/evalplus/evaluate.py',
                 'https://github.com/evalplus/evalplus', '26d6d00bb1fd0fa37f39c99d5290da67891d1c5e',
                 'official_evalplus_native_inputs_ipc', version)
    register('livecodebench', data_root / 'test/code/livecodebench_release_v6/test.jsonl',
             data_root / 'evaluators/livecodebench/LiveCodeBench/lcb_runner/evaluation/testing_util.py',
             'https://github.com/LiveCodeBench/LiveCodeBench', '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24',
             'official_lcb_native_comparator_ipc', 'release_v6', {'coverage': 'all cumulative release_v6; one final submission'})
    for identifier in ('bfcl_v3', 'acebench'):
        blocked[identifier] = {'status': 'BLOCKED_REPORT_ADAPTER',
                              'reason': 'Native multi-turn execution, official subset/output protocol not validated; no aggregate proxy score'}
    save_json(root / 'configs/report-evaluators.json', {'evaluators': specifications, 'blocked': blocked})
    return {'configured': list(specifications), 'blocked': blocked, 'model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/data/RSI_iclr2027/dataset'))
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.data_root), indent=2))
