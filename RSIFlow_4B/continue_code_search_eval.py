"""Complete the five Code/Search official evaluations after skipping ACE.

This is a reporting-only partial validation. It retains the 300-task manifest
and existing BFCL score; ACE remains pending, so no round completion is issued.
"""
import argparse
import json
import sys
from pathlib import Path

from common import ROOT, immutable, read, write, sha

sys.path.insert(0, str(ROOT / 'runtime'))

from evaluation_manifest import select_specs
from evaluation_results import official_task_scores, publish
from dynamic_predictions import generate_dynamic
from parallel_predictions import generate_parallel
from sia.task_meta.durable import load_task, task_hash
from sia.task_meta.gpu_phases import ensure_services, verify_services
from sia.task_meta.harnessforge_manifest import load_manifest
from sia.task_meta.pipeline import load_config
from sia.task_meta.reporting import OfficialEvaluatorSpec, evaluate_official
from sia.task_meta.storage import checkpoint_manifest
from validation_sources import verify


CODE_SEARCH = ('livecodebench', 'humaneval_plus', 'mbpp_plus',
               'hotpotqa_dev', '2wiki_dev')


def run(snapshot_path, pipeline_path, validation_path, manifest_path):
    snapshot_path = Path(snapshot_path).resolve()
    pipeline_path = Path(pipeline_path).resolve()
    validation_path = Path(validation_path).resolve()
    out = snapshot_path.parent
    snapshot = read(snapshot_path)
    config = load_config(pipeline_path)
    settings = read(validation_path)
    if settings['benchmark_ids'] != ['bfcl_v3', 'acebench', *CODE_SEARCH]:
        raise ValueError('Unexpected seven-benchmark order')
    verify(settings, Path(config.output_root) / 'validation_sources.json')
    manifest, specs = select_specs(settings, manifest_path, 'independent_validation',
                                   out / 'inputs')
    frozen = read(out / 'frozen_task.json')
    task = load_task(frozen['task_state'])
    if (snapshot['task_state'] != frozen['task_state']
            or frozen['source_role'] != 'independent_validation'
            or frozen['manifest_hash'] != manifest['manifest_hash']
            or frozen['state_hash'] != task_hash(task)
            or frozen['checkpoint_files'] != checkpoint_manifest(task.checkpoint_path)
            or frozen['task_harness_identity'] != load_manifest(task.harness_path).identity
            or frozen['validation_config_sha256'] != sha(validation_path)
            or frozen['system_snapshot_sha256'] != sha(snapshot_path)
            or frozen['evaluation_config_sha256'] != sha(pipeline_path)):
        raise ValueError('Frozen validation Task or manifest changed')
    if (out / 'complete.json').exists():
        raise ValueError('Independent validation is already complete')
    bfcl = read(out / 'scores/bfcl_v3/result.json')
    if bfcl.get('status') != 'completed' or bfcl.get('state_hash') != frozen['state_hash']:
        raise ValueError('BFCL official result is unavailable or not for this Task')
    if (out / 'scores/acebench/result.json').exists():
        raise ValueError('ACE result already exists; inspect before partial continuation')
    ensure_services(config, task.checkpoint_path)
    immutable(out / 'serving_fingerprint.json', verify_services(config, task.checkpoint_path))
    results = [{'benchmark_id': 'bfcl_v3', **bfcl}]
    metrics = publish(manifest, snapshot, frozen, results, out)
    for identifier in CODE_SEARCH:
        directory = out / 'scores' / identifier
        result_path = directory / 'result.json'
        spec = OfficialEvaluatorSpec(**{**specs[identifier], 'benchmark': identifier})
        spec.validate()
        if result_path.exists() and read(result_path).get('status') == 'completed':
            result = read(result_path)
            if result['state_hash'] != frozen['state_hash']:
                raise ValueError('Cached official score belongs to another Task')
        else:
            single = out / (identifier + '.spec.json')
            immutable(single, {'evaluators': {identifier: specs[identifier]}})
            predictions = out / 'predictions' / identifier
            generator = generate_parallel if identifier == 'livecodebench' else generate_dynamic
            generator(config, out / 'frozen_task.json', single, predictions)
            rows = [json.loads(line) for line in
                    (predictions / (identifier + '.jsonl')).read_text().splitlines()
                    if line.strip()]
            result = evaluate_official(spec, rows, frozen, directory)
            scores = official_task_scores(identifier, spec, rows, directory, frozen) if result.get('status') == 'completed' else {}
            result['task_results'] = [dict(
                native_id=str(row['task_id']), execution_status='completed',
                scoring_status='completed' if str(row['task_id']) in scores else 'pending',
                **scores.get(str(row['task_id']), {'success': None, 'native_scores': {}}),
                tokens=[call.get('usage') for call in row.get('transport_calls', [])],
                tool_calls=len(row.get('trajectory', {}).get('tool_calls', [])),
                latency=row.get('wall_seconds'),
                trajectory_path=str(predictions / (identifier + '.jsonl')))
                for row in rows]
            write(result_path, result)
        results.append({'benchmark_id': identifier, **result})
        write(out / 'results.json', results)
        metrics = publish(manifest, snapshot, frozen, results, out)
        print(f'[code-search-eval] {identifier}: {result.get("status")}; '
              f'officially scored {metrics["overall"]["n_scored"]}/300', flush=True)
        if result.get('status') != 'completed':
            raise RuntimeError('Official Code/Search evaluation incomplete: ' + identifier)
    if metrics['overall']['n_scored'] != 250 or metrics['overall']['complete']:
        raise ValueError('Partial validation must have 250 scored and ACE pending')
    receipt = {'status': 'code_search_completed_ace_pending',
               'source_role': 'independent_validation', 'feedback_to_meta': False,
               'task_state_hash': frozen['state_hash'],
               'benchmarks_completed': ['bfcl_v3', *CODE_SEARCH],
               'acebench_status': 'incomplete_requires_audit',
               'officially_scored': 250, 'expected_total': 300,
               'metrics_sha256': sha(out / 'task_metrics.json')}
    immutable(out / 'code_search_partial_complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--pipeline-config', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(run(args.snapshot, args.pipeline_config, args.config, args.manifest), flush=True)


if __name__ == '__main__':
    main()
