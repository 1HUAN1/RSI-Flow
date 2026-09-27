"""Evaluate a selected Task state or numbered model/Harness/Artifacts combination."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controller_tools import ControllerTools
from component_versions import ComponentVersions, write_json


def prepare(config, state, output, *, versions_root=None, model_id=None, harness_id=None, artifacts_id=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = Path(state).resolve()
    if versions_root:
        versions = ComponentVersions(Path(versions_root), PROJECT)
        original, ids, _ = versions.task(state)
        composed = versions.compose(state, model_id or ids['model'], harness_id or ids['harness'],
                                    output / 'task_state.json', artifacts_id)
        state = Path(composed['state_path'])
    else:
        write_json(output / 'task_state.json', json.loads(state.read_text()))
        state = output / 'task_state.json'
    tools = ControllerTools(PROJECT, output / 'controller_receipts')
    snapshot = tools.execute({'operation': 'prepare_validation_snapshot', 'task_state_path': str(state),
                              'source_round_path': str(state), 'destination': str(output / 'round_snapshot.json'),
                              'round': 0})
    request = {'operation': 'evaluate', 'config_path': str(Path(config).resolve()),
               'snapshot_path': snapshot.get('snapshot_path'), 'output_dir': str(output / 'job'),
               'role': 'independent_validation'}
    write_json(output / 'evaluation_request.json', request)
    return {'status': 'prepared', 'snapshot': snapshot, 'request': request,
            'purpose': 'report_only', 'executed': False, 'output_dir': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/train_180_a0_v1.json')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--versions-root', type=Path)
    for flag in ('model-id', 'harness-id', 'artifacts-id'):
        parser.add_argument('--' + flag)
    parser.add_argument('--execute', action='store_true', help='Run existing four-GPU official evaluator; otherwise only prepare.')
    args = parser.parse_args(argv)
    if any((args.model_id, args.harness_id, args.artifacts_id)) and not args.versions_root:
        parser.error('Numbered components require --versions-root.')
    report = prepare(args.config, args.state, args.output, versions_root=args.versions_root,
                     model_id=args.model_id, harness_id=args.harness_id, artifacts_id=args.artifacts_id)
    if args.execute and report['snapshot'].get('status') == 'prepared':
        report['result'] = ControllerTools(PROJECT, args.output / 'controller_receipts').execute(report['request'])
        report['executed'] = True
    write_json(args.output / 'task_evaluation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
