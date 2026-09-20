"""Bounded real Codex routing on frozen Task evidence; no Task execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sia.task_meta.meta import MetaAgent
from sia.task_meta.pipeline import backend_for, load_config, source_identity
from sia.task_meta.storage import digest, save_json
from sia.task_meta.types import MetaAgentState, MetaObservation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--source-run', required=True)
    args = parser.parse_args()
    directory = (ROOT / args.run_dir).resolve()
    source = (ROOT / args.source_run).resolve()
    if not directory.is_relative_to(ROOT / 'runs') or directory.exists() or not source.is_relative_to(ROOT / 'runs'):
        raise ValueError('A new diagnostic directory and contained historical source are required')
    config = load_config(ROOT / args.config)
    if config.mode != 'api_smoke' or config.meta.run_mode != 'api_smoke':
        raise ValueError('Explicit bounded API diagnostic profile required')
    report = json.loads(Path(config.meta.compatibility_report).read_text())
    if report.get('status') != 'API_SMOKE_PASSED' or not all(report['checks'].values()):
        raise ValueError('Complete native compatibility evidence required')
    observation_file = source / 'gen_0/meta_observation.json'
    observation = MetaObservation(**json.loads(observation_file.read_text()))
    old_request = next(json.loads(p.read_text()) for p in (source / 'meta/calls').glob('*/request.json')
                       if json.loads(p.read_text())['operation'] == 'routing')
    directory.mkdir()
    client, bundles, bundle = backend_for(config, directory)
    if bundle.hash != old_request['meta_harness_hash']:
        raise ValueError('Historical routing evidence uses a different G seed')
    client.bind_context(directory.name, observation.generation, old_request['task_state_hash'])
    state = MetaAgentState(config.meta.model, str(bundle.path / 'instructions.md'),
                          version=bundle.version, bundle_hash=bundle.hash, bundle_path=str(bundle.path))
    provenance = {'source_run': str(source), 'observation_sha256': digest(observation_file),
                  'historical_request_id': old_request['request_id'], 'source_identity': source_identity(),
                  'scope': 'real Meta v2 routing on frozen real T0; no Task execution, update, or SFT',
                  'started_at': datetime.now(timezone.utc).isoformat(), 'api_cost_usd': None}
    save_json(directory / 'diagnostic_status.json', {**provenance, 'status': 'running'})
    try:
        decision = MetaAgent(client, {}).diagnose_and_route(state, observation)
        save_json(directory / 'routing_response.json', decision.model_dump(mode='json'))
        result = {**provenance, 'status': 'ROUTING_RESPONSE_VERIFIED', 'action': decision.action.value,
                  'ended_at': datetime.now(timezone.utc).isoformat(), 'operation': client.last_operation_identity}
        save_json(directory / 'diagnostic_status.json', result)
        print(json.dumps(result))
    except Exception as exc:
        save_json(directory / 'diagnostic_status.json', {**provenance, 'status': 'failed',
                  'error_type': type(exc).__name__, 'ended_at': datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
