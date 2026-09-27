"""Independent read-only evidence checks, with no model calls or selection."""
import json
from pathlib import Path

from sia.task_meta.durable import load_task, task_hash
from sia.task_meta.storage import save_json


def audit_run(run_dir):
    root = Path(run_dir)
    if not (root / 'final_state.json').exists():
        phase = json.loads((root / 'phase.json').read_text()) if (root / 'phase.json').exists() else None
        return {'status': 'pending_no_completed_run', 'phase': phase, 'metrics': None, 'research_complete': False}
    final = json.loads((root / 'final_state.json').read_text())
    errors = []
    scores = final['performance_history']
    n = len(scores)
    if final['status'] == 'completed':
        experiences = [json.loads(line) for line in (root / 'meta/experiences.jsonl').read_text().splitlines()]
        if len(experiences) != max(0, n - 1) or len({e['experience_id'] for e in experiences}) != len(experiences):
            errors.append('Experience count or uniqueness violated')
        if len({r.get('probe_identity') for r in scores}) != 1:
            errors.append('Progress probe changed across generations')
        for generation in range(n):
            directory = root / f'gen_{generation}'
            task_in = load_task(json.loads((directory / 'evaluated_state.json').read_text()))
            train = [json.loads(line) for line in (directory / 'train_trajectories.jsonl').read_text().splitlines()]
            for row in train:
                if row['split'] != 'evolve_train' or row['state_hash'] != task_hash(task_in) or row.get('infrastructure_error'):
                    errors.append(f'Invalid or mixed training evidence at generation {generation}')
                for call in row['transport_calls']:
                    response = call['response']
                    if (response.get('model_ref_response') != task_in.model_ref or
                            response.get('binding', {}).get('weights') != task_in.checkpoint_manifest):
                        errors.append(f'Model checkpoint binding mismatch at generation {generation}')
            if generation < n - 1 and not (directory / 'intervention_receipt.json').exists():
                errors.append(f'Missing committed intervention {generation}')
        if (root / f'gen_{n}').exists():
            errors.append('Unevaluated extra successor exists')
    coverage = json.loads((root / 'coverage.json').read_text()) if (root / 'coverage.json').exists() else {}
    result = {'status': 'evidence_consistent' if not errors else 'invalid', 'errors': errors,
              'run_status': final['status'], 'generations': n, 'experiences': final['experiences'],
              'full_coverage': coverage.get('full_coverage', False),
              'research_complete': False, 'report_eval': 'separate_frozen_evaluation_required'}
    save_json(root / 'pipeline_audit.json', result)
    return result
