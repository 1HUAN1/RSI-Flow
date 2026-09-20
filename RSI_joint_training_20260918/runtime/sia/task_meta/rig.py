"""Window RIG with explicit incomplete-cost and incomparable-set states."""
import math
import json
from pathlib import Path


def compute_rig(before, after, denominator, tokens, *, paired=True, complete=True):
    if not paired:
        return {'value': None, 'reason': 'different_evaluation_identity'}
    if not complete:
        return {'value': None, 'reason': 'incomplete_evaluation_or_usage'}
    if any(type(x) is not int for x in (before, after, denominator)) or not (0 <= before <= denominator and 0 <= after <= denominator):
        return {'value': None, 'reason': 'invalid_counts'}
    if denominator == before:
        return {'value': None, 'reason': 'zero_remaining_success_gap'}
    if type(tokens) not in (int, float) or not math.isfinite(tokens) or tokens <= 0:
        return {'value': None, 'reason': 'missing_or_nonpositive_usage'}
    return {'value': ((after - before) / (denominator - before)) / (tokens / 100000),
            'reason': None, 'before': before, 'after': after, 'denominator': denominator,
            'inference_tokens': tokens}


def report_run(directory):
    """Read actual receipts. Never substitute reservations for measured usage."""
    from .storage import save_json
    root = Path(directory)
    generations = []
    task_tokens = task_calls = task_unknown = 0
    evaluation_dirs = set(root.glob('gen_*'))
    evaluation_dirs.update(p.parent for p in root.glob('round_*/candidates/*/gen_*/execution_receipt.json'))
    for gen in sorted(evaluation_dirs, key=lambda p: p.relative_to(root).as_posix()):
        groups = {}
        for split in ('train', 'probe'):
            rows = [json.loads(p.read_text())['row'] for p in (gen / (split+'_rollouts')).glob('*.json')
                    if '.attempt_' not in p.name]
            for row in rows:
                task_tokens += row.get('input_tokens', 0) + row.get('output_tokens', 0)
                task_calls += row.get('model_call_count', 0)
                task_unknown += row.get('unknown_usage_calls', 0)
            groups[split] = {'completed': len(rows),
                'successes': sum(r.get('verification', {}).get('success') is True for r in rows),
                'parse_errors': sum(r.get('error_type') == 'parse_error' for r in rows),
                'zero_tool_episodes': sum(not r.get('tool_calls') for r in rows),
                'tool_calls': sum(len(r.get('tool_calls', [])) for r in rows),
                'infrastructure_errors': sum(bool(r.get('infrastructure_error')) for r in rows),
                'task_ids': sorted(r['task_id'] for r in rows)}
        update_path = gen/'meta_self_update.json'
        update = json.loads(update_path.read_text()) if update_path.exists() else {}
        generations.append({'generation': int(gen.name.split('_')[1]), 'evaluation_path': gen.relative_to(root).as_posix(), **groups,
            'meta_status': update.get('status'), 'meta_version_changed': update.get('version_changed')})
    records = []
    for call in (root/'meta/calls').glob('*'):
        path = call/'transport_records.json'
        fallback = call/'transport.json'
        if path.exists(): values = json.loads(path.read_text())
        elif fallback.exists(): values = json.loads(fallback.read_text()).get('requests', [])
        else: continue
        seen = set()
        for value in values:
            ordinal = value.get('ordinal')
            if ordinal in seen: continue
            seen.add(ordinal); records.append(value)
    budgets = [json.loads(p.read_text()) for p in (root/'meta/operations').glob('*/budget.json')]
    request_count = sum(b.get('requests', 0) for b in budgets)
    pending = sum(b.get('pending_requests', 0) for b in budgets)
    reserve = sum(b.get('unresolved_output_token_reserve', 0) for b in budgets)
    meta_tokens = known_meta = 0
    for row in records:
        usage = row.get('usage') or {}
        if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')):
            meta_tokens += usage['input_tokens'] + usage['output_tokens']; known_meta += 1
    full_cost = known_meta == request_count == len(records) and not task_unknown and not pending and not reserve
    final_path = root/'final_state.json'
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    sequential = final.get('task_update_policy') == 'sequential_same_parent_first_positive_v1' or any(root.glob('round_*'))
    complete = final.get('status') == 'completed' and (final.get('rounds_completed') == 5 if sequential else len(generations) == 5)
    before, after = (generations[0]['probe'], generations[-1]['probe']) if generations else ({}, {})
    paired = bool(before) and before.get('task_ids') == after.get('task_ids')
    paired = paired and before.get('completed') == after.get('completed') == 10
    result = {'status': final.get('status', 'running'), 'generations': generations,
        'cost': {'task_requests': task_calls, 'task_known_tokens': task_tokens, 'task_unknown_usage': task_unknown,
            'meta_requests': request_count, 'meta_known_usage_requests': known_meta, 'meta_known_tokens': meta_tokens,
            'meta_pending': pending, 'unknown_output_reservations_not_usage': reserve,
            'inference_tokens_complete': full_cost, 'api_cost_usd': None},
        'rig': compute_rig(before.get('successes'), after.get('successes'), before.get('completed'),
            task_tokens + meta_tokens, paired=paired, complete=complete and full_cost),
        'rig_scope': 'same fixed probe: window T0 to deployed T4; all actual in-run inference costs, including final consolidation; excludes separate engineering smoke and SFT training computation',
        'final_test_used': False}
    if sequential:
        result['rounds'] = [json.loads(p.read_text()) for p in sorted(root.glob('round_*/deployment.json'))]
        result['rig'] = None
        result['rig_scope'] = 'Sequential same-parent candidates; see per-attempt paired feedback. No legacy best-generation selection.'
        result['imported_candidate'] = json.loads((root/'initial_candidate.json').read_text()) if (root/'initial_candidate.json').exists() else None
    save_json(root/'method2_progress.json', result)
    return result
