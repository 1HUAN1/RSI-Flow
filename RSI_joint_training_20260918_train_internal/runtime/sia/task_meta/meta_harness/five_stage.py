"""Five-stage feedback contracts. Uses the existing G snapshot as the atomic G/K store."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

PROTOCOL = 'five-stage-principle-update-v1'
LEGACY_PROTOCOL_HASH = 'a02f286e9742ce06988d1343943feb1ab572e88ac5d26f03c68e6be1059d12f3'
PREVIOUS_PROTOCOL_HASH = '8e2107e7574acb5114f40e6ca7fe74825bf0070255de1daa57865c736c836ee0'
PROTOCOL_HASH = '8500806eda9009b13013c1670aeb8dcdfba5684b0ef916cdd138af95972b325b'

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class PerformanceExpectation(Strict):
    metric: str
    unit: str
    direction: Literal['increase', 'decrease', 'unchanged', 'unknown']
    baseline: float | None
    value_source: str
    predicted_delta: float | None
    interval: list[float] | None
    rationale: str = Field(min_length=1)

class ErrorExpectation(Strict):
    error_type: str
    baseline_count: int | None = Field(ge=0)
    denominator: int | None = Field(ge=0)
    denominator_kind: Literal['tasks', 'tool_calls', 'events', 'unknown']
    target_condition: str
    observable_behavior: str
    evidence: list[str]

class CostExpectation(Strict):
    scope: Literal['one_time_update', 'per_rollout']
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    calls: int | None = Field(ge=0)
    seconds: float | None = Field(ge=0)
    rationale: str = Field(min_length=1)

class Expectations(Strict):
    performance: PerformanceExpectation
    errors: list[ErrorExpectation] = Field(min_length=1, max_length=8)
    implementation_events: list[str] = Field(min_length=1, max_length=16)
    costs: list[CostExpectation] = Field(min_length=2, max_length=2)
    preserve: list[str]
    risks: list[str]
    unknowns: list[str]
    evidence: list[str]
    principle_ids: list[str]

class Principle(Strict):
    principle_id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_.-]{0,95}$')
    revision: int = Field(ge=1)
    active: bool = True
    evidence_state: Literal['tentative', 'supported', 'challenged', 'legacy_inherited']
    library_tags: list[Literal['success', 'failure']] = Field(min_length=1, max_length=2)
    applicability: str = Field(min_length=1, max_length=2000)
    statement: str = Field(min_length=1, max_length=2400)
    procedure: str = Field(min_length=1, max_length=2400)
    scope: Literal['terminal', 'local_behavior', 'cost', 'implementation']
    supporting_evidence: list[str] = Field(max_length=64)
    counter_evidence: list[str] = Field(max_length=64)
    unknowns: list[str] = Field(max_length=16)
    source_experiences: list[str] = Field(max_length=64)
    source_decisions: list[str] = Field(max_length=64)
    source_events: list[str] = Field(max_length=64)
    invalid_when: str = Field(min_length=1)
    g_targets: list[str] = Field(max_length=16)
    expected_next_behavior: str = Field(min_length=1)

class PrincipleOperation(Strict):
    operation: Literal['ADD', 'MERGE', 'REVISE', 'RETIRE']
    principle_id: str
    record: Principle | None
    rationale: str = Field(min_length=1)
    compared_ids: list[str] = Field(max_length=128)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)

class Finding(Strict):
    status: Literal['met', 'partially_met', 'not_met', 'unknown', 'not_applicable']
    explanation: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(max_length=64)
    unknowns: list[str] = Field(max_length=16)

class HarnessBinding(Strict):
    principle_ids: list[str] = Field(min_length=1, max_length=32)
    target: str
    purpose: str
    expected_event: str
    layer: Literal['instruction', 'executable_structure']

class StructuralMismatch(Strict):
    issue_id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_.-]{0,95}$')
    target_workflow: str
    explanation: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)
    missing_information: str = Field(min_length=1, max_length=1000)

class SlowEdit(Strict):
    operation: Literal['Insert', 'Prune', 'Rewire']
    workflow: str
    issue_id: str
    experience_ids: list[str] = Field(min_length=2, max_length=16)
    steps: list[dict] = Field(min_length=1, max_length=16)
    graph: dict
    replay_requirements: list[Literal['schema', 'single_proposal', 'input_presence', 'bounded_paths']] = Field(min_length=1)

class FiveStageReview(Strict):
    protocol: Literal['five-stage-principle-update-v1']
    expectation_id: str | None
    outcome_hash: str | None
    implementation: Finding
    activation: Finding
    outcome: Finding
    principle_operations: list[PrincipleOperation] = Field(max_length=8)
    harness_bindings: list[HarnessBinding] = Field(max_length=16)
    no_change_reason: str | None
    subsequent_verification: str = Field(min_length=1)
    structural_mismatches: list[StructuralMismatch] = Field(default_factory=list, max_length=4)
    slow_edit: SlowEdit | None = None

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(files):
    if 'principles.json' not in files:
        return None
    return {'protocol_hash': hashlib.sha256(files['self_update_protocol.md'].encode()).hexdigest(),
            'principles_hash': hashlib.sha256(files['principles.json'].encode()).hexdigest()}

def validate_library(value):
    if set(value) != {'schema_version', 'records', 'revisions'} or value['schema_version'] != 1:
        raise ValueError('Invalid Meta principle library schema')
    if len(value['records']) > 128 or len(value['revisions']) > 1024:
        raise ValueError('Bounded principle library exceeded')
    seen = set()
    for row in value['records']:
        p = Principle.model_validate(row)
        if p.principle_id in seen:
            raise ValueError('Duplicate principle ID')
        seen.add(p.principle_id)
        for name in ('supporting_evidence', 'counter_evidence', 'source_experiences', 'source_decisions', 'source_events'):
            if len(getattr(p, name)) != len(set(getattr(p, name))):
                raise ValueError('Duplicate evidence must not count as independent support')
    return copy.deepcopy(value)

def library_views(value):
    records = validate_library(value)['records']
    return {tag: [p['principle_id'] for p in records if p['active'] and tag in p['library_tags']]
            for tag in ('success', 'failure')} | {'retired': [p['principle_id'] for p in records if not p['active']]}

def resolve(files, target):
    name, sep, pointer = target.partition('#')
    if name not in {'instructions.md', 'context.json', 'workflow.json', 'evolution.json'}:
        raise ValueError('G target outside existing executable/imperative fields')
    value = json.loads(files[name]) if name.endswith('.json') else files[name].rstrip()
    if sep:
        if not pointer.startswith('/'):
            raise ValueError('G targets use file#/JSON/pointer')
        for key in pointer[1:].split('/'):
            key = key.replace('~1', '/').replace('~0', '~')
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                value = value[int(key)]
            else:
                return None
    return value

def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from _strings(item)
    elif isinstance(value, list):
        for item in value: yield from _strings(item)

def materialize(update, files, *, allowed_evidence=None):
    """Validate K' and G' together; no mutation before the existing atomic commit."""
    before = validate_library(json.loads(files['principles.json']))
    if update.five_stage is None:
        raise ValueError('Native five-stage update requires a structured review')
    review = update.five_stage
    if set(update.bundle_files) - {'instructions.md', 'context.json', 'workflow.json', 'evolution.json'}:
        raise ValueError('Use principle_operations for K; fixed protocol cannot be rewritten')
    after_files = {**files, **update.bundle_files, 'instructions.md': update.harness}
    records = {p['principle_id']: copy.deepcopy(p) for p in before['records']}
    prior = copy.deepcopy(records)
    parent_policy = json.loads(files['evolution.json'])
    method2 = parent_policy.get('schema_version') == 2
    if method2:
        from .graph import fingerprint, topology
        candidate_policy = json.loads(after_files['evolution.json'])
        if fingerprint(topology(parent_policy)) != fingerprint(topology(candidate_policy)):
            raise ValueError('Fast update cannot change node interfaces, handlers, edges, guards or loop bounds; use slow_edit')
        if parent_policy['method2'] != candidate_policy.get('method2'):
            raise ValueError('Graph versions and mismatch history are controller-maintained')
    events = copy.deepcopy(before['revisions'])
    touched = set()
    for operation in review.principle_operations:
        op = operation.model_dump(mode='json'); key = op['principle_id']; old = records.get(key)
        if key in touched:
            raise ValueError('One maintenance operation per principle per update')
        touched.add(key)
        if not set(op['compared_ids']) <= set(prior):
            raise ValueError('Principle comparison cites nonexistent IDs: ' + ', '.join(sorted(set(op['compared_ids']) - set(prior))) + '. Existing principle IDs: ' + ', '.join(sorted(prior)) + '. G rule IDs are not principle IDs; REVISE/MERGE/RETIRE must use exact existing principle IDs.')
        if allowed_evidence is not None and not set(op['evidence_ids']) <= allowed_evidence:
            raise ValueError('Principle maintenance cites unavailable evidence')
        if op['operation'] == 'ADD':
            if old is not None or op['record'] is None:
                raise ValueError('ADD requires a new stable ID and a record')
            candidate = op['record']
            if candidate['revision'] != 1 or candidate['evidence_state'] != 'tentative' or not candidate['active']:
                raise ValueError('New principles start active/tentative at revision 1')
        elif op['operation'] == 'RETIRE':
            if old is None or not old['active'] or op['record'] is not None:
                raise ValueError('RETIRE requires an active ID and null replacement')
            candidate = {**old, 'active': False, 'revision': old['revision'] + 1,
                         'evidence_state': 'challenged',
                         'counter_evidence': sorted(set(old['counter_evidence'] + op['evidence_ids']))}
            for target in old['g_targets']:
                old_value = resolve(files, target)
                if old_value is not None and old_value == resolve(after_files, target):
                    raise ValueError('Retired principle still has an unchanged active G binding')
                if isinstance(old_value, dict) and old_value.get('instruction'):
                    if any(old_value['instruction'] in text for name in ('instructions.md','context.json','workflow.json','evolution.json') for text in _strings(resolve(after_files,name))):
                        raise ValueError('Retired instruction was copied elsewhere in G')
        else:
            if old is None or not old['active'] or op['record'] is None:
                raise ValueError('MERGE/REVISE require an active prior record')
            candidate = op['record']
            if candidate['revision'] != old['revision'] + 1 or not candidate['active']:
                raise ValueError('Principle revision must increment once and remain active')
            if op['operation'] == 'MERGE':
                for field in ('statement', 'procedure', 'applicability', 'scope', 'invalid_when'):
                    if candidate[field] != old[field]:
                        raise ValueError('MERGE preserves the rule; use REVISE for semantic changes')
                if all(set(candidate[f]) <= set(old[f]) for f in ('supporting_evidence', 'counter_evidence', 'source_experiences')):
                    raise ValueError('MERGE needs independent new evidence, not another copy')
        if old and {k:v for k,v in candidate.items() if k != 'revision'} == {k:v for k,v in old.items() if k != 'revision'}:
            raise ValueError('Memory revision must change content or attributed evidence, not only revision')
        if candidate['principle_id'] != key:
            raise ValueError('Replacement principle ID mismatch')
        for other_key, other in records.items():
            if other_key != key and other['active'] and candidate['active']:
                if (other['statement'].strip().casefold(), other['applicability'].strip().casefold()) == (candidate['statement'].strip().casefold(), candidate['applicability'].strip().casefold()):
                    raise ValueError('Equivalent active principle already exists; inspect and MERGE')
        if allowed_evidence is not None:
            for field in ('supporting_evidence', 'counter_evidence', 'source_experiences', 'source_decisions', 'source_events'):
                added = set(candidate[field]) - set((old or {}).get(field, []))
                if not added <= allowed_evidence:
                    raise ValueError('New principle source is outside the permitted evidence package')
        records[key] = candidate
        events.append({'operation': op['operation'], 'principle_id': key, 'rationale': op['rationale'],
                       'compared_ids': op['compared_ids'], 'evidence_ids': sorted(set(op['evidence_ids'])),
                       'before_hash': digest(old) if old else None, 'after_hash': digest(candidate),
                       'expectation_id': review.expectation_id, 'outcome_hash': review.outcome_hash})
    new_library = validate_library({'schema_version': 1, 'records': list(records.values()), 'revisions': events})
    changed_files = [n for n in ('instructions.md', 'context.json', 'workflow.json', 'evolution.json')
                     if resolve(files, n) != resolve(after_files, n)]
    for binding in review.harness_bindings:
        if not set(binding.principle_ids) <= {k for k, p in records.items() if p['active']}:
            # Retirement is allowed to remove a binding using its retired ID.
            if not set(binding.principle_ids) <= touched:
                raise ValueError('G patch cites no retained or explicitly retired principle')
        if resolve(files, binding.target) == resolve(after_files, binding.target):
            raise ValueError('Claimed G patch target is unchanged')
        for key in binding.principle_ids:
            if records[key]['active'] and binding.target not in records[key]['g_targets']:
                raise ValueError('G patch binding must be registered in the principle record')
    if set(changed_files) - {b.target.partition('#')[0] for b in review.harness_bindings}:
        raise ValueError('Every changed G file must bind a principle and a subsequent observable')
    k_changed = canonical(new_library) != canonical(before)
    if method2 and not k_changed:
        raise ValueError('Every round fast update must change grounded Meta Memory; facts or unresolved evidence are valid')
    if update.status == 'NO_CHANGE' and (k_changed or changed_files):
        raise ValueError('NO_CHANGE cannot alter K or G')
    if update.status == 'UPDATED' and not (k_changed or changed_files):
        raise ValueError('UPDATED cannot be formatting, version or timestamp only')
    after_files['principles.json'] = canonical(new_library)
    if update.status != 'UPDATED' or not (k_changed or changed_files):
        raise ValueError('Fast update requires real Meta content change; grounded Memory-only updates are valid, NO_CHANGE is not')
    slow_event = {'status': 'not_proposed'}
    if method2:
        from .graph import apply_slow, validate
        policy = json.loads(after_files['evolution.json'])
        for mismatch in review.structural_mismatches:
            row = mismatch.model_dump(mode='json')
            if not update.experience_id or row['target_workflow'] not in policy['graphs']:
                raise ValueError('Mismatch must bind an actual intervention and existing operation')
            if allowed_evidence is not None and not set(row['evidence_ids']) <= allowed_evidence:
                raise ValueError('Mismatch cites unavailable evidence')
            row['experience_id'] = update.experience_id
            if not any(x['issue_id'] == row['issue_id'] and x['experience_id'] == row['experience_id'] for x in policy['method2']['mismatches']):
                policy['method2']['mismatches'].append(row)
        validate(policy)
        policy, slow_event = apply_slow(policy, review.slow_edit, policy['method2']['mismatches'], experience_id=update.experience_id)
        after_files['evolution.json'] = canonical(policy)
    elif review.slow_edit or review.structural_mismatches:
        raise ValueError('Graph evolution requires an explicitly migrated Method 2 Bundle')
    return after_files, {'principle_memory_updated': k_changed, 'harness_policy_changed': bool(changed_files),
        'fast_status': 'content_updated', 'slow_update': slow_event,
        'operations': len(review.principle_operations), 'views': library_views(new_library)}

def _read(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Evidence symlinks are not permitted')
    return json.loads(path.read_text())

def _once(path, value):
    from sia.task_meta.storage import save_json
    if path.exists():
        if _read(path) != value:
            raise ValueError('Frozen expectation/outcome cannot be rewritten')
    else:
        save_json(path, value)

def freeze_expectation(run_dir, task, meta, decision, observation):
    if not meta.bundle_path or not (Path(meta.bundle_path)/'principles.json').exists():
        return None
    from sia.task_meta.meta_harness.bundle import MetaHarnessBundle
    bundle = MetaHarnessBundle(Path(meta.bundle_path), _read(Path(meta.bundle_path)/'manifest.json')).verify()
    exp = Expectations.model_validate(decision.expectations)
    if {c.scope for c in exp.costs} != {'one_time_update', 'per_rollout'}:
        raise ValueError('Expectations need one-time and per-rollout costs, unknown values allowed')
    active = {p['principle_id'] for p in json.loads(bundle.read_files()['principles.json'])['records'] if p['active']}
    if not set(exp.principle_ids) <= active:
        raise ValueError('Expectation cites unknown/retired principles')
    baseline = observation.current_performance.get(exp.performance.metric)
    if exp.performance.baseline is not None and (type(baseline) not in (int, float) or abs(baseline-exp.performance.baseline) > 1e-9):
        raise ValueError('Expectation baseline must match the trusted current development metric or be unknown')
    payload = {'protocol': PROTOCOL, **identity(bundle.read_files()), 'g_hash': bundle.hash,
        'task_state': asdict(task), 'decision': decision.model_dump(mode='json'), 'decision_id': decision.decision_id,
        'expectations': exp.model_dump(mode='json'), 'expected_effect_legacy': decision.expected_effect,
        'baseline_metrics': copy.deepcopy(observation.current_performance), 'external_budget': copy.deepcopy(observation.budget),
        'probe_hash': observation.current_performance.get('probe_identity'),
        'creation_phase': 'before_first_task_intervention_side_effect', 'counterfactual_components': 'unobserved'}
    payload['expectation_id'] = 'expectation_' + digest(payload)
    path = Path(run_dir)/f'gen_{task.generation}'/f'expectation_{decision.decision_id}.json'
    if not path.exists() and (path.parent/'intervention_receipt.json').exists():
        raise ValueError('Cannot backfill an expectation after intervention dispatch')
    _once(path, payload)
    return payload

def _probe_rows(root, generation):
    path = Path(root)/f'gen_{generation}'/'probe_trajectories.jsonl'
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Probe evidence escaped trusted run')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    allowed = ('task_id', 'rollout_id', 'seed', 'task_source_hash', 'reset_hash', 'environment_type',
               'external_task_budget', 'state_hash', 'harness_sha256', 'model_ref', 'artifact_input_manifest',
               'terminal_reward', 'error_type', 'infrastructure_error', 'model_call_count', 'input_tokens',
               'output_tokens', 'unknown_usage_calls', 'wall_time_seconds', 'messages', 'events',
               'tool_calls', 'harness_identity', 'transitions', 'chat_template_kwargs')
    result = []
    for row in rows:
        training_probe = row.get('source_role') == 'train_evolution' and row.get('purpose') == 'evolution_train'
        if row.get('split') != 'search_dev' and not (training_probe and row.get('split') == 'evolve_train'):
            raise ValueError('Paired feedback only admits registered internal development probe')
        item = {k: copy.deepcopy(row[k]) for k in allowed if k in row}
        item['success'] = row.get('verification', {}).get('success')
        item['native_partial_metrics'] = {k:v for k,v in row.get('metrics', {}).items() if type(v) in (int,float,bool)}
        item['source_id'] = f"dev:{generation}:{row['task_id']}:{row['rollout_id']}"
        item['source_hash'] = digest(row)
        item['split'] = 'training_meta_probe' if training_probe else 'internal_dev_readonly'
        if training_probe:
            item.update(round_id=row['round_id'],manifest_hash=row['manifest_hash'])
        item['excluded_from_sft_and_task_assets'] = not training_probe
        if training_probe: item['source_id'] = f"train_probe:{row['round_id']}:{generation}:{row['task_id']}:{row['rollout_id']}"
        item['mechanism_event_counts'] = dict(Counter(event['kind'] for event in row.get('events', [])))
        item['tool_error_count'] = sum(e.get('kind') == 'tool_error' for e in row.get('events', []))
        item['tool_call_denominator'] = len(row.get('tool_calls', []))
        result.append(item)
    return result

def paired_training_summary(root, before_gen, after_gen):
    """Compact same-training-task evidence; never substitute it for held-out probe."""
    def rows(g):
        path=Path(root)/f'gen_{g}'/'train_trajectories.jsonl'
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Training comparison escaped trusted run')
        values=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(row.get('split') != 'evolve_train' for row in values):
            raise ValueError('Training comparison only admits registered training rollouts')
        keyed={(r['task_id'],r['rollout_id'],r['seed']):r for r in values}
        if len(keyed)!=len(values):raise ValueError('Duplicate training rollout identity')
        return keyed
    left,right=rows(before_gen),rows(after_gen)
    if set(left)!=set(right):
        return {'status':'different_windows_not_paired','denominator':0,'pairs':[]}
    fields=('task_source_hash','reset_hash','environment_type','external_task_budget','chat_template_kwargs')
    def compact(row):
        return {'success':row.get('verification',{}).get('success'),
            'partial_score':row.get('metrics',{}).get('native_partial_score'),
            'error_type':row.get('error_type'),'infrastructure_error':bool(row.get('infrastructure_error')),
            'calls':row.get('model_call_count'),'input_tokens':row.get('input_tokens'),
            'output_tokens':row.get('output_tokens'),'unknown_usage_calls':row.get('unknown_usage_calls'),
            'mechanism_counts':dict(Counter(e['kind'] for e in row.get('events',[]))),
            'source_hash':digest(row)}
    pairs=[]
    for key in sorted(left):
        a,b=left[key],right[key]
        comparable=all(a.get(f) is not None and a.get(f)==b.get(f) for f in fields)
        available=comparable and not a.get('infrastructure_error') and not b.get('infrastructure_error')
        aa,bb=compact(a),compact(b)
        delta=int(bb['success'])-int(aa['success']) if available and type(aa['success']) is bool and type(bb['success']) is bool else None
        pairs.append({'source_id':f'train_pair:{before_gen}_{after_gen}:'+digest(list(key))[:16],
            'task_id':key[0],'rollout_id':key[1],'seed':key[2],'comparable':comparable,
            'before':aa,'after':bb,'success_delta':delta})
    complete=bool(pairs) and all(p['success_delta'] is not None for p in pairs)
    return {'status':'paired_fixed_training_tasks','scope':'training_replay_not_heldout',
        'denominator':len(pairs),'comparable_pairs':sum(p['comparable'] for p in pairs),
        'complete':complete,'success_delta':sum(p['success_delta'] for p in pairs)/len(pairs) if complete else None,
        'failed_to_success':sum(p['success_delta']==1 for p in pairs),
        'success_to_failed':sum(p['success_delta']==-1 for p in pairs),'pairs':pairs,
        'interpretation':'Observed changes on reused training tasks; fixed development probe remains the primary metric. Mechanism counts alone do not establish opportunity or causality.'}


def paired_outcome(root, experience):
    trusted_root = Path(root).resolve()
    if experience.feedback_root:
        root = Path(experience.feedback_root).resolve()
        if not root.is_relative_to(trusted_root):
            raise ValueError('Round feedback escaped the trusted run')
    before_gen = experience.generation
    after_gen = before_gen + 1
    decision_id = experience.decision.get('decision_id', '')
    expected_path = Path(root)/f'gen_{before_gen}'/f'expectation_{decision_id}.json'
    expectation = _read(expected_path) if expected_path.exists() else {'recording': 'legacy/not_recorded',
        'original_text': experience.decision.get('expected_effect'), 'expectation_id': None}
    before, after = _probe_rows(root,before_gen), _probe_rows(root,after_gen)
    key = lambda r:(r['task_id'],r['rollout_id'],r['seed'])
    left,right={key(r):r for r in before},{key(r):r for r in after}
    if len(left)!=len(before) or len(right)!=len(after):raise ValueError('Duplicate paired probe identity')
    pairs=[]
    binding_keys=('task_source_hash','reset_hash','environment_type','external_task_budget','chat_template_kwargs')
    for k in sorted(set(left)|set(right)):
        a,b=left.get(k),right.get(k)
        comparable=bool(a and b and all(a.get(f) is not None and a.get(f)==b.get(f) for f in binding_keys))
        available=comparable and not a.get('infrastructure_error') and not b.get('infrastructure_error')
        success_delta=int(b['success'])-int(a['success']) if available and type(a['success']) is bool and type(b['success']) is bool else None
        pairs.append({'key':list(k),'before':a,'after':b,'comparable':comparable,'success_delta':success_delta,
            'native_reward_delta':b['terminal_reward']-a['terminal_reward'] if available and type(a.get('terminal_reward')) in (int,float) and type(b.get('terminal_reward')) in (int,float) else None,
            'partial_delta':b['native_partial_metrics']['native_partial_score']-a['native_partial_metrics']['native_partial_score'] if available and all(type(row.get('native_partial_metrics',{}).get('native_partial_score')) in (int,float) for row in (a,b)) else None,
            'partial_metric':'native_partial_score',
            'usage_delta':{f:b.get(f,0)-a.get(f,0) for f in ('input_tokens','output_tokens','model_call_count','wall_time_seconds')} if available and not (a.get('unknown_usage_calls') or b.get('unknown_usage_calls')) else None,
            'mechanism_activation':'unknown_unless_explicit_event_and_opportunity',
            'artifact_state_changed':a.get('artifact_input_manifest')!=b.get('artifact_input_manifest') if a and b else None})
    complete=bool(pairs) and all(p['comparable'] and p['success_delta'] is not None for p in pairs)
    delta=sum(p['success_delta'] for p in pairs)/len(pairs) if complete else None
    value={'protocol':PROTOCOL,'experience_id':experience.experience_id,'expectation':expectation,
        'pairs':pairs,'paired_evidence_complete':complete,'denominator':len(pairs),
        'success_delta':delta,'failed_to_success':sum(p['success_delta']==1 for p in pairs),
        'success_to_failed':sum(p['success_delta']==-1 for p in pairs),
        'comparison':'observed_transition_not_isolated_causality','train_windows':'diagnostic_only_not_paired',
        'feedback_scope':'new_protocol_readonly_internal_dev_observations_predictions_scores_no_gold',
        'error_counts': {side:dict(Counter(r.get('error_type') or 'none' for r in rows)) for side,rows in [('before',before),('after',after)]},
        'implementation':{'status':'unknown','actual_change':experience.actual_change,'versions':experience.versions},
        'activation':{'status':'unknown','reason':'Meta must cite explicit mechanism events/opportunities'},
        'outcome':{'status':'unknown','observed_success_delta':delta,'reason':'Actual delta is distinct from expectation satisfaction'}}
    partial_pairs=[p for p in pairs if p['partial_delta'] is not None]
    value['comparable_pairs']=sum(p['comparable'] for p in pairs)
    value['partial_score_comparison']={
        'metric':'native_partial_score','denominator':len(partial_pairs),
        'complete':bool(pairs) and len(partial_pairs)==len(pairs),
        'before_mean':sum(p['before']['native_partial_metrics']['native_partial_score'] for p in partial_pairs)/len(partial_pairs) if partial_pairs else None,
        'after_mean':sum(p['after']['native_partial_metrics']['native_partial_score'] for p in partial_pairs)/len(partial_pairs) if partial_pairs else None,
        'mean_delta':sum(p['partial_delta'] for p in partial_pairs)/len(partial_pairs) if partial_pairs else None}
    protocol_file=Path(root)/'protocol.json'
    if protocol_file.exists() and _read(protocol_file).get('config',{}).get('training_schedule')=='fixed_subset':
        value['training_comparison']=paired_training_summary(root,before_gen,after_gen)
        value['train_windows']='same_fixed_training_tasks_paired_diagnostic_not_heldout'
    if experience.candidate_attempts:
        value['candidate_attempts'] = copy.deepcopy(experience.candidate_attempts)
        value['deployment_status'] = experience.deployment_status
        value['learning_scope'] = 'all_attempts_including_rejected_candidates_not_only_deployed_state'
        value['gain_attribution'] = 'Meta snapshot that generated each candidate; newer Meta assessed on later candidates'
        value['expectation'] = {'expectation_id': None, 'recording': 'per_candidate_before_side_effects',
                                'per_candidate': [a.get('expectation') for a in experience.candidate_attempts]}
    if protocol_file.exists() and _read(protocol_file).get('config',{}).get('training_schedule') == 'full_cohort':
        value['feedback_scope'] = 'in_training_monitor_not_external_validation_no_gold'
        value['train_windows'] = 'full_training_cohort_reused_each_round'
        for pair in value['pairs']:
            for side in ('before','after'):
                if pair.get(side): pair[side]['split'] = 'in_training_monitor_readonly'
    if protocol_file.exists() and _read(protocol_file).get('config',{}).get('training_schedule') == 'round_disjoint':
        if any(a.get('round_id') != experience.generation+1 or a.get('manifest_hash') != b.get('manifest_hash') for a,b in zip(before,after)):
            raise ValueError('Training adaptation requires the same current-round M_r')
        value['feedback_scope']='training_meta_probe_adaptation_not_held_out'
        value['train_windows']='current_B_r_only'
    value['outcome_hash']=digest(value)
    _once(Path(root)/f'gen_{after_gen}'/'outcome_review.json',value)
    return value
