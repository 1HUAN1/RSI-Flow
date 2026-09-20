"""Role, provenance and snapshot contracts shared by the existing RSI entrypoints.

This module performs no inference and does not open source datasets implicitly.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROLES = {'train_evolution', 'independent_validation', 'final_test'}
DOMAINS = ('tool_use', 'code', 'searchqa')
SFT_PRESET = dict(finetuning_type='lora', num_train_epochs=1.0, max_steps=-1,
    learning_rate=2e-6, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
    lora_target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    train_base_weights=False, replay_previous_rounds=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Frozen input changed: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never replace a previous run's artifact.
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    return value


def manifest(path, role, round_id):
    if role not in ROLES: raise ValueError('Explicit registered role required')
    value = read(path)
    if (value.get('role'), value.get('round_id')) != (role, round_id):
        raise ValueError('Manifest role/round mismatch')
    if value.get('manifest_hash') != fingerprint({k:v for k,v in value.items() if k!='manifest_hash'}):
        raise ValueError('Manifest hash mismatch')
    required = {'source','version','original_split','task_id','group_id','content_hash','role','round_id','domain'}
    seen = set()
    for row in value['tasks']:
        if not required <= row.keys() or any(row[k] in (None,'') for k in required-{'round_id'}):
            raise ValueError('Incomplete task provenance')
        if (row['role'], row['round_id']) != (role, round_id) or row['task_id'] in seen:
            raise ValueError('Task role/round/identity mismatch')
        seen.add(row['task_id'])
    if not seen: raise ValueError('Empty task manifest; missing data must not become an empty score')
    return value


def make_manifest(role, round_id, tasks, **metadata):
    value = dict(schema_version=2, role=role, round_id=round_id, tasks=tasks, **metadata)
    return {**value, 'manifest_hash':fingerprint(value)}


def assert_disjoint(manifests):
    owners = {}
    for item in manifests:
        owner = (item['role'], item['round_id'])
        for row in item['tasks']:
            for field in ('task_id','group_id','content_hash'):
                key = (field, row[field])
                if key in owners and owners[key] != owner:
                    raise ValueError('Cross-role/round overlap: '+field)
                owners[key] = owner


def authorize_evidence(entries, registry, *, current_round):
    """Transitive provenance allowlist; renaming a final-test artifact cannot launder it."""
    visited = set()
    def visit(identifier, stack):
        if identifier in stack: raise ValueError('Cyclic evidence provenance')
        if identifier in visited: return
        if identifier not in registry: raise ValueError('Evidence is not in the explicit input allowlist')
        item = registry[identifier]
        if item.get('source_role') != 'train_evolution' or item.get('purpose') != 'evolution_train':
            raise ValueError('Meta forbids external validation/test and unknown provenance')
        if item.get('round_id', current_round) > current_round:
            raise ValueError('Future-round evidence is unavailable')
        if not item.get('sha256') or not item.get('path'):
            raise ValueError('Evidence needs a frozen file hash')
        path = Path(item['path'])
        if path.is_symlink() or file_hash(path) != item['sha256']:
            raise ValueError('Evidence bytes changed')
        for parent in item.get('derived_from', []): visit(parent, stack | {identifier})
        visited.add(identifier)
    for identifier in entries: visit(identifier, set())
    return [registry[k] for k in sorted(visited)]


def snapshot(task, meta, *, execution, memory, evaluator, retrieval, prompts):
    """All behavioral inputs, not just a model path. Caller freezes/checks actual files."""
    value = dict(task=task, meta_harness=meta, execution=execution,
                 active_experience=memory, evaluator=evaluator, retrieval=retrieval, prompts=prompts)
    if not all(x is not None for x in value.values()): raise ValueError('Incomplete system snapshot')
    return {**value, 'snapshot_hash':fingerprint(value)}


def cache_key(system, task_manifest, *, role, round_id, seeds, branch_id):
    if role != task_manifest['role']: raise ValueError('Cache role mismatch')
    if system['snapshot_hash'] != fingerprint({k:v for k,v in system.items() if k!='snapshot_hash'}):
        raise ValueError('Snapshot mutated')
    return fingerprint(dict(snapshot=system['snapshot_hash'], manifest=task_manifest['manifest_hash'],
        role=role, round_id=round_id, branch_id=branch_id, seeds=seeds))


def training_provenance(rows, current_manifest, *, round_id, branch_id, harness_hash, parent_policy_hash):
    if (current_manifest['role'],current_manifest['round_id']) != ('train_evolution',round_id):
        raise ValueError('SFT requires this round train_evolution manifest')
    members = {t['task_id']:t for t in current_manifest['tasks']}
    for row in rows:
        expected = dict(source_role='train_evolution', purpose='evolution_train', round_id=round_id, branch_id=branch_id,
                        harness_hash=harness_hash, parent_policy_hash=parent_policy_hash)
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('SFT role/round/branch/harness/parent mismatch')
        task = members.get(row.get('task_id'))
        if not task or row.get('manifest_hash')!=current_manifest['manifest_hash']:
            raise ValueError('SFT trajectory is outside B_r')
        if not row.get('trajectory_id') or not row.get('success_verifier_version'):
            raise ValueError('Missing trajectory/verifier provenance')
        if row.get('derived_from') or row.get('meta_generated'):
            raise ValueError('Derived summaries and Meta text are not Task supervision')
    return rows


def success_filter(rows, *, minimum_samples, native_selector):
    """Trusted native verifier rules remain authoritative. No copied rebalancing."""
    if type(minimum_samples) is not int or minimum_samples < 1: raise ValueError('Invalid SFT threshold')
    reasons = Counter(); candidates = []
    for row in rows:
        verification = row.get('verification', {})
        if verification.get('status')!='completed' or verification.get('success') is not True:
            reasons['not_verified_success'] += 1
        else:
            samples = native_selector([row], profile='multidomain')
            if not samples: reasons['native_supervision_rejected'] += 1
            candidates.extend(samples)
    from_seen = {}; selected = []
    for sample in candidates:
        key = fingerprint({k:sample.get(k) for k in ('messages','tools','chat_template_kwargs','sft_supervision')})
        if key in from_seen: reasons['duplicate_supervision'] += 1
        else: from_seen[key]=True; selected.append(sample)
    report = dict(input_trajectories=len(rows), supervised_samples=len(selected),
        filtering_reasons=dict(reasons), domains=dict(Counter(x['domain'] for x in selected)),
        status='eligible' if len(selected)>=minimum_samples else 'skipped_insufficient_success',
        minimum_samples=minimum_samples, resampling_applied=False)
    return (selected if report['status']=='eligible' else []), report


def feedback_delta(current, previous):
    if previous is None: return None
    if (current['manifest_hash'],current['round_id']) != (previous['manifest_hash'],previous['round_id']):
        raise ValueError('Adaptation deltas require the SAME M_r before and after; never compare M_r with M_(r-1)')
    delta = {}
    for benchmark, scores in current['benchmarks'].items():
        prior = previous['benchmarks'][benchmark]
        delta[benchmark] = {k:v-prior[k] for k,v in scores.items()
                            if type(v) in (int,float) and type(prior.get(k)) in (int,float)}
    return delta


def representative_rows(rows, *, per_domain_error=4, maximum=48, seed=42):
    groups=defaultdict(list)
    for row in rows:
        group=(row['domain'],row.get('error_type') or ('success' if row.get('verification',{}).get('success') else 'failure'))
        groups[group].append(row)
    selected=[]
    for group in sorted(groups):
        ranked=sorted(groups[group],key=lambda r:fingerprint([seed,r['task_id'],r.get('trajectory_id')]))
        groups[group]=ranked[:per_domain_error]
    # Round robin prevents alphabetically early domains from exhausting the cap.
    for i in range(per_domain_error):
        for group in sorted(groups):
            if i<len(groups[group]) and len(selected)<maximum: selected.append(groups[group][i])
    return selected


def assert_reusable_edit(before, after, tasks):
    """Reject introduced benchmark/task lookup keys; evidence citations stay in audit fields."""
    old=json.dumps(before,ensure_ascii=False,sort_keys=True)
    new=json.dumps(after,ensure_ascii=False,sort_keys=True)
    for task in tasks:
        for token in (task['task_id'], task.get('native_id')):
            if token and len(str(token))>=4 and str(token) in new and str(token) not in old:
                raise ValueError('Harness edit introduces an evaluation-task identifier')
    if any(key in new.lower() and key not in old.lower() for key in ('answer_lookup','reference_answers','task_answers')):
        raise ValueError('Harness edit introduces an answer lookup table')
