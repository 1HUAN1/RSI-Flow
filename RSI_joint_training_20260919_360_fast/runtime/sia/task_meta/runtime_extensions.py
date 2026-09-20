"""Controller helpers installed into the isolated runtime, never a historical run."""
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from sia.task_meta.data import ManifestStore, DOMAINS

class FullTrainingStore(ManifestStore):
    compact_rollouts=True
    def window(self, cursor, quotas):
        counts=self.counts()
        return TaskWindow(self),{d:counts[d]['evolve_train'] for d in DOMAINS}

class TaskWindow:
    """Replayable task view; payloads are loaded only for the current GPU batch."""
    def __init__(self,store):self.store=store
    def __iter__(self):return self.store.iter_split('evolve_train')
    def __len__(self):return sum(v['evolve_train'] for v in self.store.counts().values())

def compact_rollout(row,path):
    """Full controller receipt stays on disk; all scoring fields remain exact."""
    from sia.task_meta.storage import digest
    from sia.task_meta.observations import action_failure
    fields=('task_id','question_id','source','domain','split','rollout_id','seed','state_hash','model_ref',
            'harness_sha256','artifact_input_manifest','infrastructure_error','task_source_hash','reset_hash',
            'environment_type','external_task_budget','terminal_reward','metrics','verification','error_type',
            'execution_error_type','valid_answer','output_truncated','chat_template_kwargs','model_call_count',
            'wall_time_seconds','input_tokens','output_tokens','unknown_usage_calls','usage_complete','notes',
            'error','api_failure','api_error','model_call_failed','parse_failure','parse_failed','finish_reason')
    summary={k:row[k] for k in fields if k in row}
    if row.get('model_answer')=='':summary['model_answer']=''
    summary['_action_contract_failure']=action_failure(row)
    summary['_full_rollout_ref']={'path':str(Path(path).resolve()),'sha256':digest(Path(path))}
    return summary

def hydrate_rollout(row, *, for_sft=False):
    """Verify the original bytes and identity before exposing recorded messages."""
    reference=row.get('_full_rollout_ref')
    if reference is None:return row
    from sia.task_meta.storage import digest
    path=Path(reference['path']);root=Path(__file__).resolve().parents[2]/'runs'
    if (not path.resolve().is_relative_to(root.resolve()) or path.parent.name!='train_rollouts'
            or path.is_symlink() or any(p.is_symlink() for p in path.parents) or digest(path)!=reference['sha256']):
        raise ValueError('Full rollout evidence path or hash changed')
    full=json.loads(path.read_text(encoding='utf-8'))['row']
    for field in ('task_id','rollout_id','state_hash','verification','terminal_reward','split','domain'):
        if full.get(field)!=row.get(field):raise ValueError('Compact rollout identity differs: '+field)
    if for_sft:
        # These are duplicate execution logs, not the supervised messages. The
        # complete source hash stays attached to every resulting SFT sample.
        for key in ('transport_calls','model_calls','events','transitions','tool_calls','verifier_details'):
            full.pop(key,None)
        full['_source_rollout_ref']=reference
    return full

def meta_sample(rows, limit=96):
    """Deterministic task/outcome strata; full SFT input stays with the controller."""
    if rows and all(r.get('purpose')=='evolution_train' for r in rows):
        from sia.task_meta.evolution_protocol import representative_rows
        return representative_rows(rows,maximum=limit,per_domain_error=4)
    groups=defaultdict(list)
    for row in rows:
        key=(row.get('domain'),bool(row.get('verification',{}).get('success')))
        group=groups[key]
        ident=str(row.get('task_id',row.get('question_id')))+':'+str(row.get('rollout_id'))
        rank=hashlib.sha256(ident.encode()).hexdigest()
        group.append((rank,row))
    selected=[]
    for group in groups.values():group.sort(key=lambda x:x[0])
    for index in range(limit):
        for key in sorted(groups,key=str):
            if index<len(groups[key]):selected.append(groups[key][index][1])
            if len(selected)==limit:return selected
    return selected

def epoch_complete(epoch, steps, expected_steps, loss):
    import math
    if not math.isfinite(loss) or steps<=0:raise RuntimeError('SFT did not complete valid optimization')
    if expected_steps==-1:
        if epoch is None or not 0.999999<=epoch<=1.000001:raise RuntimeError('SFT did not finish exactly one epoch')
    elif steps!=expected_steps:raise RuntimeError('SFT step budget not completed')
