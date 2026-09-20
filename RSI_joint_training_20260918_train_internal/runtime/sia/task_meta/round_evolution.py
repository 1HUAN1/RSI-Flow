"""Training-only round bindings for the existing native RSI controller and executors."""
import copy
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from sia.task_meta.evolution_protocol import (manifest, freeze, fingerprint, file_hash, read,
    authorize_evidence, training_provenance, success_filter, SFT_PRESET)
from sia.task_meta.data import ManifestStore, TaskRecord, _prompt, content_hash


class RoundStore(ManifestStore):
    compact_rollouts = False

    def __init__(self, path):
        super().__init__(path)
        self.root = Path(path).parent
        self.round_id = None
        self.mode = 'probe'

    def select_round(self, number):
        if number not in (1, 2, 3): raise ValueError('Exactly three training rounds')
        self.current = manifest(self.root/f'B{number}/manifest.json', 'train_evolution', number)
        self.probes = manifest(self.root/f'B{number}/meta_probe.json', 'train_evolution', number)
        self.members = {r['task_id']: r for r in self.current['tasks']}
        if self.probes['parent_manifest_hash'] != self.current['manifest_hash']:
            raise ValueError('M_r parent B_r changed')
        if any(self.members.get(r['task_id']) != r or not r['meta_probe'] for r in self.probes['tasks']):
            raise ValueError('M_r must be the marked subset of B_r')
        self.round_id = number
        self.verified_files = {}

    def _task(self, row):
        if row['round_id'] != self.round_id or row['purpose'] != 'evolution_train':
            raise ValueError('Future or external task unavailable')
        path=Path(row['record_file'])
        if path.is_symlink() or not path.resolve().is_relative_to((self.root/f'B{self.round_id}').resolve()):
            raise ValueError('Training payload escaped current B_r')
        if path not in self.verified_files: self.verified_files[path]=file_hash(path)
        if self.verified_files[path]!=row['record_sha256']:
            raise ValueError('Training payload differs from frozen manifest')
        with path.open('rb') as f:
            f.seek(row['record_offset']); payload = json.loads(f.read(row['record_length']))
        return TaskRecord(row['task_id'], row['domain'], row['source'], 'evolve_train',
            _prompt(payload, row['domain']), payload, row['record_sha256'], content_hash(_prompt(payload,row['domain'])))

    def window(self, cursor, quotas):
        if self.round_id is None: raise ValueError('Explicit current round is required')
        tasks = [self._task(r) for r in self.current['tasks']] if self.mode == 'training' else []
        return tasks, dict(Counter(t.domain for t in tasks))

    def probe(self):
        if self.round_id is None: raise ValueError('Explicit current round is required')
        return [self._task(r) for r in self.probes['tasks']]

    def iter_split(self, split, domain=None):
        if self.round_id is None or split != 'evolve_train': raise ValueError('Round-local training access only')
        for r in self.current['tasks']:
            if domain is None or r['domain'] == domain: yield self._task(r)

    def coverage(self, cursor):
        return dict(allocated_tasks=len(self.current['tasks']), all_tasks_scheduled=self.mode=='training',
                    round_id=self.round_id, meta_probe_is_training_subset=True)


def scoped_rollout(executor, state, task, rollout, directory, binding):
    """One physical receipt per task/config, shared by M_r and B_r, never across states."""
    protocol = getattr(executor, 'round_protocol', None)
    if protocol is None: return directory, binding
    member = executor.store.members.get(task.task_id)
    if member is None: raise ValueError('Task is outside current B_r')
    binding = {**binding, 'round_scope': protocol.execution_scope,
               'task_record': member['content_hash']}
    # Native scratch artifact paths and generation labels are not behavioral changes.
    binding['state_hash'] = fingerprint(protocol.execution_scope)
    target = protocol.root/'rollout_cache'/fingerprint(protocol.execution_scope)/'train_rollouts'
    target.mkdir(parents=True, exist_ok=True)
    return target, binding


def annotate_row(executor, state, row):
    protocol = getattr(executor, 'round_protocol', None)
    if protocol is None: return
    m = executor.store.current
    row.update(source_role='train_evolution', purpose='evolution_train', round_id=m['round_id'],
        meta_probe=executor.store.members[row['task_id']]['meta_probe'],
        branch_id='parent', manifest_hash=m['manifest_hash'], harness_hash=file_hash(state.harness_path),
        parent_policy_hash=fingerprint(state.checkpoint_manifest),
        success_verifier_version=protocol.execution_scope['protocol_hash'],
        trajectory_id=fingerprint([protocol.execution_scope, row['task_id'], row['rollout_id']]))
    row['notes'] = []  # Evaluation never promotes task-local notes into active memory.


def select_sft(updater, task, context):
    from sia.task_meta.sft import select_positive_rows
    if not updater.training.get('round_protocol'):
        return select_positive_rows(context.evaluation.trajectories, profile=updater.sft_profile)
    config = updater.training['round_protocol']
    r = context.generation
    current = manifest(Path(config['data_dir'])/f'B{r}/manifest.json', 'train_evolution', r)
    rows = context.evaluation.trajectories
    training_provenance(rows, current, round_id=r, branch_id='parent',
        harness_hash=file_hash(task.harness_path), parent_policy_hash=fingerprint(task.checkpoint_manifest))
    positives, report = success_filter(rows, minimum_samples=config['minimum_sft_samples'], native_selector=select_positive_rows)
    freeze(context.directory/'sft_filter.json', report)
    return positives


def validate_trainer_rows(rows, request):
    config = request['training'].get('round_protocol')
    if not config: return
    for key, expected in SFT_PRESET.items():
        if request['training'].get(key) != expected: raise ValueError('SFT preset changed: '+key)
    source = request['source_binding']
    current = manifest(Path(config['data_dir'])/f"B{source['round_id']}/manifest.json", 'train_evolution', source['round_id'])
    training_provenance(rows, current, **source)
    if len(rows)<config['minimum_sft_samples']: raise ValueError('SFT minimum not met')
    keys=[fingerprint({k:r.get(k) for k in ('messages','tools','chat_template_kwargs')}) for r in rows]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate supervision')


def check_lora_modules(model, options):
    targets = options['lora_target_modules']
    available={name.rsplit('.',1)[-1] for name, module in model.named_modules()
               if hasattr(module,'weight') and getattr(module.weight,'ndim',0)==2}
    if not targets or set(targets)-available: raise ValueError('LoRA targets absent from actual model: '+str(set(targets)-available))
    if getattr(model,'is_loaded_in_4bit',False) or getattr(model,'is_loaded_in_8bit',False):
        raise ValueError('Quantized parent merging is not supported by this trainer')
    for p in model.parameters(): p.requires_grad_(False)


def validate_meta_edit(candidate, meta, tasks):
    from sia.task_meta.meta import validate_meta_candidate
    from sia.task_meta.types import MetaHarnessUpdate
    from sia.task_meta.evolution_protocol import assert_reusable_edit
    update=MetaHarnessUpdate.model_validate(candidate)
    result=validate_meta_candidate(candidate,meta) if meta.bundle_path else update
    assert_reusable_edit(Path(meta.harness_path).read_text(),update.harness,tasks)
    for name,content in update.bundle_files.items():
        assert_reusable_edit((Path(meta.bundle_path)/name).read_text(),content,tasks)
    if update.five_stage:
        previous={p['principle_id']:p for p in read(Path(meta.bundle_path)/'principles.json')['records']}
        # Citations may identify evidence; reusable policy content may not encode task lookups.
        fields=('applicability','statement','procedure','unknowns','invalid_when','expected_next_behavior')
        for op in update.five_stage.principle_operations:
            if op.record is not None:
                old=previous.get(op.principle_id,{})
                current=op.record.model_dump()
                assert_reusable_edit({k:old.get(k) for k in fields},{k:current.get(k) for k in fields},tasks)
    return result


class RoundProtocol:
    """Hooks only: native loop, Meta backend, candidate policy and trainer remain authoritative."""
    def __init__(self, config, root, executor):
        self.config=config; self.root=Path(root); self.executor=executor
        self.native=executor.executor; self.store=self.native.store
        self.native.round_protocol=self
        self.execution_scope=None; self.registry={}; self.active_memory=[]

    def begin(self, number, meta, history):
        self.store.select_round(number+1)
        self.meta=copy.deepcopy(meta)
        self.active_memory=[asdict(h) for h in history]
        self.registry={}
        # Only prior, explicitly committed training-side summaries are eligible.
        for n in range(number):
            before=self.root/f'round_{n}/feedback/gen_{n}/agent_execution.json'
            after=self.root/f'round_{n}/feedback/gen_{n+1}/agent_execution.json'
            parents=[self.register_rows(before,n+1),self.register_rows(after,n+1)]
            self.register(self.root/f'round_{n}/experience.json',n+1,parents)

    def register_rows(self,path,round_id):
        current=manifest(self.store.root/f'B{round_id}/manifest.json','train_evolution',round_id)
        members={r['task_id'] for r in current['tasks']}
        rows=read(path)
        for row in rows:
            if (row.get('source_role')!='train_evolution' or row.get('purpose')!='evolution_train'
                or row.get('round_id')!=round_id or row.get('manifest_hash')!=current['manifest_hash']
                or row.get('task_id') not in members or row.get('derived_from')):
                raise ValueError('Meta rejects non-native, external, derived or wrong-round rollout evidence')
        return self.register(path,round_id)

    def register(self, path, round_id, derived_from=()):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError('Evidence outside training run')
        ident=fingerprint([str(path),file_hash(path)])
        self.registry[ident]=dict(path=str(path),sha256=file_hash(path),source_role='train_evolution',
            purpose='evolution_train',round_id=round_id,derived_from=list(derived_from))
        return ident

    def guard(self):
        authorize_evidence(list(self.registry),self.registry,current_round=self.store.round_id)

    def bind_execution(self, state, directory, mode='probe'):
        from sia.task_meta.storage import artifact_manifest
        from sia.task_meta.pipeline import source_identity
        self.store.mode=mode
        self.execution_scope=dict(role='train_evolution',purpose='evolution_train',round_id=self.store.round_id,
            manifest_hash=self.store.current['manifest_hash'],model=state.checkpoint_manifest,
            model_ref=state.model_ref,task_harness=file_hash(state.harness_path),meta_harness=asdict(self.meta),
            active_experience=fingerprint(self.active_memory),artifacts=artifact_manifest(state.artifacts.directory),
            rollout_seed=self.config.seed,protocol_hash=file_hash(self.root/'protocol.json'),
            implementation=fingerprint(source_identity()))
        freeze(Path(directory)/'execution_scope.json',dict(scope=self.execution_scope,mode=mode,
            selected_manifest=self.store.current['manifest_hash'] if mode=='training' else self.store.probes['manifest_hash']))
        cache=self.root/'rollout_cache'/fingerprint(self.execution_scope)/'train_rollouts'
        self.existing_receipts={p.name for p in cache.glob('*.json')}

    def observed(self, result, directory):
        rows=[json.loads(line) for line in (Path(directory)/'probe_trajectories.jsonl').read_text().splitlines() if line.strip()]
        if self.store.mode=='probe': result.trajectories=rows
        result.performance['feedback_role']='train_evolution'
        result.performance['meta_probe_manifest']=self.store.probes['manifest_hash']
        result.performance['round_id']=self.store.round_id
        result.performance['evaluation_protocol']='same_round_training_meta_probe_v1'
        by_source={}
        for source in sorted({r['source'] for r in rows}):
            group=[r for r in rows if r['source']==source]
            by_source[source]=dict(count=len(group),successes=sum(r['verification'].get('success') is True for r in group),
                errors=dict(Counter(r.get('execution_error_type') or r.get('error_type') or 'none' for r in group)))
        result.performance['benchmarks']=by_source
        cache=self.root/'rollout_cache'/fingerprint(self.execution_scope)/'train_rollouts'
        fresh=[read(p)['row'] for p in cache.glob('*.json') if p.name not in self.existing_receipts]
        result.cost.update(physical_rollout_attempts=len(fresh),executed_unique_tasks=len({r['task_id'] for r in fresh}),
            model_calls=sum(r.get('model_call_count',0) for r in fresh),input_tokens=sum(r.get('input_tokens',0) for r in fresh),
            output_tokens=sum(r.get('output_tokens',0) for r in fresh),wall_time_seconds=sum(r.get('wall_time_seconds',0) for r in fresh),
            unknown_usage_calls=sum(r.get('unknown_usage_calls',0) for r in fresh),
            tool_calls=sum(len(r.get('tool_calls',[])) for r in fresh))
        return result

    def adapt_meta(self, agent, meta, observation, directory, history, handler, resume):
        agent.round_protocol=self
        from sia.task_meta.observations import operation_input
        from sia.task_meta.types import FiveStageMetaUpdate, MetaHarnessUpdate
        from sia.task_meta.loop import _accept_meta_update
        evidence=self.register_rows(Path(directory)/'agent_execution.json',self.store.round_id)
        self.guard()
        if not getattr(agent.client,'supports_evolution',False): raise ValueError('Native structured Meta backend required')
        envelope=operation_input(observation)
        envelope['trusted_facts']['feedback_protocol']={
            'role':'train_evolution','purpose':'evolution_train','round_id':self.store.round_id,
            'manifest_hash':self.store.probes['manifest_hash'],'evidence_ids':list(self.registry),
            'current_round_delta':None,'historical_training_deltas':[h.performance_delta for h in history],
            'external_evaluation_available':False,'phase':'before_task_update'}
        # No post-update outcome exists yet; do not fabricate an expectation/outcome pair.
        envelope['outcome_review']=None
        learned=agent.client.complete(
            'Update your Meta harness from current training diagnostics and committed history. '
            'Current-round improvement is unknown. Preserve reusable strategies; never encode answers or task-ID lookup rules.',
            FiveStageMetaUpdate if agent._five_stage(meta) else MetaHarnessUpdate,
            meta_state=meta,operation='learn',experience_id=f'training_diagnosis_{self.store.round_id}',
            operation_input=envelope,validate_candidate=lambda candidate:validate_meta_edit(candidate,meta,self.store.current['tasks']))
        validate_meta_edit(learned,meta,self.store.current['tasks'])
        accepted=_accept_meta_update(self.root,meta,learned,[],self.root/f'round_{self.store.round_id-1}/meta_self_update.json',
            'training_diagnosis_before_task_update',handler,resume)
        self.meta=copy.deepcopy(accepted)
        freeze(self.root/f'round_{self.store.round_id-1}/meta_input_allowlist.json',self.registry)
        return accepted

    def model_context(self, state, context, evaluate):
        directory=context.directory.parent/'sft_collection'/f'gen_{state.generation}'
        self.bind_execution(state,directory,'training')
        context.evaluation=evaluate(state,directory)
        freeze(context.directory/'sft_source_binding.json',dict(round_id=self.store.round_id,branch_id='parent',
            harness_hash=file_hash(state.harness_path),parent_policy_hash=fingerprint(state.checkpoint_manifest)))
        self.store.mode='probe'

    def decision(self, decision, directory):
        self.guard()
        freeze(Path(directory)/'decision_protocol.json',dict(decision_id=decision.decision_id,
            round_id=self.store.round_id,evidence_ids=decision.evidence,input_evidence_allowlist=list(self.registry),update_model=decision.action.value=='MODEL',
            update_targets=decision.target_components,reason=decision.rationale,
            budget={'candidate_limit':len(getattr(self.config,'allowed_task_components',['HARNESS','MODEL'])),
                    'sft_epochs_if_selected':1,'replay_previous_rounds':False}))

    def freeze_system(self,record,round_dir):
        from sia.task_meta.storage import artifact_manifest
        task=record['task_after'];meta=record['meta_after']
        value=dict(task_policy={'checkpoint':task['checkpoint_path'],'weights':task['checkpoint_manifest']},
            task_harness={'path':task['harness_path'],'sha256':file_hash(task['harness_path'])},
            meta_harness=meta,active_experience_hash=fingerprint(self.active_memory),
            prompts_and_execution_protocol=file_hash(self.root/'protocol.json'),
            artifacts=artifact_manifest((task.get('artifacts') or {}).get('directory')))
        freeze(Path(round_dir)/'system_snapshot.json',{**value,'snapshot_hash':fingerprint(value)})
        record['system_snapshot_hash']=fingerprint(value)
        record['active_experience_hash']=fingerprint(self.active_memory)

    def check_task_edit(self,parent,candidate):
        from sia.task_meta.evolution_protocol import assert_reusable_edit
        assert_reusable_edit(read(parent.harness_path),read(candidate.harness_path),self.store.current['tasks'])

    def completed(self, record, round_dir, baseline, outcome):
        from sia.task_meta.meta_harness.five_stage import paired_outcome
        from sia.task_meta.types import ImprovementExperience
        experience=ImprovementExperience(**read(Path(round_dir)/'experience.json'))
        paired_outcome(Path(round_dir)/'feedback',experience)
        freeze(Path(round_dir)/'adaptation_feedback_metrics.json',dict(source_role='train_evolution',purpose='evolution_train',
            round_id=self.store.round_id,meta_probe_manifest=self.store.probes['manifest_hash'],
            before=baseline.performance,after=outcome.performance,delta=outcome.performance['macro_success']-baseline.performance['macro_success'],
            future_use='next_round_only' if self.store.round_id<3 else 'archived_no_fourth_update'))
        rows=[read(p)['row'] for p in (self.root/'rollout_cache').glob('*/train_rollouts/*.json')]
        unique={r['trajectory_id']:r for r in rows}
        training=[read(p) for p in self.root.glob('round_*/candidates/MODEL/gen_*/model_update/training_metrics.json')
                  if (p.parent/'checkpoint.json').is_file() or (p.parent/'checkpoint_trained.json').is_file()]
        freeze(Path(round_dir)/'cumulative_usage.json',dict(allocated_tasks=3600,
            executed_unique_tasks=len({r['task_id'] for r in unique.values()}),rollout_attempts=len(rows),
            successful_trajectories=sum(r.get('verification',{}).get('success') is True for r in unique.values()),
            model_calls=sum(r.get('model_call_count',0) for r in rows),
            input_tokens=sum(r.get('input_tokens',0) for r in rows),
            output_tokens=sum(r.get('output_tokens',0) for r in rows),
            task_wall_seconds=sum(r.get('wall_time_seconds',0) for r in rows),
            tool_calls=sum(len(r.get('tool_calls',[])) for r in rows),
            actual_sft_calls=len(training),sft_samples=sum(t['available_positive_samples'] for t in training),
            supervised_training_tokens=sum(t['sampled_supervised_tokens'] for t in training),
            optimizer_steps=sum(t['optimizer_steps'] for t in training),
            sft_wall_seconds=sum(t['training_wall_seconds'] for t in training),
            usage_complete=all(r.get('usage_complete',False) for r in rows)))


def require_round_release(config):
    from sia.task_meta.pipeline import project_path
    root=project_path(config.data_dir)
    if not (root/'tasks.sqlite').is_file(): raise FileNotFoundError('Prepared round manifests required; no full-pool fallback')
    for r in (1,2,3): manifest(root/f'B{r}/manifest.json','train_evolution',r)
    return {'status':'PREPARED_NOT_TRAINED','allocated_tasks':3600,'rounds':3,'model_calls':0}
