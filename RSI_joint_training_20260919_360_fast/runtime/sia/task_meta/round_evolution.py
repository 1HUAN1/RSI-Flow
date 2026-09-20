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
        self.mode = 'parent_pre_update'

    def select_round(self, number):
        if number not in (1, 2, 3): raise ValueError('Exactly three training rounds')
        self.current = manifest(self.root/f'B{number}/manifest.json', 'train_evolution', number)
        self.members = {r['task_id']: r for r in self.current['tasks']}
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
        tasks = [self._task(r) for r in self.current['tasks']]
        return tasks, dict(Counter(t.domain for t in tasks))

    def probe(self):
        if self.round_id is None: raise ValueError('Explicit current round is required')
        return [self._task(r) for r in self.current['tasks']]

    def iter_split(self, split, domain=None):
        if self.round_id is None or split != 'evolve_train': raise ValueError('Round-local training access only')
        for r in self.current['tasks']:
            if domain is None or r['domain'] == domain: yield self._task(r)

    def coverage(self, cursor):
        return dict(allocated_tasks=len(self.current['tasks']), all_tasks_scheduled=True,
                    round_id=self.round_id, feedback_tasks=len(self.current['tasks']))


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
        branch_id='parent' if executor.store.mode=='parent_pre_update' else 'child',
        collection_stage=executor.store.mode, manifest_hash=m['manifest_hash'], harness_hash=file_hash(state.harness_path),
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


MEMORY_POLICY = 'append_component_skill_then_general_principle'


def memory_view(meta):
    """Views of the existing atomic K store; no second memory database."""
    from sia.task_meta.meta_harness.five_stage import validate_library
    if not meta.bundle_path:
        raise ValueError('Append-only memory requires the native principles bundle')
    library=validate_library(read(Path(meta.bundle_path)/'principles.json'))
    records=library['records']
    skills={c:[r for r in records if r['principle_id'].startswith('skill.'+c+'.')]
            for c in ('MODEL','HARNESS','ARTIFACTS')}
    return dict(library_hash=fingerprint(library),component_skills=skills,
        general_principles=[r for r in records if not r['principle_id'].startswith('skill.')])


def validate_memory_append(candidate, meta, component, decision_id, experience_id):
    """Reject destructive K operations before the native atomic materialization."""
    from sia.task_meta.types import MetaHarnessUpdate
    update=MetaHarnessUpdate.model_validate(candidate)
    review=update.five_stage
    if review is None or review.slow_edit or review.structural_mismatches:
        raise ValueError('This protocol updates Meta through append-only Memory')
    if update.harness != Path(meta.harness_path).read_text(encoding='utf-8'):
        raise ValueError('Memory update cannot rewrite Meta instructions')
    for name,content in update.bundle_files.items():
        if name not in {'instructions.md','context.json','workflow.json','evolution.json'}:
            raise ValueError('Memory update cannot replace protected files')
        if content != (Path(meta.bundle_path)/name).read_text(encoding='utf-8'):
            raise ValueError('Memory update cannot rewrite existing bundle files')
    if review.harness_bindings:
        raise ValueError('Append-only Memory has no direct harness rewrite')
    previous=read(Path(meta.bundle_path)/'principles.json')
    seen={r['principle_id'] for r in previous['records']};skills=[];principles=[]
    for op in review.principle_operations:
        record=op.record
        if op.operation!='ADD' or record is None or op.principle_id in seen:
            raise ValueError('Memory only permits ADD of a new unique entry')
        seen.add(op.principle_id)
        if record.principle_id!=op.principle_id or record.revision!=1 or record.g_targets:
            raise ValueError('New memory must be revision 1 without direct G edits')
        if decision_id not in record.source_decisions or experience_id not in record.source_experiences:
            raise ValueError('Memory must cite the actual decision and paired experience')
        if op.principle_id.startswith('skill.'+component+'.') and not principles:
            skills.append(op.principle_id)
        elif op.principle_id.startswith('principle.') and skills and any(s in op.rationale for s in skills):
            principles.append(op.principle_id)
        else:
            raise ValueError('Append selected-component skill first, then principles citing that skill in rationale')
    if not skills or not principles:
        raise ValueError('Both a component skill and a derived general principle are required')
    return dict(component=component,skills=skills,general_principles=principles,
                previous_library_hash=fingerprint(previous))


def training_pairs(current, pre, post, experience, selected):
    """Account for every frozen task, including unscored/infra cases, before summarizing."""
    expected={r['task_id']:r for r in current['tasks']}
    def index(rows, phase):
        values={r['task_id']:r for r in rows}
        if len(values)!=len(rows) or set(values)!=set(expected):
            raise ValueError('Pre/post must each contain every B_r task exactly once')
        if any(r.get('collection_stage')!=phase or r.get('manifest_hash')!=current['manifest_hash']
               or r.get('source_role')!='train_evolution' for r in rows):
            raise ValueError('Paired evidence has incorrect purpose/phase/manifest')
        return values
    left=index(pre,'parent_pre_update');right=index(post,'child_post_update');pairs=[]
    fixed=('task_source_hash','reset_hash','environment_type','external_task_budget','chat_template_kwargs','seed')
    def side(row):
        verified=row.get('verification',{})
        success=verified.get('success') if verified.get('status')=='completed' and not row.get('infrastructure_error') else None
        if type(success) is not bool:success=None
        return dict(success=success,score=row.get('terminal_reward'),native_metrics=row.get('metrics',{}),
            error=row.get('execution_error_type') or row.get('error_type'),
            scoring_status=verified.get('status','pending'),
            cost={k:row.get(k) for k in ('input_tokens','output_tokens','unknown_usage_calls','model_call_count','wall_time_seconds')},
            tool_calls=len(row.get('tool_calls',[])),trajectory_id=row.get('trajectory_id'),
            trajectory_hash=fingerprint(row),system_fingerprint=row.get('state_hash'),
            behavior_harness=row.get('harness_hash'))
    for task_id in sorted(expected):
        a,b=left[task_id],right[task_id];aa,bb=side(a),side(b)
        comparable=all(a.get(k) is not None and a.get(k)==b.get(k) for k in fixed)
        scored=comparable and aa['success'] is not None and bb['success'] is not None
        transition=(('success' if aa['success'] else 'failure')+'_to_'+('success' if bb['success'] else 'failure')) if scored else 'unscored_or_incomparable'
        pairs.append(dict(task_id=task_id,source=expected[task_id]['source'],domain=expected[task_id]['domain'],
            parent=aa,child=bb,comparable=comparable,transition=transition,
            success_delta=int(bb['success'])-int(aa['success']) if scored else None,
            pre_trajectory_path=experience.trajectory_before,post_trajectory_path=experience.trajectory_after))
    def summarize(rows):
        complete=all(r['success_delta'] is not None for r in rows)
        counts=dict(Counter(r['transition'] for r in rows))
        return dict(n_expected=len(rows),n_paired=len(rows),n_scored=sum(r['success_delta'] is not None for r in rows),
            complete=complete,transitions=counts,
            success_delta=sum(r['success_delta'] for r in rows)/len(rows) if complete and rows else None)
    return dict(source_role='train_evolution',purpose='evolution_train',round_id=current['round_id'],
        manifest_hash=current['manifest_hash'],decision_id=selected['decision_id'],actual_change=experience.actual_change,
        versions=experience.versions,parent_system=selected['parent_hash'],child_system=selected['child_hash'],
        pairs=pairs,overall=summarize(pairs),
        by_source={s:summarize([p for p in pairs if p['source']==s]) for s in sorted({p['source'] for p in pairs})},
        by_domain={s:summarize([p for p in pairs if p['domain']==s]) for s in sorted({p['domain'] for p in pairs})},
        coverage=dict(included_task_ids=sorted(expected),included_count=len(pairs),omitted_count=0,
                      batches=[{'task_ids':[p['task_id'] for p in pairs[i:i+100]],
                                'summary':summarize(pairs[i:i+100])} for i in range(0,len(pairs),100)]))


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
        self.meta_stage='decision'
        if self.append_memory:
            self.loaded_memory=memory_view(meta)
            freeze(self.root/f'round_{number}/memory_input.json',self.loaded_memory)
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
        identifier=self.register(path,round_id)
        self.registry[identifier]['collection_stages']=sorted({r.get('collection_stage','unknown') for r in rows})
        return identifier

    def register(self, path, round_id, derived_from=()):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError('Evidence outside training run')
        ident=fingerprint([str(path),file_hash(path)])
        self.registry[ident]=dict(path=str(path),sha256=file_hash(path),source_role='train_evolution',
            purpose='evolution_train',round_id=round_id,derived_from=list(derived_from))
        return ident

    def guard(self):
        authorize_evidence(list(self.registry),self.registry,current_round=self.store.round_id)
        if self.meta_stage=='decision':
            if any(v['round_id']==self.store.round_id and
                   set(v.get('collection_stages',[]))-{'parent_pre_update'} for v in self.registry.values()):
                raise ValueError('Decision cannot consume current-round post/future evidence')

    def enrich_effect(self,envelope):
        if self.meta_stage!='effect':raise ValueError('Effects are unavailable before paired post scoring')
        value=read(self.root/f'round_{self.store.round_id-1}/paired_training_effect.json')
        envelope['trusted_facts']['all_training_task_effects']={k:v for k,v in value.items() if k!='pairs'}
        if self.append_memory:
            envelope['trusted_facts']['memory_update_contract']=dict(
                policy=MEMORY_POLICY,component=self.selected_component,
                decision_id=self.selected_decision,experience_id=self.selected_experience,
                instructions='Use only current paired training evidence and prior training Memory. '
                'ADD at least one skill.<COMPONENT>.<new_id> first; then ADD principle.<new_id>. '
                'Each general principle operation rationale must cite its newly added skill ID. '
                'Each record must cite the actual decision and experience. Include failures, limitations and applicability. '
                'Keep existing entries, instructions, bundle files and graph byte-for-byte unchanged; no G targets or harness_bindings. '
                'No MERGE/REVISE/RETIRE, slow_edit, structural_mismatches or second Task update. '
                'New records remain tentative. Native next-round memory loading makes them effective.')
            envelope['trusted_facts']['meta_memory']=self.loaded_memory

    @property
    def append_memory(self):
        return (getattr(self.config,'round_protocol',None) or {}).get('memory_policy')==MEMORY_POLICY

    def enrich_decision(self,envelope):
        self.guard()
        if self.append_memory:
            envelope['trusted_facts']['meta_memory']=self.loaded_memory
            envelope['trusted_facts']['component_decision_contract']={
                'candidate_limit':1,'instructions':'Read current full pre statistics and training evidence. '
                'Consult component_skills for MODEL/HARNESS/ARTIFACTS and general_principles. '
                'Choose exactly one component and one proposal; no alternative-component retry. '
                'MODEL uses all eligible current parent pre successes, one epoch LoRA; 120 is a domain quota, not an SFT cap. '
                'Post and independent evaluation are unavailable to this decision.'}

    def bind_execution(self, state, directory, mode='parent_pre_update'):
        from sia.task_meta.storage import artifact_manifest
        from sia.task_meta.pipeline import source_identity
        if mode not in {'parent_pre_update','child_post_update'}: raise ValueError('Invalid collection stage')
        self.store.mode=mode
        self.execution_scope=dict(role='train_evolution',purpose='evolution_train',round_id=self.store.round_id,
            manifest_hash=self.store.current['manifest_hash'],collection_stage=mode,model=state.checkpoint_manifest,
            model_ref=state.model_ref,task_harness=file_hash(state.harness_path),meta_harness=asdict(self.meta),
            active_experience=fingerprint(self.active_memory),artifacts=artifact_manifest(state.artifacts.directory),
            rollout_seed=self.config.seed,protocol_hash=file_hash(self.root/'protocol.json'),
            implementation=fingerprint(source_identity()))
        freeze(Path(directory)/'execution_scope.json',dict(scope=self.execution_scope,mode=mode,
            selected_manifest=self.store.current['manifest_hash']))
        freeze(self.root/f'round_{self.store.round_id-1}'/f'{mode}_binding.json',self.execution_scope)
        cache=self.root/'rollout_cache'/fingerprint(self.execution_scope)/'train_rollouts'
        self.existing_receipts={p.name for p in cache.glob('*.json')}

    def observed(self, result, directory):
        rows=[json.loads(line) for line in (Path(directory)/'probe_trajectories.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=len(self.store.members) or {r['task_id'] for r in rows}!=set(self.store.members):
            raise ValueError('Each parent/child pass must execute the complete current B_r exactly once')
        result.trajectories=rows
        result.performance['feedback_role']='train_evolution'
        result.performance['round_id']=self.store.round_id
        result.performance['evaluation_protocol']='same_round_full_B_parent_child_v2'
        result.performance['training_manifest']=self.store.current['manifest_hash']
        result.performance['collection_stage']=self.store.mode
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

    def prepare_meta(self, agent, experience, round_dir):
        agent.round_protocol=self
        self.meta_stage='effect'
        self.selected_component=experience.decision['action']
        self.selected_decision=experience.decision['decision_id']
        self.selected_experience=experience.experience_id
        parents=[self.register_rows(Path(path),self.store.round_id)
                 for path in (experience.trajectory_before,experience.trajectory_after)]
        self.register(Path(round_dir)/'experience.json',self.store.round_id,parents)
        self.register(Path(round_dir)/'paired_training_effect.json',self.store.round_id,parents)
        self.guard()
        freeze(Path(round_dir)/'meta_input_allowlist.json',self.registry)

    def validate_meta(self, candidate, meta):
        self.guard()
        if self.append_memory:
            if self.meta_stage!='effect':raise ValueError('Memory append requires completed paired post evidence')
            validate_memory_append(candidate,meta,self.selected_component,self.selected_decision,self.selected_experience)
        return validate_meta_edit(candidate,meta,self.store.current['tasks'])

    def model_context(self, state, context, evaluate):
        # SFT always consumes the frozen PRE-decision parent pass, never a new rollout.
        rows=context.evaluation.trajectories
        if len(rows)!=len(self.store.members) or {r['task_id'] for r in rows}!=set(self.store.members):
            raise ValueError('SFT requires complete parent B_r rollout evidence')
        source=dict(round_id=self.store.round_id,branch_id='parent',
            harness_hash=file_hash(state.harness_path),parent_policy_hash=fingerprint(state.checkpoint_manifest))
        training_provenance(rows,self.store.current,**source)
        freeze(context.directory/'sft_source_binding.json',source)

    def select_child(self, parent, child, decision, round_dir):
        from sia.task_meta.durable import task_hash
        self.check_task_edit(parent,child)
        freeze(Path(round_dir)/'selected_update.json',dict(
            policy='legal_single_update_then_measure',decision_id=decision.decision_id if decision else None,
            parent_hash=task_hash(parent),child_hash=task_hash(child),
            selected_before_child_evaluation=True,selection_uses_child_score=False))
        self.store.mode='child_post_update'

    def decision(self, decision, directory):
        self.guard()
        freeze(Path(directory)/'decision_protocol.json',dict(decision_id=decision.decision_id,
            round_id=self.store.round_id,evidence_ids=decision.evidence,input_evidence_allowlist=list(self.registry),update_model=decision.action.value=='MODEL',
            update_targets=decision.target_components,reason=decision.rationale,
            budget={'evaluated_candidate_limit':1,'task_passes':2,'tasks_per_pass':len(self.store.members),
                    'sft_epochs_if_selected':1,'replay_previous_rounds':False}))

    def freeze_system(self,record,round_dir):
        from sia.task_meta.storage import artifact_manifest
        task=record['task_after'];meta=record['meta_after']
        from sia.task_meta.durable import task_hash, load_task
        selected=read(Path(round_dir)/'selected_update.json')
        if task_hash(load_task(task))!=selected['child_hash']:
            raise ValueError('Meta harness update modified the frozen child Task')
        if self.append_memory:
            previous=read(Path(self.meta.bundle_path)/'principles.json')
            following=read(Path(meta['bundle_path'])/'principles.json')
            for key in ('records','revisions'):
                if following[key][:len(previous[key])]!=previous[key]:
                    raise ValueError('Committed Memory changed or removed an existing entry')
            added=following['records'][len(previous['records']):]
            if len(added)<2:raise ValueError('Committed Memory lacks new skill and principle')
            freeze(Path(round_dir)/'memory_append_receipt.json',dict(
                before_hash=fingerprint(previous),after_hash=fingerprint(following),
                preserved_entries=len(previous['records']),added_ids=[r['principle_id'] for r in added],
                effective_from_round=self.store.round_id+1 if self.store.round_id<3 else None,
                final_round_archive_only=self.store.round_id==3))
        self.meta=copy.deepcopy(meta)
        self.active_memory.append(read(Path(round_dir)/'experience.json'))
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
        native_pairs=paired_outcome(Path(round_dir)/'feedback',experience)
        paired=training_pairs(self.store.current,read(experience.trajectory_before),read(experience.trajectory_after),
            experience,read(Path(round_dir)/'selected_update.json'))
        freeze(Path(round_dir)/'paired_training_effect.json',paired)
        freeze(Path(round_dir)/'adaptation_feedback_metrics.json',dict(source_role='train_evolution',purpose='evolution_train',
            round_id=self.store.round_id,training_manifest=self.store.current['manifest_hash'],
            before=baseline.performance,after=outcome.performance,delta=paired['overall']['success_delta'],
            paired_report='paired_training_effect.json',complete=paired['overall']['complete'],
            future_use='current_round_meta_update_then_next_round' if self.store.round_id<3 else 'current_round_meta_update_then_archive'))
        rows=[read(p)['row'] for p in (self.root/'rollout_cache').glob('*/train_rollouts/*.json')]
        unique={r['trajectory_id']:r for r in rows}
        training=[read(p) for p in self.root.glob('round_*/candidates/MODEL/gen_*/model_update/training_metrics.json')
                  if (p.parent/'checkpoint.json').is_file() or (p.parent/'checkpoint_trained.json').is_file()]
        allocated=sum(len(manifest(self.store.root/f'B{r}/manifest.json','train_evolution',r)['tasks']) for r in (1,2,3))
        freeze(Path(round_dir)/'cumulative_usage.json',dict(allocated_tasks=allocated,
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
    from sia.task_meta.evolution_protocol import TRAIN_QUOTAS, assert_disjoint
    rounds=[manifest(root/f'B{r}/manifest.json','train_evolution',r) for r in (1,2,3)]
    assert_disjoint(rounds)
    if any(Counter(t['source'] for t in m['tasks'])!=TRAIN_QUOTAS for m in rounds):
        raise ValueError('Expected registered 360-task round manifests; no old/full-pool fallback')
    return {'status':'PREPARED_NOT_TRAINED','allocated_tasks':sum(len(m['tasks']) for m in rounds),'rounds':3,'model_calls':0}
