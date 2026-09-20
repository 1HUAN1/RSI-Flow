"""CPU-only protocol fixtures; all model/trainer substitutions are explicit mocks."""
import copy
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import test_contracts as native
from evolution_protocol import (make_manifest,assert_disjoint,freeze,fingerprint,file_hash,manifest,
    authorize_evidence,training_provenance,success_filter,feedback_delta,cache_key,snapshot,SFT_PRESET)
from data_protocol import allocate,TRAIN_QUOTAS,exposure,scan
from sia.task_meta.round_evolution import RoundStore,RoundProtocol,select_sft,scoped_rollout,annotate_row
from sia.task_meta.durable import DurableExecutor,DurableClient,DurableUpdater,StageJournal
from sia.task_meta.types import TaskAgentState,MetaAgentState,MetaDecision,MetaHarnessUpdate,TaskUpdate,TaskUpdateAction,EvaluationResult,MetaObservation
from sia.task_meta import sequential_loop as loop


def task(source,n,kind='train'):
    domain='tool_use' if source=='envscaler' else 'code' if source=='deepcoder_taco' else 'searchqa'
    return dict(source=source,domain=domain,version='fixture-v1',original_split='train' if kind=='train' else 'test',
        task_id=f'{source}:{n}',native_id=str(n),group_id=f'group-{source}-{n}',content_hash=f'content-{source}-{n}',
        source_sha256='fixture',kind=kind,strata={'difficulty':n%3})


class Allocation(unittest.TestCase):
    def test_reproducible_quotas_round_disjoint_and_probe_containment(self):
        config=native.read(native.BUNDLE/'configs/train.json')
        config['validation_limits']={}
        rows=[task(s,i) for s,q in TRAIN_QUOTAS.items() for i in range(q*3+30)]
        first,_=allocate(rows,config,{'events':[],'coverage':{}})
        second,_=allocate(list(reversed(rows)),config,{'events':[],'coverage':{}})
        self.assertEqual(first,second);assert_disjoint(first)
        self.assertEqual(sum(len(x['tasks']) for x in first[:3]),1080)
        for value in first[:3]:
            self.assertEqual(Counter(x['source'] for x in value['tasks']),TRAIN_QUOTAS)
            self.assertEqual(value['feedback_scope'],'all_B_r')
        with self.assertRaises(ValueError):allocate(rows[:20],config,{})

    def test_external_overlap_and_exposure_unknown_are_not_relabelled(self):
        config=native.read(native.BUNDLE/'configs/train.json');config['validation_limits']={'benchmark':1}
        rows=[task(s,i) for s,q in TRAIN_QUOTAS.items() for i in range(q*3+30)]
        external=task('benchmark',0,'evaluation');external['group_id']=rows[0]['group_id']
        result,quarantine=allocate(rows+[external,task('benchmark',1,'evaluation')],config,{'events':[],'coverage':{}})
        self.assertNotIn(rows[0]['task_id'],{r['task_id'] for m in result[:3] for r in m['tasks']})
        self.assertFalse(result[-2]['clean_independence_audited']);self.assertFalse(result[-1]['tasks'])
        self.assertTrue(all(r['exposure']=='exposure_unknown' for r in quarantine))
        exposed={'events':[dict(source='benchmark',all_tasks=True,evidence='past-full-score-used-for-tuning')],'coverage':{}}
        self.assertEqual(exposure(external,exposed),'exposed')


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        t={**task('nq_open',1), 'role':'train_evolution','purpose':'evolution_train','round_id':1,'meta_probe':True}
        self.m=make_manifest('train_evolution',1,[t])
        self.row=dict(task_id=t['task_id'],source_role='train_evolution',purpose='evolution_train',round_id=1,branch_id='parent',collection_stage='parent_pre_update',
            manifest_hash=self.m['manifest_hash'],harness_hash='h',parent_policy_hash='p',trajectory_id='t',success_verifier_version='v',
            domain='searchqa',verification={'status':'completed','success':True})

    def check(self,rows):
        return training_provenance(rows,self.m,round_id=1,branch_id='parent',harness_hash='h',parent_policy_hash='p')

    def test_sft_accepts_M_and_rejects_external_wrong_round_harness_parent(self):
        self.check([self.row])
        for change in ({'source_role':'independent_validation'},{'source_role':'final_test'},
                       {'purpose':'report_only'},{'collection_stage':'child_post_update'},{'round_id':2},{'harness_hash':'old'},{'parent_policy_hash':'old'}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.check([{**self.row,**change}])
        from sia.task_meta.sft import select_positive_rows
        for role in ('independent_validation','final_test'):
            with self.assertRaises(ValueError):select_positive_rows([{**self.row,'source_role':role}],profile='multidomain')

    def test_meta_rejects_external_and_derived_external_even_when_renamed(self):
        p=self.root/'renamed.json';p.write_text('{}')
        train=dict(path=str(p),sha256=file_hash(p),source_role='train_evolution',purpose='evolution_train',round_id=1)
        for role in ('independent_validation','final_test'):
            registry={'external':{**train,'source_role':role},'summary':{**train,'derived_from':['external']}}
            with self.assertRaises(ValueError):authorize_evidence(['summary'],registry,current_round=1)
        with self.assertRaises(ValueError):authorize_evidence(['unknown'],{},current_round=1)
        with self.assertRaises(ValueError):authorize_evidence(['future'],{'future':{**train,'round_id':2}},current_round=1)
        from sia.task_meta.meta_harness.runtime import _source_records
        with self.assertRaises(ValueError):_source_records([{'split':'evolve_train','source_role':'final_test'}],'trajectory')

    def test_verified_success_minimum_and_no_duplicate_supervision(self):
        selector=lambda rows,profile:[{**r,'messages':[{'role':'assistant','content':'recorded'}]} for r in rows]
        selected,report=success_filter([self.row,self.row],minimum_samples=1,native_selector=selector)
        self.assertEqual(len(selected),1);self.assertEqual(report['filtering_reasons']['duplicate_supervision'],1)
        selected,report=success_filter([{**self.row,'verification':{'success':True,'status':'self_score'}}],minimum_samples=1,native_selector=selector)
        self.assertFalse(selected);self.assertEqual(report['status'],'skipped_insufficient_success')
        self.assertFalse(success_filter([self.row],minimum_samples=2,native_selector=selector)[0])

    def test_delta_same_M_only_cache_all_behavioral_state(self):
        a={'manifest_hash':'M1','round_id':1,'benchmarks':{'nq':{'success':0.5}}}
        b={**a,'benchmarks':{'nq':{'success':0.7}}}
        self.assertIsNone(feedback_delta(a,None));self.assertAlmostEqual(feedback_delta(b,a)['nq']['success'],.2)
        with self.assertRaises(ValueError):feedback_delta({**b,'manifest_hash':'M2','round_id':2},a)
        args=dict(task={'weights':'p'},meta={'rules':'G'},execution={'decode':1},memory=[],evaluator='v',retrieval='i',prompts='h')
        def key(args):return cache_key(snapshot(**args),self.m,role='train_evolution',round_id=1,seeds={'rollout':42},branch_id='parent')
        before=key(args)
        for field in args:
            self.assertNotEqual(before,key({**args,field:'changed'}))

    def test_actual_lora_modules_and_parent_checkpoint_inheritance(self):
        spec=importlib.util.spec_from_file_location('trainer_protocol',native.RUNTIME/'scripts/train_task_meta_sft.py')
        trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
        options=trainer.resolve_training_options({'sft_profile':'multidomain','training':SFT_PRESET,'supervision':'final_assistant'},SimpleNamespace())
        self.assertEqual((options['num_train_epochs'],options['lora_rank'],options['lora_alpha'],options['lora_dropout'],options['learning_rate']),(1,8,16,.05,2e-6))
        for name in ('parent','initial'):(self.root/name).mkdir();(self.root/name/'config.json').write_text('{}')
        request={'checkpoint_path':str(self.root/'parent')}
        self.assertEqual(trainer.resolve_training_base(request),self.root/'parent')
        with self.assertRaises(ValueError):trainer.resolve_training_base(request,self.root/'initial')
        from sia.task_meta.round_evolution import check_lora_modules
        model=SimpleNamespace(named_modules=lambda:[('layer.q_proj',SimpleNamespace(weight=SimpleNamespace(ndim=2)))],parameters=lambda:[])
        with self.assertRaises(ValueError):check_lora_modules(model,options)


class NativeRoundLoop(unittest.TestCase):
    def test_each_component_is_selectable_and_exactly_one_candidate_is_evaluated(self):
        for action in ('HARNESS','MODEL','ARTIFACTS'):
            with self.subTest(action=action):
                tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
                self.run_fixture(Path(tmp.name),[action]*3,search=[(action,1)])

    def test_strict_positive_model_candidate_deploys_and_resume_is_idempotent(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.run_fixture(Path(tmp.name),['MODEL']*3,interrupt=True,search=[('MODEL',1)])

    def test_zero_negative_and_incomplete_candidates_retain_parent(self):
        for action,score,incomplete,outcome in (
            ('HARNESS',.5,False,'zero_gain'),
            ('MODEL',0,False,'negative_gain'),
            ('ARTIFACTS',1,True,'incomplete_comparison'),
        ):
            with self.subTest(outcome=outcome):
                tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
                self.run_fixture(Path(tmp.name),[action]*3,search=[(action,score)],
                                 incomplete=incomplete,expected_outcome=outcome)

    def test_unavailable_candidate_retains_parent_without_second_decision(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.run_fixture(Path(tmp.name),['MODEL']*3,unavailable=True,search=[('MODEL',1)],
                         expected_outcome='candidate_unavailable')

    def test_three_full_360_rounds_use_one_candidate_each(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.run_fixture(Path(tmp.name),['HARNESS']*3,
                         search=[('HARNESS',1)],full_cohort=True)

    def run_fixture(self,root,actions,interrupt=False,unavailable=False,search=None,full_cohort=False,incomplete=False,expected_outcome=None):
        if not search or len(search)!=1: raise ValueError('Fixture requires exactly one candidate')
        sequential=False
        (root/'meta').mkdir();h=root/'h.json'
        from sia.task_meta.harnessforge_production import initialize_base_manifest
        initialize_base_manifest(h)
        g=root/'g.md';g.write_text('Initial reusable rules')
        freeze(root/'protocol.json',{'config':{'training_schedule':'round_disjoint'}})
        data=root/'data';data.mkdir();db=data/'tasks.sqlite';sqlite3.connect(db).close()
        for r in (1,2,3):
            rows=[{**task(source,(r-1)*1000+i),'role':'train_evolution','purpose':'evolution_train','round_id':r} for source,q in TRAIN_QUOTAS.items() for i in range(q)] if sequential or full_cohort else [{**task('nq_open',r*10+i),'role':'train_evolution','purpose':'evolution_train','round_id':r,'meta_probe':True} for i in range(2)]
            value=make_manifest('train_evolution',r,rows);freeze(data/f'B{r}/manifest.json',value)
            freeze(data/f'B{r}/meta_probe.json',make_manifest('train_evolution',r,rows,parent_manifest_hash=value['manifest_hash']))
        store=RoundStore(db);self.addCleanup(store.close);events=[]
        interrupted=[False]
        class Executor:
            def execute(executor,state,directory):
                if interrupt and not interrupted[0] and store.round_id==1 and 'MODEL' in directory.parts and store.mode=='child_post_update':
                    interrupted[0]=True
                    raise RuntimeError('mock interruption after committed SFT')
                events.append(('execute',store.round_id,store.mode,state.checkpoint_path))
                directory.mkdir(parents=True,exist_ok=True)
                candidate=state.harness_path!=str(h) or state.checkpoint_path!='parent'
                row=dict(task_id=store.current['tasks'][0]['task_id'],question_id=store.current['tasks'][0]['task_id'],domain='tool_use',source='nq_open',
                    rollout_id=0,seed=42,split='evolve_train',task_source_hash='fixture',
                    reset_hash='incomparable-child' if incomplete and store.mode=='child_post_update' else 'fixed',environment_type='mock',
                    external_task_budget={},chat_template_kwargs={},verification={'success':candidate},terminal_reward=int(candidate),
                    metrics={'native_partial_score':int(candidate)},model_call_count=0,wall_time_seconds=0,input_tokens=0,output_tokens=0)
                annotate_row(executor,state,row)
                freeze(directory/'window.json',{'tasks':[row['task_id']]})
                (directory/'probe_trajectories.jsonl').write_text(json.dumps(row)+'\n')
                (directory/'train_trajectories.jsonl').write_text(json.dumps(row)+'\n')
                # A negative measured gain must NOT trigger rollback or more candidates.
                score=0 if store.mode=='child_post_update' else 1
                row['verification']['success']=bool(score);row['terminal_reward']=score
                (directory/'probe_trajectories.jsonl').write_text(json.dumps(row)+'\n')
                rows=[{**row,'task_id':t['task_id'],'question_id':t['task_id'],'source':t['source'],'domain':t['domain']} for t in store.current['tasks']]
                if search:
                    if store.mode=='parent_pre_update': score=.5
                    else:
                        component=next(c for c in ('MODEL','HARNESS','ARTIFACTS') if c in directory.parts)
                        score=dict(search)[component]
                    for i,item in enumerate(rows):
                        item['verification']={'status':'completed','success':i<int(score*len(rows))}
                        item['terminal_reward']=int(item['verification']['success'])
                for item in rows:annotate_row(executor,state,item)
                (directory/'probe_trajectories.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                (directory/'train_trajectories.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                return EvaluationResult({'macro_success':score,'probe_identity':store.current['manifest_hash'],'domains':{'tool_use':{'count':len(rows),'successes':score*len(rows)}}},rows)
        engine=Executor();engine.store=store;durable=DurableExecutor(engine,StageJournal(root))
        protocol=RoundProtocol(SimpleNamespace(seed=42,round_protocol={'sequential_domains':False,
            'candidate_policy':'single_candidate_strict_positive_gain'}),root,durable)
        class Provider:
            supports_evolution=True
            def bind_context(self,*args):pass
            def complete(self,prompt,schema,**kwargs):
                events.append(('meta_update',store.round_id))
                self_outer.assertTrue(any(e[:3]==('execute',store.round_id,'parent_pre_update') for e in events))
                return MetaHarnessUpdate(harness=f'Reusable strategy round {store.round_id}',rationale='mock training diagnosis',changed_rules=['rule'],experience_id=kwargs['experience_id'])
        self_outer=self
        class Meta:
            capabilities={};_five_stage=staticmethod(lambda state:False)
            client=DurableClient(Provider(),StageJournal(root))
            def diagnose_and_route(self,state,observation,feedback):
                events.append(('decision',store.round_id,state.version))
                a=search[len(feedback)][0] if search else actions[store.round_id-1]
                operation,target=(('produce_harness','harness_bundle') if a=='HARNESS' else
                                  ('sft','current_checkpoint') if a=='MODEL' else ('write_asset','helper.py'))
                return MetaDecision(action=a,diagnosis='mock',evidence=[],rationale='mock',proposed_change='mock',expected_effect='mock',target_components=[a],
                    requested_changes=[dict(id='x',component=a,operation=operation,target=target,instruction='mock')])
            run_directory=root
            def learn_from_experience(self,*args):
                from sia.task_meta.meta import MetaAgent
                with patch('sia.task_meta.meta.harness_evidence',return_value=({'seed.json':'{}'},{'fixture':True})):
                    return MetaAgent.learn_from_experience(self,*args)
        class Updater:
            def __init__(self,action):self.action=action
            def apply(self,parent,decision,context):
                events.append(('update_parent',store.round_id,self.action.value,parent.checkpoint_path,parent.harness_path))
                after=copy.deepcopy(parent);after.generation+=1
                if self.action==TaskUpdateAction.MODEL:
                    if unavailable:
                        from sia.task_meta.types import DecisionConstraintError
                        raise DecisionConstraintError('MODEL/SFT unavailable: insufficient verified successes')
                    events.append(('trainer',store.round_id,parent.checkpoint_path,store.mode))
                    self_outer.assertTrue(context.evaluation.trajectories)
                    if full_cohort:
                        self_outer.assertEqual(len(context.evaluation.trajectories),360)
                        self_outer.assertEqual(Counter(r['source'] for r in context.evaluation.trajectories),TRAIN_QUOTAS)
                    self_outer.assertTrue(all(r['collection_stage']=='parent_pre_update' and r['branch_id']=='parent' for r in context.evaluation.trajectories))
                    after.checkpoint_path=f'child{store.round_id}';after.model_ref=after.checkpoint_path
                    after.checkpoint_manifest=[{'sha256':after.checkpoint_path}]
                elif self.action==TaskUpdateAction.ARTIFACTS:
                    from sia.task_meta.types import ArtifactState
                    from sia.task_meta.storage import artifact_manifest
                    path=context.directory/'assets';path.mkdir(parents=True,exist_ok=True)
                    (path/'helper.py').write_text('answer = '+str(store.round_id))
                    after.artifacts=ArtifactState(directory=str(path),manifest=artifact_manifest(str(path)))
                else:
                    from sia.task_meta.harnessforge_manifest import HarnessBundleManifest,load_manifest,save_manifest
                    path=context.directory/'h.json';manifest=load_manifest(parent.harness_path);files=dict(manifest.files)
                    files['Description.md'] += f'\nfixture round {store.round_id}\n'
                    save_manifest(path,HarnessBundleManifest(f'fixture_{store.round_id}',files))
                    after.harness_path=str(path)
                return after,TaskUpdate(self.action,'explicit mock')
        observation=MetaObservation(0,'T_0',0,{},[],{},[],[],'fake','',[],None,None,None,[],[],
            available_actions={a.value:{'available':(search is not None and any(x[0]=='ARTIFACTS' for x in search)) or a!=TaskUpdateAction.ARTIFACTS} for a in TaskUpdateAction})
        initial=TaskAgentState(0,'parent',str(h),checkpoint_path='parent');meta=MetaAgentState('fake',str(g))
        updaters={a:DurableUpdater(Updater(a),StageJournal(root)) for a in TaskUpdateAction if a!=TaskUpdateAction.ARTIFACTS or search and any(x[0]=='ARTIFACTS' for x in search)}
        with patch.object(loop,'build_observation',side_effect=lambda *a,**k:copy.deepcopy(observation)),patch.object(loop,'updater_capabilities',return_value={'MODEL':{'available':True}}):
            if interrupt:
                with self.assertRaisesRegex(RuntimeError,'mock interruption'):
                    loop.run_sequential_task_meta(root,initial,meta,durable,Meta(),updaters,max_generations=len(actions),round_protocol=protocol)
            result=loop.run_sequential_task_meta(root,initial,meta,durable,Meta(),updaters,max_generations=len(actions),round_protocol=protocol,resume=interrupt)
            before=list(events)
            again=loop.run_sequential_task_meta(root,initial,meta,durable,Meta(),updaters,max_generations=len(actions),round_protocol=protocol,resume=True)
        self.assertEqual(result['rounds_completed'],3);self.assertEqual(again['rounds_completed'],3);self.assertEqual(events,before)
        action,score=search[0]
        accepted=(not unavailable and not incomplete and score>.5)
        self.assertEqual(sum(e[0]=='decision' for e in events),len(actions))
        self.assertEqual(sum(e[0]=='meta_update' for e in events),len(actions))
        self.assertEqual(sum(e[0]=='trainer' for e in events),len(actions) if action=='MODEL' and not unavailable else 0)
        for stage in range(len(actions)):
            directory=root/f'round_{stage}'
            deployment=native.read(directory/'deployment.json')
            self.assertEqual(deployment['status'],'deployed' if accepted else 'parent_retained')
            self.assertEqual(deployment['attempted_components'],[action])
            self.assertFalse(native.read(directory/'selected_update.json')['selected_before_child_evaluation'])
            report=native.read(directory/'all_candidate_effects.json')
            self.assertEqual(len(report['attempts']),1)
            executed=0 if unavailable else 1
            n_tasks=360 if full_cohort else 2
            self.assertEqual(report['candidate_task_pairs'],executed*n_tasks)
            self.assertEqual(sum(e[:2]==('execute',stage+1) for e in events),1+executed)
            experience=native.read(directory/'experience.json')
            attempt=experience['candidate_attempts'][0]
            expected=expected_outcome or ('accepted_positive_gain' if accepted else 'zero_gain')
            self.assertEqual(attempt['outcome_class'],expected)
            self.assertIs(attempt['accepted_for_deployment'],accepted)
            self.assertEqual(experience['chosen_action'],action)
            self.assertEqual(experience['attempted_component'],action)
            self.assertEqual(experience['state_after'],deployment['task_after'])
            if unavailable:
                self.assertIsNone(experience['performance_delta'])
                self.assertIsNone(experience['trajectory_after'])
                self.assertFalse(experience['attempted_state_after'])
                self.assertFalse(native.read(directory/'paired_training_effect.json')['overall']['complete'])
            else:
                self.assertTrue(experience['attempted_state_after'])
                self.assertEqual(experience['attempted_trajectory_after'],attempt['trajectory_after'])
                self.assertEqual(experience['attempted_outcome'],attempt['paired_outcome'])
                if incomplete:
                    self.assertIsNone(experience['performance_delta'])
                    self.assertFalse(native.read(directory/'paired_training_effect.json')['overall']['complete'])
                else:
                    self.assertAlmostEqual(experience['performance_delta'],score-.5)
            if not accepted and not unavailable:
                self.assertNotEqual(experience['attempted_state_after'],experience['state_after'])
            if full_cohort:
                expected_manifest=native.read(data/f'B{stage+1}/manifest.json')
                self.assertEqual(native.read(directory/'stage_manifest.json'),expected_manifest)
                self.assertEqual({p['task_id'] for p in report['attempts'][0]['pairs']},
                                 {p['task_id'] for p in expected_manifest['tasks']})
                self.assertEqual(Counter(p['source'] for p in report['attempts'][0]['pairs']),TRAIN_QUOTAS)
        self.assertEqual(result['task_update_policy'],'single_candidate_strict_positive_gain_v1')


class NativeRolloutReuse(unittest.TestCase):
    def test_parent_child_two_passes_resume_and_state_binding(self):
        from sia.task_meta.pipeline_execution import MultiDomainExecutor
        from sia.task_meta.durable import load_task
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();h=root/'h.json'
            from sia.task_meta.harnessforge_production import initialize_base_manifest
            initialize_base_manifest(h)
            freeze(root/'protocol.json',{'fixture':'native-execution-mock'})
            db=data/'tasks.sqlite';sqlite3.connect(db).close()
            rows=[];pack=data/'B1/records.jsonl';pack.parent.mkdir()
            with pack.open('wb') as f:
                for domain,source in [('tool_use','envscaler'),('code','deepcoder_taco'),('searchqa','nq_open')]:
                    for i in range(2):
                        line=(json.dumps({'task':'mock','problem':'mock','question':'mock'})+'\n').encode();offset=f.tell();f.write(line)
                        rows.append({**task(source,i),'domain':domain,'role':'train_evolution','purpose':'evolution_train','round_id':1,
                            'meta_probe':i==0,'record_file':str(pack),'record_offset':offset,'record_length':len(line)})
            for row in rows:row['record_sha256']=file_hash(pack)
            m=make_manifest('train_evolution',1,rows);freeze(data/'B1/manifest.json',m)
            freeze(data/'B1/meta_probe.json',make_manifest('train_evolution',1,[r for r in rows if r['meta_probe']],parent_manifest_hash=m['manifest_hash']))
            store=RoundStore(db)
            calls=[]
            class Environment:
                def reset(self,*args):return {'public':'only'}
                def evaluate(self,answer):return SimpleNamespace(infrastructure_error=False,reward=1,metrics={'native_partial_score':1,'f1':1},
                    verification={'status':'completed','success':True},error_type=None,details={})
                def close(self):pass
            def run_seed(spec,model,environment,*args,**kwargs):
                calls.append('mock_task');model([{'role':'user','content':'only public'}],seed=42,max_tokens=2,temperature=0)
                return {'final_answer':'mock','sft_conversations':[],'notes':['must not enter persistent memory']}
            def model_factory(*args,**kwargs):
                return lambda *a,**k:{'content':'mock','usage':{'prompt_tokens':1,'completion_tokens':1}}
            engine=MultiDomainExecutor(store,lambda d:Environment(),model_factory,quotas={'tool_use':2,'code':2,'searchqa':2})
            durable=DurableExecutor(engine,StageJournal(root));protocol=RoundProtocol(SimpleNamespace(seed=42),root,durable)
            state=TaskAgentState(0,'parent',str(h),checkpoint_path='parent');meta=MetaAgentState('mock',str(h))
            try:
                protocol.begin(0,meta,[])
                with patch('sia.task_meta.seed.load_seed',return_value={'schema_version':1}),patch('sia.task_meta.seed.run_seed',side_effect=run_seed):
                    before=root/'before';protocol.bind_execution(state,before)
                    first=durable.execute(state,before);self.assertEqual(len(calls),6)
                    self.assertEqual(len(first.trajectories),6)
                    protocol.bind_execution(state,before)
                    resumed=durable.execute(state,before);self.assertEqual(len(calls),6)
                    child=root/'child';protocol.bind_execution(state,child,'child_post_update')
                    collected=durable.execute(state,child);self.assertEqual(len(calls),12)
                    self.assertEqual(len(collected.trajectories),6);self.assertEqual(collected.cost['physical_rollout_attempts'],6)
                    self.assertTrue(all(r['purpose']=='evolution_train' and r['collection_stage']=='child_post_update' and not r['notes'] for r in collected.trajectories))
                    protocol.meta.version=1
                    with self.assertRaises(ValueError):protocol.bind_execution(state,before)
                    pack.write_text('{}\n')
                    protocol.begin(0,meta,[])
                    with self.assertRaisesRegex(ValueError,'differs from frozen manifest'):store.probe()
            finally:store.close()


if __name__=='__main__':unittest.main()
