"""CPU tests with explicit fake models; no training results are produced here."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import asdict
from unittest.mock import patch

BUNDLE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BUNDLE))
from common import read,write
from install_runtime import patched_files

SOURCE=Path(os.environ.get('RSI_TEST_RUNTIME',str(BUNDLE.parent/'RSI_method2_20260915/four_gpu')))
TEMP=tempfile.TemporaryDirectory()
RUNTIME=Path(TEMP.name)/'runtime'
shutil.copytree(SOURCE/'sia',RUNTIME/'sia',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(SOURCE/'scripts',RUNTIME/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
(RUNTIME/'pyproject.toml').write_text('# CPU test fixture, not an installable production runtime\n')
for name,content in patched_files(SOURCE).items():
    p=RUNTIME/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
for name in ('runtime_extensions.py','round_validation.py','round_evolution.py','evolution_protocol.py'):
    shutil.copy2(BUNDLE/name,RUNTIME/'sia/task_meta'/name)
sys.path.insert(0,str(RUNTIME))
from sia.task_meta.runtime_extensions import FullTrainingStore,meta_sample,epoch_complete
from sia.task_meta import sequential_loop as loop
from sia.task_meta.durable import DurableUpdater,StageJournal
from sia.task_meta.storage import save_json
from sia.task_meta.types import EvaluationResult,MetaAgentState,MetaDecision,MetaObservation,TaskAgentState,TaskUpdate,TaskUpdateAction

# Shared native fixture also exercises the round protocol module; clean after the suite.
import atexit
atexit.register(TEMP.cleanup)

class DataAndEpoch(unittest.TestCase):
    def test_counts_and_separate_configs(self):
        train=read(BUNDLE/'configs/train.json');valid=read(BUNDLE/'configs/validation.json')
        self.assertEqual(sum(train['train_quotas_per_round'].values())*3,3600)
        self.assertEqual(train['allocated_tasks'],3600)
        self.assertEqual((train['rounds'],train['sft_epochs_per_update'],train['sft_max_steps']),(3,1,-1))
        self.assertEqual(len(set(valid['benchmark_ids'])),7)
        self.assertFalse(valid['feedback_to_training_or_meta'])
        self.assertTrue(all('test/' not in p['path'] for d in train['datasets'] for p in d['files']))

    def fixture(self,path):
        datasets=[]
        for domain,name in [('tool_use','envscaler'),('code','deepcoder_taco'),('searchqa','hotpotqa')]:
            rows=[]
            for i in range(4):
                rows.append({'id':str(i),'task':'tool '+str(i),'env_id':str(i),'problem':'code '+str(i),
                             'question':'search '+str(i),'answer':'SECRET_GOLD','context':[['title',['public sentence']]]})
            relative=name+'.jsonl';(path/relative).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            datasets.append(dict(name=name,domain=domain,count=4,paths=[relative]))
        return dict(dataset_root=str(path),datasets=datasets,seed=42,expected_total=12,training_probe_per_domain=1)

    def test_full_cohort_repeats_and_source_changes_are_rejected(self):
        from prepare_data import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            out=prepare(config,root/'runtime');store=FullTrainingStore(out/'tasks.sqlite')
            first,cursor=store.window(None,{})
            second,cursor2=store.window(cursor,{})
            self.assertEqual(len(first),12);self.assertEqual(list(first),list(second));self.assertEqual(cursor,cursor2)
            self.assertEqual(len(store.probe()),3);store.close()
            # Public passage index has no answers or question text.
            import sqlite3
            conn=sqlite3.connect(out/'search.sqlite')
            text=str(conn.execute('SELECT title,body FROM documents').fetchall());conn.close()
            self.assertNotIn('SECRET_GOLD',text);self.assertNotIn('search 0',text)
            with (root/'hotpotqa.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaises(ValueError):prepare(config,root/'runtime')

    def test_counts_do_not_silently_shrink(self):
        from prepare_data import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root);config['datasets'][0]['count']=5
            with self.assertRaises(ValueError):prepare(config,root/'runtime')
            self.assertFalse((root/'runtime/data/joint_full/tasks.sqlite').exists())

    def test_full_manifest_enters_native_prepare_without_resplitting(self):
        from prepare_data import prepare
        from sia.task_meta import pipeline
        folder=RUNTIME/'data/native_prepare_fixture';folder.mkdir(parents=True,exist_ok=True)
        config=self.fixture(folder)
        out=prepare(config,folder/'prepared')
        seed=folder/'seed.json';seed.write_text('{}')
        cfg=pipeline.PipelineConfig(data_dir=str(out),task_checkpoint=str(folder/'fake_model'),
            seed_harness=str(seed),training_schedule='full_cohort',search_dev_fraction=0,
            probe_per_domain=dict.fromkeys(('tool_use','code','searchqa'),1)).checked()
        with patch('sia.task_meta.seed.load_seed',return_value={'reference':'test_override'}),\
             patch.object(pipeline,'model_identity',return_value={'path':str(folder/'fake_model')}):
            result=pipeline.prepare(cfg)
        self.assertEqual(result['status'],'PREPARED_NOT_TRAINED')
        self.assertEqual(sum(d['evolve_train'] for d in result['counts'].values()),12)

    def test_meta_sampling_preserves_full_training_list(self):
        rows=[dict(task_id=str(i),rollout_id=0,domain=['tool_use','code','searchqa'][i%3],verification={'success':i%2==0}) for i in range(300)]
        before=copy.deepcopy(rows);sample=meta_sample(rows)
        self.assertEqual(len(sample),96);self.assertEqual(rows,before)
        self.assertEqual({(x['domain'],x['verification']['success']) for x in sample},
                         {(x['domain'],x['verification']['success']) for x in rows})
        self.assertEqual({x['task_id'] for x in sample},{x['task_id'] for x in meta_sample(list(reversed(rows)))})

    def test_disk_rollout_preserves_scores_and_full_sft_messages(self):
        from sia.task_meta.runtime_extensions import compact_rollout,hydrate_rollout
        from sia.task_meta.observations import trajectory_statistics
        from sia.task_meta.sft import select_positive_rows
        root=RUNTIME/'runs/test_override/gen_0/train_rollouts';root.mkdir(parents=True,exist_ok=True)
        full=dict(task_id='a',question_id='a',rollout_id=0,seed=42,domain='searchqa',split='evolve_train',state_hash='bound',
                  messages=[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}],
                  terminal_reward=1,verification={'status':'completed','success':True,'verifier_id':'test_override','exact_match':True},
                  model_answer='answer',valid_answer=True,model_calls=[],transport_calls=[{'large_log':'x'*10000}],
                  model_call_count=1,wall_time_seconds=1,input_tokens=2,output_tokens=1)
        path=root/'sample.json';write(path,{'row':full})
        compact=compact_rollout(full,path)
        self.assertNotIn('transport_calls',compact);self.assertEqual(hydrate_rollout(compact),full)
        self.assertEqual(trajectory_statistics([compact]),trajectory_statistics([full]))
        samples=select_positive_rows([compact],profile='multidomain')
        self.assertEqual(samples[0]['messages'],full['messages']);self.assertEqual(samples[0]['verification'],full['verification'])
        self.assertNotIn('transport_calls',samples[0]);self.assertIn('_source_rollout_ref',samples[0])
        changed=copy.deepcopy(full);changed['messages'][-1]['content']='tampered';write(path,{'row':changed})
        with self.assertRaises(ValueError):select_positive_rows([compact],profile='multidomain')

    def test_one_epoch_not_fixed_step_override(self):
        spec=importlib.util.spec_from_file_location('test_trainer',RUNTIME/'scripts/train_task_meta_sft.py')
        trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
        request={'sft_profile':'multidomain','supervision':'final_assistant',
                 'training':{'num_train_epochs':1,'max_steps':-1,'max_samples':10000000}}
        options=trainer.resolve_training_options(request,SimpleNamespace())
        self.assertEqual((options['num_train_epochs'],options['max_steps']),(1,-1))
        request['training']['max_steps']=32
        with self.assertRaises(ValueError):trainer.resolve_training_options(request,SimpleNamespace())
        epoch_complete(1.0,3,-1,0.5)
        for epoch,steps,loss in [(0.7,3,0.5),(2.0,6,0.5),(1.0,0,0.5),(1.0,3,float('nan'))]:
            with self.assertRaises(RuntimeError):epoch_complete(epoch,steps,-1,loss)

class RoundOrdering(unittest.TestCase):
    def test_three_native_commits_validate_in_order_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h=root/'h.json';h.write_text('{}')
            task=TaskAgentState(0,'fake',str(h),checkpoint_path='fake')
            meta=MetaAgentState('fake',str(h));save_json(root/'protocol.json',{'config':{}})
            events=[]
            class Executor:
                def execute(self,state,directory):
                    directory.mkdir(parents=True,exist_ok=True)
                    candidate=directory.parent.name=='HARNESS'
                    events.append(('evaluate',state.generation,candidate))
                    rows=[dict(task_id='q',rollout_id=0,seed=42,split='search_dev',task_source_hash='source',
                        reset_hash='reset',environment_type='test_override',external_task_budget={},chat_template_kwargs={},
                        verification={'success':candidate},terminal_reward=int(candidate),metrics={'native_partial_score':int(candidate)})]
                    (directory/'probe_trajectories.jsonl').write_text(json.dumps(rows[0])+'\n')
                    (directory/'train_trajectories.jsonl').write_text('');save_json(directory/'window.json',{})
                    return EvaluationResult({'macro_success':int(candidate),'probe_identity':'fixed',
                        'domains':{'tool_use':{'successes':int(candidate),'count':1}}},[])
            class Updater:
                def apply(self,parent,decision,context):
                    candidate=copy.deepcopy(parent);candidate.generation+=1
                    path=context.directory.parent/'h.json';path.write_text(json.dumps({'generation':candidate.generation}))
                    candidate.harness_path=str(path)
                    return candidate,TaskUpdate(TaskUpdateAction.HARNESS,'test_override')
            class Meta:
                capabilities={};client=SimpleNamespace(bind_context=lambda *args:None)
                def diagnose_and_route(self,state,observation,feedback):
                    events.append(('route',state.version))
                    return MetaDecision(action='HARNESS',diagnosis='test_override',evidence=[],rationale='test_override',
                        proposed_change='test_override',expected_effect='test_override',target_components=['HARNESS'],
                        requested_changes=[dict(id='x',component='HARNESS',operation='update',target='test_override',instruction='test_override')])
                def learn_from_experience(self,state,*args):return {}
            def accept(root,state,*args):
                state=copy.deepcopy(state);state.version+=1;state.bundle_hash=f'fake_meta_{state.version}';return state
            observation=MetaObservation(0,'T_0',0,{},[],{},[],[],'fake','',[],None,None,None,[],[],
                available_actions={a.value:{'available':a==TaskUpdateAction.HARNESS} for a in TaskUpdateAction})
            def callback(number,record):
                self.assertTrue((root/f'round_{number}/complete.json').exists())
                events.append(('validate',number+1))
                if number==0 and fail[0]:raise RuntimeError('test validation outage')
            updaters={TaskUpdateAction.HARNESS:DurableUpdater(Updater(),StageJournal(root))}
            fail=[True]
            with patch.object(loop,'build_observation',return_value=observation),patch.object(loop,'updater_capabilities',return_value={}),patch.object(loop,'_accept_meta_update',side_effect=accept):
                with self.assertRaisesRegex(RuntimeError,'validation outage'):
                    loop.run_sequential_task_meta(root,task,meta,Executor(),Meta(),updaters,max_generations=3,after_round=callback)
                self.assertFalse((root/'round_1/complete.json').exists())
                before=len([e for e in events if e[0]=='evaluate']);fail[0]=False
                result=loop.run_sequential_task_meta(root,task,meta,Executor(),Meta(),updaters,max_generations=3,after_round=callback,resume=True)
            self.assertEqual(result['rounds_completed'],3)
            self.assertEqual([e[1] for e in events if e[0]=='validate'],[1,1,2,3])
            self.assertEqual(len([e for e in events if e[0]=='evaluate'])-before,4)
            for n in (1,2):self.assertLess(events.index(('validate',n)),events.index(('route',n)))

class EvaluationReceipts(unittest.TestCase):
    def test_validation_sources_cannot_change_between_rounds(self):
        from validation_sources import freeze,verify
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data.json';data.write_text('[]')
            specs=root/'spec.json';write(specs,{'evaluators':{name:{'data_path':str(data),'entrypoint':str(data),'task_ids_path':str(data)}
                for name in ('livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev')}})
            settings={'official_specs':str(specs),'tool_benchmarks':{}}
            path=root/'frozen.json';freeze(settings,path);verify(settings,path)
            data.write_text('[1]')
            with self.assertRaises(ValueError):verify(settings,path)

    def test_all_seven_evaluators_run_and_completed_scores_resume(self):
        import validate as evaluation
        settings=read(BUNDLE/'configs/validation.json')
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'validation/fake/round_01';out.mkdir(parents=True)
            h=root/'h.json';h.write_text('{}');source=root/'complete.json';write(source,{'test_override':True})
            state=TaskAgentState(1,'test_override',str(h),checkpoint_path='fake')
            snapshot={'round':1,'task_state':asdict(state),'meta_state':{'version':1},'source_round':str(source),'chosen_component':'HARNESS'}
            write(out/'round_snapshot.json',snapshot)
            specs=root/'specs.json';write(specs,{'evaluators':{k:{} for k in settings['benchmark_ids'] if k not in settings['tool_benchmarks']}})
            settings['official_specs']=str(specs);config=root/'validation.json';write(config,settings)
            fail=[True]
            def score(identifier,binding,directory):
                calls.append(identifier)
                if identifier=='livecodebench' and fail[0]:raise RuntimeError('test_override evaluator outage')
                result={'status':'completed','state_hash':binding['state_hash'],'metrics':{'test_override':0.0},'expected':2,'completed':2}
                write(Path(directory)/'result.json',result);return result
            def predict(p,f,s,out,**kwargs):
                name=next(iter(read(s)['evaluators']));out.mkdir(parents=True,exist_ok=True)
                (out/(name+'.jsonl')).write_text('{}\n')
            with patch.object(evaluation,'ROOT',root),patch('validation_sources.verify'),patch('sia.task_meta.pipeline.load_config',return_value=SimpleNamespace()),\
                 patch('sia.task_meta.storage.checkpoint_manifest',return_value=[]),patch('sia.task_meta.task_harness.harness_identity',return_value={}),\
                 patch('sia.task_meta.gpu_phases.ensure_services'),patch('sia.task_meta.report_predictions.generate_report_predictions',side_effect=predict),\
                 patch('sia.task_meta.reporting.OfficialEvaluatorSpec',side_effect=lambda **kw:SimpleNamespace(**kw)),\
                 patch('sia.task_meta.reporting.evaluate_official',side_effect=lambda spec,rows,binding,out:score(spec.benchmark,binding,out)),\
                 patch('tool_validation.evaluate_tool',side_effect=lambda name,c,b,s,u,out:score(name,b,out)):
                with self.assertRaisesRegex(RuntimeError,'evaluator outage'):evaluation.validate(out/'round_snapshot.json','unused',config)
                self.assertFalse((out/'complete.json').exists())
                fail[0]=False;done=evaluation.validate(out/'round_snapshot.json','unused',config)
                self.assertEqual(done['benchmarks_completed'],7);self.assertFalse(done['feedback_to_meta'])
                self.assertEqual(calls.count('bfcl_v3'),1);self.assertEqual(calls.count('acebench'),1)
                self.assertEqual(set(calls),set(settings['benchmark_ids']))
                count=len(calls);evaluation.validate(out/'round_snapshot.json','unused',config)
                self.assertEqual(len(calls),count)
                self.assertTrue((out.parent/'all_rounds.csv').is_file())

    def test_full_denominator_and_no_nan_or_duplicate_ids(self):
        from tool_validation import metric_rows
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred=root/'BFCL_v3_simple_result.json';score=root/'BFCL_v3_simple_score.json'
            write(pred,[{'id':'a'},{'id':'b'}]);write(score,[{'accuracy':0.5,'total_count':2}])
            self.assertEqual(metric_rows('bfcl_v3',root,{'simple':['a','b']})['expected'],2)
            write(pred,[{'id':'a'}])
            with self.assertRaises(RuntimeError):metric_rows('bfcl_v3',root,{'simple':['a','b']})
            write(pred,[{'id':'a'},{'id':'a'}])
            with self.assertRaises(RuntimeError):metric_rows('bfcl_v3',root,{'simple':['a','b']})
            write(pred,[{'id':'a'},{'id':'b'}]);write(score,[{'accuracy':float('nan'),'total_count':2}])
            with self.assertRaises(RuntimeError):metric_rows('bfcl_v3',root,{'simple':['a','b']})

    def test_interrupted_bridge_request_is_never_redispatched(self):
        from harness_api import HarnessAPI
        from sia.task_meta.durable import value_hash
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            api=HarnessAPI.__new__(HarnessAPI);api.directory=Path(tmp);api.frozen={'state_hash':'frozen'};api.lock=threading.Lock()
            body={'model':'gpt-rsi-frozen','messages':[{'role':'user','content':'test'}]}
            identity={'state_hash':'frozen','request':body,'protocol':'official_outer_saved_harness_inner_v1'}
            path=api.directory/(value_hash(identity)+'.json');write(path,{'status':'dispatching'})
            with self.assertRaisesRegex(RuntimeError,'no blind retry'):api.complete(body)

class StartupRecovery(unittest.TestCase):
    def test_failed_host_preflight_can_reenter_native_startup(self):
        import train
        from sia.task_meta import pipeline
        from sia.task_meta.meta_backends.contracts import BackendUnavailable
        directory=RUNTIME/'runs'/'startup_recovery_fixture'
        attempts=[]
        def fail_host_check(prepared):
            attempts.append(prepared.directory)
            raise BackendUnavailable('REMOTE_META_CONNECTION_UNAVAILABLE','test fixture; no API')
        backend=SimpleNamespace(bind_context=lambda *a:None,validate=fail_host_check)
        bundle=SimpleNamespace(hash='0'*64)
        config=SimpleNamespace(mode='full',meta=SimpleNamespace(compatibility_mode='in_run'))
        with patch.object(sys,'argv',['train','--config','fixture','--run-dir',str(directory)]),\
             patch.object(pipeline,'load_config',return_value=config),\
             patch.object(pipeline,'backend_for',return_value=(backend,None,bundle)):
            for _ in range(2):
                with self.assertRaises(BackendUnavailable):train.main()
        self.assertEqual(len(attempts),2)
        self.assertFalse((directory/'protocol.json').exists())
        self.assertEqual(read(directory/'phase.json')['phase'],'REMOTE_META_CONNECTION_UNAVAILABLE')

if __name__=='__main__':unittest.main(verbosity=2)
