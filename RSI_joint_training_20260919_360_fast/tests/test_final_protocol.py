"""Final protocol acceptance tests: fixtures only, no services, models or paid APIs."""
import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_contracts as native
from test_training_protocol import task
from evolution_protocol import make_manifest,freeze,VALIDATION_QUOTAS,success_filter
from data_protocol import TRAIN_QUOTAS,scan,materialize
from sia.task_meta.round_evolution import training_pairs
from evaluation_results import publish,counts


class FinalProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def pairs(self):
        tasks=[{**task(s,i),'role':'train_evolution','round_id':1} for s,n in TRAIN_QUOTAS.items() for i in range(n)]
        m=make_manifest('train_evolution',1,tasks);pre=[];post=[]
        for i,t in enumerate(tasks):
            row=dict(task_id=t['task_id'],source_role='train_evolution',manifest_hash=m['manifest_hash'],
                collection_stage='parent_pre_update',task_source_hash=t['content_hash'],reset_hash=t['content_hash'],
                seed=42,environment_type=t['source'],external_task_budget={},chat_template_kwargs={},
                verification={'status':'completed','success':i%4 in (1,2)})
            pre.append(row);post.append({**row,'collection_stage':'child_post_update',
                'verification':{'status':'completed','success':i%4 in (0,2)}})
        args=(SimpleNamespace(actual_change={},versions={},trajectory_before='pre',trajectory_after='post'),
              dict(decision_id='d',parent_hash='p',child_hash='c'))
        return m,pre,post,args

    def test_all_360_paired_tasks_and_four_transitions(self):
        m,a,b,args=self.pairs();result=training_pairs(m,a,b,*args)
        self.assertEqual(len(result['pairs']),360)
        self.assertEqual(result['overall']['transitions'],dict(failure_to_success=90,success_to_failure=90,
            success_to_success=90,failure_to_failure=90))
        self.assertEqual(result['overall']['success_delta'],0)
        self.assertEqual({s:r['n_expected'] for s,r in result['by_source'].items()},TRAIN_QUOTAS)
        self.assertEqual({s:r['n_expected'] for s,r in result['by_domain'].items()},dict(tool_use=120,code=120,searchqa=120))
        covered=[i for batch in result['coverage']['batches'] for i in batch['task_ids']]
        self.assertEqual(len(set(covered)),360)
        b[-1]['verification']['status']='pending'
        partial=training_pairs(m,a,b,*args)
        self.assertIsNone(partial['overall']['success_delta']);self.assertEqual(partial['overall']['n_scored'],359)
        for changed in (b[:-1],b+[b[0]],[{**r,'collection_stage':'parent_pre_update'} for r in b]):
            with self.assertRaises(ValueError):training_pairs(m,a,changed,*args)

    def test_sft_has_no_400_or_240_cap(self):
        rows=[dict(task_id=str(i),trajectory_id=str(i),domain='code',verification={'status':'completed','success':True},
              messages=[{'role':'assistant','content':str(i)}]) for i in range(600)]
        selected,report=success_filter(rows,minimum_samples=1,native_selector=lambda rs,profile:rs)
        self.assertEqual(len(selected),600);self.assertEqual(len(report['trajectory_decisions']),600)

    def test_complete_ace_conversations_are_not_split(self):
        p=self.root/'ace.json';p.write_text(json.dumps([
            {'id':f'normal_multi_turn_user_adjust_{family}_{turn}','question':f'public {family} {turn}'}
            for family in range(2) for turn in range(3)]))
        cfg=native.read(native.BUNDLE/'configs/train.json')
        rows,audit=scan([dict(path=str(p),source='acebench',domain='tool_use',kind='evaluation',
            original_split='test',stratify=[])],cfg['grouping'])
        self.assertEqual((len(rows),audit['valid_unique_native_ids']),(2,6))
        packed=materialize(make_manifest('independent_validation',None,[{**r,'role':'independent_validation','round_id':None} for r in rows]),self.root/'pack')
        self.assertTrue(all(len(t['native_ids'])==3 for t in packed['tasks']))

    def test_reporting_has_all_300_rows_and_incomplete_is_not_a_final_rate(self):
        tasks=[{**task(s,i,'evaluation'),'source':s,'domain':'tool_use' if s in ('bfcl_v3','acebench') else 'searchqa' if s.endswith('_dev') else 'code'}
               for s,n in VALIDATION_QUOTAS.items() for i in range(n)]
        m=make_manifest('independent_validation',None,tasks);out=self.root/'round_01';out.mkdir()
        frozen=dict(source_role='independent_validation',state_hash='child');snapshot={'round':1}
        result=publish(m,snapshot,frozen,[],out)
        self.assertEqual(result['overall']['n_pending'],300);self.assertIsNone(result['overall']['success_rate'])
        blocked=publish(m,snapshot,frozen,[dict(benchmark_id='bfcl_v3',status='blocked')],out)
        self.assertEqual(blocked['overall']['n_infra_error'],0)
        self.assertEqual(blocked['overall']['n_pending'],300)
        scores=[dict(benchmark_id=s,task_results=[dict(native_id=str(i),success=i%2==0,scoring_status='completed',
            execution_status='completed',native_scores={'official':int(i%2==0)}) for i in range(n)]) for s,n in VALIDATION_QUOTAS.items()]
        result=publish(m,snapshot,frozen,scores,out)
        self.assertEqual(result['overall']['n_scored'],300);self.assertEqual(len(result['benchmarks']),7)
        self.assertEqual({s:v['n_expected'] for s,v in result['domains'].items()},dict(tool_use=100,code=100,searchqa=100))
        self.assertEqual(len((out/'task_results.jsonl').read_text().splitlines()),300)
        scores[-1]['task_results'].pop();partial=publish(m,snapshot,frozen,scores,out)
        self.assertIsNone(partial['overall']['success_rate']);self.assertEqual(partial['overall']['n_scored'],299)

    def test_actual_tool_runner_queue_is_50_not_the_full_pool(self):
        import tool_validation as tool
        repo=self.root/'repo';(repo/'bfcl_eval/constants').mkdir(parents=True)
        (repo/'bfcl_eval/constants/model_config.py').write_text('# fixture')
        (repo/'bfcl_eval/data').mkdir()
        freeze(repo/'bfcl_eval/data/BFCL_v3_simple.json',[{'id':str(i),'question':'fixture'} for i in range(77)])
        spec=dict(repository_root=str(repo),selected_task_ids=[str(i) for i in range(50)],
            selected_tasks=[dict(task_id=str(i)) for i in range(50)])
        out=self.root/'scores';frozen=dict(source_role='independent_validation',state_hash='child')
        phases=[]
        def run(command,work,output,log,**kwargs):
            phases.append(command[4]);queued=tool.records(work/'bfcl_eval/data/BFCL_v3_simple.json')
            self.assertEqual([r['id'] for r in queued],spec['selected_task_ids'])
            freeze(output/'BFCL_v3_simple_result.json',[{'id':r['id']} for r in queued])
            freeze(output/'BFCL_v3_simple_score.json',[{'accuracy':1.,'correct_count':50,'total_count':50}])
        class API:
            user_client=None
            def __init__(self,*args):pass
            def complete(self,*args):raise AssertionError('No model calls in this test')
        with patch.object(tool,'HarnessAPI',API),patch.object(tool,'run_native',side_effect=run):
            result=tool.evaluate_tool('bfcl_v3',SimpleNamespace(),frozen,spec,{},out)
            self.assertEqual(result['expected'],50);self.assertEqual(len(result['task_results']),50)
            self.assertEqual(native.read(out/'task_queue.json')['complete_tasks'],50)
            tool.evaluate_tool('bfcl_v3',SimpleNamespace(),frozen,spec,{},out)
            self.assertEqual(len(phases),2)
            with self.assertRaises(ValueError):tool.evaluate_tool('bfcl_v3',SimpleNamespace(),frozen,
                {'repository_root':str(repo)}, {},self.root/'blocked')

    def test_actual_served_weights_and_all_four_devices_are_checked(self):
        from sia.task_meta.gpu_phases import verify_services
        checkpoint=self.root/'model';checkpoint.mkdir()
        replicas=[{'gpu':i,'base_url':f'http://127.0.0.1:{4500+i}/v1'} for i in range(4)]
        expected={'checkpoint_path':str(checkpoint.resolve()),'weights':{'weights':'child'}}
        health=[dict(ready=True,visible_devices=str(i),bindings={str(checkpoint):expected}) for i in range(4)]
        def response(*args,**kwargs):return io.StringIO(json.dumps(health.pop(0)))
        with patch('sia.task_meta.storage.checkpoint_manifest',return_value=expected['weights']),\
             patch('sia.task_meta.gpu_phases.build_opener',return_value=SimpleNamespace(open=response)):
            self.assertEqual(len(verify_services(SimpleNamespace(task_replicas=replicas),checkpoint)),4)
            health.append(dict(ready=True,visible_devices='0',bindings={str(checkpoint):{**expected,'weights':{'weights':'parent'}}}))
            with self.assertRaises(RuntimeError):verify_services(SimpleNamespace(task_replicas=replicas),checkpoint)

    def test_real_dry_run_counts_include_optional_A0_only_when_enabled(self):
        from launch import dry_run
        config=native.read(native.BUNDLE/'configs/train.json');config['validation_vault']=str(self.root/'vault')
        for r in (1,2,3):
            rows=[{**task(s,i+r*10000),'role':'train_evolution','round_id':r} for s,n in TRAIN_QUOTAS.items() for i in range(n)]
            freeze(self.root/'data'/config['data_release']/f'B{r}/manifest.json',make_manifest('train_evolution',r,rows))
        valid=[{**task(s,i,'evaluation'),'role':'independent_validation','round_id':None} for s,n in VALIDATION_QUOTAS.items() for i in range(n)]
        freeze(self.root/'vault/validation/manifest.json',make_manifest('independent_validation',None,valid))
        result=dry_run(config,self.root)
        self.assertEqual(result['base_task_executions'],3060);self.assertEqual(result['external_validation_checkpoints'],['A1','A2','A3'])
        config['evaluate_initial_system']=True
        self.assertEqual(dry_run(config,self.root)['base_task_executions'],3360)


if __name__=='__main__':unittest.main()
