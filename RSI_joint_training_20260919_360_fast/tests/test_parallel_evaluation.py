"""CPU-only checks for routing, response correlation and unchanged task allocation."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runtime'))


class ParallelEvaluationTests(unittest.TestCase):
    def test_four_replicas_and_duplicate_cache(self):
        from harness_api import HarnessAPI
        barrier=threading.Barrier(4);endpoints=[]
        class Client:
            def __init__(self,state,endpoint,**kwargs):self.endpoint=endpoint
            def __call__(self,messages,**kwargs):
                endpoints.append(self.endpoint);barrier.wait(timeout=10)
                return {'usage':{'prompt_tokens':1,'completion_tokens':1},'endpoint':self.endpoint}
        def run_seed(spec,model,env,prompt,memory,seed):
            model([{'role':'user','content':prompt}],seed=seed,max_tokens=32,temperature=0)
            return {'final_answer':prompt}
        config=SimpleNamespace(task_replicas=[{'base_url':f'http://127.0.0.1:{8171+i}/v1'} for i in range(4)],
            task_timeout=10,task_enable_thinking=False,model_call_limit=2,max_output_tokens=32,seed=42)
        state=SimpleNamespace(harness_path='mock-harness')
        with tempfile.TemporaryDirectory() as directory, \
             patch('sia.task_meta.durable.load_task',return_value=state), \
             patch('sia.task_meta.seed.load_seed',return_value={}), \
             patch('sia.task_meta.seed.run_seed',side_effect=run_seed), \
             patch('sia.task_meta.task_client.LocalTaskClient',Client):
            api=HarnessAPI(config,{'state_hash':'frozen','task_state':{}},directory,{'user_model':'user-simulator'})
            bodies=[{'model':'task','messages':[{'role':'user','content':str(i)}]} for i in range(4)]
            with ThreadPoolExecutor(4) as pool:responses=list(pool.map(api.complete,bodies))
            self.assertEqual(len(set(endpoints)),4)
            with ThreadPoolExecutor(4) as pool:cached=list(pool.map(api.complete,bodies))
            self.assertEqual(cached,responses);self.assertEqual(len(endpoints),4)
            self.assertTrue(all(json.loads(p.read_text())['status']=='completed' for p in Path(directory).glob('*.json')))

    def test_rpc_out_of_order_reply_correlation(self):
        from tool_sandbox import serve_rpc
        gate=threading.Barrier(4);errors=[]
        frames=b''.join(json.dumps({'rpc_id':i,'request':{'task':i}}).encode()+b'\n' for i in range(4))
        result=io.BytesIO()
        def handle(body):
            gate.wait(timeout=10);time.sleep((3-body['task'])*.01)
            return {'task':body['task']}
        serve_rpc(io.BytesIO(frames),result,handle,errors)
        replies=[json.loads(line) for line in result.getvalue().splitlines()]
        self.assertFalse(errors);self.assertEqual(len(replies),4)
        self.assertTrue(all(row['rpc_id']==row['response']['task'] for row in replies))
        self.assertNotEqual([r['rpc_id'] for r in replies],list(range(4)))

    def test_official_sdk_concurrent_calls(self):
        import openai
        from official_tool_entry import sdk_bridge
        from tool_sandbox import serve_rpc
        request_read,request_write=os.pipe();response_read,response_write=os.pipe();errors=[]
        barrier=threading.Barrier(4)
        def handle(body):
            barrier.wait(timeout=10);value=body['messages'][0]['content']
            return {'id':'fixture','object':'chat.completion','created':1,'model':'fixture',
                'choices':[{'index':0,'message':{'role':'assistant','content':value},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        def serve():
            with os.fdopen(request_read,'rb') as a,os.fdopen(response_write,'wb') as b:serve_rpc(a,b,handle,errors)
        thread=threading.Thread(target=serve,daemon=True);thread.start();original=openai.OpenAI
        try:
            with patch.dict(os.environ,{'RSI_RPC_READ_FD':str(response_read),'RSI_RPC_WRITE_FD':str(request_write)}):sdk_bridge()
            client=openai.OpenAI()
            def call(i):return client.chat.completions.create(model='fixture',messages=[{'role':'user','content':str(i)}]).choices[0].message.content
            with ThreadPoolExecutor(4) as pool:self.assertEqual(list(pool.map(call,range(4))),list(map(str,range(4))))
            self.assertFalse(errors)
        finally:
            openai.OpenAI=original;os.close(request_write);thread.join(timeout=3)

    def test_task_shards_keep_fixed_order_and_denominator(self):
        from parallel_predictions import partition_rows
        rows=[{'task_id':str(i),'prompt':'fixture','entry_point':'solution'} for i in range(13)]
        ids=[r['task_id'] for r in rows]
        parts=partition_rows('humaneval_plus',rows,ids,4)
        self.assertEqual([len(p) for p in parts],[4,3,3,3])
        self.assertEqual({r['task_id'] for part in parts for r in part},set(ids))
        self.assertEqual(sum(map(len,parts)),len(ids))
        with self.assertRaises(ValueError):partition_rows('humaneval_plus',rows+[rows[0]],ids,4)
        with self.assertRaises(ValueError):partition_rows('humaneval_plus',rows,ids[:-1],4)

    def test_prediction_workers_aggregate_and_resume(self):
        from parallel_predictions import generate_parallel
        from sia.task_meta.durable import value_hash
        calls=[];workers=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'data.jsonl';ids=[str(i) for i in range(9)]
            source.write_text(''.join(json.dumps({'task_id':i,'prompt':'fixture','entry_point':'solution'})+'\n' for i in ids))
            frozen=root/'frozen.json';frozen.write_text(json.dumps({'state_hash':'fixture'}))
            specs=root/'spec.json';specs.write_text(json.dumps({'evaluators':{'humaneval_plus':{
                'repository':'fixture','commit':'a'*40,'entrypoint':'fixture','entrypoint_sha256':'fixture',
                'python_executable':sys.executable,'data_path':str(source),'data_sha256':'fixture',
                'task_ids_path':'fixture','task_ids_sha256':'fixture','protocol':'fixture'}}}))
            config=SimpleNamespace(task_replicas=[{'base_url':str(i)} for i in range(4)],model_dump=lambda:{})
            class FakeProcess:
                def __init__(self,command,**kwargs):
                    job=json.loads(Path(command[-1]).read_text());workers.append(job['config']['task_base_url'])
                    spec=json.loads(Path(job['spec']).read_text())['evaluators']['humaneval_plus']
                    target=Path(job['output'])/'humaneval_plus.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        rows=[]
                        for task_id in json.loads(Path(spec['task_ids_path']).read_text()):
                            calls.append(task_id);row={'task_id':task_id,'state_hash':'fixture','final_answer':'fixture'}
                            rows.append({**row,'record_hash':value_hash(row)})
                        target.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                def wait(self):return 0
                def poll(self):return 0
            with patch('sia.task_meta.reporting.OfficialEvaluatorSpec.validate',return_value=ids), \
                 patch('parallel_predictions.subprocess.Popen',FakeProcess):
                output=root/'out'
                generate_parallel(config,frozen,specs,output)
                self.assertEqual(set(workers),{'0','1','2','3'});self.assertEqual(len(calls),9)
                result=(output/'humaneval_plus.jsonl').read_bytes()
                self.assertEqual([json.loads(line)['task_id'] for line in result.splitlines()],ids)
                generate_parallel(config,frozen,specs,output)
                self.assertEqual(len(calls),9);self.assertEqual(result,(output/'humaneval_plus.jsonl').read_bytes())


if __name__=='__main__':unittest.main()
