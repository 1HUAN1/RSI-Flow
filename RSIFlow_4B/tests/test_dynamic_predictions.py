"""CPU-only checks for the four-card dynamic evaluation queue."""
import importlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))


class DynamicPredictionTests(unittest.TestCase):
    def test_work_stealing_and_resume_without_new_model_calls(self):
        import dynamic_predictions
        generate_dynamic=importlib.reload(dynamic_predictions).generate_dynamic
        from sia.task_meta.durable import value_hash
        ids=[str(i) for i in range(12)]
        barrier=threading.Barrier(4)
        guard=threading.Lock()
        calls=[]
        seen=set()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'data.jsonl'
            source.write_text(''.join(json.dumps({'task_id':i,'prompt':'fixture','entry_point':'solution'})+'\n' for i in ids))
            frozen=root/'frozen.json'
            frozen.write_text(json.dumps({'state_hash':'fixture'}))
            spec=root/'spec.json'
            spec.write_text(json.dumps({'evaluators':{'humaneval_plus':{
                'repository':'fixture','commit':'a'*40,'entrypoint':'fixture','entrypoint_sha256':'fixture',
                'python_executable':sys.executable,'data_path':str(source),'data_sha256':'fixture',
                'task_ids_path':'fixture','task_ids_sha256':'fixture','protocol':'fixture'}}}))
            config=SimpleNamespace(task_replicas=[{'base_url':str(i)} for i in range(4)],model_dump=lambda:{})
            def fake_run(command,**kwargs):
                job=json.loads(Path(command[-1]).read_text())
                endpoint=job['config']['task_base_url']
                shard=json.loads(Path(job['spec']).read_text())['evaluators']['humaneval_plus']
                task_id=json.loads(Path(shard['task_ids_path']).read_text())[0]
                with guard:
                    first=endpoint not in seen
                    seen.add(endpoint)
                    calls.append((endpoint,task_id))
                if first:barrier.wait(timeout=5)
                time.sleep(.03 if endpoint=='0' else .001)
                row={'task_id':task_id,'state_hash':'fixture','final_answer':'fixture'}
                out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
                (out/'humaneval_plus.jsonl').write_text(json.dumps({**row,'record_hash':value_hash(row)})+'\n')
                return SimpleNamespace(returncode=0)
            with patch('sia.task_meta.reporting.OfficialEvaluatorSpec.validate',return_value=ids), \
                 patch('dynamic_predictions.subprocess.run',side_effect=fake_run):
                out=root/'out'
                receipt=generate_dynamic(config,frozen,spec,out)
                self.assertEqual(receipt['count'],12)
                self.assertEqual(len(calls),12)
                self.assertEqual({gpu for gpu,_ in calls},set('0123'))
                self.assertLess(sum(gpu=='0' for gpu,_ in calls),3)
                first=(out/'humaneval_plus.jsonl').read_bytes()
                self.assertEqual([json.loads(line)['task_id'] for line in first.splitlines()],ids)
                generate_dynamic(config,frozen,spec,out)
                self.assertEqual(len(calls),12)
                self.assertEqual(first,(out/'humaneval_plus.jsonl').read_bytes())


if __name__=='__main__':unittest.main()
