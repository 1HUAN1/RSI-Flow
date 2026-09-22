"""Four-GPU dynamic queue for official answer generation."""
import json
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, immutable, read, sha, write
from sia.task_meta.durable import value_hash
from sia.task_meta.report_predictions import public_final_task
from sia.task_meta.reporting import OfficialEvaluatorSpec


def generate_dynamic(config, frozen_path, specs_path, output_dir):
    endpoints=[r['base_url'] if isinstance(r,dict) else r.base_url for r in config.task_replicas]
    if len(endpoints)!=4 or len(set(endpoints))!=4:
        raise ValueError('Dynamic evaluation requires four distinct GPU replicas')
    configured=read(specs_path)['evaluators']
    if len(configured)!=1:raise ValueError('One benchmark per dynamic prediction stage')
    identifier,spec=next(iter(configured.items()))
    expected=OfficialEvaluatorSpec(**spec,benchmark=identifier).validate()
    source=Path(spec['data_path'])
    rows=read(source) if source.suffix=='.json' else [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    members={}
    for row in rows:
        task_id=public_final_task(identifier,row).task_id
        if task_id in members:raise ValueError('Duplicate evaluation task')
        members[task_id]=row
    if set(members)!=set(expected):raise ValueError('Evaluation denominator changed')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'parallel_protocol.json').exists():raise ValueError('Static shard cache cannot be changed to dynamic')
    immutable(output/'dynamic_protocol.json',{'frozen_sha256':sha(frozen_path),'spec_sha256':sha(specs_path),
        'endpoints':endpoints,'task_ids':expected,'scheduler':'one_task_per_claim','feedback_to_meta':False})
    claims_path=output/'dynamic_claims.json'
    claims=read(claims_path) if claims_path.exists() else {}
    if set(claims)-set(expected) or any(type(i) is not int or i not in range(4) for i in claims.values()):
        raise ValueError('Dynamic claim receipt changed')
    state_hash=read(frozen_path)['state_hash']

    def task_dir(task_id,gpu):
        return output/f'gpu_{gpu}'/'tasks'/value_hash(task_id)

    def completed(task_id,gpu):
        out=task_dir(task_id,gpu)/'predictions'
        path=out/(identifier+'.jsonl')
        cache=out/identifier/(value_hash(task_id)+'.json')
        if path.exists():
            lines=path.read_text().splitlines()
            if len(lines)!=1:raise ValueError('One task must produce exactly one submission')
            row=json.loads(lines[0])
        elif cache.exists():
            row=read(cache)
        else:
            calls=out/identifier/(value_hash(task_id)+'.calls')
            if calls.exists() and any(calls.iterdir()):
                raise RuntimeError('Incomplete model-call receipts require audit: '+task_id)
            return None
        record_hash=row.pop('record_hash',None)
        if record_hash!=value_hash(row) or row['task_id']!=task_id or row['state_hash']!=state_hash:
            raise ValueError('Prediction receipt identity mismatch')
        return {**row,'record_hash':record_hash}

    records={}
    for task_id,gpu in list(claims.items()):
        record=completed(task_id,gpu)
        if record is None:del claims[task_id]
        else:records[task_id]=record
    write(claims_path,claims)
    pending=queue.Queue()
    for task_id in expected:
        if task_id not in records:pending.put(task_id)
    claim_lock=threading.Lock()
    records_lock=threading.Lock()
    failed=threading.Event()

    def worker(gpu):
        while not failed.is_set():
            try:task_id=pending.get_nowait()
            except queue.Empty:return
            try:
                root=task_dir(task_id,gpu);root.mkdir(parents=True,exist_ok=True)
                data=root/('task'+source.suffix)
                row=members[task_id]
                content=(json.dumps([row],ensure_ascii=False) if source.suffix=='.json'
                    else json.dumps(row,ensure_ascii=False)+'\n')
                if data.exists():
                    if data.read_text()!=content:raise ValueError('Frozen evaluation task changed')
                else:data.write_text(content)
                ids=root/'ids.json';immutable(ids,[task_id])
                shard=root/'spec.json'
                immutable(shard,{'evaluators':{identifier:{**spec,'data_path':str(data),'data_sha256':sha(data),
                    'task_ids_path':str(ids),'task_ids_sha256':sha(ids)}}})
                job=root/'job.json'
                immutable(job,{'config':{**config.model_dump(),'task_base_url':endpoints[gpu]},
                    'frozen':str(frozen_path),'spec':str(shard),'output':str(root/'predictions')})
                with claim_lock:
                    claims[task_id]=gpu
                    write(claims_path,claims)
                with (output/f'gpu_{gpu}'/'worker.log').open('ab',buffering=0) as log:
                    code=subprocess.run([sys.executable,'-B',str(ROOT/'parallel_predictions.py'),str(job)],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log).returncode
                if code:raise RuntimeError(f'GPU {gpu} failed for {task_id}; inspect worker.log')
                record=completed(task_id,gpu)
                if record is None:raise ValueError('Worker exited without a completed submission')
                with records_lock:records[task_id]=record
            except Exception:
                failed.set()
                raise
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu) for gpu in range(4)]
        errors=[]
        for future in futures:
            try:future.result()
            except Exception as exc:errors.append(exc)
    if errors:raise RuntimeError('Dynamic prediction failed; preserve receipts and audit before resume') from errors[0]
    if set(records)!=set(expected):raise ValueError('Incomplete dynamic predictions')
    target=output/(identifier+'.jsonl')
    with target.with_suffix('.jsonl.tmp').open('w') as stream:
        for task_id in expected:stream.write(json.dumps(records[task_id])+'\n')
    target.with_suffix('.jsonl.tmp').replace(target)
    return {'status':'submissions_completed_unscored','count':len(expected),'workers':4,
            'scheduler':'dynamic_one_task_per_claim'}
