"""Partition reporting tasks across verified replicas; reuse the native generator."""
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT, read, immutable, sha


def partition_rows(identifier, rows, expected, workers):
    from sia.task_meta.report_predictions import public_final_task
    members={}
    for row in rows:
        key=public_final_task(identifier,row).task_id
        if key in members:raise ValueError('Duplicate evaluation task')
        members[key]=row
    if set(members)!=set(expected):raise ValueError('Evaluation denominator changed')
    return [[members[key] for key in expected[i::workers]] for i in range(workers)]


def generate_parallel(config, frozen_path, specs_path, output_dir):
    from sia.task_meta.reporting import OfficialEvaluatorSpec
    from sia.task_meta.report_predictions import generate_report_predictions
    from sia.task_meta.durable import value_hash
    replicas=config.task_replicas
    if len(replicas)<2:
        return generate_report_predictions(config,frozen_path,specs_path,output_dir,enabled=True)
    endpoints=[r['base_url'] if isinstance(r,dict) else r.base_url for r in replicas]
    if len(set(endpoints))!=len(endpoints):raise ValueError('Duplicate GPU replica endpoint')
    configured=read(specs_path)['evaluators']
    if len(configured)!=1:raise ValueError('One benchmark per parallel prediction stage')
    identifier,spec=next(iter(configured.items()))
    expected=OfficialEvaluatorSpec(**spec,benchmark=identifier).validate()
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    # Existing serial partial calls need reconciliation, never silent duplication.
    if (output/identifier).exists():raise ValueError('Serial prediction cache requires explicit migration')
    source=Path(spec['data_path'])
    rows=read(source) if source.suffix=='.json' else [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    partitions=partition_rows(identifier,rows,expected,len(endpoints))
    immutable(output/'parallel_protocol.json',{'frozen_sha256':sha(frozen_path),'spec_sha256':sha(specs_path),
        'endpoints':endpoints,'task_ids':expected,'shards':[expected[i::len(endpoints)] for i in range(len(endpoints))],
        'feedback_to_meta':False})
    jobs=[]
    for i,(endpoint,subset) in enumerate(zip(endpoints,partitions)):
        if not subset:continue
        directory=output/f'gpu_{i}';directory.mkdir(exist_ok=True)
        data=directory/('tasks'+source.suffix)
        content=json.dumps(subset,ensure_ascii=False) if source.suffix=='.json' else ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in subset)
        if data.exists():
            if data.read_text()!=content:raise ValueError('Frozen evaluation shard changed')
        else:data.write_text(content)
        ids=directory/'ids.json';immutable(ids,expected[i::len(endpoints)])
        shard_spec=directory/'spec.json'
        immutable(shard_spec,{'evaluators':{identifier:{**spec,'data_path':str(data),'data_sha256':sha(data),
            'task_ids_path':str(ids),'task_ids_sha256':sha(ids)}}})
        job=directory/'job.json'
        immutable(job,{'config':{**config.model_dump(),'task_base_url':endpoint},'frozen':str(frozen_path),
            'spec':str(shard_spec),'output':str(directory/'predictions')})
        jobs.append((directory,job))
    processes=[]
    try:
        for directory,job in jobs:
            with (directory/'worker.log').open('ab',buffering=0) as log:
                processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),str(job)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError('Parallel prediction worker failed; preserve receipts and audit before resume')
    finally:
        for p in processes:
            if p.poll() is None:p.wait()
    records={}
    for directory,_ in jobs:
        path=directory/'predictions'/(identifier+'.jsonl')
        for line in path.read_text().splitlines():
            row=json.loads(line);key=row['task_id'];record_hash=row.pop('record_hash')
            if key in records or record_hash!=value_hash(row):raise ValueError('Duplicate or changed prediction')
            if row['state_hash']!=read(frozen_path)['state_hash']:raise ValueError('Wrong frozen prediction state')
            records[key]={**row,'record_hash':record_hash}
    if set(records)!=set(expected):raise ValueError('Incomplete parallel predictions')
    target=output/(identifier+'.jsonl');temporary=target.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for key in expected:stream.write(json.dumps(records[key])+'\n')
    temporary.replace(target)
    return {'status':'submissions_completed_unscored','count':len(expected),'workers':len(jobs)}


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'runtime'))
    from sia.task_meta.pipeline import PipelineConfig
    from sia.task_meta.report_predictions import generate_report_predictions
    job=read(sys.argv[1]);config=PipelineConfig.model_validate(job['config']).checked()
    generate_report_predictions(config,job['frozen'],job['spec'],job['output'],enabled=True)
