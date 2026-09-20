"""Native BFCL v3 / ACE scoring, fixed IDs and isolated tool execution."""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, read, write, sha, immutable
from harness_api import HarnessAPI
from tool_sandbox import run_native

def records(path):
    text=Path(path).read_text(encoding='utf-8')
    try:value=json.loads(text)
    except json.JSONDecodeError:return [json.loads(line) for line in text.splitlines() if line.strip()]
    return value if isinstance(value,list) else [value]

def native_layout(identifier,spec):
    repo=Path(spec['repository_root']).resolve()
    if identifier=='bfcl_v3':
        matches=list(repo.glob('**/bfcl_eval/constants/model_config.py'))
        if len(matches)!=1:raise ValueError('BFCL package root is ambiguous or missing')
        repo=matches[0].parents[2]
        paths=sorted((repo/'bfcl_eval/data').glob('BFCL_v3_*.json'))
        if not paths:raise ValueError('Pinned BFCL-v3 data missing; V4 substitution is forbidden')
        categories=[p.stem.removeprefix('BFCL_v3_') for p in paths]
        spec={**spec,'categories':categories}
    else:
        paths=sorted((repo/'data_all'/('data_'+spec['language'])).glob('data_*.json'))
        if not paths:raise ValueError('ACE native language data missing')
    ids={}
    for p in paths:
        rows=records(p)
        if not rows or not all(isinstance(r,dict) and 'id' in r for r in rows):raise ValueError('Unsupported native data file: '+str(p))
        category=p.stem.removeprefix('BFCL_v3_').removeprefix('data_')
        ids[category]=[str(r['id']) for r in rows]
        if len(ids[category])!=len(set(ids[category])):raise ValueError('Duplicate native benchmark ID')
    return repo,spec,ids,paths

def check_native(identifier,spec,output):
    """Import/register the actual official runners in isolation; zero inference."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    repo,spec,ids,files=native_layout(identifier,spec)
    work=output/'native_work'
    if not work.exists():shutil.copytree(repo,work,ignore=shutil.ignore_patterns('.git','__pycache__','.env*','result*','score*'))
    write(output/'spec.json',spec)
    command=[str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'official_tool_entry.py'),identifier,'check','gpt-rsi-preflight',str(work),str(output),str(output/'spec.json')]
    calls=[]
    def fake_only(body):
        if body['model']!='gpt-rsi-preflight' or len(calls)>=2:raise ValueError('Unexpected preflight RPC')
        calls.append(body)
        return {'id':'test_override','object':'chat.completion','created':0,'model':body['model'],
                'choices':[{'index':0,'message':{'role':'assistant','content':'test_override_cpu_only'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    with (output/'check.log').open('wb') as log:
        run_native(command,work,output,log,handler=fake_only,timeout=90)
    if len(calls)!=2:raise RuntimeError('Official SDK bridge did not complete the CPU test')

def metric_rows(identifier, output, ids):
    summaries=[]
    for category,expected in ids.items():
        stem=('BFCL_v3_' if identifier=='bfcl_v3' else 'data_')+category
        preds=list(output.glob('**/'+stem+'_result.json'))
        scores=list(output.glob('**/'+stem+'_score.json'))
        if len(preds)!=1 or len(scores)!=1:raise RuntimeError('Missing/ambiguous native category: '+category)
        actual=[str(r['id']) for r in records(preds[0])]
        if len(actual)!=len(set(actual)) or set(actual)!=set(expected):raise RuntimeError('Incomplete official denominator: '+category)
        first=records(scores[0])[0]
        metric=first.get('accuracy',first.get('end_to_end_accuracy'))
        if type(metric) not in (int,float) or not math.isfinite(metric) or not 0<=metric<=1:
            raise RuntimeError('Invalid official metric: '+category)
        count=first.get('total_count',first.get('total'))
        if count!=len(expected):raise RuntimeError('Official score denominator mismatch: '+category)
        summaries.append({'category':category,'accuracy':metric,'denominator':count,
                          'score_sha256':sha(scores[0]),'predictions_sha256':sha(preds[0])})
    total=sum(x['denominator'] for x in summaries)
    return {'status':'completed','expected':total,'completed':total,'categories':summaries,
        'metrics':{'category_macro_accuracy':sum(x['accuracy'] for x in summaries)/len(summaries),
                   'sample_weighted_accuracy':sum(x['accuracy']*x['denominator'] for x in summaries)/total},
        'aggregation_note':'Derived named aggregates; original official category scores and CSVs are retained. No cross-benchmark overall.'}

def evaluate_tool(identifier, config, frozen, spec, user_spec, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    repo,spec,ids,files=native_layout(identifier,spec)
    selected=spec.get('selected_task_ids')
    if selected is not None:
        if not selected or len(selected)!=len(set(selected)) or set(selected)-{i for v in ids.values() for i in v}:
            raise ValueError('Invalid fixed tool evaluation subset')
        ids={k:[i for i in v if i in set(selected)] for k,v in ids.items()}
        ids={k:v for k,v in ids.items() if v}
        if identifier=='bfcl_v3':spec={**spec,'categories':list(ids)}
    source_hashes={str(p):sha(p) for p in [*repo.rglob('*.py'),*repo.rglob('*.json')] if '__pycache__' not in p.parts and '.git' not in p.parts}
    binding={'state_hash':frozen['state_hash'],'native_data_ids':ids,'source_hashes':source_hashes,
             'spec':spec,'bridge':'official_outer_saved_harness_inner_v1',
             'evaluation_scope':{k:frozen.get(k) for k in ('manifest_hash','source_role','system_snapshot_sha256','evaluation_config_sha256')}}
    immutable(output/'protocol.json',binding)
    if (output/'result.json').exists():
        previous=read(output/'result.json')
        if previous['status']=='completed':return previous
    # ACE uses relative paths and writes dialogue files; copy public code/data to its own sandbox scratch.
    work=output/'native_work'
    if not work.exists():shutil.copytree(repo,work,ignore=shutil.ignore_patterns('.git','__pycache__','.env*','result*','score*'))
    if selected is not None:
        for path in files:
            target=work/path.relative_to(repo)
            rows=[r for r in records(path) if str(r['id']) in set(selected)]
            # Preserve complete native records/dialogues. Source data is read-only.
            text=path.read_text(encoding='utf-8').lstrip()
            content=json.dumps(rows,ensure_ascii=False) if text.startswith('[') else ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
            target.write_text(content,encoding='utf-8')
    api=HarnessAPI(config,frozen,output/'transport',user_spec)
    alias='gpt-rsi-'+frozen['state_hash'][:16]
    write(output/'spec.json',spec)
    try:
        for phase in ['generate','evaluate']:
            marker=output/(phase+'.completed.json')
            if marker.exists():continue
            command=[str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'official_tool_entry.py'),identifier,phase,alias,str(work),str(output),str(output/'spec.json')]
            with (output/(phase+'.log')).open('ab') as log:
                run_native(command,work,output,log,handler=api.complete)
            pending=[p for p in (output/'transport').glob('*.json') if '.call_' not in p.name and read(p).get('status')!='completed']
            if pending:raise RuntimeError('Benchmark inference has unreconciled calls; scores cannot be accepted')
            write(marker,{'status':'completed','state_hash':frozen['state_hash']})
        result={**metric_rows(identifier,output,ids),'benchmark':identifier,'state_hash':frozen['state_hash'],
                'feedback_to_meta':False,'harness_graph_executed':True}
        write(output/'result.json',result);return result
    finally:
        if api.user_client is not None:api.user_client.close()
