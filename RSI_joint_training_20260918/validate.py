"""Frozen per-round evaluation on seven external benchmarks; no Meta feedback."""
import argparse
import csv
import json
import sys
from pathlib import Path
from common import ROOT, read, write, sha, immutable

def validate(snapshot_path,pipeline_path,validation_path):
    sys.path.insert(0,str(ROOT/'runtime'))
    from sia.task_meta.pipeline import load_config
    from sia.task_meta.durable import load_task,task_hash
    from sia.task_meta.storage import checkpoint_manifest
    from sia.task_meta.task_harness import harness_identity
    from sia.task_meta.gpu_phases import ensure_services
    from sia.task_meta.report_predictions import generate_report_predictions
    from sia.task_meta.reporting import OfficialEvaluatorSpec,evaluate_official,BENCHMARK_IDS
    from tool_validation import evaluate_tool
    snapshot=read(snapshot_path);config=load_config(pipeline_path);settings=read(validation_path)
    from validation_sources import verify
    verify(settings,ROOT/'validation_sources.json')
    out=Path(snapshot_path).parent
    state=load_task(snapshot['task_state'])
    binding={'task_state':snapshot['task_state'],'state_hash':task_hash(state),
             'checkpoint_files':checkpoint_manifest(state.checkpoint_path),
             'task_harness_identity':harness_identity(state.harness_path),
             'protocol_hash':sha(snapshot['source_round']),'status':'frozen_for_report_eval',
             'selection':{'rule':'committed_round_snapshot_without_external_score_selection','round':snapshot['round']},
             'validation_config_sha256':sha(validation_path)}
    immutable(out/'frozen_task.json',binding)
    if (out/'complete.json').exists():
        done=read(out/'complete.json')
        if done['state_hash']!=binding['state_hash']:raise ValueError('Completed validation state changed')
        for path,digest in done['result_files'].items():
            if sha(out/path)!=digest:raise ValueError('Completed validation result modified')
        return done
    ensure_services(config,state.checkpoint_path)
    specs=read(settings['official_specs'])['evaluators']
    expected={'livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev'}
    if not expected<=set(specs):raise RuntimeError('All five Code/Search official evaluators are required')
    results=[]
    for identifier in settings['benchmark_ids']:
        result_path=out/'scores'/identifier/'result.json'
        if result_path.exists() and read(result_path).get('status')=='completed':
            result=read(result_path)
            if result.get('state_hash')!=binding['state_hash']:raise ValueError('Cached score state mismatch')
        elif identifier in settings['tool_benchmarks']:
            result=evaluate_tool(identifier,config,binding,settings['tool_benchmarks'][identifier],
                                 settings['tool_benchmarks']['acebench'],result_path.parent)
        else:
            single=out/(identifier+'.spec.json')
            immutable(single,{'evaluators':{identifier:specs[identifier]}})
            predictions=out/'predictions'
            generate_report_predictions(config,out/'frozen_task.json',single,predictions/identifier,enabled=True)
            rows=[json.loads(line) for line in (predictions/identifier/(identifier+'.jsonl')).read_text().splitlines() if line.strip()]
            result=evaluate_official(OfficialEvaluatorSpec(**{**specs[identifier],'benchmark':identifier}),rows,binding,result_path.parent)
        results.append({'benchmark_id':identifier,**result})
        write(out/'results.json',results)
        update_tables(out,snapshot,results)
        if result.get('status')!='completed':raise RuntimeError('Validation incomplete: '+identifier)
    done={'status':'completed','round':snapshot['round'],'state_hash':binding['state_hash'],
          'benchmarks_completed':len(results),'feedback_to_meta':False,
          'result_files':{str(p.relative_to(out)):sha(p) for p in out.glob('scores/*/result.json')}}
    if len(results)!=7:raise RuntimeError('All seven benchmarks are mandatory')
    write(out/'complete.json',done)
    return done

def update_tables(out,snapshot,results):
    rows=[]
    for result in results:
        for metric,value in (result.get('metrics') or {'not_completed':None}).items():
            rows.append({'round':snapshot['round'],'benchmark':result['benchmark_id'],'metric':metric,
                         'value':value,'percent':None if value is None else 100*value,
                         'completed':result.get('completed'),'denominator':result.get('expected'),
                         'status':result['status'],'task_state_hash':result.get('state_hash'),
                         'meta_version':snapshot['meta_state']['version'],
                         'task_update':snapshot['chosen_component'] or 'parent_retained'})
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    all_rows=[]
    for path in sorted(out.parent.glob('round_*/results.csv')):
        with path.open(encoding='utf-8-sig') as f:all_rows.extend(csv.DictReader(f))
    with (out.parent/'all_rounds.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(all_rows)
    text=['|轮次|基准|指标|结果 (%)|完成数/分母|状态|','|---|---|---|---:|---:|---|']
    for row in all_rows:
        score=f"{float(row['percent']):.4f}" if row['percent'] else '未完成'
        text.append(f"|{row['round']}|{row['benchmark']}|{row['metric']}|{score}|{row['completed']}/{row['denominator']}|{row['status']}|")
    (out.parent/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--pipeline-config',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();print(validate(a.snapshot,a.pipeline_config,a.config))

if __name__=='__main__':main()
