"""Frozen per-round evaluation on seven external benchmarks; no Meta feedback."""
import argparse
import csv
import json
import sys
from pathlib import Path
from common import ROOT, read, write, sha, immutable

def validate(snapshot_path,pipeline_path,validation_path, *, role=None, manifest_path=None, full_benchmark=False):
    sys.path.insert(0,str(ROOT/'runtime'))
    from sia.task_meta.pipeline import load_config
    from sia.task_meta.durable import load_task,task_hash
    from sia.task_meta.harnessforge_manifest import load_manifest
    from sia.task_meta.storage import checkpoint_manifest
    from sia.task_meta.gpu_phases import ensure_services, verify_services
    from parallel_predictions import generate_parallel
    from sia.task_meta.reporting import OfficialEvaluatorSpec,evaluate_official,BENCHMARK_IDS
    from tool_validation import evaluate_tool
    snapshot=read(snapshot_path);config=load_config(pipeline_path);settings=read(validation_path)
    from validation_sources import verify
    validation_sources=Path(getattr(config,'output_root',ROOT))/'validation_sources.json'
    verify(settings,validation_sources)
    out=Path(snapshot_path).parent
    selected_manifest=None;selected_specs=None
    if getattr(config,'round_protocol',None):
        if not manifest_path or role not in {'independent_validation','final_test'}:
            raise ValueError('Explicit reporting role and frozen manifest required')
        if out.resolve().is_relative_to(Path(pipeline_path).resolve().parent):
            raise ValueError('External results must be outside training run')
        from evaluation_manifest import select_specs
        selected_manifest,selected_specs=select_specs(settings,manifest_path,role,out/'inputs',full_benchmark=full_benchmark)
        if role=='final_test':
            final=read(Path(pipeline_path).parent/'final_state.json')
            if final.get('rounds_completed')!=3 or final.get('status')!='completed':
                raise ValueError('final_test requires completed, frozen three-round evolution')
    state=load_task(snapshot['task_state'])
    binding={'task_state':snapshot['task_state'],'state_hash':task_hash(state),
             'checkpoint_files':checkpoint_manifest(state.checkpoint_path),
             'task_harness_identity':load_manifest(state.harness_path).identity,
             'protocol_hash':sha(snapshot['source_round']),'status':'frozen_for_report_eval',
             'selection':{'rule':'committed_round_snapshot_without_external_score_selection','round':snapshot['round']},
             'validation_config_sha256':sha(validation_path),'explicit_full_benchmark':full_benchmark,
             **({'source_role':role,'purpose':'report_only','manifest_hash':selected_manifest['manifest_hash'],
                 'system_snapshot_sha256':sha(snapshot_path),'evaluation_config_sha256':sha(pipeline_path)} if selected_manifest else {})}
    immutable(out/'frozen_task.json',binding)
    if (out/'complete.json').exists():
        done=read(out/'complete.json')
        if done['state_hash']!=binding['state_hash']:raise ValueError('Completed validation state changed')
        for path,digest in done['result_files'].items():
            if sha(out/path)!=digest:raise ValueError('Completed validation result modified')
        return done
    ensure_services(config,state.checkpoint_path)
    immutable(out/'serving_fingerprint.json',verify_services(config,state.checkpoint_path))
    specs=selected_specs or read(settings['official_specs'])['evaluators']
    expected={'livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev'}
    if not expected<=set(specs):raise RuntimeError('All five Code/Search official evaluators are required')
    results=[]
    if selected_manifest:
        from evaluation_results import publish
        # Persist the entire expected queue before starting any benchmark.
        # A benchmark-level exception must not turn unattempted tasks into failures.
        publish(selected_manifest,snapshot,binding,results,out)
    for identifier in settings['benchmark_ids']:
        result_path=out/'scores'/identifier/'result.json'
        if result_path.exists() and read(result_path).get('status')=='completed':
            result=read(result_path)
            if result.get('state_hash')!=binding['state_hash']:raise ValueError('Cached score state mismatch')
        elif identifier in settings['tool_benchmarks']:
            spec=settings['tool_benchmarks'][identifier]
            if selected_manifest:
                tasks=[t for t in selected_manifest['tasks'] if t['source']==identifier]
                spec={**spec,'selected_tasks':tasks,'selected_task_ids':[i for t in tasks for i in t.get('native_ids',[t['native_id']])]}
            result=evaluate_tool(identifier,config,binding,spec,
                                 settings['tool_benchmarks']['acebench'],result_path.parent)
        else:
            single=out/(identifier+'.spec.json')
            immutable(single,{'evaluators':{identifier:specs[identifier]}})
            predictions=out/'predictions'
            generate_parallel(config,out/'frozen_task.json',single,predictions/identifier)
            rows=[json.loads(line) for line in (predictions/identifier/(identifier+'.jsonl')).read_text().splitlines() if line.strip()]
            result=evaluate_official(OfficialEvaluatorSpec(**{**specs[identifier],'benchmark':identifier}),rows,binding,result_path.parent)
            from evaluation_results import official_task_scores
            scores=official_task_scores(identifier,OfficialEvaluatorSpec(**{**specs[identifier],'benchmark':identifier}),rows,result_path.parent,binding) if result.get('status')=='completed' else {}
            result['task_results']=[dict(native_id=str(row['task_id']),
                execution_status='completed',scoring_status='completed' if str(row['task_id']) in scores else 'pending',
                **scores.get(str(row['task_id']),{'success':None,'native_scores':{}}),
                tokens=[r.get('usage') for r in row.get('transport_calls',[])],
                tool_calls=len(row.get('trajectory',{}).get('tool_calls',[])),latency=row.get('wall_seconds'),
                trajectory_path=str(predictions/identifier/(identifier+'.jsonl'))) for row in rows]
            write(result_path,result)
        results.append({'benchmark_id':identifier,**result})
        write(out/'results.json',results)
        update_tables(out,snapshot,results)
        if selected_manifest:
            from evaluation_results import publish
            metrics=publish(selected_manifest,snapshot,binding,results,out)
        if result.get('status')!='completed':raise RuntimeError('Validation incomplete: '+identifier)
    done={'status':'completed','round':snapshot['round'],'state_hash':binding['state_hash'],
          'benchmarks_completed':len(results),'feedback_to_meta':False,
          'result_files':{str(p.relative_to(out)):sha(p) for p in out.glob('scores/*/result.json')}}
    if selected_manifest:
        done.update(source_role=role,purpose='report_only',manifest_hash=selected_manifest['manifest_hash'],
            metrics_kind='held_out_test_metrics' if role=='final_test' else 'independent_validation_metrics',
            official_full_benchmark=False,clean_independence_audited=selected_manifest.get('clean_independence_audited',False))
    if len(results)!=7:raise RuntimeError('All seven benchmarks are mandatory')
    if selected_manifest and not metrics['overall']['complete']:
        raise RuntimeError('Generated predictions are not a complete scored evaluation')
    if selected_manifest:
        done['completion']=metrics['overall']
        done['result_files'].update({name:sha(out/name) for name in ('task_results.jsonl','task_metrics.json','serving_fingerprint.json')})
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--pipeline-config',required=True);p.add_argument('--config',required=True)
    p.add_argument('--role',required=True,choices=['independent_validation','final_test']);p.add_argument('--manifest',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--full-benchmark',action='store_true',help='Explicit reporting-only full manifest; never a round-loop default')
    a=p.parse_args()
    from evaluation_manifest import evaluation_budget
    print(evaluation_budget(a.manifest,a.role),flush=True)
    if a.execute:print(validate(a.snapshot,a.pipeline_config,a.config,role=a.role,manifest_path=a.manifest,full_benchmark=a.full_benchmark))

if __name__=='__main__':main()

