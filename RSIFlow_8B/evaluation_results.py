"""Reporting-only adapters over pinned official scores; never imported by training."""
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from common import read, write, immutable


def official_task_scores(identifier, spec, predictions, directory, frozen):
    directory=Path(directory)
    if identifier in {'humaneval_plus','mbpp_plus'}:
        raw=read(directory/'official_results.json')['eval']
        return {str(k):dict(success=(v[0]['base_status']=='pass' and v[0]['plus_status']=='pass'),
                           native_scores={'base_status':v[0]['base_status'],'plus_status':v[0]['plus_status']})
                for k,v in raw.items() if len(v)==1}
    if identifier=='livecodebench':
        return {str(r['question_id']):dict(success=r['passed'],native_scores={'passed_tests':r['outcomes']})
                for r in read(directory/'official_samples_codegeneration_output_eval.json')[1]}
    # Replay the *scorer*, never the Task: official QA scripts expose aggregate-only output.
    # A singleton keeps their exact normalization/aliases/support scoring, including native scale.
    from sia.task_meta.reporting import export_predictions, official_command, parse_official_metrics
    data={str(r['_id']):r for r in read(spec.data_path)}
    result={}
    for row in predictions:
        identifier=str(row['task_id']);out=directory/'per_task'/__import__('hashlib').sha256(identifier.encode()).hexdigest()
        out.mkdir(parents=True,exist_ok=True)
        if (out/'score.json').exists():result[identifier]=read(out/'score.json');continue
        gold=out/'scorer_input.json';immutable(gold,[data[identifier]])
        pred=export_predictions(spec.benchmark,[row],[identifier],out,frozen['state_hash'])
        single=replace(spec,data_path=str(gold))
        completed=subprocess.run(official_command(single,pred,out),cwd=out,
            env={'PATH':str(Path(spec.python_executable).parent),'LANG':'C.UTF-8','PYTHONNOUSERSITE':'1'},
            capture_output=True,text=True,timeout=min(spec.timeout_seconds,120),check=True)
        metrics,_=parse_official_metrics(single,out,completed.stdout)
        result[identifier]=dict(success=metrics['EM']==1.0,native_scores=metrics,
            success_rule='official_normalized_answer_exact_match')
        immutable(out/'score.json',result[identifier])
    return result


def task_rows(manifest, snapshot, frozen, results, output):
    """One row for every expected task, even before generation/scoring finishes."""
    output=Path(output);by_result={r['benchmark_id']:r for r in results};rows=[]
    for task in manifest['tasks']:
        benchmark=task['source'];result=by_result.get(benchmark,{})
        native={str(r['native_id']):r for r in result.get('task_results',[])}
        parts=task.get('native_ids',[task['native_id']]);members=[native.get(str(i)) for i in parts]
        known=[m for m in members if m is not None]
        scored=len(known)==len(parts) and all(m.get('scoring_status')=='completed' and type(m.get('success')) is bool for m in known)
        success=all(m['success'] for m in known) if scored else None
        generated=len(known)==len(parts) and all(m.get('execution_status')=='completed' for m in known)
        failures=[m.get('failure_type') for m in known if m.get('failure_type')]
        def total(key):
            values=[m.get(key) for m in members if m]
            return sum(values) if len(values)==len(parts) and all(type(v) in (int,float) for v in values) else None
        rows.append(dict(run_id=output.parent.name,round_id=snapshot['round'],role=frozen.get('source_role'),phase='independent_eval',
            task_id=task['task_id'],native_ids=parts,benchmark=benchmark,domain=task['domain'],
            system_fingerprint=frozen['state_hash'],data_version=task['version'],manifest_hash=manifest['manifest_hash'],
            evaluator=result.get('evaluator'),execution_config_sha256=frozen.get('evaluation_config_sha256'),
            execution_status='completed' if generated else 'infra_error' if any(m.get('execution_status')=='infra_error' for m in known) else 'pending',
            scoring_status='completed' if scored else 'pending',success=int(success) if success is not None else None,
            native_scores=known[0].get('native_scores',{}) if len(parts)==1 and known else
                {'complete_task_all_turns_pass':success,'turns':{i:m.get('native_scores',{}) for i,m in zip(parts,members) if m}},
            failure_type=failures or ('task_failure' if success is False else None),
            tokens=[m.get('tokens') for m in known] or None,tool_calls=total('tool_calls'),latency=total('latency'),
            cost_coverage='native_fields_only; missing values are unknown, never zero',
            trajectory_path=sorted({m['trajectory_path'] for m in known if m.get('trajectory_path')}) or None))
    return rows


def counts(rows):
    scored=[r for r in rows if r['scoring_status']=='completed' and r['success'] in (0,1)]
    complete=len(scored)==len(rows) and bool(rows)
    n_success=sum(r['success'] for r in scored)
    return dict(n_expected=len(rows),n_generated=sum(r['execution_status']=='completed' for r in rows),
        n_scored=len(scored),n_success=n_success,n_task_failure=len(scored)-n_success,
        n_timeout=sum('timeout' in str(r.get('failure_type')).lower() for r in rows),
        n_infra_error=sum(r['execution_status']=='infra_error' for r in rows),n_pending=len(rows)-len(scored),
        complete=complete,success_rate=n_success/len(rows) if complete else None,
        provisional_success_fraction=f'{n_success}/{len(rows)}',denominator_is_expected=True)


def publish(manifest,snapshot,frozen,results,output):
    output=Path(output);rows=task_rows(manifest,snapshot,frozen,results,output)
    if len({r['task_id'] for r in rows})!=len(rows):raise ValueError('Duplicate reporting task')
    (output/'task_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    metrics=dict(metrics_kind='held_out_test_metrics' if frozen['source_role']=='final_test' else 'independent_validation_metrics',
        round_id=snapshot['round'],source_role=frozen['source_role'],purpose='report_only',
        official_full_benchmark=False,clean_independence_audited=manifest.get('clean_independence_audited',False),
        benchmarks={s:counts([r for r in rows if r['benchmark']==s]) for s in sorted({r['benchmark'] for r in rows})},
        domains={s:counts([r for r in rows if r['domain']==s]) for s in sorted({r['domain'] for r in rows})},overall=counts(rows),
        native_metrics={r['benchmark_id']:r.get('metrics') for r in results},
        benchmark_status={r['benchmark_id']:r.get('status') for r in results},feedback_to_meta=False)
    write(output/'task_metrics.json',metrics)
    table=[dict(round=snapshot['round'],group=k,**v) for k,v in {**metrics['benchmarks'],**metrics['domains'],'Overall':metrics['overall']}.items()]
    with (output/'task_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report=['Fixed subset; reporting only. Historical independence audited: '+str(metrics['clean_independence_audited']),
            '| Source/domain | Success / expected | Scored | Complete |','|---|---:|---:|---|']
    report += [f"| {r['group']} | {r['n_success']}/{r['n_expected']} | {r['n_scored']} | {r['complete']} |" for r in table]
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    all_rows=[]
    for path in sorted(output.parent.glob('round_*/task_metrics.csv')):
        with path.open(encoding='utf-8') as f:all_rows.extend(csv.DictReader(f))
    with (output.parent/'all_round_task_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(all_rows)
    return metrics
