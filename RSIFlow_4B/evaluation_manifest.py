"""Explicit reporting-only manifests; never imported by Meta or the SFT trainer."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from evolution_protocol import manifest, file_hash, freeze, VALIDATION_QUOTAS


def select_specs(settings, path, role, out, *, full_benchmark=False):
    value=manifest(path,role,None)
    if role not in {'independent_validation','final_test'}: raise ValueError('External evaluation role required')
    if role=='independent_validation' and not full_benchmark and Counter(t['source'] for t in value['tasks'])!=VALIDATION_QUOTAS:
        raise ValueError('Independent evaluation must use exactly the registered 300-task manifest')
    for task in value['tasks']:
        if task['source']=='acebench' and task['native_id'].startswith('normal_multi_turn_') and not task.get('native_ids'):
            raise ValueError('Legacy ACE turn-level manifest is not a complete-task evaluation')
    if role=='final_test' and (not value.get('clean_independence_audited') or any(t.get('exposure')!='unexposed' for t in value['tasks'])):
        raise ValueError('Clean final_test unavailable: historical exposure is not verified')
    groups=defaultdict(list)
    for task in value['tasks']:groups[task['source']].append(task)
    if set(groups)!=set(settings['benchmark_ids']):raise ValueError('Missing benchmark tasks; cannot claim complete evaluation')
    original=json.loads(Path(settings['official_specs']).read_text())['evaluators'];selected={}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,tasks in groups.items():
        if name in settings['tool_benchmarks']:continue
        raw=[];seen_files=set()
        for t in tasks:
            p=Path(t['record_file'])
            if p not in seen_files:
                if p.is_symlink() or file_hash(p)!=t['record_sha256']:raise ValueError('External pack changed')
                seen_files.add(p)
            with p.open('rb') as f:f.seek(t['record_offset']);raw.append(json.loads(f.read(t['record_length'])))
        data=out/(name+'.json' if name.endswith('_dev') else name+'.jsonl')
        content=json.dumps(raw,ensure_ascii=False) if name.endswith('_dev') else ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw)
        if data.exists():
            if data.read_text(encoding='utf-8')!=content:raise ValueError('Frozen evaluation input changed')
        else:data.write_text(content,encoding='utf-8')
        ids=out/(name+'.ids.json');freeze(ids,[t['native_id'] for t in tasks])
        selected[name]={**original[name],'data_path':str(data),'data_sha256':file_hash(data),
            'task_ids_path':str(ids),'task_ids_sha256':file_hash(ids),'subset':role+'_frozen_subset',
            'metadata':{**original[name].get('metadata',{}),'manifest_hash':value['manifest_hash'],
                'official_full_benchmark':False,'role':role}}
    return value,selected


def evaluation_budget(path,role):
    value=manifest(path,role,None)
    return dict(role=role,manifest_hash=value['manifest_hash'],tasks=len(value['tasks']),
        by_source=dict(Counter(t['source'] for t in value['tasks'])),checkpoints=['A1','A2','A3'] if role=='independent_validation' else ['A0','A3'],
        model_calls='bounded by task budgets; not yet executed',independence_audited=value.get('clean_independence_audited',False))
