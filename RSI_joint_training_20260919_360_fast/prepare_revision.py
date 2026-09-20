"""Offline 360-task release: seeded subsets of each frozen B, unchanged evaluation IDs.

Never signal jobs, start services, call models, or overwrite an existing release.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from common import ROOT
from evolution_protocol import read,freeze,manifest,make_manifest,file_hash,assert_disjoint
from data_protocol import balanced_sample,native_index



def repack(old, selected, directory):
    """Copy only selected records; preserve task/group identities and original content."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    tasks=[];checked=set()
    for source in sorted({r['source'] for r in selected}):
        target=directory/(source+'.jsonl');packed=[]
        with target.open('xb') as stream:
            for row in selected:
                if row['source']!=source:continue
                src=Path(row['record_file'])
                if src not in checked:
                    if file_hash(src)!=row['record_sha256']:raise ValueError('Frozen source pack changed')
                    checked.add(src)
                with src.open('rb') as f:
                    f.seek(row['record_offset']);line=f.read(row['record_length'])
                if len(line)!=row['record_length']:raise ValueError('Incomplete frozen task record')
                offset=stream.tell();stream.write(line)
                packed.append({**row,'record_file':str(target),'record_offset':offset})
        tasks.extend({**r,'record_sha256':file_hash(target)} for r in packed)
    return make_manifest(old['role'],old['round_id'],tasks,split_seed=old['split_seed'],
        feedback_scope='all_B_r',historical_manifest_hash=old['manifest_hash'])


def prepare(previous):
    previous=Path(previous).resolve();config=read(ROOT/'configs/train.json')
    from install_runtime import install
    runtime=install(config);output=runtime/'data'/config['data_release'];vault=Path(config['validation_vault'])
    if output.exists() or vault.exists():raise FileExistsError('Existing release must never be replaced')
    rounds=[];originals={};selected={};old_config=read(previous/'configs/train.json')
    old_vault=Path(old_config['validation_vault'])
    old_validation=manifest(old_vault/'validation/manifest.json','independent_validation',None)
    # Preserve the exact reporting cohort and every recorded exposure label.
    for source in config['validation_limits']:
        selected[source]=[t for t in old_validation['tasks'] if t['source']==source]
        if len(selected[source])!=config['validation_limits'][source]:
            raise ValueError('Previous fixed independent cohort quota mismatch')
    old_data=previous/'runtime/data'/old_config.get('data_release','rounds')
    for r in (1,2,3):
        old=manifest(old_data/f'B{r}/manifest.json','train_evolution',r)
        tasks=[]
        for source,count in config['train_quotas_per_round'].items():
            cap=max(1,int(count*config['grouping']['max_environment_fraction_per_round'])) if source=='envscaler' else None
            tasks+=balanced_sample([t for t in old['tasks'] if t['source']==source],count,config['split_seed']+r,cap=cap)
        current=repack(old,tasks,output/f'B{r}')
        freeze(output/f'B{r}/manifest.json',current);rounds.append(current);originals[f'B{r}']=old['manifest_hash']
    validation=[];directory=vault/'validation';directory.mkdir(parents=True)
    for source,tasks in selected.items():
        dest=directory/(source+'.jsonl');packed=[];checked=set()
        with dest.open('xb') as stream:
            for t in tasks:
                src=Path(t['record_file'])
                if src not in checked:
                    if file_hash(src)!=t['record_sha256']:raise ValueError('Prior validation pack changed')
                    checked.add(src)
                with src.open('rb') as f:f.seek(t['record_offset']);line=f.read(t['record_length'])
                offset=stream.tell();stream.write(line)
                packed.append({**t,'record_file':str(dest),'record_offset':offset,'record_length':len(line)})
        validation.extend({**t,'record_sha256':file_hash(dest)} for t in packed)
    current=make_manifest('independent_validation',None,validation,fixed_across_rounds=True,
        split_seed=config['split_seed'],clean_independence_audited=False,
        historical_manifest_hash=old_validation['manifest_hash'])
    assert_disjoint([*rounds,current]);freeze(directory/'manifest.json',current)
    # Preserve historical exposure evidence; unselected or regrouped tasks are never promoted.
    for name in ('full','quarantine.json','budget_limited_manifest.json','validation_history.json'):
        src=old_vault/name;dest=vault/name
        if src.is_dir():shutil.copytree(src,dest)
        elif src.is_file():shutil.copy2(src,dest)
    freeze(vault/'revision_exposure_history.json',dict(previous_manifest=str(old_vault/'validation/manifest.json'),
        previous_manifest_sha256=file_hash(old_vault/'validation/manifest.json'),
        evaluation_ids_unchanged=True,no_independence_reset=True,prior_history_vault=str(old_vault)))
    freeze(output/'protocol.json',config);native_index(output,rounds,config)
    freeze(output/'preparation_audit.json',dict(status='prepared_not_executed',model_calls=0,
        original_training_manifests=originals,selected_training_ids_groups_content_preserved=True,
        historical_source_audit=str(old_data/'preparation_audit.json'),
        historical_source_audit_sha256=file_hash(old_data/'preparation_audit.json'),
        allocated_tasks=sum(len(m['tasks']) for m in rounds),manifest_hashes={**{f'B{i+1}':m['manifest_hash'] for i,m in enumerate(rounds)},
            'independent_validation':current['manifest_hash']}))
    from launch import dry_run,make_pipeline
    sys.path.insert(0,str(runtime))
    from sia.task_meta.pipeline import PipelineConfig
    resolved=PipelineConfig.model_validate(make_pipeline(config,runtime)).checked()
    freeze(ROOT/'preflight/resolved_pipeline.json',resolved.model_dump())
    report=dry_run(config,runtime);freeze(ROOT/'preflight/dry_run.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previous',required=True)
    print(json.dumps(prepare(parser.parse_args().previous),ensure_ascii=False,indent=2))
