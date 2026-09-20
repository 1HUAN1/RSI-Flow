"""One command: preflight -> isolated runtime -> full training -> per-round validation."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, read, write, immutable, sha

def make_pipeline(config, runtime):
    source=Path(config['runtime_source'])
    p=read(source/config['base_config'])
    p.update(mode='full',experiment_scope='multidomain',training_schedule='full_cohort',
             task_update_policy='sequential_positive_gain',artifact_evaluation='rerollout',
             allowed_task_components=['HARNESS','MODEL'],
             data_dir=str(runtime/'data/joint_full'),max_generations=config['rounds'],seed=config['seed'],
             search_dev_fraction=0,
             envscaler_root=str(Path(config['dataset_root'])/'train/tool_use/envscaler'),
             max_wall_seconds=config['max_wall_seconds'],rollouts_per_task=1,probe_rollouts=1,
             gpu_execution='phased_four',training_timeout_seconds=config['training_timeout_seconds'],
             round_validation_config=str((ROOT/config['validation_config']).resolve()),
             trainer_python=sys.executable,task_base_url=f"http://127.0.0.1:{config['ports'][0]}/v1")
    p['meta']['run_mode']='full'
    p['meta']['remote_worker_socket']=config['meta_worker_socket']
    p['meta']['remote_relay_socket']=config['meta_relay_socket']
    if p['meta']['model']!='gpt-5.6-sol':raise ValueError('Expected the previously authorized gpt-5.6-sol Meta runtime')
    # Runtime pin/catalog paths remain server-local and are never read from the validation split.
    for name in ['codex_source','codex_executable','provenance_file','model_catalog_json','compatibility_report','harness_root']:
        value=p['meta'].get(name)
        if value and not Path(value).is_absolute():p['meta'][name]=str(source/value)
    if p.get('initial_meta_bundle') and not Path(p['initial_meta_bundle']).is_absolute():
        p['initial_meta_bundle']=str(source/p['initial_meta_bundle'])
    p['probe_per_domain']=dict.fromkeys(['tool_use','code','searchqa'],config['training_probe_per_domain'])
    p['window_quotas']={'tool_use':2550,'code':6600,'searchqa':337069}
    p['task_replicas']=[{'gpu':i,'base_url':f'http://127.0.0.1:{port}/v1'} for i,port in enumerate(config['ports'])]
    p['training'].update(num_train_epochs=1,max_steps=-1,max_length=config['max_length'],max_samples=config['max_sft_samples'])
    return p

def check(config, runtime):
    from tool_validation import native_layout
    from sia.task_meta.reporting import OfficialEvaluatorSpec
    from sia.task_meta.sandbox import probe_isolation
    from sia.task_meta.pipeline import PipelineConfig
    errors=[];details={}
    for package in ['torch','transformers','peft','openai','pydantic','datasets']:
        if importlib.util.find_spec(package) is None:errors.append('Missing Python dependency: '+package)
    details['candidate_isolation']=probe_isolation()
    if not details['candidate_isolation']['available']:errors.append('Code evaluator isolation unavailable')
    settings=read(ROOT/config['validation_config'])
    try:
        official=read(settings['official_specs'])['evaluators']
        for name in ['livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev']:
            details[name]={'count':len(OfficialEvaluatorSpec(**{**official[name],'benchmark':name}).validate())}
        for name,spec in settings['tool_benchmarks'].items():
            repo,native,ids,paths=native_layout(name,spec)
            details[name]={'count':sum(map(len,ids.values())),'categories':list(ids),'source':str(repo)}
            from tool_validation import check_native
            check_native(name,spec,ROOT/'preflight'/name)
    except Exception as exc:errors.append('Official evaluation preflight: '+str(exc))
    pipeline=make_pipeline(config,runtime)
    native_config=PipelineConfig.model_validate(pipeline).checked()
    for name in [pipeline['meta']['api_key_env'],pipeline['meta']['remote_worker_token_env'],
                 settings['tool_benchmarks']['acebench']['user_api_key_env'],settings['tool_benchmarks']['acebench']['user_base_url_env']]:
        if not os.environ.get(name):errors.append('Required environment variable is not set: '+name)
    if not errors:
        try:
            from sia.task_meta.meta_backends.remote_execution import validate_worker
            validate_worker(native_config.meta)
            details['meta_worker']={'healthy':True,'isolation_and_version_verified':True}
        except Exception as exc:errors.append('Meta worker preflight: '+str(exc))
    if errors:
        result={'status':'blocked_before_training','errors':errors,'details':details,'model_calls':0}
        write(ROOT/'preflight/status.json',result)
        raise RuntimeError('\n'.join(errors))
    from prepare_data import audit_validation_overlap,prepare
    details['leakage_audit']=audit_validation_overlap(config,settings)
    prepare(config,runtime)
    result={'status':'ready_for_explicit_start','details':details,'model_calls':0,
            'train_count':config['expected_total'],'rounds':config['rounds'],'sft_epochs_per_update':1}
    write(ROOT/'preflight/status.json',result)
    return pipeline,result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/train.json'))
    p.add_argument('--rounds',type=int,choices=[3,5],help='Override before creating a new run')
    p.add_argument('--check',action='store_true',help='CPU/file checks only; no inference, SFT or Meta API')
    a=p.parse_args()
    if sys.platform!='linux':raise SystemExit('Run start.sh on the four-GPU Linux training server; this bundle is portable as a directory.')
    config=read(a.config)
    if a.rounds:config['rounds']=a.rounds
    if config['rounds'] not in (3,5):raise ValueError('Rounds must be 3 or 5')
    if sum(x['count'] for x in config['datasets'])!=config['expected_total']:raise ValueError('Dataset counts do not add up')
    if (config['sft_epochs_per_update'],config['sft_max_steps'],config['dataset_passes_per_round'])!=(1,-1,1):
        raise ValueError('This protocol requires one full training cohort per round and one SFT epoch per model update')
    from install_runtime import install
    runtime=install(config);sys.path.insert(0,str(runtime))
    from sia.task_meta.file_lock import exclusive_lock
    with exclusive_lock(ROOT/'.launch.lock'):
        pipeline,result=check(config,runtime)
        from validation_sources import freeze
        freeze(read(ROOT/config['validation_config']),ROOT/'validation_sources.json')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if a.check:return
        name=config['run_name']+f"_r{config['rounds']}"
        config_path=runtime/'configs'/('joint_'+name+'.json')
        immutable(config_path,pipeline)
        immutable(ROOT/('experiment_'+name+'.json'),{'training':config,'validation':read(ROOT/config['validation_config']),
            'validation_sources_sha256':sha(ROOT/'validation_sources.json'),
            'script_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'validation_never_used_for_selection':True})
        command=[sys.executable,'-u',str(ROOT/'train.py'),'--config',str(config_path),'--run-dir',str(runtime/'runs'/name)]
        logs=ROOT/'logs';logs.mkdir(exist_ok=True)
        write(ROOT/'active_run.json',{'run':str(runtime/'runs'/name),'command':command,'status':'starting'})
        with (logs/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdout=log,stderr=log,cwd=ROOT)
            write(ROOT/'active_run.json',{'run':str(runtime/'runs'/name),'pid':child.pid,'status':'running','log':str(logs/(name+'.log'))})
            code=child.wait()
        if code:
            write(ROOT/'active_run.json',{'run':str(runtime/'runs'/name),'status':'failed','exit_code':code,'log':str(logs/(name+'.log'))})
            raise SystemExit('Training stopped with preserved receipts. Inspect logs; rerun start.sh after resolving the reported error. No budgets are reset.')
        write(ROOT/'active_run.json',{'run':str(runtime/'runs'/name),'status':'completed','rounds':config['rounds'],
              'results':str(ROOT/'validation'/name/'all_rounds.csv')})

if __name__=='__main__':main()
