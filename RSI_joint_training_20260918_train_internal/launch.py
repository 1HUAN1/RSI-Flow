"""Explicit preparation/start of the existing RSI runtime with frozen round manifests."""
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
    p.update(mode='full',experiment_scope='multidomain',training_schedule='round_disjoint',
             task_update_policy='sequential_positive_gain',artifact_evaluation='rerollout',
             allowed_task_components=['HARNESS','MODEL'],
             data_dir=str(runtime/'data/rounds'),max_generations=3,seed=config['rollout_seed'],
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
    p['probe_per_domain']=dict.fromkeys(['tool_use','code','searchqa'],80)
    p['window_quotas']=dict.fromkeys(['tool_use','code','searchqa'],400)
    p['task_replicas']=[{'gpu':i,'base_url':f'http://127.0.0.1:{port}/v1'} for i,port in enumerate(config['ports'])]
    p['training'].update(num_train_epochs=1,max_steps=-1,max_length=config['max_length'],max_samples=config['max_sft_samples'])
    from evolution_protocol import SFT_PRESET
    p['training'].update(SFT_PRESET,seed=config['training_seed'])
    p['round_protocol']={'data_dir':p['data_dir'],'minimum_sft_samples':config['minimum_sft_samples'],
        'validation_manifest':str(Path(config['validation_vault'])/'validation/manifest.json'),
        'split_seed':config['split_seed'],'rollout_seed':config['rollout_seed'],'training_seed':config['training_seed']}
    p['training']['round_protocol']={k:v for k,v in p['round_protocol'].items() if k!='validation_manifest'}
    return p


def dry_run(config, runtime):
    from evolution_protocol import manifest, assert_disjoint
    from collections import Counter
    rounds=[manifest(Path(runtime)/f'data/rounds/B{r}/manifest.json','train_evolution',r) for r in (1,2,3)]
    assert_disjoint(rounds)
    details=[]
    for value in rounds:
        probe=manifest(Path(runtime)/f"data/rounds/B{value['round_id']}/meta_probe.json",'train_evolution',value['round_id'])
        expected={t['task_id']:t for t in value['tasks']}
        if (Counter(t['source'] for t in value['tasks'])!=config['train_quotas_per_round'] or
            Counter(t['source'] for t in probe['tasks'])!=config['meta_probe_quotas'] or
            any(expected.get(t['task_id'])!=t or not t['meta_probe'] for t in probe['tasks'])):
            raise ValueError('Round allocation or M_r subset mismatch')
        details.append({'round':value['round_id'],'allocated_tasks':len(expected),'meta_probe_tasks':len(probe['tasks']),
                        'by_source':dict(Counter(t['source'] for t in value['tasks'])),'manifest_hash':value['manifest_hash']})
    return dict(status='dry_run_no_model_calls',rounds=details,allocated_tasks=3600,
        distinct_training_diagnostic_tasks=720,paired_diagnostic_logical_evaluations=1440,
        candidate_limit_per_round=2,selected_lineages=1,training_executions='conditional_on_MODEL_and_cache; at most one B_r collection per round',
        candidate_evaluations='additional, charged separately',sft_calls='0..3, conditional on Meta MODEL and verified-success minimum',
        sft_epochs_per_call=1,external_validation_checkpoints=['A0','A1','A2','A3'],final_test='separate explicit entry')

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
    details['round_data']=dry_run(config,runtime)
    result={'status':'ready_for_explicit_start','details':details,'model_calls':0,
            'allocated_tasks':3600,'rounds':3,'sft_epochs_per_update':1}
    write(ROOT/'preflight/status.json',result)
    return pipeline,result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/train.json'))
    p.add_argument('--prepare',action='store_true',help='Prepare a new frozen data release only')
    p.add_argument('--execute',action='store_true',help='Explicitly start or resume real execution')
    p.add_argument('--check','--dry-run',action='store_true',help='CPU/file checks only; no inference, SFT or Meta API')
    a=p.parse_args()
    config=read(a.config)
    if config['rounds']!=3:raise ValueError('Protocol fixes three rounds')
    if (config['sft_epochs_per_update'],config['sft_max_steps'],config['dataset_passes_per_round'])!=(1,-1,1):
        raise ValueError('One current-round dataset epoch per selected SFT update')
    if a.prepare:
        if a.execute:raise ValueError('Prepare and execute are separate explicit steps')
        from data_protocol import prepare_protocol
        from install_runtime import install
        install(config)
        print(json.dumps(prepare_protocol(config,read(ROOT/config['validation_config']),ROOT/'runtime/data/rounds',config['validation_vault']),ensure_ascii=False))
        return
    if not a.execute:
        print(json.dumps(dry_run(config,ROOT/'runtime'),ensure_ascii=False,indent=2));return
    if a.check:raise ValueError('--check cannot be combined with --execute')
    if sys.platform!='linux':raise SystemExit('Execute from the Linux server work directory')
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
        experiment={'training':config,'validation':read(ROOT/config['validation_config']),
            'validation_sources_sha256':sha(ROOT/'validation_sources.json'),
            'script_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'validation_never_used_for_selection':True}
        frozen_experiment=ROOT/('experiment_'+name+'.json')
        if frozen_experiment.exists() and read(frozen_experiment)!=experiment:
            audit=read(ROOT/'preflight/four_gpu_parallel/deployment.json')
            previous=read(frozen_experiment)
            if (sha(frozen_experiment)!=audit['original_experiment_sha256'] or
                experiment['script_hashes']!=audit['deployed_script_hashes'] or
                {k:v for k,v in previous.items() if k!='script_hashes'}!={k:v for k,v in experiment.items() if k!='script_hashes'}):
                raise ValueError('Unapproved execution revision or changed experiment configuration')
            changed={k for k in previous['script_hashes'].keys()|experiment['script_hashes'].keys()
                     if previous['script_hashes'].get(k)!=experiment['script_hashes'].get(k)}
            if changed!={'launch.py','harness_api.py','official_tool_entry.py','tool_sandbox.py','parallel_predictions.py','validate.py'}:
                raise ValueError('Four-GPU revision changed unrelated source')
        else:immutable(frozen_experiment,experiment)
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
