"""Explicit preparation/start of the existing RSI runtime with frozen round manifests."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, read, write, immutable, sha, resolve_output_root

def make_pipeline(config, runtime):
    source=(ROOT/Path(config.get('runtime_source','runtime'))).resolve()
    output=resolve_output_root(config)
    p=read(source/config['base_config'])
    # The normalized project always initializes Meta G/K from its own seed.
    p.pop('initial_meta_bundle', None)
    p.pop('initial_meta_bundle_hash', None)
    p.update(mode='full',experiment_scope='multidomain',training_schedule='round_disjoint',
             task_update_policy=config.get('candidate_policy','single_candidate_strict_positive_gain'),artifact_evaluation=config.get('artifact_evaluation','direct_submission'),
             allowed_task_components=['HARNESS','MODEL','ARTIFACTS'],
             output_root=str(output),
             data_dir=config.get('frozen_data_dir') or str(runtime/'data'/config.get('data_release','rounds')),max_generations=9 if config.get('sequential_domains') else 3,seed=config['rollout_seed'],
             search_dev_fraction=0,
             envscaler_root=str(Path(config['dataset_root'])/'train/tool_use/envscaler'),
             max_wall_seconds=config['max_wall_seconds'],rollouts_per_task=1,probe_rollouts=1,
             gpu_execution='phased_four',training_timeout_seconds=config['training_timeout_seconds'],
             round_validation_config=str((ROOT/config['validation_config']).resolve()),
             trainer_python=sys.executable,task_base_url=f"http://127.0.0.1:{config['ports'][0]}/v1")
    p['meta']['run_mode']='full'
    p['seed_harness']=str((ROOT/'seed_harness/harnessforge_base_manifest.json').resolve())
    p['meta']['remote_worker_socket']=config['meta_worker_socket']
    p['meta']['remote_relay_socket']=config['meta_relay_socket']
    if p['meta']['model']!='DeepSeek-V4.1-Flash':raise ValueError('Expected the user-selected DeepSeek-V4.1-Flash Meta runtime')
    # Runtime pin/catalog paths remain server-local and are never read from the validation split.
    for name in ['codex_source','codex_executable','provenance_file','model_catalog_json','compatibility_report','harness_root']:
        value=p['meta'].get(name)
        if value and not Path(value).is_absolute():p['meta'][name]=str(source/value)
    p['probe_per_domain']=dict.fromkeys(['tool_use','code','searchqa'],120)
    p['window_quotas']=dict.fromkeys(['tool_use','code','searchqa'],120)
    p['task_replicas']=[{'gpu':i,'base_url':f'http://127.0.0.1:{port}/v1'} for i,port in enumerate(config['ports'])]
    p['training'].update(num_train_epochs=1,max_steps=-1,max_length=config['max_length'],max_samples=config['max_sft_samples'])
    from evolution_protocol import SFT_PRESET
    p['training'].update(SFT_PRESET,seed=config['training_seed'])
    p['round_protocol']={'data_dir':p['data_dir'],'evaluate_initial_system':config.get('evaluate_initial_system',False),'minimum_sft_samples':config['minimum_sft_samples'],
        'memory_policy':config['memory_policy'], 'single_decision_per_round':True,
        'candidate_policy':config.get('candidate_policy','single_candidate_strict_positive_gain'),
        'sequential_domains':config.get('sequential_domains',False),
        'evidence_bound_interventions':config.get('evidence_bound_interventions',True),
        'validation_manifest':str(Path(config['validation_vault'])/'validation/manifest.json'),
        'split_seed':config['split_seed'],'rollout_seed':config['rollout_seed'],'training_seed':config['training_seed']}
    if config.get('pause_after_round') is not None:
        p['round_protocol']['pause_after_round']=config['pause_after_round']
    p['training']['round_protocol']={k:v for k,v in p['round_protocol'].items() if k!='validation_manifest'}
    return p


def dry_run(config, runtime):
    from evolution_protocol import manifest, assert_disjoint
    from collections import Counter
    from evolution_protocol import TRAIN_QUOTAS
    if config['train_quotas_per_round'] != TRAIN_QUOTAS or config['allocated_tasks'] != 3*sum(TRAIN_QUOTAS.values()):
        raise ValueError('Expected three disjoint 360-task rounds with registered source quotas')
    data_root=Path(config.get('frozen_data_dir') or Path(runtime)/'data'/config.get('data_release','rounds'))
    rounds=[manifest(data_root/f'B{r}/manifest.json','train_evolution',r) for r in (1,2,3)]
    assert_disjoint(rounds)
    details=[]
    for value in rounds:
        expected={t['task_id']:t for t in value['tasks']}
        if Counter(t['source'] for t in value['tasks'])!=config['train_quotas_per_round']:
            raise ValueError('Round allocation mismatch')
        details.append({'round':value['round_id'],'allocated_tasks':len(expected),'meta_feedback_tasks':len(expected),
                        'pre':len(expected),'post':len(expected),
                        'by_source':dict(Counter(t['source'] for t in value['tasks'])),'manifest_hash':value['manifest_hash']})
    from evolution_protocol import VALIDATION_QUOTAS
    validation=manifest(Path(config['validation_vault'])/'validation/manifest.json','independent_validation',None)
    if Counter(t['source'] for t in validation['tasks'])!=VALIDATION_QUOTAS:
        raise ValueError('Actual validation manifest must contain the fixed 300 tasks')
    assert_disjoint([*rounds,validation])
    initial=bool(config.get('evaluate_initial_system',False))
    if len(config['ports'])!=4 or len(set(config['ports']))!=4:raise ValueError('Four distinct GPU endpoints required')
    allocated=sum(d['allocated_tasks'] for d in details)
    sequential=bool(config.get('sequential_domains',False))
    from evolution_protocol import stage_manifest
    stages=[stage_manifest(data_root,i,sequential_domains=sequential) for i in range(1,10 if sequential else 4)]
    if config.get('candidate_policy','single_candidate_strict_positive_gain')!='single_candidate_strict_positive_gain':
        raise ValueError('Only the single-candidate strict-positive policy is executable')
    limit=1
    external=900+300*initial
    result=dict(status='dry_run_no_model_calls',rounds=details,allocated_tasks=allocated,
        harness_initialization='upstream_harnessforge_bundle',
        training_tasks_per_round=sum(TRAIN_QUOTAS.values()),training_passes_per_round={'min':1,'max':2},
        training_executions={'min':allocated,'max':2*allocated},
        meta_memory_policy=config['memory_policy'],
        candidate_limit_per_stage=limit,candidate_limit_per_round=limit*(3 if sequential else 1),
        execution_stages=[dict(stage=m['round_id'],allocation_round=m.get('allocation_round_id',m['round_id']),domain=m.get('domain','all'),pre=len(m['tasks']),post=len(m['tasks']),manifest_hash=m['manifest_hash']) for m in stages],
        selected_lineages=1,deployment_policy=config.get('candidate_policy','single_candidate_strict_positive_gain'),
        candidate_evaluations='one child pass per domain stage' if sequential else 'one child pass included per round',external_validation_tasks_per_checkpoint=300,sft_calls=('0..9' if sequential else '0..3')+', conditional on Meta MODEL and verified-success minimum',
        sft_epochs_per_call=1,external_validation_checkpoints=(['A0'] if initial else [])+['A1','A2','A3'],
        validation_manifest_hash=validation['manifest_hash'],validation_quotas=VALIDATION_QUOTAS,
        external_validation_executions=external,
        base_task_executions={'min':allocated+external,'max':2*allocated+external},
        inference_gpus=4,sft_gpus=4,queues_are_global_not_per_gpu=True,final_test='separate explicit entry')
    for item in result['execution_stages']:
        item['post']=item['pre']
    for item in result['rounds']:
        item['post']=item['pre']
    return result

def check(config, runtime, *, external_readiness=False):
    """Run the complete local preflight without implicitly starting the run.

    The standalone mode validates all local dependencies and frozen inputs but
    deliberately does not contact the Meta worker or any model/service
    endpoint. Real execution repeats these checks with external readiness
    enabled immediately before starting.
    """
    from tool_validation import native_layout
    from sia.task_meta.reporting import OfficialEvaluatorSpec
    from sia.task_meta.sandbox import probe_isolation
    from sia.task_meta.pipeline import PipelineConfig
    errors=[];details={};output=resolve_output_root(config)
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
            check_native(name,spec,output/'preflight'/name)
    except Exception as exc:errors.append('Official evaluation preflight: '+str(exc))
    pipeline=make_pipeline(config,runtime)
    native_config=PipelineConfig.model_validate(pipeline).checked()
    required_env = [pipeline['meta']['api_key_env'],
                    settings['tool_benchmarks']['acebench']['user_api_key_env'],
                    settings['tool_benchmarks']['acebench']['user_base_url_env']]
    if pipeline['meta'].get('execution_location', 'ssh_worker') == 'ssh_worker':
        required_env.append(pipeline['meta']['remote_worker_token_env'])
    for name in required_env:
        if not os.environ.get(name):errors.append('Required environment variable is not set: '+name)
    if not errors and external_readiness:
        try:
            if native_config.meta.execution_location == 'local_chroot':
                from sia.task_meta.meta_backends.local_execution import validate_local
                details['meta_worker'] = validate_local(native_config.meta)
            else:
                from sia.task_meta.meta_backends.remote_execution import validate_worker
                validate_worker(native_config.meta)
                details['meta_worker']={'healthy':True,'isolation_and_version_verified':True}
        except Exception as exc:errors.append('Meta worker preflight: '+str(exc))
    elif not errors:
        details['meta_worker']={'checked':False,'reason':'offline preflight performs no network or model calls'}
    if errors:
        result={'status':'blocked_before_training','errors':errors,'details':details,'model_calls':0,
                'network_calls':0 if not external_readiness else None,'started':False,
                'external_readiness_checked':external_readiness}
        write(output/'preflight/status.json',result)
        raise RuntimeError('\n'.join(errors))
    details['round_data']=dry_run(config,runtime)
    result={'status':'ready_for_explicit_start' if external_readiness else 'offline_preflight_passed',
            'details':details,'model_calls':0,'network_calls':0 if not external_readiness else None,
            'started':False,'external_readiness_checked':external_readiness,
            'allocated_tasks':config['allocated_tasks'],'rounds':3,'sft_epochs_per_update':1,
            'output_root':str(output)}
    write(output/'preflight/status.json',result)
    return pipeline,result

def _finish_child(config, runtime, name, log, code):
    """Commit launcher status only from the child's durable final state."""
    output=resolve_output_root(config)
    active_path=output/'active_run.json'
    run=output/'runs'/name
    final_path=run/'final_state.json'

    def invalid(reason, final=None):
        value={'run':str(run),'status':'failed','exit_code':0,'reason':reason,
               'final_state':str(final_path),'receipts_preserved':True,'log':str(log),
               'output_root':str(output)}
        if isinstance(final,dict):
            value.update(final_status=final.get('status'),
                         rounds_completed=final.get('rounds_completed'))
        write(active_path,value)
        raise RuntimeError(reason+' Receipts are preserved; rerun --execute after resolving it.')

    if code:
        write(active_path,{'run':str(run),'status':'failed','exit_code':code,
              'receipts_preserved':True,'log':str(log),'output_root':str(output)})
        raise SystemExit('Training stopped with preserved receipts. Inspect logs; rerun --execute after resolving the reported error. No budgets are reset.')
    if not final_path.is_file():
        invalid('Training exited zero without final_state.json.')
    try:
        final=read(final_path)
    except Exception:
        invalid('Training exited zero with an unreadable final_state.json.')

    if final.get('status')=='paused':
        expected=config.get('pause_after_round')
        rounds=final.get('rounds_completed')
        marker_text=final.get('pause_marker')
        marker=Path(marker_text).resolve() if isinstance(marker_text,str) and marker_text else None
        expected_marker=(run/'pause_after_round.json').resolve()
        if (type(expected) is not int or rounds!=expected
                or final.get('paused_at_round')!=expected
                or marker!=expected_marker or not marker.is_file()):
            invalid('Training exited zero with invalid durable pause evidence.',final)
        value={'run':str(run),'status':'paused','rounds_completed':rounds,
               'pause_marker':str(marker),'resume_starts_at_round':rounds+1,
               'log':str(log),'output_root':str(output)}
        write(active_path,value)
        return value

    if (final.get('status')=='completed' and config.get('rounds')==3
            and final.get('rounds_completed')==config['rounds']):
        value={'run':str(run),'status':'completed','rounds':config['rounds'],
               'rounds_completed':final['rounds_completed'],
               'results':str(output/'validation'/name/'all_rounds.csv'),'log':str(log),
               'output_root':str(output)}
        write(active_path,value)
        return value

    invalid('Training exited zero without a valid paused or completed final state.',final)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/train.json'))
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true',help='Prepare a new frozen data release only')
    mode.add_argument('--execute',action='store_true',help='Explicitly start or resume real execution')
    mode.add_argument('--check',action='store_true',help='Complete offline preflight; no model/network calls and no execution')
    mode.add_argument('--dry-run',action='store_true',help='Manifest/count summary only; no inference, SFT or Meta API')
    a=p.parse_args(argv)
    config=read(a.config)
    output=resolve_output_root(config)
    if config['rounds']!=3:raise ValueError('Protocol fixes three rounds')
    if (config['sft_epochs_per_update'],config['sft_max_steps'])!=(1,-1):
        raise ValueError('One SFT epoch is required')
    if config.get('candidate_policy')!='single_candidate_strict_positive_gain':
        raise ValueError('Only the single-candidate strict-positive policy is supported')
    if config.get('max_dataset_passes_per_stage')!=2:
        raise ValueError('One parent and at most one candidate pass is required')
    if a.prepare:
        from data_protocol import prepare_protocol
        from install_runtime import install
        install(config)
        print(json.dumps(prepare_protocol(config,read(ROOT/config['validation_config']),ROOT/'runtime/data'/config.get('data_release','rounds'),config['validation_vault']),ensure_ascii=False))
        return
    if a.check:
        from install_runtime import install
        runtime=install(config);sys.path.insert(0,str(runtime))
        _,result=check(config,runtime,external_readiness=False)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    if not a.execute:
        print(json.dumps(dry_run(config,ROOT/'runtime'),ensure_ascii=False,indent=2))
        return
    if sys.platform!='linux':raise SystemExit('Execute from the Linux server work directory')
    from install_runtime import install
    runtime=install(config);sys.path.insert(0,str(runtime))
    from sia.task_meta.file_lock import exclusive_lock
    with exclusive_lock(output/'locks/launch.lock'):
        pipeline,result=check(config,runtime,external_readiness=True)
        from validation_sources import freeze
        freeze(read(ROOT/config['validation_config']),output/'validation_sources.json')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        name=config['run_name']+f"_r{config['rounds']}"
        config_path=output/'configs'/('joint_'+name+'.json')
        immutable(config_path,pipeline)
        experiment={'training':config,'validation':read(ROOT/config['validation_config']),
            'validation_sources_sha256':sha(output/'validation_sources.json'),
            'script_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'validation_never_used_for_selection':True}
        frozen_experiment=output/'experiments'/('experiment_'+name+'.json')
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
        run=output/'runs'/name
        command=[sys.executable,'-u',str(ROOT/'train.py'),'--config',str(config_path),'--run-dir',str(run)]
        logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
        write(output/'active_run.json',{'run':str(run),'command':command,'status':'starting','output_root':str(output)})
        with (logs/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdout=log,stderr=log,cwd=ROOT)
            write(output/'active_run.json',{'run':str(run),'pid':child.pid,'status':'running','log':str(logs/(name+'.log')),'output_root':str(output)})
            code=child.wait()
        _finish_child(config,runtime,name,logs/(name+'.log'),code)

if __name__=='__main__':main()
