"""Called at a committed round boundary; validation results are not returned to Meta."""
import json
import os
import subprocess
import sys
from pathlib import Path
from sia.task_meta.storage import save_json

def validate_round(config, run_dir, number, record):
    bundle=Path(config.round_validation_config).resolve().parents[1]
    output_root=Path(getattr(config,'output_root',bundle)).resolve()
    sequential=bool((getattr(config,'round_protocol',None) or {}).get('sequential_domains',False))
    if sequential and number>=0 and (number+1)%3: return
    outer_number=(number+1)//3 if sequential and number>=0 else number+1
    out=output_root/'validation'/Path(run_dir).name/f'round_{outer_number:02d}'
    out.mkdir(parents=True,exist_ok=True)
    binding={'round':outer_number,'task_state':record['task_after'],'meta_state':record['meta_after'],
             'deployment_status':record['status'],'chosen_component':record['chosen_component'],
             'source_round':str(Path(run_dir)/f'round_{number}/complete.json') if number>=0 else str(Path(run_dir)/'initial_meta_state.json')}
    if getattr(config,'round_protocol',None):
        from sia.task_meta.evolution_protocol import file_hash, fingerprint
        ledger=Path(run_dir)/'meta/experiences.jsonl'
        binding.update(source_role='independent_validation',purpose='report_only',
            execution_config_hash=fingerprint(config.model_dump()),
            active_experience_sha256=record.get('active_experience_hash',fingerprint([])),
            system_snapshot_hash=record.get('system_snapshot_hash'),
            task_harness_hash=file_hash(record['task_after']['harness_path']),
            meta_harness_hash=record['meta_after'].get('bundle_hash'))
        if number>=0:
            from sia.task_meta.durable import task_hash,load_task
            selected=json.loads((Path(run_dir)/f'round_{number}/selected_update.json').read_text())
            if selected['child_hash']!=task_hash(load_task(record['task_after'])):
                raise ValueError('Independent evaluation must use the exact child selected before post')
    path=out/'round_snapshot.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('Validation round snapshot changed')
    save_json(path,binding)
    env={k:v for k,v in os.environ.items() if k not in {'AUTODL_API_KEY','OPENROUTER_API_KEY','RSI_REMOTE_WORKER_TOKEN'}}
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+str(bundle)
    command=[sys.executable,'-u',str(bundle/'validate.py'),'--snapshot',str(path),
             '--pipeline-config',str(Path(run_dir)/'effective_config.json'),
             '--config',config.round_validation_config]
    if getattr(config,'round_protocol',None):
        command+=['--role','independent_validation','--manifest',config.round_protocol['validation_manifest'],'--execute']
    subprocess.run(command,env=env,cwd=bundle,check=True)
    done=json.loads((out/'complete.json').read_text())
    if done['benchmarks_completed']!=7 or done['feedback_to_meta'] is not False:
        raise RuntimeError('Round validation is incomplete')
