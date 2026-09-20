"""Called at a committed round boundary; validation results are not returned to Meta."""
import json
import os
import subprocess
import sys
from pathlib import Path
from sia.task_meta.storage import save_json

def validate_round(config, run_dir, number, record):
    bundle=Path(config.round_validation_config).resolve().parents[1]
    out=bundle/'validation'/Path(run_dir).name/f'round_{number+1:02d}'
    out.mkdir(parents=True,exist_ok=True)
    binding={'round':number+1,'task_state':record['task_after'],'meta_state':record['meta_after'],
             'deployment_status':record['status'],'chosen_component':record['chosen_component'],
             'source_round':str(Path(run_dir)/f'round_{number}/complete.json')}
    path=out/'round_snapshot.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('Validation round snapshot changed')
    save_json(path,binding)
    env={k:v for k,v in os.environ.items() if k not in {'AUTODL_API_KEY','OPENROUTER_API_KEY','RSI_REMOTE_WORKER_TOKEN'}}
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+str(bundle)
    command=[sys.executable,'-u',str(bundle/'validate.py'),'--snapshot',str(path),
             '--pipeline-config',str(Path(run_dir)/'effective_config.json'),
             '--config',config.round_validation_config]
    subprocess.run(command,env=env,cwd=bundle,check=True)
    done=json.loads((out/'complete.json').read_text())
    if done['benchmarks_completed']!=7 or done['feedback_to_meta'] is not False:
        raise RuntimeError('Round validation is incomplete')
