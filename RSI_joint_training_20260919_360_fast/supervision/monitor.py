"""Read-only supervision of the authorized run; never retries training or provider calls."""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

root=Path(sys.argv[1]).resolve()
directory=root/'supervision';directory.mkdir(exist_ok=True)
lock=(directory/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(root/'runtime'))
from sia.task_meta.evolution_protocol import fingerprint
cache={}

def read(path):
    return json.loads(path.read_text())

def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def snapshot():
    config=read(root/'configs/train.json')
    active=read(root/'active_run.json') if (root/'active_run.json').exists() else {}
    run=root/'runtime/runs'/(config['run_name']+'_r3')
    rounds=[]
    for n in range(3):
        r=run/f'round_{n}';item={'round':n+1,'meta_update_committed':(r/'complete.json').exists()}
        for phase in ('parent_pre_update','child_post_update'):
            binding=r/(phase+'_binding.json');stats={'expected':config['training_tasks_per_pass'],'receipts':0,'scored':0,'success':0,'infra_error':0}
            if binding.exists():
                folder=run/'rollout_cache'/fingerprint(read(binding))/'train_rollouts'
                for p in folder.glob('*.json'):
                    identity=(str(p),p.stat().st_mtime_ns)
                    if identity not in cache:
                        row=read(p)['row'];v=row.get('verification',{})
                        scored=v.get('status')=='completed' and type(v.get('success')) is bool and not row.get('infrastructure_error')
                        cache[identity]=(scored,scored and v['success'],bool(row.get('infrastructure_error')),row['source'],row.get('wall_time_seconds',0),row.get('model_call_count',0))
                    scored,success,infra,source,seconds,calls=cache[identity]
                    bucket=stats.setdefault('by_source',{}).setdefault(source,dict(scored=0,success=0,infra=0,task_seconds=0,model_calls=0))
                    bucket['scored']+=int(scored);bucket['success']+=int(success);bucket['infra']+=int(infra)
                    bucket['task_seconds']+=seconds;bucket['model_calls']+=calls
                    stats['receipts']+=1;stats['scored']+=int(scored);stats['success']+=int(success);stats['infra_error']+=int(infra)
            stats['phase_started_at']=binding.stat().st_mtime if binding.exists() else None
            item[phase]=stats
        evaluation=root/'validation'/run.name/f'round_{n+1:02d}'
        item['independent_eval_complete']=(evaluation/'complete.json').exists()
        rounds.append(item)
    live=alive(active.get('pid')) if active.get('pid') else False
    completed=sum(r['meta_update_committed'] and r['independent_eval_complete'] for r in rounds)
    state='completed' if completed==3 and active.get('status')=='completed' else 'running' if live else 'needs_attention'
    return dict(state=state,controller_pid=active.get('pid'),controller_alive=live,
        launcher_status=active.get('status','not_started'),complete_rounds=completed,rounds=rounds,
        run=str(run),log=active.get('log'),automatic_restarts=0)

previous=None
while True:
    try: value=snapshot()
    except Exception as exc: value={'state':'monitor_error','error_type':type(exc).__name__}
    signature=json.dumps(value,sort_keys=True)
    value['checked_at']=time.time()
    tmp=directory/'status.pending';tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,directory/'status.json')
    if signature!=previous:
        with (directory/'events.jsonl').open('a') as out:out.write(json.dumps(value,ensure_ascii=False)+'\n')
        previous=signature
    if value['state']=='completed':break
    time.sleep(60)
