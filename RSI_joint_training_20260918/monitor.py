"""Persist one-minute progress snapshots without restarting any paid work."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from common import ROOT,read,write


def identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def snapshot(launcher_pid,launcher_start):
    active=read(ROOT/'active_run.json') if (ROOT/'active_run.json').exists() else {}
    out={'checked_at':time.time(),'launcher_alive':identity(launcher_pid)==launcher_start,
         'status':active.get('status','preflight'),'free_disk_bytes':shutil.disk_usage(ROOT).free,
         'joint_rounds_committed':0,'validation_rounds_completed':0,'rollouts':[]}
    if active.get('run'):
        run=Path(active['run']);validation=ROOT/'validation'/run.name
        out['run']=str(run)
        out['joint_rounds_committed']=len(list(run.glob('round_*/complete.json')))
        out['validation_rounds_completed']=len(list(validation.glob('round_*/complete.json')))
        out['phase']=read(run/'phase.json') if (run/'phase.json').exists() else None
        directories=list(run.glob('gen_*'))+list(run.glob('round_*/candidates/*/gen_*'))
        for directory in directories:
            counts={}
            for split in ('train','probe'):
                path=directory/(split+'_rollouts')
                if path.exists():
                    with os.scandir(path) as files:
                        counts[split]=sum(len(f.name)==69 and f.name.endswith('.json') for f in files)
            if counts:out['rollouts'].append({'path':str(directory.relative_to(run)),**counts})
        tables=validation/'all_rounds.csv'
        if tables.exists():out['results_table']=str(tables)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    out['gpu']=gpu.stdout.strip().splitlines()
    out['requires_attention']=not out['launcher_alive'] and out['status']!='completed'
    if out['requires_attention'] and (ROOT/'preflight/status.json').exists():
        out['preflight']=read(ROOT/'preflight/status.json')
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launcher-pid',type=int,required=True)
    parser.add_argument('--launcher-start',required=True);args=parser.parse_args()
    while True:
        try:current=snapshot(args.launcher_pid,args.launcher_start)
        except Exception as exc:
            current={'checked_at':time.time(),'monitor_error_type':type(exc).__name__,
                     'launcher_alive':identity(args.launcher_pid)==args.launcher_start}
        write(ROOT/'monitor.json',current)
        with (ROOT/'logs/monitor.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(current,ensure_ascii=False)+'\n')
        if not current['launcher_alive']:return
        time.sleep(60)


if __name__=='__main__':main()
