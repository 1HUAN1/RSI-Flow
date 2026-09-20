"""Read-only status; never restarts inference, SFT or Meta requests."""
import json
from pathlib import Path
from common import ROOT,read

def main():
    path=ROOT/'active_run.json'
    if not path.exists():
        monitor=read(ROOT/'monitor.json') if (ROOT/'monitor.json').exists() else None
        print(json.dumps({'status':monitor.get('status','preflight') if monitor else 'not_started',
          'monitor':monitor,'preflight':read(ROOT/'preflight/status.json') if (ROOT/'preflight/status.json').exists() else None},ensure_ascii=False,indent=2));return
    active=read(path);run=Path(active['run']);validation=ROOT/'validation'/run.name
    active['joint_rounds_committed']=len(list(run.glob('round_*/complete.json')))
    active['validation_rounds_completed']=len(list(validation.glob('round_*/complete.json')))
    active['latest_phase']=read(run/'phase.json') if (run/'phase.json').exists() else None
    active['results_table']=str(validation/'RESULTS.md')
    if (ROOT/'monitor.json').exists():active['monitor']=read(ROOT/'monitor.json')
    print(json.dumps(active,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
