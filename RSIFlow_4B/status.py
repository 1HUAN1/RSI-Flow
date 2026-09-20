"""Read-only status; never restarts inference, SFT or Meta requests."""
import json
from pathlib import Path
from common import ROOT,read,resolve_output_root

def main():
    output=resolve_output_root(read(ROOT/'configs/train.json'))
    path=output/'active_run.json'
    if not path.exists():
        monitor=read(output/'monitor.json') if (output/'monitor.json').exists() else None
        print(json.dumps({'status':monitor.get('status','preflight') if monitor else 'not_started',
          'output_root':str(output),'monitor':monitor,'preflight':read(output/'preflight/status.json') if (output/'preflight/status.json').exists() else None},ensure_ascii=False,indent=2));return
    active=read(path);run=Path(active['run']);validation=output/'validation'/run.name
    active['joint_rounds_committed']=len(list(run.glob('round_*/complete.json')))
    active['validation_rounds_completed']=len(list(validation.glob('round_*/complete.json')))
    active['latest_phase']=read(run/'phase.json') if (run/'phase.json').exists() else None
    active['results_table']=str(validation/'RESULTS.md')
    if (output/'monitor.json').exists():active['monitor']=read(output/'monitor.json')
    print(json.dumps(active,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
