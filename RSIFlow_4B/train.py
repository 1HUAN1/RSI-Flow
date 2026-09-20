"""Real Task/Meta evolution entrypoint; validation is a separate subprocess."""
import argparse
import sys
from pathlib import Path
from common import ROOT, read, write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'runtime'))
    from sia.task_meta.pipeline import load_config,run
    config=load_config(args.config)
    # Native startup creates its journal before writing protocol.json. A failed
    # host preflight must resume that directory; native identity checks still run.
    result=run(config,args.run_dir,resume=Path(args.run_dir).exists())
    if result.get('status') == 'paused':
        expected = (config.round_protocol or {}).get('pause_after_round')
        marker = Path(result.get('pause_marker', ''))
        if (result.get('paused_at_round') != expected or result.get('rounds_completed') != expected
                or not marker.is_file()):
            raise RuntimeError('Paused result is missing its durable round-boundary evidence')
        print({'status':'paused_at_round_boundary','rounds':result['rounds_completed'],
               'resume_starts_at_round':expected + 1,'pause_marker':str(marker)},flush=True)
        return
    if result.get('rounds_completed')!=config.max_generations:
        raise RuntimeError('Requested evolution rounds are incomplete; receipts are preserved for resume')
    print({'status':'all_joint_rounds_completed','rounds':result['rounds_completed']},flush=True)

if __name__=='__main__':main()
