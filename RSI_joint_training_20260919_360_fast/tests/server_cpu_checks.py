"""Run against the actual Linux runtime/evaluators without any model requests."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-source')
    parser.add_argument('--output',type=Path,default=ROOT/'preflight')
    args=parser.parse_args();source=args.runtime_source or read(ROOT/'configs/train.json')['runtime_source']
    sys.path.insert(0,source)
    args.output.mkdir(parents=True,exist_ok=True)
    from install_runtime import patched_files
    from tool_validation import check_native
    from tool_sandbox import run_native
    results=[]
    try:
        from transformers import Trainer
        import peft
        import numpy
        results.append({'check':'training_stack','status':'passed','python':sys.executable,'numpy':numpy.__version__})
    except Exception as exc:results.append({'check':'training_stack','status':'failed','reason':type(exc).__name__+': '+str(exc)})
    try:
        names=list(patched_files(Path(source)))
        results.append({'check':'runtime_extensions','status':'passed','files':names})
    except Exception as exc:results.append({'check':'runtime_extensions','status':'failed','reason':str(exc)})
    try:
        config=read(ROOT/'configs/train.json');counts={}
        for spec in config['datasets']:
            count=0
            for part in spec['files']:
                with (Path(config['dataset_root'])/part['path']).open('rb') as stream:
                    count+=sum(bool(line.strip()) for line in stream)
            counts[spec['name']]=count
        results.append({'check':'training_counts','status':'passed','counts':counts,'total':sum(counts.values())})
    except Exception as exc:results.append({'check':'training_counts','status':'failed','reason':str(exc)})
    try:
        from sia.task_meta.reporting import OfficialEvaluatorSpec
        settings=read(ROOT/'configs/validation.json')
        specs=read(settings['official_specs'])['evaluators']
        counts={}
        for name in ('livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev'):
            counts[name]=len(OfficialEvaluatorSpec(**{**specs[name],'benchmark':name}).validate())
        results.append({'check':'code_search_official_specs','status':'passed','counts':counts})
    except Exception as exc:results.append({'check':'code_search_official_specs','status':'failed','reason':str(exc)})
    try:
        from launch import dry_run
        results.append({'check':'frozen_queues_and_overlap',**dry_run(read(ROOT/'configs/train.json'),ROOT/'runtime'),'status':'passed'})
    except Exception as exc:results.append({'check':'data_overlap','status':'failed','reason':str(exc)})
    for name,spec in read(ROOT/'configs/validation.json')['tool_benchmarks'].items():
        output=args.output/name
        try:
            check_native(name,spec,output)
            results.append({'check':name,'status':'passed','real_model_calls':0})
        except Exception as exc:
            log=output/'check.log'
            lines=log.read_text(errors='replace').splitlines() if log.exists() else []
            allowed=('ModuleNotFoundError:','ImportError:','PermissionError:','RuntimeError:','TypeError:','ValueError:','ValidationError:','AssertionError:','isolation_setup:')
            results.append({'check':name,'status':'failed','reason':type(exc).__name__,
                            'diagnostic':[line for line in lines if any(s in line for s in allowed)][-4:]})
    output=args.output/'isolation'
    output.mkdir(parents=True,exist_ok=True)
    secret=args.output/'outside_sandbox_fixture.txt';secret.write_text('test_override_only')
    entry=output/'check_isolation.py'
    entry.write_text('''import json,os,socket,sys
from pathlib import Path
denied={}
try:Path(sys.argv[1]).read_text();denied['file']=False
except PermissionError:denied['file']=True
try:socket.socket();denied['network']=False
except PermissionError:denied['network']=True
denied['provider_environment']=all(k not in os.environ for k in ('AUTODL_API_KEY','OPENROUTER_API_KEY','ACE_USER_API_KEY','RSI_REMOTE_WORKER_TOKEN'))
Path('result.json').write_text(json.dumps(denied))
assert all(denied.values()),denied
''')
    try:
        with (output/'check.log').open('wb') as log:
            run_native([sys.executable,'-B',str(entry),str(secret)],output,output,log,timeout=20)
        results.append({'check':'isolation','status':'passed','denials':read(output/'result.json')})
    except Exception as exc:results.append({'check':'isolation','status':'failed','reason':type(exc).__name__})
    report={'checks':results,'real_model_calls':0,'training_started':False}
    write(args.output/'server_cpu_checks.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if any(r['status']!='passed' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
