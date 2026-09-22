"""Wait for the ACE-skipped A1 report, then resume the same three-round run.

This entrypoint never marks a 250/300 report complete. It archives the pinned
controller and completed-rollout authorization before an explicit evaluator
hotfix revision, verifies the old B2 rollout, and then invokes the normal
three-round resume launcher.
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, read, sha, write

sys.path.insert(0,str(ROOT/'runtime'))
from sia.task_meta.completed_rollout_reuse import completed_request_reusable
from sia.task_meta.durable import load_task, task_hash, value_hash
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.pipeline import PipelineConfig, source_identity
from sia.task_meta.round_validation import validate_round
from sia.task_meta.storage import digest


BENCHMARKS=['bfcl_v3','livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev']
HOTFIX_PATHS={'sia/task_meta/evalplus_codec.py','sia/task_meta/lcb_isolated.py','sia/task_meta/round_validation.py'}


def evaluator_alive(pid, snapshot):
    try:
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'continue_code_search_eval.py' in command and str(snapshot) in command


def wait_for_report(directory, pid):
    partial=directory/'code_search_partial_complete.json'
    snapshot=directory/'round_snapshot.json'
    while not partial.exists():
        if not evaluator_alive(pid,snapshot):
            raise RuntimeError('Code/Search evaluator exited without a partial-completion receipt')
        time.sleep(15)
    while evaluator_alive(pid,snapshot):
        time.sleep(2)
    return partial


def prepare_resume(root, partial):
    root=Path(root).resolve(strict=True)
    output=root.parent.parent
    report=output/'validation'/root.name/'round_01'
    if Path(partial).resolve()!=report/'code_search_partial_complete.json':
        raise ValueError('Partial report is not the A1 report for this run')
    with exclusive_lock(output/'locks/launch.lock'):
        receipt=read(partial)
        metrics_path=report/'task_metrics.json'
        metrics=read(metrics_path)
        round_record=read(root/'round_0/complete.json')
        state_hash=task_hash(load_task(round_record['task_after']))
        if (receipt.get('status')!='code_search_completed_ace_pending'
                or receipt.get('source_role')!='independent_validation'
                or receipt.get('feedback_to_meta') is not False
                or receipt.get('benchmarks_completed')!=BENCHMARKS
                or receipt.get('task_state_hash')!=state_hash
                or receipt.get('officially_scored')!=250
                or receipt.get('expected_total')!=300
                or receipt.get('metrics_sha256')!=sha(metrics_path)
                or metrics['overall']['n_scored']!=250 or metrics['overall']['complete']
                or (report/'complete.json').exists()
                or (report/'scores/acebench/result.json').exists()):
            raise ValueError('A1 is not an authentic ACE-pending 250/300 report')
        for name in BENCHMARKS:
            result=read(report/'scores'/name/'result.json')
            if (result.get('status')!='completed' or result.get('state_hash')!=state_hash
                    or result.get('feedback_to_meta') is not False):
                raise ValueError('A1 official score is missing or belongs to another Task: '+name)
        status=read(root/'round_1/early_rollout/status.json')
        if status.get('status')!='completed':
            raise ValueError('B2 parent rollout is not complete')

        archive=root/'recovery/partial_eval_resume'
        archive.mkdir(parents=True,exist_ok=True)
        protocol_before=archive/'protocol_before.json'
        proof_before=archive/'completed_rollout_reuse_before.json'
        protocol_path=root/'protocol.json'
        proof_path=root/'recovery/completed_rollout_reuse.json'
        if not protocol_before.exists():shutil.copy2(protocol_path,protocol_before)
        if not proof_before.exists():shutil.copy2(proof_path,proof_before)
        old=read(protocol_before)
        old_proof=read(proof_before)
        current_controller=source_identity()
        changed={name for name in set(old['controller'])|set(current_controller)
                 if old['controller'].get(name)!=current_controller.get(name)}
        if changed!=HOTFIX_PATHS:
            raise ValueError('Unexpected controller revision; only evaluator repair and explicit ACE policy are permitted: '+str(sorted(changed)))
        if old_proof['controller']!=old['controller']:
            raise ValueError('Archived B2 rollout authorization does not match archived protocol')
        for entry in old_proof['entries'].values():
            for name, expected in entry['files'].items():
                if digest(root/name)!=expected:
                    raise ValueError('Completed B2 rollout evidence changed: '+name)

        updated=copy.deepcopy(old)
        updated['controller']=current_controller
        updated.pop('hash',None)
        updated['hash']=value_hash(updated)
        proof=copy.deepcopy(old_proof)
        proof['controller']=current_controller
        if read(protocol_path) not in (old,updated) or read(proof_path) not in (old_proof,proof):
            raise ValueError('Protocol or rollout authorization has an unrelated revision')
        config=PipelineConfig.model_validate(old['config']).checked()
        manifest=Path(config.round_protocol['validation_manifest'])
        authorization={
            'schema_version':'ace-skipped-round-validation-v1',
            'run_name':root.name,
            'manifest_sha256':digest(manifest),
            'validation_config_sha256':digest(Path(config.round_validation_config)),
            'initial_partial_sha256':digest(partial),
            'expected_total':300,'scored_total':250,
            'reason':'ACE user-simulator failed; preserve incomplete ACE result and proceed only after six authentic official scores',
            'feedback_to_meta':False,
        }
        auth_path=root/'recovery/ace_skip_authorization.json'
        if auth_path.exists() and read(auth_path)!=authorization:
            raise ValueError('ACE skip authorization changed')
        write(auth_path,authorization)
        if read(proof_path)!=proof:write(proof_path,proof)
        if read(protocol_path)!=updated:write(protocol_path,updated)
        receipt_path=archive/'revision.json'
        revision={
            'status':'authorized_evaluator_hotfix_and_partial_validation',
            'source_protocol_sha256':digest(protocol_before),
            'source_rollout_reuse_sha256':digest(proof_before),
            'new_protocol_sha256':digest(protocol_path),
            'new_rollout_reuse_sha256':digest(proof_path),
            'a1_partial_receipt_sha256':digest(partial),
            'changed_controller_paths':sorted(changed),
            'full_300_task_validation_completed':False,
            'B2_parent_rollout_reused':True,
        }
        if receipt_path.exists() and read(receipt_path)!=revision:
            raise ValueError('Partial validation continuation revision changed')
        write(receipt_path,revision)
        validate_round(config,root,0,round_record)
        request=read(root/'round_1/early_rollout/request.json')
        if not completed_request_reusable(root,root/'round_1/before/gen_1',request['task_hash']):
            raise ValueError('Completed B2 parent rollout cannot be reused')
        return revision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--evaluator-pid',required=True,type=int)
    args=parser.parse_args()
    root=Path(args.run_dir).resolve(strict=True)
    report=root.parent.parent/'validation'/root.name/'round_01'
    partial=wait_for_report(report,args.evaluator_pid)
    print('[partial-eval] A1 250/300 completed, ACE remains pending',flush=True)
    revision=prepare_resume(root,partial)
    print('[partial-eval] verified revision and completed B2 parent rollout: '+json.dumps(revision),flush=True)
    environment=os.environ.copy()
    environment['RSIFLOW_RESUME_DEPLOYED_RUN']=str(root)
    subprocess.run(['bash',str(ROOT/'start_3round_training.sh')],env=environment,cwd=ROOT,check=True)


if __name__=='__main__':main()
