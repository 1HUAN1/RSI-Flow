"""Resume an accepted Task with pending Meta learning, preserving historical evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'runtime'))
    from sia.task_meta.pipeline import PipelineConfig, source_identity, run
    from sia.task_meta.file_lock import exclusive_lock
    from sia.task_meta.durable import value_hash
    from sia.task_meta.storage import digest, save_json
    from sia.task_meta.storage_budget import check_system_disk, RUNTIME_RESERVE_BYTES
    from sia.task_meta.deployed_recovery import read, verify_evidence
    check_system_disk(RUNTIME_RESERVE_BYTES)
    root = Path(args.run_dir).resolve(strict=True)
    original = read(root/'protocol.json')
    config = PipelineConfig.model_validate(original['config'])
    output = Path(config.output_root).resolve()
    if not root.is_relative_to(output/'runs'):
        raise ValueError('Recovery must use a registered output run')
    with exclusive_lock(output/'locks/launch.lock'):
        # Preparation uses the same native run lock; release it before run() takes it.
        with exclusive_lock(output/'locks/run'/(value_hash(str(root))+'.lock')):
            recovery = root/'recovery'
            recovery.mkdir(exist_ok=True)
            authorization = recovery/'authorization.json'
            if not authorization.exists():
                if (root/'round_0/complete.json').exists():
                    raise ValueError('This entrypoint requires a pending first-round Meta commit')
                d = read(root/'round_0/deployment.json')
                if d['status'] != 'deployed' or not d['gain'] > 0:
                    raise ValueError('No accepted positive-gain Task to recover')
                prior = recovery/'protocol_before.json'
                if prior.exists() and digest(prior) != digest(root/'protocol.json'):
                    raise ValueError('Recovery preparation has conflicting original protocol')
                if not prior.exists():
                    shutil.copy2(root/'protocol.json', prior)
                evidence = [root/'initial_meta_state.json']
                evidence += [root/'round_0'/name for name in (
                    'deployment.json','experience.json','selected_update.json','paired_training_effect.json',
                    'recursive_summary.json','all_candidate_effects.json','stage_manifest.json')]
                for generation in (0,1):
                    directory = root/f'round_0/feedback/gen_{generation}'
                    evidence += [directory/name for name in ('agent_execution.json','train_trajectories.jsonl','results.json')]
                state = d['task_after']
                from sia.task_meta.storage import checkpoint_manifest
                # Weight bytes, not only their previously stored manifest, must still match.
                actual = checkpoint_manifest(state['checkpoint_path'])
                if actual != state['checkpoint_manifest']:
                    raise ValueError('Accepted checkpoint weights changed')
                save_json(authorization, {'operation':'finish_deployed_round','round_index':0,
                    'source_protocol_sha256':digest(prior),'controller':source_identity(),
                    'evidence_sha256':{str(p.relative_to(root)):digest(p) for p in evidence}})
            else:
                verify_evidence(root)
            save_json(recovery/'effective_config.json',config.model_dump())
        active = {'run':str(root),'pid':os.getpid(),'status':'running',
                  'recovery':'finish_meta1_then_validation_then_round2',
                  'log':str(output/'logs'/(root.name+'_resume.log')),'output_root':str(output)}
        save_json(output/'active_run.json',active)
        try:
            result = run(config,root,resume=True)
            if result.get('rounds_completed') != config.max_generations:
                raise RuntimeError('Recovery stopped before all three rounds completed')
        except BaseException:
            save_json(output/'active_run.json',{**active,'status':'failed'})
            raise
        save_json(output/'active_run.json',{**active,'status':'completed'})


if __name__ == '__main__':
    main()
