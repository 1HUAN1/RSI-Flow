"""Compare two Meta packages on one fixed Task, parent rollout and evidence.

Default: freeze inputs and write plans, without API/GPU work.
--execute: independent persistent Codex threads, same skill/context inputs, one
first candidate per arm, same Task batch. Do not deploy or learn from this report.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from component_versions import write_json
from controller_tools import ControllerTools
from launch_meta import (_autodl_key, prepare_codex_home, prepare_meta_program,
                         prepare_context_path, prepare_skills_path, SubprocessToolExecutor)
from meta_harness_runtime import FILES, MetaHarnessRuntime
from persistent_meta import (CodexAppServer, ExperimentJobs, LONG_OPERATIONS,
                             dynamic_tools, native_thread_parameters)
from meta_loop import render_receipt_for_codex


def prepare(config_path, task, parent, old_package, new_package, skills, context, output, round_number=1):
    output = Path(output).resolve()
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(config_path).read_text())
    write_json(inputs / 'task_state.json', json.loads(Path(task).read_text()))
    shutil.copy2(skills, inputs / 'skills.jsonl')
    write_json(inputs / 'context.json', json.loads(Path(context).read_text()) if context else {})
    source = {'task_state_path': str(inputs / 'task_state.json'),
              'parent_rollout_dir': str(Path(parent).resolve()), 'skills_path': str(inputs / 'skills.jsonl'),
              'context_path': str(inputs / 'context.json'), 'round_number': int(round_number)}
    arms = {}
    for name, package in (('old', old_package), ('new', new_package)):
        arm = output / name
        arm.mkdir(exist_ok=True)
        seed = arm / 'seed'
        seed.mkdir(exist_ok=True)
        for filename in FILES:
            shutil.copy2(Path(package) / filename, seed / filename)
        config = {**base, 'run_name': output.name + '_' + name,
                  'meta_harness_seed_path': str(seed), 'meta_harness_mode': 'fixed',
                  'meta_fixed_skills': True}
        write_json(arm / 'config.json', config)
        write_json(arm / 'active_task.json', json.loads(Path(task).read_text()))
        prepare_skills_path(arm, inputs / 'skills.jsonl')
        prepare_context_path(arm)
        shutil.copy2(inputs / 'context.json', arm / 'meta/context.json')
        prepare_meta_program(arm, config)
        arms[name] = {**source, 'config_path': str(arm / 'config.json'),
                      'arm_dir': str(arm), 'used_package': str(arm / 'meta_harness/G000'),
                      'supplied_package': str(Path(package).resolve())}
    plan = {'status': 'prepared', 'purpose': 'report_only', 'executed': False,
            'comparison': 'One first intervention per arm; identical Task/evidence/skills/context.',
            'arms': arms, 'output_dir': str(output)}
    write_json(output / 'pair_plan.json', plan)
    return plan


class PairTools:
    """Use the SAME executor/jobs/modules as production, without the multi-round finish rule."""
    def __init__(self, arm):
        self.arm = arm
        self.run = Path(arm['arm_dir'])
        self.executor = SubprocessToolExecutor(project_root=PROJECT, run_dir=self.run)
        self.jobs = ExperimentJobs(PROJECT, self.run)
        self.program = MetaHarnessRuntime(PROJECT, self.run)
        self.finished = False
        turns = self.run / 'meta_session/turns'
        self.number = max((int(p.name) for p in turns.iterdir() if p.name.isdigit()), default=-1) + 1 if turns.exists() else 0
        saved = self.run / 'comparison.json'
        self.comparison = json.loads(saved.read_text()) if saved.exists() else None
        self.phase = 'feedback' if self.comparison else 'route'
        self.last_anchor = None

    def guidance(self):
        return self.program.phase({**self.arm, 'round': self.arm['round_number'], 'phase': self.phase,
                                   'evidence_paths': [self.arm['parent_rollout_dir'], self.arm['skills_path']]})

    def guidance_text(self):
        policy = self.guidance()
        path = self.run / 'meta_harness/phase_context.json'
        write_json(path, policy)
        return render_receipt_for_codex(policy, path, max_inline_bytes=24000)

    def __call__(self, tool, arguments, params=None):
        turn = self.run / 'meta_session/turns' / f'{self.number:04d}'
        self.number += 1
        turn.mkdir(parents=True, exist_ok=True)
        request = arguments.get('request', {}) if tool == 'experiment' else {'operation': tool, **arguments}
        request = {**request, '_meta_round': self.arm['round_number']}
        write_json(turn / 'command.json', {'action': 'tool', 'request': request})
        try:
            if tool == 'finish_experiment':
                # Missing outcomes are facts, never evidence of zero gain.
                if self.comparison is not None:
                    self.finished = True
                    receipt = {'status': 'complete', 'summary': arguments.get('summary'),
                               'comparison_path': str(self.run / 'comparison.json')}
                else:
                    receipt = {'status': 'incomplete', 'remaining': 'Obtain paired compare_scores facts, or record the failed build/run through report_failed_attempt.'}
            elif tool == 'wait_job':
                receipt = self.jobs.wait(arguments['job_id'], arguments.get('seconds', 60))
            elif tool == 'job_status':
                receipt = self.jobs.status(arguments['job_id']) if arguments.get('job_id') else {'jobs': self.jobs.all()}
            elif request['operation'] == 'report_failed_attempt':
                self.comparison = {'status': 'unmeasured', 'failure': request.get('failure'),
                                   'evidence_refs': request.get('evidence_refs', []), 'delta': None}
                write_json(self.run / 'comparison.json', self.comparison)
                receipt = self.comparison
            elif request['operation'] in LONG_OPERATIONS:
                receipt = self.jobs.start(request, turn)
            else:
                receipt = self.executor.execute(request)
            if request['operation'] == 'compare_scores' and receipt.get('status') == 'compared':
                self.comparison = receipt
                write_json(self.run / 'comparison.json', receipt)
                self.phase = 'feedback'
            elif request['operation'] in {'materialize_harness', 'sft', 'artifacts', 'run_candidate'}:
                self.phase = 'candidate'
        except Exception as exc:
            receipt = {'status': 'tool_error', 'error': f'{type(exc).__name__}: {exc}'}
        write_json(turn / 'receipt.json', receipt)
        text = render_receipt_for_codex(receipt, turn / 'receipt.json')
        if self.phase != self.last_anchor:
            try:
                text += '\nActive comparison Meta method:\n' + self.guidance_text()
            except Exception as exc:
                text += f'\nMeta hook error: {exc}; inspect it and continue the same comparison arm.'
            self.last_anchor = self.phase
        return text


PAIR_PROTOCOL = """Report-only Meta comparison, not a training run. Codex stays alive while tools execute.
Native tools and experiment/job_status/wait_job helpers are available. Diagnose errors and retry
the affected step. Fixed scorer/model/API/research protocol; component choice is yours.
Use supplied frozen parent Task, parent rollout, evidence, skill seed and context; do not rerun
the parent, deploy into the source run, update skills, or advance rounds. Read all statistics,
48 selected excerpts and original evidence as needed; prepare_meta_evidence exposes these.
The active three-file package supplies your method. Choose HARNESS/MODEL/ARTIFACTS freely,
record prospective reasons and prediction, then generate ONLY ONE first candidate.
HARNESS: read pinned templates → localization report → improvement direction → complete bundle
→ at most three executable checks/repairs of the SAME bundle. MODEL: verified successful
parent trajectories only, one-epoch four-GPU LoRA SFT. ARTIFACTS: direct submission edits/rescore.
Run candidate on the SAME supplied round/batch with unchanged budgets; compare_scores returns
paired measurements, compare_task_differences exposes successes/regressions. Write selection.json
with component, reasons and proposed accept/retain decision but DO NOT activate_task.
For an unexecutable attempt, call report_failed_attempt(failure,evidence_refs); do not report
zero gain or switch to another candidate. Finish through finish_experiment once measurement
or explicit unmeasured failure has been saved. No independent validation feeds learning here.
"""


def execute_arm(arm):
    os.environ['AUTODL_API_KEY'] = _autodl_key()
    run = Path(arm['arm_dir'])
    home = prepare_codex_home(run)
    tools = PairTools(arm)
    server = CodexAppServer(codex_home=home, workspace=PROJECT.parent,
                            journal=run / 'meta_session/app_server', on_tool=tools)
    state_path = run / 'pair_session.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get('status') == 'complete':
        return previous
    parameters = native_thread_parameters(PROJECT.parent)
    parameters['developerInstructions'] = PAIR_PROTOCOL
    definitions = dynamic_tools()
    definitions[-1]['description'] = 'Finish this report-only one-intervention comparison arm.'
    try:
        server.start()
        method = 'thread/resume' if previous.get('thread_id') else 'thread/start'
        request = {**parameters, 'threadId': previous['thread_id']} if method.endswith('resume') else {**parameters, 'dynamicTools': definitions}
        thread = server.request(method, request)['thread']['id']
        state = {'status': 'running', 'thread_id': thread, 'purpose': 'report_only'}
        write_json(state_path, state)
        prompt = PAIR_PROTOCOL + '\nFrozen comparison inputs:\n' + json.dumps(arm)
        prompt += '\nActive method:\n' + tools.guidance_text()
        while not tools.finished:
            server.turn(thread, prompt, environments=[{'environmentId': 'local', 'cwd': str(PROJECT.parent),
                                                       'runtimeWorkspaceRoots': [str(PROJECT.parent)]}])
            prompt = 'Continue the same report-only arm; attach to existing jobs and finish its first-candidate measurement.'
        state.update(status='complete', comparison_path=str(run / 'comparison.json'))
        write_json(state_path, state)
        return state
    finally:
        server.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/train_180_a0_v1.json')
    for name in ('task-state', 'parent-rollout', 'old-package', 'new-package', 'skills', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--context', type=Path)
    parser.add_argument('--round-number', type=int, default=1)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan_path = args.output / 'pair_plan.json'
    # Resume an already frozen comparison instead of overwriting input/program/skill snapshots.
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else prepare(
        args.config, args.task_state, args.parent_rollout, args.old_package, args.new_package,
        args.skills, args.context, args.output, args.round_number)
    if args.execute:
        plan['sessions'] = {name: execute_arm(arm) for name, arm in plan['arms'].items()}
        plan['executed'] = True
        plan['measurements'] = {name: json.loads((Path(arm['arm_dir']) / 'comparison.json').read_text())
                                for name, arm in plan['arms'].items()}
        old, new = (plan['measurements'][name].get('delta') for name in ('old', 'new'))
        plan['new_minus_old_task_gain'] = new - old if type(old) in (int, float) and type(new) in (int, float) else None
        plan['status'] = 'reported'
        write_json(args.output / 'pair_report.json', plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
