"""Same-parent, first-positive Task candidates using the existing execution stack."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

from sia.task_meta.durable import DurableUpdater, StageJournal, load_task, record_experience, task_hash, value_hash
from sia.task_meta.loop import BudgetManager, _accept_meta_update, _assert_single_component, _intervention_diff, _validate_decision
from sia.task_meta.meta_harness.five_stage import freeze_expectation, paired_outcome
from sia.task_meta.observations import build_observation
from sia.task_meta.storage import artifact_manifest, digest, save_json
from sia.task_meta.types import (DecisionConstraintError, GenerationContext, ImprovementExperience,
    MetaAgentState, MetaDecision, MetaObservation, TaskUpdate, TaskUpdateAction)
from sia.task_meta.updaters import updater_capabilities

POLICY = 'sequential_same_parent_first_positive_v1'
CONTRACT = (
    'Select one currently available, untried M/H/A component. Each candidate starts from the SAME frozen parent. '
    'At most one attempt per component per round. The controller evaluates the SAME feedback tasks and deploys '
    'only the FIRST strictly positive full-success gain; zero/negative candidates are not deployed. '
    'After rejection choose an untried component using all attempt feedback; no fixed order or parallel search. '
    'If none improves or resources end, retain the parent Task. All attempts, including unavailable or rejected '
    'ones, enter this round Meta learning. Every round fast update must change usable Meta Memory; facts, '
    'counterexamples and unresolved evidence are valid. Optional structural edits follow residual mismatch evidence. '
    'Candidate improvement is attributed to its generating Meta snapshot, not the subsequently updated Meta.')


def content_identity(task):
    return value_hash({'model': task.checkpoint_manifest, 'harness': digest(Path(task.harness_path)),
                       'artifacts': artifact_manifest(task.artifacts.directory)})


def positive_gain(outcome):
    if not outcome.get('paired_evidence_complete'):
        raise ValueError('Candidate cannot be accepted with missing or incomparable feedback pairs')
    delta = outcome['success_delta']
    if type(delta) not in (int, float):
        raise ValueError('Candidate feedback gain is unavailable')
    return delta > 0


def link_evidence(source, target, names):
    """Link immutable evidence only; never link mutable state or acceptance receipts."""
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        left, right = source / name, target / name
        if not left.is_file():
            raise ValueError('Missing paired source: ' + str(left))
        if right.exists():
            if digest(left) != digest(right):
                raise ValueError('Paired evidence changed during resume')
        else:
            os.link(left, right)


EVIDENCE = ('window.json', 'train_trajectories.jsonl', 'probe_trajectories.jsonl',
            'agent_execution.json', 'results.json', 'cost.json')


def write_evaluation(path, task, result):
    for name, value in [('task_state.json', asdict(task)), ('evaluated_state.json', asdict(task)),
                        ('results.json', result.performance), ('cost.json', result.cost),
                        ('agent_execution.json', result.trajectories)]:
        target = path / name
        if target.exists() and name in EVIDENCE:
            if json.loads(target.read_text()) != value:
                raise ValueError('Completed evaluation evidence changed: ' + str(target))
        else:
            save_json(target, value)


def make_experience(round_number, parent, after, baseline, result, decision, update, meta, root,
                    *, attempts=None, deployed=True):
    delta = result.performance['macro_success'] - baseline.performance['macro_success']
    action = decision.action.value if decision else 'NO_CHANGE'
    return ImprovementExperience(
        generation=round_number, state_before=asdict(parent), state_after=asdict(after),
        decision=decision.model_dump(mode='json') if decision else {}, modification=asdict(update) if update else {},
        performance_before=baseline.performance, performance_after=result.performance, performance_delta=delta,
        cost_before=baseline.cost, cost_after=result.cost, update_cost=update.cost if update else {},
        trajectory_before=str(root / f'gen_{round_number}/agent_execution.json'),
        trajectory_after=str(root / f'gen_{round_number + 1}/agent_execution.json'),
        experience_id=f'experience_{round_number}_{round_number + 1}',
        evaluated_state_before=asdict(parent), intervention_base_state=asdict(parent), evaluated_state_after=asdict(after),
        chosen_action=action if deployed else 'NO_CHANGE',
        requested_change=[c.model_dump(mode='json') for c in decision.requested_changes] if decision else [],
        actual_change=asdict(update) if deployed and update else {}, intervention_diff=_intervention_diff(parent, after),
        observed_performance_delta=delta,
        versions={'task_before': parent.version, 'task_after': after.version, 'meta_at_decision': meta.version,
                  'meta_bundle_hash_at_decision': meta.bundle_hash, 'probe_before': baseline.performance['probe_identity'],
                  'probe_after': result.performance['probe_identity']},
        candidate_attempts=attempts or [], feedback_root=str(root) if attempts is not None else None,
        deployment_status=('deployed' if deployed else 'parent_retained') if attempts is not None else None)


def run_sequential_task_meta(run_dir, task_state, meta_state, executor, meta_agent, updaters,
                            max_generations=5, primary_metric_name='macro_success', primary_metric_mode='max',
                            max_wall_time=None, *, resume=False, meta_update_handler=None, evaluated_prefix=None,
                            before_evaluation=None, after_round=None, round_protocol=None):
    if round_protocol and max_generations != 3: raise ValueError('Round protocol requires exactly three rounds')
    if primary_metric_name != 'macro_success' or primary_metric_mode != 'max' or evaluated_prefix:
        raise ValueError('Sequential policy requires fixed full-success feedback and a fresh round prefix')
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    budget = BudgetManager(max_generations, max_wall_time); budget.persist(root / 'budget.json', resume=resume)
    ledger = root / 'meta/experiences.jsonl'; ledger.parent.mkdir(exist_ok=True); ledger.touch()
    def evaluate(state, directory):
        if round_protocol:
            round_protocol.bind_execution(state, directory, round_protocol.store.mode)
        if before_evaluation and not (directory / 'execution_receipt.json').exists():
            before_evaluation(state)
        return executor.execute(copy.deepcopy(state), directory)
    task, meta = copy.deepcopy(task_state), copy.deepcopy(meta_state)
    history, scores, rounds = [], [], []
    client = meta_agent.client
    for number in range(max_generations):
        round_dir = root / f'round_{number}'; round_dir.mkdir(exist_ok=True)
        complete = round_dir / 'complete.json'
        if complete.exists():
            record = json.loads(complete.read_text())
            if record['input_content_hash'] != content_identity(task) or record['meta_before_hash'] != meta.bundle_hash:
                raise ValueError('Round resume input changed')
            task = load_task(record['task_after']); meta = MetaAgentState(**record['meta_after'])
            if content_identity(task) != record['output_content_hash']:
                raise ValueError('Committed round Task content changed')
            history.append(ImprovementExperience(**json.loads((round_dir / 'experience.json').read_text())))
            scores.append(record['score']); rounds.append(record)
            if after_round: after_round(number, record)
            continue
        if budget.stop_after(number):
            break
        if round_protocol:
            round_protocol.begin(number, meta, history)
            round_protocol.store.mode = 'parent_pre_update'
        task.generation = number  # Round index is distinct from accepted content identity/version.
        parent = copy.deepcopy(task); parent_hash = content_identity(parent); meta_before = copy.deepcopy(meta)
        baseline_dir = (round_dir / 'before' if round_protocol else root) / f'gen_{number}'
        baseline = evaluate(parent, baseline_dir)
        write_evaluation(baseline_dir, parent, baseline)
        save_json(baseline_dir / 'meta_state_before.json', meta)
        if round_protocol:
            meta_agent.round_protocol = round_protocol
            round_protocol.register_rows(baseline_dir/'agent_execution.json',number+1)
            round_protocol.guard()
        base_observation = None
        direct_submissions = meta_agent.capabilities.get('artifact_evaluation') == 'direct_submission'
        attempts, tried, chosen = [], set(), None
        last_decision = last_update = None
        for index in range(1 if round_protocol else 3):
            if budget.stop_after(number): break
            observation_path = round_dir / f'observation_{index}.json'
            if observation_path.exists():
                observation = MetaObservation(**json.loads(observation_path.read_text()))
            else:
                if base_observation is None:
                    capabilities = copy.deepcopy(meta_agent.capabilities)
                    capabilities.update(updater_capabilities(parent, baseline, True, sft_profile=capabilities.get('sft_profile', 'multidomain')))
                    for action in TaskUpdateAction:
                        if action not in updaters:
                            capabilities[action.value] = {'available': False, 'reason': 'Component excluded by registered experiment scope'}
                    if round_protocol and TaskUpdateAction.MODEL in updaters:
                        capabilities['MODEL']['available'] = True
                        capabilities['MODEL']['reason'] = 'Use only frozen pre-decision parent B_r successes; no additional SFT collection'
                    base_observation = build_observation(parent, parent, meta, baseline, [] if round_protocol else scores, history, [], capabilities, retain_raw=True)
                    if direct_submissions:
                        from sia.task_meta.submissions import CONTRACT as SUBMISSION_CONTRACT, baseline_outputs, submission
                        submission_rows = baseline_outputs(baseline_dir)
                        base_observation.available_actions['ARTIFACTS'] = {
                            'available': bool(submission_rows), 'evaluation_mode': 'direct_submission',
                            'operations': [{'operation': 'write_asset', 'targets': sorted(submission_rows)}],
                            'submission_targets': sorted(submission_rows),
                            'submission_identity': [{'target': name, 'task_id': row['task_id'], 'split': row['split'],
                                'domain': row['domain'], 'success': row.get('verification', {}).get('success')}
                                for name, row in submission_rows.items()], 'constraints': [SUBMISSION_CONTRACT]}
                        for name, row in submission_rows.items():
                            base_observation.current_artifact_files['assets/' + name] = json.dumps(submission(row), ensure_ascii=False)
                    base_observation.budget.update(policy='legal_single_update_then_measure' if round_protocol else POLICY, candidate_limit=1 if round_protocol else 3, max_generations=max_generations,
                                                   round=number, global_deadline=budget.wall_started + max_wall_time if max_wall_time else None)
                observation = copy.deepcopy(base_observation)
                observation.budget.update(candidate_index=index, attempted_components=sorted(tried), immutable_task_policy=('Choose one legal untried component from the frozen parent. MODEL uses only existing parent B_r successful trajectories. A legal update is saved before child evaluation, without a positive-gain selection gate. Only one component decision is permitted; unavailable MODEL is skipped without reselection. Run the child once on the SAME full B_r; then update Meta harness from measured before/after experience. External validation never enters optimization.' if round_protocol else CONTRACT))
                for action in tried:
                    observation.available_actions[action].update(available=False, reason='Already attempted this round')
                save_json(observation_path, observation)
            if not any(v.get('available') for v in observation.available_actions.values()): break
            client.bind_context(root.name, number, task_hash(parent))
            imported_path = root / 'initial_candidate.json'
            imported = json.loads(imported_path.read_text()) if number == index == 0 and imported_path.exists() else None
            recorded = [json.loads(p.read_text()) for p in (round_dir / 'candidates').glob('*/attempt.json')]
            recorded = [a for a in recorded if a.get('decision', {}).get('decision_id') == f'generation_{number}_decision_{index}']
            if len(recorded) > 1:
                raise ValueError('Multiple candidates claim the same sequential decision')
            if imported:
                source_receipt = Path(imported['receipt'])
                if digest(source_receipt) != imported['receipt_sha256']:
                    raise ValueError('Imported real candidate receipt changed')
                source = json.loads(source_receipt.read_text())
                if source['status'] != 'committed' or source['binding']['meta']['bundle_hash'] != imported['generating_meta_hash']:
                    raise ValueError('Imported candidate generating Meta identity changed')
                decision = MetaDecision.model_validate(source['binding']['decision'])
            elif recorded:
                prior = recorded[0]
                if prior['parent_content_hash'] != parent_hash or prior['generating_G']['bundle_hash'] != meta.bundle_hash:
                    raise ValueError('Recovered candidate has a different parent or generating Meta')
                decision = MetaDecision.model_validate(prior['decision'])
            else:
                decision = MetaDecision.model_validate(meta_agent.diagnose_and_route(meta, observation, feedback=attempts))
            decision.decision_id = f'generation_{number}_decision_{index}'
            _validate_decision(decision)
            if round_protocol: round_protocol.decision(decision,round_dir/f'decision_{index}')
            action = decision.action.value
            if action in tried or not observation.available_actions[action].get('available'):
                raise DecisionConstraintError('Meta chose an unavailable or already attempted component')
            tried.add(action); last_decision = decision
            candidate_root = round_dir / 'candidates' / action
            before_dir, candidate_dir = candidate_root / f'gen_{number}', candidate_root / f'gen_{number + 1}'
            link_evidence(baseline_dir, before_dir, EVIDENCE)
            link_evidence(root, candidate_root, ('protocol.json',))
            context = GenerationContext(number + 1, candidate_dir, observation, baseline, copy.deepcopy(meta))
            attempt_path = candidate_root / 'attempt.json'
            if attempt_path.exists():
                attempt = json.loads(attempt_path.read_text())
                if attempt['parent_content_hash'] != parent_hash or attempt['decision'] != decision.model_dump(mode='json'):
                    raise ValueError('Candidate replay input changed')
            else:
                if imported:
                    expectation_path = Path(imported['expectation'])
                    if digest(expectation_path) != imported['expectation_sha256']:
                        raise ValueError('Imported prospective expectation changed')
                    link_evidence(expectation_path.parent, before_dir, (expectation_path.name,))
                    expectation = json.loads(expectation_path.read_text())
                else:
                    expectation = freeze_expectation(candidate_root, parent, meta, decision, observation)
                attempt = {'attempt_id': f'round_{number}_{action}', 'parent_content_hash': parent_hash,
                    'action': action, 'decision': decision.model_dump(mode='json'), 'expectation': expectation,
                    'generating_G': source['binding']['meta'] if imported else asdict(meta),
                    'imported_prior_candidate': bool(imported), 'status': 'prepared', 'deployed': False}
                save_json(attempt_path, attempt)
            if attempt['status'] not in {'evaluated', 'unavailable'}:
                try:
                    if imported:
                        candidate = load_task(source['successor'])
                        update_data = dict(source['update']); update_data['action'] = TaskUpdateAction(update_data['action'])
                        update = TaskUpdate(**update_data)
                        candidate_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        if round_protocol and decision.action == TaskUpdateAction.MODEL:
                            round_protocol.model_context(parent,context,evaluate)
                        updater = DurableUpdater(updaters[decision.action].updater, StageJournal(candidate_root))
                        candidate, update = updater.apply(copy.deepcopy(parent), decision, context)
                    if round_protocol: round_protocol.check_task_edit(parent,candidate)
                    _assert_single_component(parent, candidate, decision.action, digest(Path(parent.harness_path)), artifact_manifest(parent.artifacts.directory))
                except DecisionConstraintError as exc:
                    attempt.update(status='unavailable', reason=str(exc), candidate_executed=False)
                    save_json(attempt_path, attempt)
                else:
                    if content_identity(parent) != parent_hash:
                        raise ValueError('Candidate mutated the frozen parent')
                    if round_protocol: round_protocol.select_child(parent,candidate,decision,round_dir)
                    if direct_submissions and decision.action == TaskUpdateAction.ARTIFACTS:
                        from sia.task_meta.submissions import SubmissionEvaluator
                        from sia.task_meta.durable import DurableExecutor
                        evaluator = SubmissionEvaluator(executor.executor, baseline_dir,
                            [c.target for c in decision.requested_changes])
                        result = DurableExecutor(evaluator, executor.journal).execute(candidate, candidate_dir)
                    else:
                        result = evaluate(candidate, candidate_dir)
                    write_evaluation(candidate_dir, candidate, result)
                    candidate_meta = MetaAgentState(**attempt['generating_G'])
                    experience = make_experience(number, parent, candidate, baseline, result, decision, update, candidate_meta, candidate_root)
                    outcome = paired_outcome(candidate_root, experience)
                    accepted = positive_gain(outcome)
                    attempt.update(status='evaluated', candidate_executed=True, candidate_state=asdict(candidate),
                        update=asdict(update), performance=result.performance, cost=result.cost,
                        trajectory_after=str(candidate_dir / 'agent_execution.json'),
                        paired_outcome=outcome, gain=outcome['success_delta'], positive_gain=accepted, accepted_for_deployment=True if round_protocol else accepted)
                    save_json(attempt_path, attempt)
            attempts.append(attempt)
            print('[candidate-result] ' + json.dumps({'round': number, 'component': action,
                'status': attempt['status'], 'before': baseline.performance['domains']['tool_use'],
                'after': attempt.get('performance', {}).get('domains', {}).get('tool_use'),
                'gain': attempt.get('gain'), 'accepted': attempt.get('accepted_for_deployment',attempt.get('positive_gain', False)),
                'generating_G': attempt['generating_G']['version']}), flush=True)
            if attempt.get('accepted_for_deployment',attempt.get('positive_gain')):
                chosen = attempt; break
            if content_identity(parent) != parent_hash:
                raise ValueError('Rejected candidate mutated the frozen parent')
        after = load_task(chosen['candidate_state']) if chosen else copy.deepcopy(parent)
        after.generation = number + 1
        feedback_root = round_dir / 'feedback'
        link_evidence(root, feedback_root, ('protocol.json',))
        link_evidence(baseline_dir, feedback_root / f'gen_{number}', EVIDENCE)
        retained_result = None
        if round_protocol and not chosen:
            retained_dir = round_dir/'after'/f'gen_{number+1}'
            round_protocol.select_child(parent,after,last_decision,round_dir)
            retained_result = evaluate(after,retained_dir)
            write_evaluation(retained_dir,after,retained_result)
        result_dir = Path(chosen['trajectory_after']).parent if chosen else retained_dir if retained_result else baseline_dir
        link_evidence(result_dir, feedback_root / f'gen_{number + 1}', EVIDENCE)
        from sia.task_meta.types import EvaluationResult
        outcome_result = EvaluationResult(chosen['performance'], [], chosen['cost']) if chosen else retained_result or baseline
        if chosen:
            last_decision = MetaDecision.model_validate(chosen['decision'])
            item = dict(chosen['update']); item['action'] = TaskUpdateAction(item['action']); last_update = TaskUpdate(**item)
        deployment = {'status': 'deployed' if chosen else 'parent_retained', 'round': number,
            'parent_content_hash': parent_hash, 'task_after': asdict(after), 'output_content_hash': content_identity(after),
            'chosen_component': chosen['action'] if chosen else None, 'attempted_components': sorted(tried),
            'rule': 'legal_single_update_then_measure' if round_protocol else POLICY, 'gain': chosen['gain'] if chosen else 0.0}
        save_json(round_dir / 'deployment.json', deployment)
        experience = make_experience(number, parent, after, baseline, outcome_result, last_decision, last_update,
            meta_before, feedback_root, attempts=attempts, deployed=chosen is not None)
        experience.update_cost = {'candidate_update_costs': [a.get('update', {}).get('cost') for a in attempts],
                                  'candidate_evaluation_costs': [a.get('cost') for a in attempts]}
        save_json(round_dir / 'experience.json', experience)
        history.append(experience); record_experience(ledger, experience)
        client.bind_context(root.name, number + 1, task_hash(after))
        if round_protocol:
            round_protocol.completed(None,round_dir,baseline,outcome_result)
            round_protocol.prepare_meta(meta_agent,experience,round_dir)
        learned = meta_agent.learn_from_experience(meta, experience, history, after)
        if round_protocol: round_protocol.validate_meta(learned,meta)
        meta = _accept_meta_update(root, meta, learned, [experience], round_dir / 'meta_self_update.json',
                                   'learn_from_experience', meta_update_handler, resume)
        record = {**deployment, 'input_content_hash': parent_hash, 'meta_before_hash': meta_before.bundle_hash,
                  'meta_after': asdict(meta), 'score': {'generation': number, **outcome_result.performance},
                  'meta_content_changed': meta.bundle_hash != meta_before.bundle_hash,
                  'next_use': 'pending_next_round' if number + 1 < max_generations else 'no_later_round_in_this_run'}
        if round_protocol: round_protocol.freeze_system(record,round_dir)
        save_json(complete, record); scores.append(record['score']); rounds.append(record); task = after
        if after_round: after_round(number, record)
        print('[round-complete] ' + json.dumps({'round': number, 'deployment': deployment['status'],
            'component': deployment['chosen_component'], 'probe': outcome_result.performance['domains']['tool_use'],
            'G_before': meta_before.version, 'G_after': meta.version, 'all_attempts': len(attempts)}), flush=True)
    final = {'status': 'completed' if len(rounds) == max_generations else 'budget_exhausted',
        'task_state': asdict(task), 'meta_state': asdict(meta), 'rounds_completed': len(rounds),
        'generations_executed': len(rounds), 'experiences': len(history), 'performance_history': scores,
        'primary_metric': primary_metric_name, 'primary_metric_mode': 'max', 'task_update_policy': 'legal_single_update_then_measure' if round_protocol else POLICY,
        'rounds': rounds, 'wall_time_seconds': budget.elapsed(), 'decision_mode': 'autonomous',
        'final_consolidation_status': ('third_round_meta_update_completed_no_fourth' if round_protocol else 'last_round_memory_update_completed') if len(rounds) == max_generations else 'pending'}
    save_json(root / 'final_state.json', final)
    return final
