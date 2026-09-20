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
from sia.task_meta.observations import build_observation, round_context
from sia.task_meta.storage import artifact_manifest, digest, save_json
from sia.task_meta.types import (DecisionConstraintError, GenerationContext, ImprovementExperience,
    MetaAgentState, MetaDecision, MetaObservation, TaskUpdate, TaskUpdateAction)
from sia.task_meta.updaters import updater_capabilities

POLICY = 'single_candidate_strict_positive_gain_v1'
CONTRACT = (
    'Select one currently available M/H/A component and produce exactly one candidate from the frozen parent. '
    'The controller evaluates the SAME complete feedback tasks and deploys only a strictly positive full-success gain; '
    'zero/negative/incomparable candidates are not deployed and never trigger a second component or HARNESS candidate. '
    'On rejection retain the parent Task. The attempt, including unavailable or rejected outcomes, enters this round '
    'Meta learning. Every round fast update must change usable Meta Memory; facts, '
    'counterexamples and unresolved evidence are valid. Optional structural edits follow residual mismatch evidence. '
    'Candidate improvement is attributed to its generating Meta snapshot, not the subsequently updated Meta.')


def content_identity(task):
    return value_hash({'model': task.checkpoint_manifest, 'harness': digest(Path(task.harness_path)),
                       'artifacts': artifact_manifest(task.artifacts.directory)})


def positive_gain(outcome):
    """Only complete, numeric, strictly positive paired evidence is deployable."""
    if not outcome.get('paired_evidence_complete'):
        return False
    delta = outcome.get('success_delta')
    return type(delta) in (int, float) and delta > 0


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
PAUSE_PENDING = 'pause_after_round.pending.json'
PAUSE_MARKER = 'pause_after_round.json'
PAUSE_CONSUMED = 'pause_after_round.consumed.json'
PAUSE_INTENT_FIELDS = {
    'schema_version', 'status', 'pause_after_round', 'round_index', 'round_complete',
    'round_complete_sha256', 'task_content_hash', 'meta_bundle_hash',
}


def _sha256_hex(value):
    return (type(value) is str and len(value) == 64 and value == value.lower()
            and all(character in '0123456789abcdef' for character in value))


def _reject_pause_paths(*paths):
    if any(path.is_symlink() for path in paths):
        raise ValueError('Durable pause evidence cannot be a symlink')
    if any(path.exists() and not path.is_file() for path in paths):
        raise ValueError('Durable pause evidence must be a regular file')


def _pause_intent(root, pause_after_round, complete, task, meta):
    value = {
        'schema_version': 1,
        'status': 'pending_after_round_validation',
        'pause_after_round': pause_after_round,
        'round_index': pause_after_round - 1,
        'round_complete': complete.relative_to(root).as_posix(),
        'round_complete_sha256': digest(complete),
        'task_content_hash': content_identity(task),
        'meta_bundle_hash': meta.bundle_hash,
    }
    path = root / PAUSE_PENDING
    _reject_pause_paths(path)
    if (type(pause_after_round) is not int or pause_after_round != 1
            or not _sha256_hex(value['round_complete_sha256'])
            or not _sha256_hex(value['task_content_hash'])
            or not _sha256_hex(value['meta_bundle_hash'])):
        raise ValueError('Durable pause intent contains invalid boundary or hashes')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Durable pause intent belongs to a different completed round')
    else:
        save_json(path, value)
    return value


def _pause_marker_value(root, intent):
    pending = root / PAUSE_PENDING
    return {
        'schema_version': 1,
        'status': 'paused_at_round_boundary',
        'pause_after_round': intent['pause_after_round'],
        'round_index': intent['round_index'],
        'round_complete': intent['round_complete'],
        'round_complete_sha256': intent['round_complete_sha256'],
        'task_content_hash': intent['task_content_hash'],
        'meta_bundle_hash': intent['meta_bundle_hash'],
        'after_round_validation_completed': True,
        'pending_sha256': digest(pending),
        'resume_starts_at_round': intent['pause_after_round'] + 1,
    }


def _finish_pause(root, intent):
    pending = root / PAUSE_PENDING
    path = root / PAUSE_MARKER
    _reject_pause_paths(pending, path)
    value = _pause_marker_value(root, intent)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Durable pause marker changed')
    else:
        save_json(path, value)
    return value


def _pause_state(root, pause_after_round, resume):
    pending_path = root / PAUSE_PENDING
    marker_path = root / PAUSE_MARKER
    consumed_path = root / PAUSE_CONSUMED
    _reject_pause_paths(pending_path, marker_path, consumed_path)
    if pause_after_round is None:
        if pending_path.exists() or marker_path.exists() or consumed_path.exists():
            raise ValueError('Durable pause evidence exists but pause_after_round is disabled')
        return None, None
    if type(pause_after_round) is not int or pause_after_round != 1:
        raise ValueError('pause_after_round currently supports only the B1 boundary')
    intent = json.loads(pending_path.read_text(encoding='utf-8')) if pending_path.exists() else None
    marker = json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else None
    for value in (intent, marker):
        if value is not None and type(value) is not dict:
            raise ValueError('Durable pause evidence must be a JSON object')
        if value is not None and value.get('pause_after_round') != pause_after_round:
            raise ValueError('Durable pause evidence belongs to a different boundary')
    if intent is not None:
        expected_complete = f'round_{pause_after_round - 1}/complete.json'
        complete = root / expected_complete
        if (set(intent) != PAUSE_INTENT_FIELDS or type(intent.get('schema_version')) is not int
                or intent.get('schema_version') != 1
                or type(intent.get('pause_after_round')) is not int
                or type(intent.get('round_index')) is not int
                or intent.get('round_index') != pause_after_round - 1
                or intent.get('round_complete') != expected_complete
                or intent.get('status') != 'pending_after_round_validation' or not complete.is_file()
                or digest(complete) != intent.get('round_complete_sha256')
                or not _sha256_hex(intent.get('round_complete_sha256'))
                or not _sha256_hex(intent.get('task_content_hash'))
                or not _sha256_hex(intent.get('meta_bundle_hash'))):
            raise ValueError('Durable pause intent failed round-complete verification')
    if marker is not None:
        expected_marker = _pause_marker_value(root, intent) if intent is not None else None
        if (intent is None or type(marker.get('schema_version')) is not int
                or type(marker.get('pause_after_round')) is not int
                or type(marker.get('round_index')) is not int
                or type(marker.get('resume_starts_at_round')) is not int
                or marker.get('after_round_validation_completed') is not True
                or marker != expected_marker):
            raise ValueError('Durable pause marker failed integrity verification')
        if not resume:
            raise ValueError('A paused run must be resumed explicitly')
        consumed = {
            'schema_version': 1,
            'status': 'pause_consumed_for_resume',
            'pause_after_round': pause_after_round,
            'pause_marker_sha256': digest(marker_path),
            'resume_starts_at_round': pause_after_round + 1,
        }
        if consumed_path.exists():
            actual_consumed = json.loads(consumed_path.read_text(encoding='utf-8'))
            if (type(actual_consumed) is not dict
                    or type(actual_consumed.get('schema_version')) is not int
                    or type(actual_consumed.get('pause_after_round')) is not int
                    or type(actual_consumed.get('resume_starts_at_round')) is not int
                    or not _sha256_hex(actual_consumed.get('pause_marker_sha256'))
                    or actual_consumed != consumed):
                raise ValueError('Durable pause-consumption marker changed')
        else:
            _reject_pause_paths(consumed_path)
            save_json(consumed_path, consumed)
        return 'consumed', marker
    if consumed_path.exists():
        raise ValueError('Pause consumption exists without a completed pause marker')
    return ('pending', intent) if intent is not None else (None, None)



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
    deployed_delta = result.performance['macro_success'] - baseline.performance['macro_success']
    action = decision.action.value if decision else 'NO_CHANGE'
    attempt = attempts[-1] if attempts else None
    attempted_state = attempt.get('candidate_state', {}) if attempt else {}
    attempted_candidate = load_task(attempted_state) if attempted_state else None
    attempted_delta = attempt.get('gain') if attempt else deployed_delta
    attempted_performance = attempt.get('performance', {}) if attempt else result.performance
    attempted_cost = attempt.get('cost', {}) if attempt else result.cost
    attempted_trajectory = attempt.get('trajectory_after') if attempt else str(root / f'gen_{round_number + 1}/agent_execution.json')
    attempted_outcome = attempt.get('paired_outcome', {}) if attempt else {}
    attempted_diff = _intervention_diff(parent, attempted_candidate) if attempted_candidate else {}
    return ImprovementExperience(
        generation=round_number, state_before=asdict(parent), state_after=asdict(after),
        decision=decision.model_dump(mode='json') if decision else {},
        modification=asdict(update) if update else {},
        performance_before=baseline.performance, performance_after=attempted_performance,
        performance_delta=attempted_delta,
        cost_before=baseline.cost, cost_after=attempted_cost, update_cost=update.cost if update else {},
        trajectory_before=str(root / f'gen_{round_number}/agent_execution.json'),
        trajectory_after=attempted_trajectory,
        experience_id=f'experience_{round_number}_{round_number + 1}',
        evaluated_state_before=asdict(parent), intervention_base_state=asdict(parent), evaluated_state_after=asdict(after),
        chosen_action=action,
        requested_change=[c.model_dump(mode='json') for c in decision.requested_changes] if decision else [],
        actual_change=asdict(update) if update else {}, intervention_diff=_intervention_diff(parent, after),
        observed_performance_delta=attempted_delta, attempted_component=action,
        attempted_state_after=attempted_state, attempted_intervention_diff=attempted_diff,
        attempted_performance_after=attempted_performance, attempted_performance_delta=attempted_delta,
        attempted_cost_after=attempted_cost, attempted_trajectory_after=attempted_trajectory,
        attempted_outcome=attempted_outcome,
        versions={'task_before': parent.version, 'task_after': after.version, 'meta_at_decision': meta.version,
                  'meta_bundle_hash_at_decision': meta.bundle_hash, 'probe_before': baseline.performance['probe_identity'],
                  'probe_after': result.performance['probe_identity']},
        candidate_attempts=attempts or [], feedback_root=str(root) if attempts is not None else None,
        deployment_status=('deployed' if deployed else 'parent_retained') if attempts is not None else None)


def run_sequential_task_meta(run_dir, task_state, meta_state, executor, meta_agent, updaters,
                            max_generations=5, primary_metric_name='macro_success', primary_metric_mode='max',
                            max_wall_time=None, *, resume=False, meta_update_handler=None,
                            before_evaluation=None, after_round=None, round_protocol=None):
    if round_protocol and max_generations != round_protocol.total_stages: raise ValueError('Execution stages must match the frozen three-round schedule')
    if primary_metric_name != 'macro_success' or primary_metric_mode != 'max':
        raise ValueError('Sequential policy requires the fixed full-success metric')
    positive_search = bool(round_protocol and round_protocol.positive_search)
    pause_after_round = getattr(round_protocol, 'pause_after_round', None) if round_protocol else None
    if pause_after_round is not None and max_generations <= pause_after_round:
        raise ValueError('pause_after_round cannot replace or shorten the frozen generation budget')
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    budget = BudgetManager(max_generations, max_wall_time); budget.persist(root / 'budget.json', resume=resume)
    pause_state, pause_record = _pause_state(root, pause_after_round, resume)
    paused_at_round = None
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
            from sia.task_meta.round_checkpoint import commit_round_checkpoint
            commit_round_checkpoint(round_dir, number, task, meta)
            history.append(ImprovementExperience(**json.loads((round_dir / 'experience.json').read_text())))
            scores.append(record['score']); rounds.append(record)
            if pause_record and pause_record['pause_after_round'] == number + 1:
                if (pause_record['task_content_hash'] != content_identity(task)
                        or pause_record['meta_bundle_hash'] != meta.bundle_hash):
                    raise ValueError('Durable pause marker does not match the committed Task/Meta boundary')
            if pause_state == 'pending' and pause_after_round == number + 1:
                if after_round:
                    after_round(number, record)
                pause_record = _finish_pause(root, pause_record)
                paused_at_round = pause_after_round
                print('[round-paused] ' + json.dumps({'pause_after_round': paused_at_round,
                    'resume_starts_at_round': paused_at_round + 1}), flush=True)
                break
            if after_round and not (pause_state == 'consumed' and pause_after_round == number + 1):
                after_round(number, record)
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
        direct_submissions = (meta_agent.capabilities.get('artifact_evaluation') == 'direct_submission'
                              and TaskUpdateAction.ARTIFACTS in updaters)
        attempts, tried, chosen = [], set(), None
        last_decision = last_update = None
        for index in range(1):
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
                    base_observation.budget.update(policy=POLICY, candidate_limit=round_protocol.candidate_limit if round_protocol else 1, max_generations=max_generations,
                                                   round=number, global_deadline=budget.wall_started + max_wall_time if max_wall_time else None)
                observation = copy.deepcopy(base_observation)
                observation.budget.update(candidate_index=index, attempted_components=sorted(tried), immutable_task_policy=CONTRACT)
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
                    'imported_prior_candidate': bool(imported), 'status': 'prepared'}
                save_json(attempt_path, attempt)
            if attempt['status'] not in {'evaluated', 'unavailable'}:
                try:
                    if imported:
                        candidate = load_task(source['successor'])
                        update_data = dict(source['update']); update_data['action'] = TaskUpdateAction(update_data['action'])
                        update = TaskUpdate(**update_data)
                        candidate_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        if round_protocol and round_protocol.aligned_interventions:
                            from sia.task_meta.intervention_evidence import branch_binding
                            branch_binding(parent,decision,context)
                        if round_protocol and decision.action == TaskUpdateAction.MODEL:
                            round_protocol.model_context(parent,context,evaluate)
                        updater = DurableUpdater(updaters[decision.action].updater, StageJournal(candidate_root))
                        candidate, update = updater.apply(copy.deepcopy(parent), decision, context)
                    if round_protocol: round_protocol.check_task_edit(parent,candidate)
                    _assert_single_component(parent, candidate, decision.action, digest(Path(parent.harness_path)), artifact_manifest(parent.artifacts.directory))
                except DecisionConstraintError as exc:
                    attempt.update(status='unavailable', reason=str(exc), candidate_executed=False,
                                   accepted_for_deployment=False, outcome_class='candidate_unavailable')
                    save_json(attempt_path, attempt)
                else:
                    if content_identity(parent) != parent_hash:
                        raise ValueError('Candidate mutated the frozen parent')
                    if round_protocol: round_protocol.select_child(parent,candidate,decision,candidate_root if positive_search else round_dir)
                    if direct_submissions and decision.action == TaskUpdateAction.ARTIFACTS:
                        if round_protocol: round_protocol.bind_execution(candidate,candidate_dir,'child_post_update')
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
                    raw_delta = outcome.get('success_delta')
                    comparable = bool(outcome.get('paired_evidence_complete')) and type(raw_delta) in (int, float)
                    delta = raw_delta if comparable else None
                    outcome_class = ('incomplete_comparison' if not comparable else
                                     'accepted_positive_gain' if accepted else
                                     'zero_gain' if delta == 0 else 'negative_gain')
                    attempt.update(status='evaluated', candidate_executed=True, candidate_state=asdict(candidate),
                        update=asdict(update), performance=result.performance, cost=result.cost,
                        trajectory_after=str(candidate_dir / 'agent_execution.json'),
                        paired_outcome=outcome, gain=delta, positive_gain=accepted,
                        accepted_for_deployment=accepted, outcome_class=outcome_class)
                    save_json(attempt_path, attempt)
            if positive_search: round_protocol.observe_attempt(attempt,candidate_root)
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
        if round_protocol and not positive_search and not chosen:
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
        elif attempts and attempts[-1].get('update'):
            item = dict(attempts[-1]['update']); item['action'] = TaskUpdateAction(item['action']); last_update = TaskUpdate(**item)
        if positive_search: round_protocol.commit_selection(parent,after,chosen,round_dir)
        deployment = {'status': 'deployed' if chosen else 'parent_retained', 'round': number,
            'parent_content_hash': parent_hash, 'task_after': asdict(after), 'output_content_hash': content_identity(after),
            'chosen_component': chosen['action'] if chosen else None, 'attempted_components': sorted(tried),
            'rule': POLICY, 'gain': (chosen or (attempts[-1] if attempts else {})).get('gain')}
        save_json(round_dir / 'deployment.json', deployment)
        experience = make_experience(number, parent, after, baseline, outcome_result, last_decision, last_update,
            meta_before, feedback_root, attempts=attempts, deployed=chosen is not None)
        experience.update_cost = {'candidate_update_costs': [a.get('update', {}).get('cost') for a in attempts],
                                  'candidate_evaluation_costs': [a.get('cost') for a in attempts]}
        save_json(round_dir / 'experience.json', experience)
        save_json(round_dir / 'meta_context.json', round_context(experience))
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
        save_json(complete, record)
        from sia.task_meta.round_checkpoint import commit_round_checkpoint
        commit_round_checkpoint(round_dir, number, after, meta)
        scores.append(record['score']); rounds.append(record); task = after
        pause_intent = (_pause_intent(root, pause_after_round, complete, task, meta)
                        if pause_after_round == number + 1 else None)
        if after_round:
            after_round(number, record)
        if pause_intent:
            pause_record = _finish_pause(root, pause_intent)
            paused_at_round = pause_after_round
        print('[round-complete] ' + json.dumps({'round': number, 'deployment': deployment['status'],
            'component': deployment['chosen_component'], 'probe': outcome_result.performance['domains']['tool_use'],
            'G_before': meta_before.version, 'G_after': meta.version, 'all_attempts': len(attempts)}), flush=True)
        if paused_at_round:
            print('[round-paused] ' + json.dumps({'pause_after_round': paused_at_round,
                'resume_starts_at_round': paused_at_round + 1}), flush=True)
            break
    run_status = ('paused' if paused_at_round else
                  'completed' if len(rounds) == max_generations else 'budget_exhausted')
    final = {'status': run_status,
        'task_state': asdict(task), 'meta_state': asdict(meta), 'rounds_completed': len(rounds),
        'generations_executed': len(rounds), 'experiences': len(history), 'performance_history': scores,
        'primary_metric': primary_metric_name, 'primary_metric_mode': 'max', 'task_update_policy': POLICY,
        'rounds': rounds, 'wall_time_seconds': budget.elapsed(), 'decision_mode': 'autonomous',
        'pause_after_round': pause_after_round, 'paused_at_round': paused_at_round,
        'pause_marker': str(root / PAUSE_MARKER) if paused_at_round else None,
        'final_consolidation_status': ('third_round_meta_update_completed_no_fourth' if round_protocol else 'last_round_memory_update_completed') if len(rounds) == max_generations else 'pending'}
    if round_protocol and round_protocol.store.sequential_domains:
        final['domain_stages_completed']=len(rounds)
        final['outer_rounds_completed']=len(rounds)//3
        final['rounds_completed']=len(rounds)//3
        final['final_consolidation_status']='nine_domain_updates_completed_no_tenth' if len(rounds)==9 else 'pending'
    save_json(root / 'final_state.json', final)
    return final
