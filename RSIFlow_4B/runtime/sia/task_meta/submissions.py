"""Meta edits task outputs; the trusted evaluator resubmits without Task inference.

These task-scoped outputs are not reusable memories or Qwen SFT supervision.
EnvScaler outputs are finite tool action sequences, never environment snapshots.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from sia.task_meta.durable import task_hash, value_hash
from sia.task_meta.storage import artifact_manifest, clone_task, save_json
from sia.task_meta.types import ArtifactState, DecisionConstraintError, EvaluationResult, TaskUpdate, TaskUpdateAction, UpdatePending


CONTRACT = (
    'ARTIFACTS means directly revising existing task submissions, not reusable advice. '
    'Choose exact submissions/train|probe/<id>.json paths from submission_targets. '
    'Only write_asset is allowed. Preserve the frozen task/model/harness, task identity, '
    'reset seed, tool schemas, hidden checks and external budgets. Code/Search outputs '
    'are final_answer strings; EnvScaler outputs are bounded lists of {name,arguments} '
    'tool actions replayed in a fresh identical sandbox. No Task model is called. '
    'Unedited submissions keep their recorded baseline result. No observations, state '
    'snapshots, rewards or hidden answers may be supplied in a submission. '
    'These scores measure Meta-assisted output repair, not standalone Task policy improvement. '
    'Probe outputs never become training data, reusable assets or inputs to later tasks.'
)


def baseline_outputs(directory):
    outputs = {}
    # RoundStore mirrors the SAME tasks in probe; submit and account once.
    train=Path(directory)/'train_trajectories.jsonl'
    with train.open(encoding='utf-8') as stream:
        first=next((json.loads(line) for line in stream if line.strip()),{})
    round_scoped=first.get('purpose')=='evolution_train'
    for split in (('train',) if round_scoped else ('train','probe')):
        path = Path(directory) / (split + '_trajectories.jsonl')
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = hashlib.sha256(f'{row["task_id"]}:{row["rollout_id"]}'.encode()).hexdigest()
            name = f'submissions/{split}/{identifier}.json'
            if name in outputs:
                raise ValueError('Duplicate submission identity')
            outputs[name] = row
    return outputs


def submission(row):
    return {'final_answer': row.get('final_answer') or '',
            'actions': [{'name': c['name'], 'arguments': c['arguments']}
                        for c in row.get('tool_calls', [])] if row['domain'] == 'tool_use' else []}


def baseline_tool_schemas(row):
    tools = [tool for call in row.get('transport_calls', []) for tool in call.get('tools') or []]
    if not tools:
        # The Task JSON-action harness embeds reset input in its user prompt,
        # rather than passing OpenAI's native tools field. Only trust the exact
        # controller-fingerprinted reset object, never arbitrary prompt JSON.
        decoder = json.JSONDecoder()
        calls = row.get('transport_calls') or []
        for message in calls[0].get('messages', []) if calls else []:
            if message.get('role') != 'user' or not isinstance(message.get('content'), str):
                continue
            text = message['content']
            for offset, char in enumerate(text):
                if char != '{':
                    continue
                try:
                    public, _ = decoder.raw_decode(text, offset)
                except ValueError:
                    continue
                if isinstance(public, dict) and value_hash(public) == row.get('reset_hash'):
                    tools = public.get('environment_tools') or []
                    break
            if tools:
                break
    return {tool.get('function', tool)['name']: tool.get('function', tool).get('parameters', {}) for tool in tools}


def validate_submission(value, row, max_tools):
    if not isinstance(value, dict) or set(value) != {'final_answer', 'actions'}:
        raise ValueError('Submission must contain only final_answer and actions')
    if not isinstance(value['final_answer'], str) or len(value['final_answer']) > 12000:
        raise ValueError('Submission answer exceeds fixed text limit')
    actions = value['actions']
    if not isinstance(actions, list) or len(actions) > max_tools:
        raise ValueError('Submission exceeds original tool-call limit')
    if row['domain'] != 'tool_use' and actions:
        raise ValueError('Code/Search submissions cannot execute additional tools')
    if any(not isinstance(c, dict) or set(c) != {'name', 'arguments'} or
           not isinstance(c['name'], str) or not isinstance(c['arguments'], dict) for c in actions):
        raise ValueError('Only named tool actions with JSON argument objects are allowed')
    if len(json.dumps(value, ensure_ascii=False)) > 120000:
        raise ValueError('Submission exceeds fixed serialized size limit')
    # Check against the actual task-visible schemas recorded by the Task client.
    schemas = baseline_tool_schemas(row)
    if actions:
        import jsonschema
        for action in actions:
            if action['name'] not in schemas:
                raise ValueError('Submission names a tool not exposed in the baseline')
            try:
                jsonschema.validate(action['arguments'], schemas[action['name']])
            except jsonschema.ValidationError as exc:
                raise ValueError('Action arguments violate the task-visible tool schema') from exc
    return value


class SubmissionUpdater:
    def __init__(self, client):
        self.client = client

    def apply(self, task_state, decision, context):
        from sia.task_meta.meta import evolution_kwargs
        from sia.task_meta.updaters import ArtifactChanges, _audit, validate_decision_targets
        started = time.monotonic()
        requested = validate_decision_targets(decision, TaskUpdateAction.ARTIFACTS)
        before = context.directory.parent / f'gen_{task_state.generation}'
        rows = baseline_outputs(before)
        requested_rows=[rows.get(c.target) for c in requested]
        if any(row is None or row.get('infrastructure_error') or row.get('verification',{}).get('status')!='completed' for row in requested_rows):
            raise DecisionConstraintError('Submission unavailable: baseline is missing or unscored')
        max_tools = json.loads(Path(task_state.harness_path).read_text())['budget']['max_tool_calls']
        for change in requested:
            if change.operation != 'write_asset' or change.target not in rows:
                raise DecisionConstraintError('Only exact existing task submissions may be replaced')
        files = {name: json.dumps(submission(row), ensure_ascii=False) for name, row in rows.items()}

        def validate(output):
            output = ArtifactChanges.model_validate(output)
            expected = {c.target for c in requested}
            if len(output.edits) != len(expected) or {e.path for e in output.edits} != expected:
                raise DecisionConstraintError('Return exactly the declared submission edits')
            try:
                for edit in output.edits:
                    value = validate_submission(json.loads(edit.content), rows[edit.path], max_tools)
                    if value == submission(rows[edit.path]):
                        raise ValueError('Submission must materially change')
            except (ValueError, TypeError, KeyError) as exc:
                raise DecisionConstraintError(str(exc)) from exc
            return True

        kwargs = evolution_kwargs(self.client, context, task_state, decision, files, validate)
        for change in requested:
            row = rows[change.target]; calls = row.get('transport_calls', [])
            public = {'task_id': row['task_id'], 'split': row['split'],
                'initial_messages': calls[0].get('messages', []) if calls else [],
                'tools': calls[0].get('tools', []) if calls else [], 'tool_observations': row.get('tool_calls', [])}
            if 'operation_input' in kwargs:
                kwargs['operation_input']['current_files']['submission_context/' + change.target] = json.dumps(public, ensure_ascii=False)
        output = self.client.complete(CONTRACT + ' Return ArtifactChanges JSON; each content is serialized submission JSON.',
            ArtifactChanges, meta_state=context.meta_state, operation='artifact_patch', decision_id=decision.decision_id,
            **kwargs)
        validate(output)
        updated = clone_task(task_state, context.generation, context.directory)
        root = context.directory / 'artifacts'
        root.mkdir(exist_ok=True)
        applied = []
        for edit in output.edits:
            row = rows[edit.path]
            # Trusted identity is outside the editable payload, bound to the exact baseline.
            record = {'schema': 'task_submission_v1', 'baseline_sha256': value_hash(row),
                      'task_id': row['task_id'], 'task_source_hash': row['task_source_hash'],
                      'rollout_id': row['rollout_id'], 'split': row['split'], 'seed': row['seed'],
                      'payload': json.loads(edit.content)}
            save_json(root / edit.path, record)
            applied.append({'id': next(c.id for c in requested if c.target == edit.path),
                            'operation': 'write_asset', 'target': edit.path,
                            'before_sha256': value_hash(submission(row)), 'after_sha256': value_hash(record['payload'])})
        updated.artifacts = ArtifactState(str(root), artifact_manifest(root))
        return updated, TaskUpdate(TaskUpdateAction.ARTIFACTS, output.summary,
            {'evaluation_mode': 'direct_submission', 'targets': [e.path for e in output.edits]},
            {'wall_time_seconds': time.monotonic()-started, 'api_cost_usd': None},
            **_audit(decision, applied, [], [{'name': 'bounded_output_edits_no_task_inference', 'passed': True}]))


class SubmissionEvaluator:
    def __init__(self, executor, baseline_dir, targets):
        self.executor, self.baseline_dir, self.targets = executor, Path(baseline_dir), set(targets)

    def execute(self, state, directory):
        from sia.task_meta.pipeline_execution import aggregate
        from sia.task_meta.task_client import TaskInfrastructureError
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        source = baseline_outputs(self.baseline_dir)
        if not self.targets or not self.targets <= source.keys():
            raise ValueError('Direct evaluation requires declared baseline submission targets')
        spec = json.loads(Path(state.harness_path).read_text())
        window = json.loads((self.baseline_dir / 'window.json').read_text())
        tasks, _ = self.executor.store.window(window['cursor'], self.executor.quotas)
        tasks = {t.task_id: t for t in [*tasks, *self.executor.store.probe()]}
        rows = {'train': [], 'probe': []}
        for name, old in source.items():
            split = name.split('/')[1]
            receipt = directory / (split + '_rollouts') / Path(name).name
            scope=getattr(getattr(self.executor,'round_protocol',None),'execution_scope',None)
            binding = {'state_hash': task_hash(state), 'baseline_sha256': value_hash(old), 'edited': name in self.targets, 'round_scope':scope}
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved['binding'] != binding or saved['status'] != 'completed':
                    raise UpdatePending('Submission replay receipt requires audit; no blind tool replay')
                row = saved['row']
            else:
                row = copy.deepcopy(old)
                # Baseline model actions are evidence, not new calls or Meta-generated SFT labels.
                row.update(state_hash=task_hash(state), model_calls=[], transport_calls=[], sft_conversations=[],
                    messages=[], notes=[], input_tokens=0, output_tokens=0, model_call_count=0,
                    unknown_usage_calls=0, usage_complete=True, wall_time_seconds=0, tool_calls=[],
                    artifact_input_manifest=state.artifacts.manifest, artifact_chars_used=0,
                    submission_origin='meta_output_repair' if name in self.targets else 'baseline_result_reused',
                    baseline_row_sha256=value_hash(old), events=[], meta_generated=True,
                    derived_from=[value_hash(old)], submission_receipt=str(receipt.resolve()))
                if scope:
                    row.update(collection_stage='child_post_update',branch_id='child',
                        trajectory_id=value_hash([binding,name]),candidate_id=scope.get('candidate_id'))
                if name in self.targets:
                    record = json.loads((Path(state.artifacts.directory) / name).read_text())
                    if record['baseline_sha256'] != value_hash(old):
                        raise ValueError('Submission is bound to another baseline')
                    value = validate_submission(record['payload'], old, spec['budget']['max_tool_calls'])
                    env = self.executor.adapter_factory(old['domain'])
                    started = time.monotonic()
                    save_json(receipt, {'binding': binding, 'status': 'started'})
                    try:
                        public = env.reset(tasks[old['task_id']], str(old['rollout_id']), old['seed'])
                        if value_hash(public) != old['reset_hash']:
                            raise ValueError('Submission environment reset differs from the paired baseline')
                        for i, action in enumerate(value['actions']):
                            observed = env.step(action['name'], action['arguments'])
                            row['tool_calls'].append({**action, 'step': i, 'observation': observed})
                            save_json(receipt, {'binding': binding, 'status': 'started', 'completed_actions': i+1})
                            if observed.get('terminated'):
                                break
                        score = env.evaluate(value['final_answer'])
                        if score.infrastructure_error:
                            raise TaskInfrastructureError(score.error_type)
                        row.update(final_answer=value['final_answer'], model_answer=None,
                            terminal_reward=score.reward, metrics=score.metrics, verification=score.verification,
                            error_type=score.error_type, execution_error_type=None, infrastructure_error=False, verifier_details=score.details,
                            steps=len(row['tool_calls']), events=[{'kind': 'direct_submission_evaluated',
                            'task_model_calls': 0, 'tool_actions': len(row['tool_calls'])}])
                    finally:
                        env.close()
                    row['wall_time_seconds'] = time.monotonic()-started
                save_json(receipt, {'binding': binding, 'status': 'completed', 'row': row})
            rows[split].append(row)
        round_scoped=bool(getattr(self.executor,'round_protocol',None))
        physical_rows=rows['train']+rows['probe']
        if round_scoped: rows['probe']=copy.deepcopy(rows['train'])
        scores = aggregate(rows['probe'], expected_domains=self.executor.expected_domains)
        scores['probe_identity'] = value_hash([t.task_id for t in self.executor.store.probe()])
        scores['training_rewards'] = aggregate(rows['train'], require_domains=False, expected_domains=self.executor.expected_domains)
        scores['evaluation_protocol'] = 'direct_submission_same_tasks_v1'
        scores['attribution'] = 'meta_assisted_output_quality_not_task_policy_improvement'
        window['state_hash'] = task_hash(state); save_json(directory / 'window.json', window)
        for split, values in rows.items():
            (directory / (split + '_trajectories.jsonl')).write_text(
                ''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')
        costs = {'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'unknown_usage_calls': 0,
                 'usage_complete': True, 'api_cost_usd': None, 'gpu_hours': 0,
                 'wall_time_seconds': sum(v['wall_time_seconds'] for v in physical_rows),
                 'replayed_submissions': len(self.targets), 'reused_baseline_results': len(source)-len(self.targets)}
        if round_scoped:
            scores.update(feedback_role='train_evolution',round_id=self.executor.store.round_id,
                training_manifest=self.executor.store.current['manifest_hash'],collection_stage='child_post_update')
        return EvaluationResult(scores, rows['train'], costs, copy.deepcopy(state))
