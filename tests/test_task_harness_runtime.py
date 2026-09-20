"""Actual Task interpreter/executor tests with explicitly mocked Qwen and tools.

No checkpoint is loaded, no API is contacted and no SFT trainer is invoked.
"""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from sia.task_meta.data import DOMAINS, TaskRecord
from sia.task_meta.environments import AdapterResult
from sia.task_meta.pipeline_execution import MultiDomainExecutor
from sia.task_meta.seed import load_seed, run_seed
from sia.task_meta.sft import select_positive_rows
from sia.task_meta.storage import artifact_manifest, checkpoint_manifest, save_json
from sia.task_meta.task_client import TaskInfrastructureError
from sia.task_meta.task_harness import load_harness
from sia.task_meta.types import ArtifactState, TaskAgentState, UpdatePending

ROOT = Path(__file__).resolve().parents[1]


def spec():
    value = load_harness(ROOT / 'seed_harness/v2/seed.json')
    value['parts']['control']['planning']['enabled'] = False
    value['parts']['memory']['context']['shortterm_enabled'] = False
    return value


def action(name, **arguments):
    return json.dumps({'tools': [{'name': name, 'arguments': arguments}]})


class MockCurrentQwen:
    enable_thinking = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, messages, **kwargs):
        self.requests.append({'messages': copy.deepcopy(messages), **copy.deepcopy(kwargs)})
        output = next(self.responses)
        if isinstance(output, Exception):
            raise output
        return {'message': {'role': 'assistant', 'content': output},
                'binding': {'model_ref': 'mock-current-checkpoint'},
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3},
                'chat_template_kwargs': {'enable_thinking': self.enable_thinking}}


class MockPublicEnvironment:
    tools: ClassVar[list] = [{'name': 'search', 'description': 'Explicit mock public evidence search.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}]

    def __init__(self):
        self.calls = []

    def step(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        if arguments.get('query') == 'reject':
            raise ValueError('Explicit mock public tool error')
        return {'results': [{'id': 'mock-evidence-1', 'text': 'Public mock evidence says Paris.'}]}

    def evaluate(self, answer):
        raise AssertionError('Task runtime must never invoke official evaluation')


def run(value, outputs, *, sources=None, environment=None):
    model = MockCurrentQwen(outputs)
    result = run_seed(value, model, environment or MockPublicEnvironment(), 'Public mock question',
                      seed=89, artifact_sources=sources)
    return result, model


def add_first_role(value, role):
    value['parts']['tools']['roles'].append(role)
    graph = value['parts']['control']['graph']
    graph['nodes'].append({'id': role['name'], 'kind': 'role', 'role': role['name'], 'next': graph['entry']})
    graph['entry'] = role['name']


def test_real_roles_review_repair_and_actual_per_call_sft():
    value = spec()
    add_first_role(value, {'name': 'planner', 'instruction': 'Plan for {{task}}.', 'result': 'plan'})
    value['parts']['tools']['roles'].append({'name': 'reviewer', 'instruction': 'Review {{candidate}}.', 'result': 'review'})
    value['parts']['submission']['checks'] = [{'name': 'review', 'kind': 'model_review', 'role': 'reviewer'}]
    result, model = run(value, ['Explicit mock plan', action('final_answer', answer='wrong'),
        '{"needs_repair":true,"feedback":"Mock review requests correction"}',
        action('final_answer', answer='Paris'), '{"needs_repair":false,"feedback":"Mock review accepts"}'])
    assert result['final_answer'] == 'Paris'
    assert result['submission_count'] == 1
    assert [row['operation'] for row in result['model_calls']] == [
        'role:planner', 'action', 'submission_review:reviewer', 'action', 'submission_review:reviewer']
    assert 'Mock review requests correction' in str(model.requests[3]['messages'])
    assert [row['seed'] for row in model.requests] == list(range(89, 94))
    assert all(row['binding']['model_ref'] == 'mock-current-checkpoint' for row in result['model_calls'])
    positive = {**result, 'split': 'evolve_train', 'domain': 'searchqa', 'terminal_reward': 1,
                'task_id': 'mock-task', 'rollout_id': 0,
                'verification': {'status': 'completed', 'success': True, 'exact_match': True,
                                 'verifier_id': 'mock-trusted-verifier'}}
    selected = select_positive_rows([positive], profile='multidomain')
    assert len(selected) == len(model.requests) == 5
    for row, request, call in zip(selected, model.requests, result['model_calls'], strict=True):
        assert row['messages'][:-1] == request['messages']
        assert row['messages'][-1] == call['assistant']
        assert row['tools'] is None
        assert row['chat_template_kwargs'] == {'enable_thinking': True}
        assert row['binding'] == call['binding']
        assert row['role'] == call['role']
        assert row['sft_supervision'] == 'final_assistant'
    assert select_positive_rows([{**positive, 'split': 'search_dev'}], profile='multidomain') == []


@pytest.mark.parametrize('initial,validate,expected_calls', [(17, True, 1), ('reject', False, 2)])
def test_argument_recovery_uses_current_qwen_and_preserves_actual_error(initial, validate, expected_calls):
    if validate:
        pytest.importorskip('jsonschema', reason='Real optional validator unavailable in the local test runtime')
    value = spec()
    value['parts']['tools']['validate_arguments'] = validate
    value['parts']['tools']['recovery']['max_retries'] = 1
    environment = MockPublicEnvironment()
    result, model = run(value, [action('search', query=initial), '{"arguments":{"query":"Paris"}}',
                               action('final_answer', answer='Paris')], environment=environment)
    assert result['final_answer'] == 'Paris'
    assert len(environment.calls) == expected_calls
    assert [row['status'] for row in result['tool_calls']] == ['task_error', 'completed']
    error = result['tool_calls'][0]['error']
    assert error and any(error in message['content'] for message in model.requests[1]['messages'])
    assert model.requests[1]['seed'] == 90
    assert result['model_calls'][1]['operation'] == 'tool_repair'
    assert environment.calls[-1] == ('search', {'query': 'Paris'})
    assert any(row['kind'] == 'tool_error' and row['error'] == error for row in result['events'])


def test_recovery_cannot_switch_tools_or_overrun_shared_budget():
    value = spec()
    value['parts']['tools']['recovery']['max_retries'] = 1
    result, _ = run(value, [action('search', query='reject'), '{"name":"another_tool","arguments":{}}'])
    assert result['error_type'] == 'parse_error'
    assert len(result['tool_calls']) == 1
    value['budget']['max_model_calls'] = 1
    result, model = run(value, [action('search', query='reject')])
    assert result['error_type'] == 'budget_exhausted'
    assert len(model.requests) == 1


def test_evidence_check_consumes_public_retrieval_and_repair_branch():
    value = spec()
    value['parts']['submission']['checks'] = [{'name': 'evidence', 'kind': 'evidence_count', 'minimum': 1}]
    result, _ = run(value, [action('final_answer', answer='Paris'), action('search', query='Paris'),
                            action('final_answer', answer='Paris')])
    assert result['final_answer'] == 'Paris'
    checks = [row for row in result['events'] if row['kind'] == 'submission_check']
    assert [row['passed'] for row in checks] == [False, True]
    assert all(row['evidence_source'] == 'public_task_state_only' for row in checks)


@pytest.mark.parametrize('repair', [False, True])
def test_structured_tool_error_is_not_success_and_only_explicit_recovery_calls_model(repair):
    class StructuredErrorEnvironment(MockPublicEnvironment):
        def step(self, name, arguments):
            self.calls.append((name, copy.deepcopy(arguments)))
            if len(self.calls) == 1:
                return {'error': 'explicit mock recoverable error', 'terminated': False}
            return {'results': [{'id': 'fixed', 'text': 'Public evidence'}]}

    value = spec()
    value['parts']['tools']['recovery']['max_retries'] = int(repair)
    value['parts']['submission']['checks'] = [{'name': 'successful_tool', 'kind': 'tool_success'}]
    # End after the first candidate's inspection so a failed check cannot trigger
    # unrelated action requests in this focused fixture.
    next(node for node in value['parts']['control']['graph']['nodes'] if node['id'] == 'branch')['next'] = 'stop'
    environment = StructuredErrorEnvironment()
    value['parts']['tools']['action']['max_tools_per_step'] = 2
    first = json.dumps({'tools': [{'name': 'search', 'arguments': {'query': 'q'}},
                                  {'name': 'final_answer', 'arguments': {'answer': 'Paris'}}]})
    outputs = [first] + (['{"arguments":{"query":"fixed"}}'] if repair else [])
    result, model = run(value, outputs, environment=environment)
    assert len(model.requests) == 1 + int(repair)
    assert result['tool_calls'][0]['status'] == 'completed'
    assert result['tool_calls'][0]['observation']['error'] == 'explicit mock recoverable error'
    assert result['tool_calls'][0]['task_tool_failure'] is True
    checks = [event for event in result['events'] if event['kind'] == 'submission_check']
    assert checks[0]['passed'] is repair
    assert (result['final_answer'] == 'Paris') is repair


@pytest.mark.parametrize('terminal_key', ['terminal', 'terminated'])
def test_terminal_environment_cannot_be_reentered_by_repair_or_graph(terminal_key):
    class TerminalEnvironment(MockPublicEnvironment):
        def step(self, name, arguments):
            self.calls.append((name, arguments))
            return {terminal_key: True, 'final_answer': '', 'error': 'explicit mock terminal error'}

    value = spec()
    value['parts']['tools']['recovery']['max_retries'] = 2
    value['parts']['tools']['action']['max_tools_per_step'] = 2
    value['parts']['submission']['checks'] = [{'name': 'nonempty', 'kind': 'nonempty'}]
    # First batch itself contains a forbidden second call after termination.
    output = json.dumps({'tools': [{'name': 'search', 'arguments': {'query': 'first'}},
                                   {'name': 'search', 'arguments': {'query': 'second'}}]})
    environment = TerminalEnvironment()
    result, model = run(value, [output], environment=environment)
    assert len(environment.calls) == len(model.requests) == 1
    assert result['final_answer'] is None and result['submission_count'] == 0
    assert result['error_type'] == 'submission_rejected'
    assert any(event['kind'] == 'terminal_guard' for event in result['events'])


def test_input_rendering_and_role_cycle_obey_declared_order_and_shared_call_budget():
    value = spec()
    value['parts']['input']['task_template'] = 'Exact task follows: {{task}}'
    value['parts']['input']['section_order'] = ['action', 'history', 'guidance']
    result, model = run(value, [action('final_answer', answer='Paris')])
    assert result['final_answer'] == 'Paris'
    assert model.requests[0]['messages'][0]['content'].startswith(value['parts']['input']['action_step'].split('{{')[0])
    assert any(message['content'] == 'Exact task follows: Public mock question' for message in model.requests[0]['messages'])
    value = spec()
    add_first_role(value, {'name': 'planner', 'instruction': 'Plan again', 'result': 'plan'})
    value['parts']['control']['graph']['nodes'][-1]['next'] = 'planner'
    value['budget']['max_model_calls'] = 2
    result, model = run(value, ['Mock first plan', 'Mock second plan'])
    assert result['error_type'] == 'budget_exhausted' and len(model.requests) == 2
    assert result['submission_count'] == 0


def test_history_selection_is_read_only_and_assets_select_before_character_budget():
    value = spec()
    value['parts']['memory']['assets']['filter'] = {'path': 'item.path', 'op': 'eq', 'value': 'later.md'}
    value['parts']['memory']['history']['filter'] = {'path': 'item.role', 'op': 'ne', 'value': 'assistant'}
    add_first_role(value, {'name': 'planner', 'instruction': 'Plan {{task}}', 'result': 'plan'})
    sources = [{'path': 'first.md', 'content': 'A' * 15000, 'sha256': 'a' * 64},
               {'path': 'later.md', 'content': 'Selected later asset', 'sha256': 'b' * 64}]
    before = copy.deepcopy(sources)
    result, model = run(value, ['Mock plan should remain only in raw history', action('final_answer', answer='Paris')], sources=sources)
    assert result['final_answer'] == 'Paris'
    action_context = str(model.requests[-1]['messages'])
    assert 'Selected later asset' in action_context and 'AAAA' not in action_context
    assert 'Mock plan should remain' not in action_context
    assert 'Mock plan should remain' in str(result['final_context'])
    assert sources == before
    assert result['artifact_chars_used'] == len('Selected later asset')


@pytest.mark.parametrize('mode,answer', [('json_key', '{"answer":"Paris"}'), ('last_line', 'Reasoning\nParis')])
def test_submission_parser_changes_only_actual_output(mode, answer):
    value = spec()
    value['parts']['submission']['answer_format']['mode'] = mode
    result, _ = run(value, [action('final_answer', answer=answer)])
    assert result['final_answer'] == 'Paris'
    assert answer in result['model_calls'][0]['assistant']['content'].replace('\\n', '\n').replace('\\"', '"')
    assert result['submission_count'] == 1


def test_graph_can_insert_skip_reorder_and_loop_without_bypassing_transition_bound():
    value = spec()
    value['parts']['control']['graph'] = {'entry': 'loop', 'nodes': [
        {'id': 'loop', 'kind': 'branch', 'next': 'loop'}, {'id': 'stop', 'kind': 'stop', 'next': 'stop'}]}
    result, model = run(value, [])
    assert result['transitions'] == 512 and result['error_type'] == 'budget_exhausted'
    assert model.requests == [] and result['submission_count'] == 0
    value = spec()
    add_first_role(value, {'name': 'skipped', 'instruction': 'Must not call this role', 'result': 'context'})
    value['parts']['control']['graph']['nodes'][-1]['when'] = False
    result, model = run(value, [action('final_answer', answer='Paris')])
    assert result['final_answer'] == 'Paris' and len(model.requests) == 1
    assert any(event['primitive'] == 'role' and not event['enabled'] for event in result['events'] if event['kind'] == 'transition')


class MockScoredEnvironment(MockPublicEnvironment):
    def reset(self, task, rollout_id, seed):
        self.task = task
        return task.public_payload()

    def evaluate(self, answer):
        return AdapterResult(1, {'f1': 1}, {'status': 'completed', 'success': True,
            'verifier_id': 'explicit-mock-verifier', 'full_verifier': True, 'task_success': True, 'exact_match': True})

    def close(self):
        pass


class MockStore:
    def window(self, cursor, quotas):
        return [TaskRecord('train-' + domain, domain, 'mock', 'evolve_train', 'Mock public question', {})
                for domain in DOMAINS], dict.fromkeys(DOMAINS, 1)

    def probe(self):
        return [TaskRecord('probe-' + domain, domain, 'mock', 'search_dev', 'Mock public question', {}) for domain in DOMAINS]

    def coverage(self, cursor):
        return {'all_tasks_scheduled': True}


def test_executor_all_roles_use_same_model_and_same_generation_assets(tmp_path):
    value = spec()
    add_first_role(value, {'name': 'memorizer', 'instruction': 'Write a reusable mock note', 'result': 'memory'})
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    assets = tmp_path / 'input-assets'
    assets.mkdir()
    (assets / 'initial.md').write_text('Original active asset', encoding='utf-8')
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness), ArtifactState(str(assets), artifact_manifest(assets)))
    clients = []

    def factory(bound_state):
        assert bound_state.model_ref == state.model_ref
        index = len(clients)
        client = MockCurrentQwen([f'Private per-rollout note {index}', action('final_answer', answer='Paris')])
        clients.append(client)
        return client

    executor = MultiDomainExecutor(MockStore(), lambda domain: MockScoredEnvironment(), factory,
                                   quotas=dict.fromkeys(DOMAINS, 1))
    result = executor.execute(state, tmp_path / 'gen_0')
    assert len(clients) == 6
    assert result.cost['model_calls'] == 12 and result.cost['usage_complete']
    assert len(result.trajectories) == len(result.output_artifacts.manifest) == 3
    for client in clients:
        assert 'Original active asset' in str(client.requests[0]['messages'])
        assert 'Private per-rollout note' not in str(client.requests[0]['messages'])
    probe = [json.loads(line) for line in (tmp_path / 'gen_0/probe_trajectories.jsonl').read_text().splitlines()]
    assert all(not row['notes'] for row in probe)
    samples = select_positive_rows(result.trajectories + probe, profile='multidomain')
    assert len(samples) == 6 and all(row['split'] == 'evolve_train' for row in samples)
    assert all(row['chat_template_kwargs'] == {'enable_thinking': True} for row in samples)
    again = executor.execute(state, tmp_path / 'gen_0')
    assert len(clients) == 6 and again.performance == result.performance


def test_executor_failed_transport_attempt_is_counted_and_never_sft(tmp_path):
    value = spec()
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))
    client = MockCurrentQwen([RuntimeError('Explicit mock connection failure')])
    executor = MultiDomainExecutor(None, lambda domain: MockScoredEnvironment(), lambda state: client,
                                   quotas=dict.fromkeys(DOMAINS, 1))
    task = TaskRecord('task', 'searchqa', 'mock', 'evolve_train', 'Mock public question', {})
    row = executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert row['infrastructure_error'] and row['model_call_count'] == 1
    assert row['transport_calls'][0]['status'] == 'failed'
    assert row['model_calls'][0]['status'] == 'failed'
    assert row['unknown_usage_calls'] == 1 and not row['usage_complete']
    assert select_positive_rows([row], profile='multidomain') == []
    with pytest.raises(UpdatePending, match='dispatch evidence'):
        executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert len(client.requests) == 1


@pytest.mark.parametrize('schema,blocked', [(1, 'memory_extract'), (1, 'memory_prune'),
    (2, 'memory_extract'), (2, 'memory_prune'), (2, 'role:second')])
def test_controller_call_cap_is_normal_exhaustion_before_dispatch(tmp_path, schema, blocked):
    value = load_seed(ROOT / ('seed_harness/seed.json' if schema == 1 else 'seed_harness/v2/seed.json'))
    context = value['context'] if schema == 1 else value['parts']['memory']['context']
    outputs = ['Explicit test_override planning response']
    if blocked == 'memory_prune':
        outputs.append(json.dumps({'step_summary': '', 'key_extracts': [
            f'Explicit test_override memory item number {index}.'
            for index in range(context['max_shortterm_items'] + 1)]}))
    if blocked.startswith('role:'):
        value = spec()
        add_first_role(value, {'name': 'second', 'instruction': 'Second test_override plan', 'result': 'plan'})
        add_first_role(value, {'name': 'first', 'instruction': 'First test_override plan', 'result': 'plan'})
    harness = tmp_path / 'seed.json'
    save_json(harness, value)
    before = harness.read_bytes()
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))
    clients = []

    def factory(current):
        client = MockCurrentQwen(outputs)
        clients.append(client)
        return client

    class UnansweredEnvironment(MockScoredEnvironment):
        def evaluate(self, answer):
            assert answer == ''  # Budget exhaustion must not fabricate a submission.
            return AdapterResult(0, {}, {'status': 'completed', 'success': False,
                'verifier_id': 'explicit-test-override'}, 'incorrect_answer')

    executor = MultiDomainExecutor(MockStore(), lambda domain: UnansweredEnvironment(), factory,
        quotas=dict.fromkeys(DOMAINS, 1), model_call_limit=len(outputs))
    result = executor.execute(state, tmp_path / 'gen_0')
    probe = [json.loads(line) for line in (tmp_path / 'gen_0/probe_trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = result.trajectories + probe
    assert len(rows) == len(clients) == 6
    assert harness.read_bytes() == before
    assert result.cost['model_calls'] == 6 * len(outputs)
    assert result.cost['input_tokens'] == 6 * len(outputs) * 7
    assert result.cost['output_tokens'] == 6 * len(outputs) * 3
    assert result.cost['usage_complete'] and result.cost['unknown_usage_calls'] == 0
    assert result.performance['macro_success'] == 0
    for row, client in zip(rows, clients, strict=True):
        assert not row['infrastructure_error'] and not row['infrastructure_failure']
        assert row['error'] == 'Maximum controller Task model calls reached'
        assert row['final_answer'] == ''
        assert len(row['model_calls']) == row['model_call_count'] == len(client.requests) == len(outputs)
        assert len(row['sft_conversations']) == len(outputs)
        assert all(call['status'] == 'completed' for call in row['model_calls'])
        assert select_positive_rows([row], profile='multidomain') == []
        if schema == 2:
            blocked_events = [event for event in row['events'] if event['kind'] == 'model_budget_exhausted']
            assert len(blocked_events) == 1 and blocked_events[0]['operation'] == blocked
            assert blocked_events[0]['dispatched'] is False
            assert sum(event['kind'] == 'model_request' for event in row['events']) == len(outputs)
    receipts = list((tmp_path / 'gen_0').glob('*_rollouts/*.calls/*.json'))
    assert len(receipts) == 6 * len(outputs)
    assert all(json.loads(path.read_text(encoding='utf-8'))['status'] == 'completed' for path in receipts)


def test_controller_token_violation_remains_infrastructure_error_without_dispatch(tmp_path):
    value = spec()
    harness = tmp_path / 'seed.json'
    save_json(harness, value)
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))
    client = MockCurrentQwen([])
    executor = MultiDomainExecutor(None, lambda domain: MockScoredEnvironment(), lambda state: client,
        quotas=dict.fromkeys(DOMAINS, 1), max_output_tokens=value['budget']['max_tokens'] - 1)
    task = TaskRecord('token-limit', 'searchqa', 'test_override', 'evolve_train', 'Mock public question', {})
    row = executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert row['infrastructure_error'] and row['model_call_count'] == 0
    assert client.requests == [] and row['sft_conversations'] == []
    assert list((tmp_path / 'rollouts').glob('*.calls/*.json')) == []


@pytest.mark.parametrize('phase', ['during_dispatch', 'after_response'])
def test_incomplete_rollout_call_receipts_block_duplicate_dispatch(tmp_path, monkeypatch, phase):
    import sia.task_meta.pipeline_execution as execution
    from sia.task_meta.durable import value_hash

    value = spec()
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))
    task = TaskRecord('crash-task', 'searchqa', 'mock', 'evolve_train', 'Mock public question', {})
    calls = []

    def model(messages, **kwargs):
        calls.append(messages)
        if phase == 'during_dispatch':
            raise KeyboardInterrupt('Explicit mock crash after request dispatch')
        return {'message': {'role': 'assistant', 'content': action('final_answer', answer='Paris')}}

    original_save = execution.save_json

    def save(path, value):
        if phase == 'after_response' and 'row' in value:
            raise OSError('Explicit mock crash before completed rollout receipt')
        original_save(path, value)

    monkeypatch.setattr(execution, 'save_json', save)
    executor = MultiDomainExecutor(None, lambda domain: MockScoredEnvironment(), lambda state: model,
                                   quotas=dict.fromkeys(DOMAINS, 1))
    with pytest.raises((KeyboardInterrupt, OSError)):
        executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    receipts = list((tmp_path / 'rollouts').glob('*.calls/*.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt['status'] == ('dispatching' if phase == 'during_dispatch' else 'completed')
    assert receipt['request_sha256'] == value_hash(receipt['request'])
    assert receipt['binding']['task'] == task.task_id
    with pytest.raises(UpdatePending, match='receipts exist'):
        executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert len(calls) == 1


def test_reset_failure_without_dispatch_evidence_can_retry(tmp_path):
    value = spec()
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))
    task = TaskRecord('reset-task', 'searchqa', 'mock', 'evolve_train', 'Mock public question', {})
    attempts = []

    class Environment(MockScoredEnvironment):
        def reset(self, *args):
            attempts.append('reset')
            if len(attempts) == 1:
                raise TaskInfrastructureError('Explicit mock setup failure before any model request')
            return super().reset(*args)

    client = MockCurrentQwen([action('final_answer', answer='Paris')])
    executor = MultiDomainExecutor(None, lambda domain: Environment(), lambda state: client,
                                   quotas=dict.fromkeys(DOMAINS, 1))
    first = executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert first['infrastructure_error'] and first['model_call_count'] == 0
    assert client.requests == []
    second = executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert not second['infrastructure_error'] and second['final_answer'] == 'Paris'
    assert len(client.requests) == 1


@pytest.mark.parametrize('crash', [False, True])
def test_final_prediction_uses_v2_roles_and_receipts_without_repeating_or_scoring(tmp_path, monkeypatch, crash):
    import sia.task_meta.report_predictions as predictions
    import sia.task_meta.task_harness.policy as policy
    from sia.task_meta.durable import task_hash

    value = spec()
    add_first_role(value, {'name': 'planner', 'instruction': 'Plan final public task', 'result': 'plan'})
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    runtime_fixture = tmp_path / 'runtime_identity_fixture.py'
    runtime_fixture.write_text('# Explicit mock dependency inventory for frozen identity test\n', encoding='utf-8')
    monkeypatch.setattr(policy, 'runtime_dependencies', lambda: {'runtime_fixture.py': runtime_fixture})
    checkpoint = tmp_path / 'mock_checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'explicit mock fixture; never loaded')
    weights = checkpoint_manifest(checkpoint)
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness), checkpoint_path=str(checkpoint), checkpoint_manifest=weights)
    frozen = tmp_path / 'frozen.json'
    save_json(frozen, {'task_state': asdict(state), 'status': 'frozen_for_report_eval',
                      'state_hash': task_hash(state), 'checkpoint_files': weights,
                      'task_harness_identity': policy.harness_identity(harness)})
    source = tmp_path / 'public.json'
    save_json(source, [{'_id': 'mock-final', 'question': 'Mock public final question', 'answer': 'HIDDEN_MUST_NOT_APPEAR'}])
    specs_path = tmp_path / 'specs.json'
    save_json(specs_path, {'evaluators': {'hotpotqa_dev': {'data_path': str(source)}}})
    save_json(tmp_path / 'search.sqlite.manifest.json', {'sha256': 'explicit-mock-index'})
    config = SimpleNamespace(data_dir=str(tmp_path), model_call_limit=20, max_output_tokens=2048,
                             task_enable_thinking=True, seed=42, task_base_url='mock-unused', task_timeout=1)
    clients = []

    def client_factory(*args, **kwargs):
        client = MockCurrentQwen(['Explicit mock final plan', action('final_answer', answer='Paris')])
        clients.append(client)
        return client

    class FinalEnvironment(MockPublicEnvironment):
        def reset(self, task, *args):
            return task.public_payload()

        def close(self):
            pass

    monkeypatch.setattr(predictions, 'LocalTaskClient', client_factory)
    monkeypatch.setattr(predictions, 'SearchQAAdapter', lambda index: FinalEnvironment())
    monkeypatch.setattr(predictions, 'FrozenSearchIndex', lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(predictions, 'OfficialEvaluatorSpec', lambda **kwargs: SimpleNamespace(**kwargs, validate=lambda: ['mock-final']))
    original_save = predictions.save_json

    def save(path, value):
        if crash and value.get('final_submission_count') == 1:
            raise OSError('Explicit mock final crash after inference before submission receipt')
        original_save(path, value)

    monkeypatch.setattr(predictions, 'save_json', save)
    output = tmp_path / 'final'
    if crash:
        with pytest.raises(OSError, match='final crash'):
            predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
        with pytest.raises(UpdatePending, match='call receipts exist'):
            predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
    else:
        result = predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
        assert result['hotpotqa_dev']['count'] == 1
        again = predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
        assert again == result
        row = json.loads((output / 'hotpotqa_dev.jsonl').read_text())
        assert row['final_submission_count'] == 1
        assert [call['operation'] for call in row['trajectory']['model_calls']] == ['role:planner', 'action']
        assert 'terminal_reward' not in row
    assert sum(len(client.requests) for client in clients) == 2
    assert all('HIDDEN_MUST_NOT_APPEAR' not in str(client.requests) for client in clients)
    assert json.loads((output / 'prediction_protocol.json').read_text())['feedback_to_meta'] is False
    saved_frozen = json.loads(frozen.read_text())
    original_frozen = copy.deepcopy(saved_frozen)
    del saved_frozen['task_harness_identity']
    save_json(frozen, saved_frozen)
    with pytest.raises(ValueError, match='runtime identity is absent or changed'):
        predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
    save_json(frozen, original_frozen)
    runtime_fixture.write_text('# Actual changed fixture bytes after freezing\n', encoding='utf-8')
    with pytest.raises(ValueError, match='runtime identity is absent or changed'):
        predictions.generate_report_predictions(config, frozen, specs_path, output, enabled=True)
    assert sum(len(client.requests) for client in clients) == 2


def test_missing_partial_usage_preserves_only_known_counts(tmp_path):
    value = spec()
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    state = TaskAgentState(0, 'mock-current-checkpoint', str(harness))

    def model(messages, **kwargs):
        return {'message': {'role': 'assistant', 'content': action('final_answer', answer='Paris')},
                'usage': {'prompt_tokens': None, 'completion_tokens': 4}}

    executor = MultiDomainExecutor(None, lambda domain: MockScoredEnvironment(), lambda state: model,
                                   quotas=dict.fromkeys(DOMAINS, 1))
    task = TaskRecord('usage-task', 'searchqa', 'mock', 'evolve_train', 'Mock public question', {})
    row = executor._one(state, value, task, 0, tmp_path / 'rollouts', '', probe=False)
    assert row['input_tokens'] == 0 and row['output_tokens'] == 4
    assert row['unknown_usage_calls'] == 1 and not row['usage_complete']
    assert row['transport_calls'][0]['unknown_usage_calls'] == row['model_calls'][0]['unknown_usage_calls'] == 1
    assert not row['model_calls'][0]['usage_complete']


def test_task_smoke_default_refusal_has_no_model_or_filesystem_effects(tmp_path, monkeypatch):
    import scripts.smoke_task_harness as smoke

    def forbidden(*args, **kwargs):
        raise AssertionError('Default refusal must occur before configuration/model/data access')

    monkeypatch.setattr(smoke, 'load_config', forbidden)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='disabled'):
        smoke.run_smoke('does-not-exist.json', output)
    assert not output.exists()
    with pytest.raises(ValueError, match='exactly one'):
        smoke.run_smoke('does-not-exist.json', output, enabled=True, limit_per_domain=2)
    with pytest.raises(ValueError, match=r'\[1, 32\]'):
        smoke.run_smoke('does-not-exist.json', output, enabled=True, max_model_calls=33)


def test_task_smoke_cpu_fixture_reuses_real_executor_and_preserves_seed_budget(tmp_path, monkeypatch):
    import scripts.smoke_task_harness as smoke

    value = spec()
    harness = tmp_path / 'seed.json'
    harness.write_text(json.dumps(value), encoding='utf-8')
    config_path = tmp_path / 'config.json'
    config_path.write_text('{"explicit_mock_configuration":true}', encoding='utf-8')
    config = SimpleNamespace(seed_harness=str(harness), task_checkpoint='mock-current-checkpoint',
        data_dir=str(tmp_path), model_call_limit=128, max_output_tokens=2048, task_base_url='mock-unused',
        task_timeout=1, task_enable_thinking=True, seed=42)
    clients, closed = [], []

    class Store(MockStore):
        def validate_sources(self):
            pass

        def close(self):
            closed.append('store')

    def client_factory(*args, **kwargs):
        client = MockCurrentQwen([action('final_answer', answer='Paris')])
        clients.append(client)
        return client

    monkeypatch.setattr(smoke, 'PROJECT', tmp_path)
    monkeypatch.setattr(smoke, 'load_config', lambda path: config)
    monkeypatch.setattr(smoke, 'model_identity', lambda path: {'path': path, 'weights': []})
    monkeypatch.setattr(smoke, 'ManifestStore', lambda path: Store())
    monkeypatch.setattr(smoke, 'adapter_factory', lambda config: (
        lambda domain: MockScoredEnvironment(), SimpleNamespace(close=lambda: closed.append('index'))))
    monkeypatch.setattr(smoke, 'LocalTaskClient', client_factory)
    output = tmp_path / 'runs/smoke'
    result = smoke.run_smoke(config_path, output, enabled=True, max_model_calls=16)
    assert result['status'] == 'completed'
    assert result['train_rollouts'] == result['probe_rollouts'] == 3
    assert result['meta_api_calls'] == result['model_updates'] == 0 and not result['training_run']
    assert result['max_total_model_calls'] == 96 and result['cost']['model_calls'] == 6
    assert result['official_experiment_result'] is False
    assert len(clients) == 6 and closed == ['index', 'store']
    assert load_harness(output / 'gen_0/seed.json')['budget'] == value['budget']
    with pytest.raises(FileExistsError, match='new output'):
        smoke.run_smoke(config_path, output, enabled=True)
    assert len(clients) == 6
