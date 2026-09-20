import json
from dataclasses import asdict

import pytest
from test_task_meta_loop import components, decision

from sia.task_meta.durable import DurableExecutor, DurableUpdater, StageJournal, task_hash
from sia.task_meta.pipeline_execution import aggregate
from sia.task_meta.task_client import TaskInfrastructureError, qwen_message
from sia.task_meta.types import GenerationContext, TaskUpdateAction, UpdatePending


def test_fixed_denominator_includes_wrong_and_parse_failed():
    rows = [{'domain': d, 'terminal_reward': 0, 'verification': {'success': False}, 'metrics': {},
             'error_type': 'parse_failure'} for d in ['code','tool_use','searchqa']]
    result = aggregate(rows)
    assert result['macro_success'] == 0
    assert result['total_rollouts'] == 3
    assert all(v['denominator'] == 1 for v in result['domains'].values())
    rows[0]['infrastructure_error'] = True
    with pytest.raises(TaskInfrastructureError):
        aggregate(rows)


def test_partial_tool_reward_is_not_primary_task_success():
    rows = [{'domain': d, 'terminal_reward': .5, 'verification': {'success': False}, 'metrics': {},
             'error_type': None} for d in ['code','tool_use','searchqa']]
    assert aggregate(rows)['macro_success'] == 0


def test_qwen_native_tool_roundtrip():
    value = qwen_message('<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>')
    call = value['tool_calls'][0]
    assert json.loads(call['function']['arguments']) == {'query': 'Paris'}
    assert call['id']
    bad = '<tool_call>not JSON</tool_call>'
    assert qwen_message(bad)['content'] == bad


def test_execution_receipt_prevents_duplicate_rollouts(tmp_path):
    events, task, _meta, executor, _agent, _updaters = components(tmp_path)
    wrapped = DurableExecutor(executor, StageJournal(tmp_path))
    first = wrapped.execute(task, tmp_path / 'gen_0')
    again = wrapped.execute(task, tmp_path / 'gen_0')
    assert asdict(first) == asdict(again)
    assert events == [('execute', 0)]
    task.model_ref = 'unexpected'
    with pytest.raises(ValueError, match=r'identity|different'):
        wrapped.execute(task, tmp_path / 'gen_0')


def test_intervention_receipt_prevents_duplicate_side_effect(tmp_path):
    events, task, meta, _executor, _agent, updaters = components(tmp_path)
    context = GenerationContext(1, tmp_path / 'gen_1', None, None, meta)
    wrapped = DurableUpdater(updaters[TaskUpdateAction.HARNESS], StageJournal(tmp_path))
    selected = decision('HARNESS')
    new, _update = wrapped.apply(task, selected, context)
    restored, _ = wrapped.apply(task, selected, context)
    assert task_hash(restored) == task_hash(new)
    assert events == [('update', 0, 'HARNESS')]


def test_uncertain_intervention_never_retrained(tmp_path):
    events, task, meta, _executor, _agent, _updaters = components(tmp_path)
    class Failed:
        def apply(self, *args):
            events.append('external_started')
            raise RuntimeError('disconnect before result observed')
    context = GenerationContext(1, tmp_path / 'gen_1', None, None, meta)
    wrapped = DurableUpdater(Failed(), StageJournal(tmp_path))
    with pytest.raises(RuntimeError):
        wrapped.apply(task, decision('MODEL'), context)
    with pytest.raises(UpdatePending, match='refusing duplicate'):
        wrapped.apply(task, decision('MODEL'), context)
    assert events == ['external_started']
