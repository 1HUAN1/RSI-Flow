"""Task prefetch is Meta-independent, durable, and joined before GPU validation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sia.task_meta.early_rollout import EarlyRollout
from sia.task_meta.round_evolution import RoundProtocol
from sia.task_meta.storage import save_json
from sia.task_meta.types import TaskAgentState, MetaAgentState


def protocol(tmp_path, meta_version=0):
    harness = tmp_path / 'harness.json'
    if not harness.exists():
        save_json(harness, {})
    save_json(tmp_path / 'protocol.json', {'fixed': 'configuration'})
    p = object.__new__(RoundProtocol)
    p.root = tmp_path
    p.config = SimpleNamespace(seed=42, round_protocol={'candidate_policy': 'single_candidate_strict_positive_gain'})
    p.store = SimpleNamespace(round_id=2, current={'manifest_hash': 'round2'})
    p.meta = MetaAgentState('meta', str(harness), version=meta_version)
    p.active_memory = [{'version': meta_version}]
    p.current_candidate_id = 'candidate'
    return p, TaskAgentState(1, 'model', str(harness))


def test_parent_scope_reusable_after_meta_update_but_child_keeps_meta(tmp_path):
    first, task = protocol(tmp_path)
    first.bind_execution(task, tmp_path / 'baseline', 'parent_pre_update')
    second, _ = protocol(tmp_path, meta_version=1)
    second.bind_execution(task, tmp_path / 'baseline', 'parent_pre_update')
    assert first.execution_scope == second.execution_scope
    assert 'meta_harness' not in second.execution_scope
    second.bind_execution(task, tmp_path / 'child', 'child_post_update')
    assert second.execution_scope['meta_harness']['version'] == 1
    assert second.execution_scope['candidate_id'] == 'candidate'


@pytest.mark.parametrize('change', ['round', 'task', 'seed'])
def test_parent_scope_rejects_behavioral_changes(tmp_path, change):
    p, task = protocol(tmp_path)
    p.bind_execution(task, tmp_path / 'baseline', 'parent_pre_update')
    if change == 'round':
        p.store.current['manifest_hash'] = 'different_round'
    elif change == 'task':
        task.model_ref = 'different_model'
    else:
        p.config.seed = 43
    with pytest.raises(ValueError):
        p.bind_execution(task, tmp_path / 'baseline', 'parent_pre_update')


def test_completed_round_resume_prefetches_then_joins_before_validation(tmp_path, monkeypatch):
    import sia.task_meta.sequential_loop as loop
    from sia.task_meta.types import ImprovementExperience
    task = TaskAgentState(1, 'model', 'harness')
    meta = MetaAgentState('meta', 'instructions')
    record = {'input_content_hash': 'task', 'output_content_hash': 'task',
              'meta_before_hash': None, 'meta_after': meta.__dict__,
              'task_after': {**task.__dict__, 'artifacts': task.artifacts.__dict__}, 'score': {}}
    save_json(tmp_path / 'round_0/complete.json', record)
    save_json(tmp_path / 'round_0/experience.json', {})
    monkeypatch.setattr(loop, 'content_identity', lambda value: 'task')
    monkeypatch.setattr(loop, 'ImprovementExperience', lambda **kw: SimpleNamespace())
    monkeypatch.setattr('sia.task_meta.round_checkpoint.commit_round_checkpoint', lambda *args: None)
    events = []
    early = SimpleNamespace(start=lambda n, t: events.append(('start', n)), wait=lambda: events.append(('join',)))
    class BoundaryReached(Exception):
        pass
    def validation(n, r):
        events.append(('validation', n))
        raise BoundaryReached()
    with pytest.raises(BoundaryReached):
        loop.run_sequential_task_meta(tmp_path, task, meta, Mock(), SimpleNamespace(client=Mock()), {},
            max_generations=3, resume=True, after_round=validation, early_rollout=early)
    assert events == [('start', 1), ('join',), ('validation', 0)]


def test_recovery_starts_next_rollout_before_meta_summary(tmp_path, monkeypatch):
    import sia.task_meta.deployed_recovery as recovery
    from dataclasses import asdict
    p, task = protocol(tmp_path)
    meta = p.meta
    task.generation = 1
    events = []
    monkeypatch.setattr(recovery, 'verify_evidence', lambda *args: None)
    monkeypatch.setattr('sia.task_meta.sequential_loop.content_identity', lambda task: 'fixed')
    monkeypatch.setattr('sia.task_meta.sequential_loop.positive_gain', lambda outcome: True)
    monkeypatch.setattr(recovery, 'record_experience', lambda *args: None)
    attempt = {'paired_outcome': {}}
    monkeypatch.setattr(recovery, 'ImprovementExperience', lambda **kw: SimpleNamespace(candidate_attempts=[attempt]))
    save_json(tmp_path / 'round_0/deployment.json', {'status': 'deployed', 'task_after': asdict(task),
              'parent_content_hash': 'fixed', 'output_content_hash': 'fixed'})
    for name in ('experience', 'recursive_summary'):
        save_json(tmp_path / f'round_0/{name}.json', {})
    save_json(tmp_path / 'initial_meta_state.json', {'bundle_hash': meta.bundle_hash})
    p = SimpleNamespace(total_stages=3, pause_after_round=None, begin=Mock(), register_rows=Mock(), prepare_meta=Mock())
    class SummaryReached(Exception):
        pass
    def learn(*args):
        events.append('summary')
        raise SummaryReached()
    agent = SimpleNamespace(client=SimpleNamespace(bind_context=Mock()), learn_from_experience=learn)
    early = SimpleNamespace(start=lambda n, t: events.append(('rollout', n)))
    with pytest.raises(SummaryReached):
        recovery.finish_deployed_round(tmp_path, 0, task, meta, agent, [], p, None, True, early_rollout=early)
    assert events == [('rollout', 1), 'summary']


def test_no_fourth_round_launched(tmp_path):
    early = EarlyRollout(SimpleNamespace(max_generations=3), tmp_path)
    early.start(3, None)
    assert early.number is None
    assert not (tmp_path / 'round_3').exists()
