"""Offline coverage for five frozen 600-task rounds and final complete-Task eval."""
import json
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from evolution_protocol import read, training_quotas, assert_disjoint
from experiment_progress import ExperimentProgress
from launch_meta import DEFAULT_CONFIG, experiment_prompt
from prepare_600_744 import split_tasks
from task_adapter import TaskAdapter


def test_new_default_normalizes_600_and_keeps_old_configuration(tmp_path):
    config = read(DEFAULT_CONFIG)
    assert config['rounds'] == 5
    assert config['allocated_tasks'] == 3000
    assert sum(training_quotas(config).values()) == 600
    adapter = TaskAdapter(ROOT, tmp_path / 'receipts')
    runtime = adapter._config({'config_path': str(DEFAULT_CONFIG)})
    assert runtime.window_quotas == dict(tool_use=200, code=200, searchqa=200)
    assert runtime.max_generations == 5
    assert runtime.round_protocol['validation_manifest'] == config['validation_manifest']
    assert not runtime.round_protocol['skip_acebench']
    assert runtime.task_checkpoint.endswith('Qwen3-4B')
    assert len(runtime.task_replicas) == 4
    assert runtime.training['num_train_epochs'] == 1
    assert read(ROOT / 'configs/train_180_a0_v1.json')['training_tasks_per_pass'] == 180


def test_split_is_deterministic_and_retains_every_task():
    tasks = [dict(source='envscaler', task_id=f't{i}', content_hash=f'h{i}',
                  group_id='same_environment', round_id=1) for i in range(1000)]
    batches = split_tasks(tasks)
    assert [len(b['tasks']) for b in batches] == [200] * 5
    assert split_tasks(list(reversed(tasks))) == batches
    assert {t['task_id'] for b in batches for t in b['tasks']} == {t['task_id'] for t in tasks}


def test_final_only_ledger_requires_eval_only_at_b5(tmp_path):
    (tmp_path / 'workflow_policy.json').write_text(json.dumps({
        'rounds': 5, 'evaluation_schedule': 'final_only'}))
    progress = ExperimentProgress(tmp_path, rounds=5)
    result = progress.rebuild()
    assert all('independent_validation' not in s['required_milestones'] for s in result['stages'][:-1])
    assert 'independent_validation' in result['stages'][-1]['required_milestones']
    assert all('task_meta_snapshot' in s['required_milestones'] for s in result['stages'])
    # Complete all required milestones except final evaluation. Do not invoke any model.
    from test_experiment_progress import _turn
    mapping = {'meta_evidence': 'prepare_meta_evidence', 'decision_recorded': 'write_text',
               'candidate_built': 'harnessforge', 'candidate_rollout': 'run_candidate',
               'scores_compared': 'compare_scores', 'selection_recorded': 'write_text',
               'skills_appended': 'append_skills', 'context_updated': 'write_text',
               'task_meta_snapshot': 'snapshot_task_meta', 'parent_rollout': 'run_parent',
               'validation_snapshot': 'prepare_validation_snapshot'}
    index = 0
    for stage in result['stages']:
        number = stage['round']
        for milestone in stage['required_milestones']:
            if milestone == 'independent_validation': continue
            path = {'decision_recorded': f'/run/round_{number}/decision.json',
                    'selection_recorded': f'/run/round_{number}/selection.json',
                    'context_updated': '/run/meta/context.json'}.get(milestone)
            _turn(tmp_path, index, mapping.get(milestone, milestone), round_number=number, path=path)
            index += 1
    state = progress.rebuild()
    assert not state['finish_allowed']
    assert state['current_stage'] == 'B5'
    assert state['next_milestone'] == 'independent_validation'
    _turn(tmp_path, index, 'evaluate', round_number=5, status='exited', returncode=0)
    assert progress.rebuild()['finish_allowed']


def test_prompt_schedule_does_not_conflict(tmp_path):
    prompt = experiment_prompt(DEFAULT_CONFIG, read(DEFAULT_CONFIG), tmp_path)
    assert '"training_tasks_per_round": 600' in prompt
    assert 'Only after FINAL-round acceptance' in prompt
    assert 'saved model/checkpoint + Harness + Artifacts' in prompt
    assert 'Run 180 parent' not in prompt
    assert 'round=0, then evaluate' not in prompt
    assert 'then report-only independent validation' not in prompt
    assert 'activate, snapshot, independently validate, then advance' not in prompt


def test_prepared_real_cohorts_and_native_loader(tmp_path):
    config = read(DEFAULT_CONFIG)
    data = Path(config['frozen_data_dir'])
    if not (data / 'READY.json').is_file(): pytest.skip('prepare_600_744.py not run on this host')
    from sia.task_meta.round_evolution import RoundStore
    store = RoundStore(data / 'tasks.sqlite')
    seen = set()
    content = set()
    batches = []
    for number in range(1, 6):
        store.select_round(number)
        rows = store.current['tasks']
        assert len(rows) == 600
        assert Counter(r['source'] for r in rows) == config['train_quotas_per_round']
        assert Counter(r['strata']['interface'] for r in rows if r['domain'] == 'code') == {'stdio': 120, 'function': 80}
        assert not seen.intersection(r['task_id'] for r in rows)
        assert not content.intersection(r['content_hash'] for r in rows)
        seen.update(r['task_id'] for r in rows)
        content.update(r['content_hash'] for r in rows)
        assert all(store._task(r).task_id == r['task_id'] for r in rows)
        batches.append(store.current)
    original = read(config['source_training_manifest'])
    assert seen == {t['task_id'] for t in original['tasks']}
    heldout = read(config['validation_manifest'])
    assert_disjoint([original, heldout])
    assert {t['task_id'] for t in heldout['tasks']} == {t['task_id'] for t in read(config['source_validation_manifest'])['tasks']}
    settings = read(ROOT / config['validation_config'])
    from evaluation_manifest import select_specs
    value, specs = select_specs(settings, config['validation_manifest'], 'independent_validation', tmp_path / 'inputs')
    assert len(value['tasks']) == 744
    assert set(specs) == {'humaneval_plus', 'mbpp_plus', 'hotpotqa_dev', '2wiki_dev'}
    assert 'livecodebench' not in settings['benchmark_ids']


def test_six_benchmark_completion_uses_full_snapshot_without_gpu(tmp_path, monkeypatch):
    """Exercise evaluator completion on synthetic cached scores, not model outputs."""
    from types import SimpleNamespace
    import sia.task_meta.pipeline as pipeline
    import sia.task_meta.durable as durable
    import sia.task_meta.storage as storage
    import sia.task_meta.harnessforge_manifest as harness
    import sia.task_meta.gpu_phases as gpu
    import validation_sources
    from validate import validate
    config = read(DEFAULT_CONFIG)
    if not Path(config['validation_manifest']).exists(): pytest.skip('real cohort not prepared')
    cohort = read(config['validation_manifest'])
    settings = read(ROOT / config['validation_config'])
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    out = tmp_path / 'report'
    out.mkdir()
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    state = {'checkpoint_path': '/selected/model', 'harness_path': '/selected/harness',
             'artifacts': {'directory': '/selected/artifacts', 'manifest': []}}
    snapshot = out / 'round_snapshot.json'
    snapshot.write_text(json.dumps({'round': 5, 'task_state': state, 'meta_state': {'version': 5},
                                   'source_round': str(selection), 'chosen_component': 'HARNESS'}))
    loaded = []
    def load_task(value):
        loaded.append(value)
        return SimpleNamespace(**value)
    monkeypatch.setattr(pipeline, 'load_config', lambda _: SimpleNamespace(output_root=tmp_path, round_protocol={'enabled': True}))
    monkeypatch.setattr(durable, 'load_task', load_task)
    monkeypatch.setattr(durable, 'task_hash', lambda _: 'selected-task')
    monkeypatch.setattr(storage, 'checkpoint_manifest', lambda _: [])
    monkeypatch.setattr(harness, 'load_manifest', lambda _: SimpleNamespace(identity='selected-harness'))
    monkeypatch.setattr(validation_sources, 'verify', lambda *args: None)
    monkeypatch.setattr(gpu, 'ensure_services', lambda *args: None)
    monkeypatch.setattr(gpu, 'verify_services', lambda *args: {'mock': True})
    for source in settings['benchmark_ids']:
        rows = [t for t in cohort['tasks'] if t['source'] == source]
        scores = [{'native_id': native_id, 'execution_status': 'completed',
                   'scoring_status': 'completed', 'success': False, 'native_scores': {}}
                  for row in rows for native_id in row.get('native_ids', [row['native_id']])]
        path = out / 'scores' / source / 'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status': 'completed', 'state_hash': 'selected-task',
                                    'expected': len(scores), 'completed': len(scores), 'task_results': scores}))
    pipeline_path = tmp_path / 'config' / 'pipeline.json'
    pipeline_path.parent.mkdir()
    pipeline_path.write_text('{}')
    result = validate(snapshot, pipeline_path, settings_path, role='independent_validation',
                      manifest_path=config['validation_manifest'])
    assert result['status'] == 'completed'
    assert result['benchmarks_completed'] == 6
    assert result['completion']['n_scored'] == 744
    assert loaded == [state]
    assert read(out / 'frozen_task.json')['task_state'] == state
