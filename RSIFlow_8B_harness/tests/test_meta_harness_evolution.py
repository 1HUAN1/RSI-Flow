"""Real module wiring/version-boundary tests. No model/API/GPU calls."""
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controller_tools import ControllerTools
from meta_harness_runtime import MetaHarnessRuntime, FILES, wiring_probe, write_json
from skill_memory import SkillMemory
from launch_meta import experiment_prompt, prepare_meta_program, prepare_workflow_policy
from eval.meta_pair import prepare as prepare_pair, PairTools
from eval.task_version import prepare as prepare_task
from eval.loop_summary import compare
from persistent_meta import NativeExperimentTools
from experiment_progress import ExperimentProgress


def candidate(runtime, tmp_path, name='candidate'):
    folder = tmp_path / name
    runtime.materialize(folder, 1)
    return folder


def replace_planning(folder, value):
    (folder / 'planning.py').write_text("def prepare(context):\n    return {'selected_method': " + repr(value) + "}\n")


def test_seed_executes_all_production_hooks():
    report = wiring_probe(PROJECT / 'meta_harness/G000')
    assert report['status'] == 'checked'
    assert set(report['executed_hooks']) == {'workflow.prepare', 'workflow.review',
                                             'planning.prepare', 'memory.retrieve', 'memory.append', 'memory.maintain'}
    assert report['capability_improvement_tested'] is False


def test_sibling_imports_remain_available_during_real_hooks(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'sibling function')
    (folder / 'workflow.py').write_text(
        "from .planning import prepare as plan_prepare\n"
        "def prepare(context):\n"
        "    from .memory import retrieve\n"
        "    return {'plan': plan_prepare(context), 'memory_hook': retrieve.__name__}\n"
        "def review(context): return {}\n")
    before = {name for name in sys.modules if name.startswith('_meta_program_')}
    assert runtime.check(folder)['status'] == 'checked'
    assert runtime.update(decision='replace', round_number=1, candidate=folder)['status'] == 'queued'
    report = runtime.phase({'round': 2, 'phase': 'route'})
    assert report['workflow']['plan']['selected_method'] == 'sibling function'
    assert report['workflow']['memory_hook'] == 'retrieve'
    assert report['workflow']['meta_harness_call']['executed_version'] == 'G001'
    assert report['workflow']['meta_harness_call']['warning'] is None
    assert {name for name in sys.modules if name.startswith('_meta_program_')} == before


def test_same_candidate_directory_edit_is_loaded_without_stale_bytecode(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'first')
    from meta_harness_runtime import load_package
    with load_package(folder) as modules:
        assert modules['planning'].prepare({})['selected_method'] == 'first'
    replace_planning(folder, 'other')  # Same length, within the same filesystem timestamp tick.
    with load_package(folder) as modules:
        assert modules['planning'].prepare({})['selected_method'] == 'other'


def test_five_rounds_bind_actual_successive_packages(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    for number in range(1, 6):
        report = runtime.phase({'round': number, 'phase': 'route'})
        assert report['planning']['meta_harness_call']['executed_version'] == f'G{number - 1:03d}'
        if number > 1:
            assert report['planning']['selected_method'] == f'for round {number}'
        folder = tmp_path / f'candidate_{number}'
        runtime.materialize(folder, number)
        replace_planning(folder, f'for round {number + 1}')
        assert runtime.update(decision='replace', round_number=number, candidate=folder)['status'] == 'queued'
        snapshot = runtime.snapshot_reference(number)
        assert snapshot['used']['version'] == f'G{number - 1:03d}'
        assert snapshot['selected_next']['version'] == f'G{number:03d}'
    assert runtime.binding(1)['version'] == 'G000'


def test_workflow_policy_syncs_five_rounds_without_changing_retry_rule(tmp_path):
    run = tmp_path / 'run'
    write_json(run / 'meta_session/workflow_policy.json', {'rounds': 3, 'retry_rejected_from_round': 2})
    policy = prepare_workflow_policy(run, 5)
    assert policy == {'rounds': 5, 'retry_rejected_from_round': 2}
    progress = ExperimentProgress(run / 'meta_session', rounds=3).rebuild()
    assert progress['final_stage'] == 'B5'


def test_large_evolved_phase_output_is_saved_not_injected_in_full(tmp_path):
    run = tmp_path / 'run'
    bridge = NativeExperimentTools(project=PROJECT, run=run, executor=None,
                                   progress=ExperimentProgress(run / 'meta_session'))
    policy = {'workflow': {'full_analysis': 'x' * 100000}, 'planning': {'next': 'route'}}
    bridge.meta_program.phase = lambda context: policy
    text = bridge.phase_instructions({'current_stage': 'B1', 'next_milestone': 'decision_recorded'})
    path = run / 'meta_harness/phase_context.json'
    assert json.loads(path.read_text())['policy'] == policy
    assert str(path) in text and len(text) < 25000


def test_complete_package_queued_then_loaded_next_round(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'new actual Python function')
    result = runtime.update(decision='replace', round_number=1, candidate=folder, reason='specific missing behavior')
    assert result['status'] == 'queued'
    assert result['capability_gain_required'] is False
    assert runtime.binding(1)['version'] == 'G000'
    assert 'selected_method' not in runtime.phase({'round': 1, 'phase': 'route'})['planning']
    assert runtime.phase({'round': 2, 'phase': 'route'})['planning']['selected_method'] == 'new actual Python function'
    assert runtime.binding(1)['version'] == 'G000'  # reconnect/late receipt cannot rebind an old round
    assert runtime.binding(2)['version'] == 'G001'
    for version in ('G000', 'G001'):
        assert all((runtime.root / version / filename).is_file() for filename in FILES)
    traces = [json.loads(line) for line in (runtime.root / 'calls.jsonl').read_text().splitlines()]
    assert any(row['executed_version'] == 'G001' and row['module'] == 'planning' for row in traces)
    assert runtime.snapshot_reference(1)['selected_next']['version'] == 'G001'


def test_same_live_bridge_loads_new_workflow_and_planning_at_boundary(tmp_path):
    run = tmp_path / 'run'
    bridge = NativeExperimentTools(project=PROJECT, run=run, executor=None,
                                   progress=ExperimentProgress(run / 'meta_session', rounds=5))
    runtime = bridge.meta_program
    before = bridge.phase_instructions({'current_stage': 'B1', 'next_milestone': 'decision_recorded'})
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'wired to existing bridge')
    with (folder / 'workflow.py').open('a') as stream:
        stream.write("\n_old_prepare = prepare\ndef prepare(context):\n    result = _old_prepare(context)\n    result['new_workflow_behavior'] = True\n    return result\n")
    runtime.update(decision='replace', round_number=1, candidate=folder)
    assert 'wired to existing bridge' not in bridge.phase_instructions({'current_stage': 'B1', 'next_milestone': 'candidate_built'})
    after = bridge.phase_instructions({'current_stage': 'B2', 'next_milestone': 'decision_recorded'})
    assert 'wired to existing bridge' in after and 'new_workflow_behavior' in after
    assert 'G000' in before and 'G001' in after


@pytest.mark.parametrize('damage', ['missing', 'syntax', 'wrong_interface'])
def test_invalid_candidate_is_fact_not_experiment_exit(tmp_path, damage):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    if damage == 'missing':
        (folder / 'memory.py').unlink()
    elif damage == 'syntax':
        (folder / 'planning.py').write_text('def prepare(!!')
    else:
        (folder / 'workflow.py').write_text('def prepare(context): return []\n')
    result = runtime.update(decision='replace', round_number=1, candidate=folder)
    assert result['status'] == 'validation_failed'
    assert result['continue_same_meta'] is True
    assert runtime.binding(2)['version'] == 'G000'


def test_runtime_only_hook_failure_falls_back_with_trace(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    (folder / 'planning.py').write_text("def prepare(context):\n    if context.get('real_fail'): raise RuntimeError('runtime condition')\n    return {'ok': True}\n")
    assert runtime.update(decision='replace', round_number=1, candidate=folder)['status'] == 'queued'
    report = runtime.phase({'round': 2, 'phase': 'route', 'real_fail': True})
    trace = report['planning']['meta_harness_call']
    assert trace['requested_version'] == 'G001'
    assert trace['executed_version'] == 'G000'
    assert 'runtime condition' in trace['warning']


def test_keep_cancels_pending_without_deleting_history(tmp_path):
    runtime = MetaHarnessRuntime(PROJECT, tmp_path / 'run')
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'unused')
    runtime.update(decision='replace', round_number=1, candidate=folder)
    assert runtime.update(decision='keep', round_number=1)['status'] == 'retained'
    assert runtime.binding(2)['version'] == 'G000'
    assert (runtime.root / 'G001/planning.py').exists()


def test_fixed_program_and_fixed_skills_are_separate(tmp_path):
    run = tmp_path / 'run'
    prepare_meta_program(run, {'meta_harness_mode': 'fixed', 'meta_fixed_skills': True})
    tools = ControllerTools(PROJECT, run / 'controller_receipts')
    path = run / 'meta/skills.jsonl'
    path.touch()
    result = tools.execute({'operation': 'append_skills', 'path': str(path), 'round_number': 1,
                            'entries': [{'id': 'skill.MODEL.new', 'kind': 'case'}]})
    assert result['status'] == 'skills_frozen'
    assert path.read_text() == ''
    result = tools.execute({'operation': 'update_meta_harness', 'decision': 'replace', 'round_number': 1})
    assert result['status'] == 'meta_fixed'


def test_memory_python_change_reaches_real_tool_result(tmp_path):
    run = tmp_path / 'run'
    runtime = MetaHarnessRuntime(PROJECT, run)
    folder = candidate(runtime, tmp_path)
    with (folder / 'memory.py').open('a') as file:
        file.write("\n_original_retrieve = retrieve\ndef retrieve(path, fingerprint, options):\n    result = _original_retrieve(path, fingerprint, options)\n    result['new_memory_behavior'] = True\n    return result\n")
    runtime.update(decision='replace', round_number=1, candidate=folder)
    path = run / 'skills.jsonl'
    path.touch()
    tools = ControllerTools(PROJECT, run / 'controller_receipts')
    result = tools.execute({'operation': 'retrieve_skills', 'path': str(path), 'round_number': 2})
    assert result['new_memory_behavior'] is True
    assert result['meta_harness_call']['executed_version'] == 'G001'


def test_report_only_cases_and_dependent_rules_not_learning_input(tmp_path):
    path = tmp_path / 'skills.jsonl'
    SkillMemory(path).append([
        {'id': 'skill.MODEL.eval', 'kind': 'case', 'component': 'MODEL', 'purpose': 'report_only'},
        {'id': 'principle.eval', 'kind': 'rule', 'support': ['skill.MODEL.eval']},
        {'id': 'skill.MODEL.train', 'kind': 'case', 'component': 'MODEL', 'title': 'usable'}])
    tools = ControllerTools(PROJECT, tmp_path / 'run/controller_receipts')
    result = tools.execute({'operation': 'retrieve_skills', 'path': str(path), 'round_number': 1})
    assert 'skill.MODEL.train' in result['records']
    assert 'skill.MODEL.eval' not in result['records']
    assert 'principle.eval' not in result['records']
    assert len(SkillMemory(path).view()[0]) == 3  # source evidence is not deleted


def test_review_and_full_package_operations_available(tmp_path):
    tools = ControllerTools(PROJECT, tmp_path / 'run/controller_receipts')
    write_json(tmp_path / 'selection.json', {'component': 'HARNESS', 'delta': -0.02, 'decision': 'retain'})
    review = tools.execute({'operation': 'prepare_meta_review', 'selection_paths': [str(tmp_path / 'selection.json')], 'round_number': 1})
    assert review['review']['facts'][0]['delta'] == -0.02
    assert review['review']['decision_owner'] == 'Meta'
    folder = tmp_path / 'candidate'
    assert tools.execute({'operation': 'materialize_meta_harness', 'destination': str(folder), 'round_number': 1})['status'] == 'materialized'
    assert tools.execute({'operation': 'check_meta_harness', 'candidate_dir': str(folder), 'round_number': 1})['status'] == 'checked'
    assert tools.execute({'operation': 'update_meta_harness', 'candidate_dir': str(folder),
                          'decision': 'replace', 'round_number': 1})['status'] == 'queued'


def test_snapshot_context_and_next_package_are_immutable(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'harness.json', {'source': 'test'})
    write_json(run / 'task.json', {'model_ref': 'Qwen3-4B', 'checkpoint_path': '/weights',
                                  'checkpoint_manifest': {}, 'harness_path': str(run / 'harness.json')})
    write_json(run / 'context.json', {'round': 1})
    (run / 'skills.jsonl').touch()
    runtime = MetaHarnessRuntime(PROJECT, run)
    folder = candidate(runtime, tmp_path)
    replace_planning(folder, 'next')
    runtime.update(decision='replace', round_number=1, candidate=folder)
    tools = ControllerTools(PROJECT, run / 'controller_receipts')
    result = tools.execute({'operation': 'snapshot_task_meta', 'round_number': 1,
                            'active_task_state': str(run / 'task.json'), 'context_path': str(run / 'context.json'),
                            'skills_path': str(run / 'skills.jsonl'), 'destination': str(run / 'snapshots/B1')})
    assert result['status'] == 'snapshotted'
    write_json(run / 'context.json', {'round': 2})
    assert json.loads(Path(result['context_reference']['path']).read_text()) == {'round': 1}
    assert result['references']['meta_program']['used']['version'] == 'G000'
    assert result['references']['meta_program']['selected_next']['version'] == 'G001'
    assert result['weights_copied'] is False


def test_initial_prompt_and_default_config_do_not_resume_live_run(tmp_path):
    config = json.loads((PROJECT / 'configs/train_180_a0_v1.json').read_text())
    assert config['run_name'] == 'rsiflow_8b_harness_180_5round_8gpu_20260927'
    assert config['rounds'] == 5 and config['training_tasks_per_pass'] == 180
    assert config['evaluate_initial_system'] is True
    prompt = experiment_prompt(PROJECT / 'configs/train_180_a0_v1.json', config, tmp_path / 'run')
    assert 'update_meta_harness' in prompt and 'ALL THREE files' in prompt
    assert 'mainline' in prompt.lower()
    assert len(prompt) < 25000


def test_eval_task_prepare_does_not_launch_gpu(tmp_path):
    write_json(tmp_path / 'task.json', {'model_ref': 'Qwen3-4B', 'harness_path': 'H0.json'})
    report = prepare_task(PROJECT / 'configs/train_180_a0_v1.json', tmp_path / 'task.json', tmp_path / 'output')
    assert report['executed'] is False
    assert report['request']['operation'] == 'evaluate'
    snapshot = json.loads(Path(report['snapshot']['snapshot_path']).read_text())
    assert snapshot['purpose'] == 'report_only'


def test_meta_pair_freezes_code_skills_handoff_and_reuses_same_parent(tmp_path):
    write_json(tmp_path / 'task.json', {'model_ref': 'Qwen3-4B'})
    (tmp_path / 'skills.jsonl').write_text('')
    write_json(tmp_path / 'context.json', {'prior_round': 1})
    plan = prepare_pair(PROJECT / 'configs/train_180_a0_v1.json', tmp_path / 'task.json', tmp_path / 'parent',
                        PROJECT / 'meta_harness/G000', PROJECT / 'meta_harness/G000', tmp_path / 'skills.jsonl',
                        tmp_path / 'context.json', tmp_path / 'output', 2)
    left, right = plan['arms']['old'], plan['arms']['new']
    assert left['parent_rollout_dir'] == right['parent_rollout_dir']
    assert left['task_state_path'] == right['task_state_path']
    for arm in (left, right):
        state = json.loads((Path(arm['arm_dir']) / 'meta_harness/state.json').read_text())
        assert state['mode'] == 'fixed' and state['fixed_skills'] is True
        assert PairTools(arm).guidance()['planning']['meta_harness_call']['executed_version'] == 'G000'
    assert plan['executed'] is False


def test_pair_unmeasured_failure_is_not_zero_gain_and_can_finish(tmp_path):
    write_json(tmp_path / 'task.json', {'model_ref': 'Qwen3-4B'})
    (tmp_path / 'skills.jsonl').touch()
    plan = prepare_pair(PROJECT / 'configs/train_180_a0_v1.json', tmp_path / 'task.json', tmp_path / 'parent',
                        PROJECT / 'meta_harness/G000', PROJECT / 'meta_harness/G000', tmp_path / 'skills.jsonl',
                        None, tmp_path / 'output')
    tools = PairTools(plan['arms']['old'])
    assert 'incomplete' in tools('finish_experiment', {'summary': 'not yet'})
    tools('experiment', {'request': {'operation': 'report_failed_attempt', 'failure': 'cannot build candidate'}})
    assert tools.comparison['delta'] is None
    assert 'complete' in tools('finish_experiment', {'summary': 'unmeasured failure'})
    assert tools.finished


def test_loop_summary_does_not_interpret_missing_metrics_as_zero(tmp_path):
    write_json(tmp_path / 'fixed/round_1/selection.json', {'component': 'HARNESS', 'decision': 'retain', 'delta': -0.01})
    report = compare(tmp_path / 'fixed', tmp_path / 'evolving')
    assert report['fixed_meta']['rounds'][0]['attempt_count'] == 1
    assert report['evolving_meta']['validations'] == []
    assert report['purpose'] == 'report_only'
