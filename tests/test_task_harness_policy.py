import copy
import json
from pathlib import Path

import pytest

from sia.task_meta.task_harness import (
    edit_value,
    legacy_view,
    migrate_seed,
    task_harness_capabilities,
    validate_harness,
    validate_task_harness_request,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return migrate_seed(json.loads((ROOT / 'seed_harness/seed.json').read_text(encoding='utf-8')))


def test_representation_migration_preserves_old_parameters_and_state(policy):
    old = json.loads((ROOT / 'seed_harness/seed.json').read_text(encoding='utf-8'))
    assert legacy_view(policy) == old
    assert policy['parts']['submission']['checks'] == []
    assert policy['parts']['tools']['roles'] == []
    assert policy['parts']['tools']['recovery']['max_retries'] == 0
    assert policy['parts']['memory']['history']['max_chars'] is None
    assert policy['initialization']['artifacts'] == []


def test_capability_targets_are_accepted_by_same_request_validator(policy, tmp_path):
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(policy), encoding='utf-8')
    capabilities = task_harness_capabilities(path)
    requests = [{'component': 'HARNESS', 'operation': 'replace_config', 'target': target, 'harness_part': part}
                for target, part in capabilities['target_parts'].items()]
    validate_task_harness_request(policy, requests)
    assert {item['harness_part'] for item in requests} == {'input', 'control', 'tools', 'memory', 'submission'}
    assert {r['target'] for r in requests} == set(capabilities['operations'][0]['targets'])


@pytest.mark.parametrize('target', ['budget.max_model_calls', 'reference.commit', 'initialization.artifacts',
                                   '../seed.json', 'parts.tools.model', 'parts.memory.asset_contents'])
def test_protected_and_unregistered_targets_rejected(policy, target):
    with pytest.raises(ValueError):
        edit_value(policy, target, 'changed')


@pytest.mark.parametrize('bad_path', ['reward', 'state.reward', 'state.terminal_reward', 'item.task_id',
                                     'item.answers', 'state.private.tests', 'environment.evaluate'])
def test_predicates_cannot_query_ids_scores_or_hidden_data(policy, bad_path):
    policy['parts']['memory']['history']['filter'] = {'path': bad_path, 'op': 'exists', 'value': True}
    with pytest.raises(ValueError):
        validate_harness(policy)


def test_arbitrary_code_and_unknown_graph_destinations_are_rejected(policy):
    invalid = copy.deepcopy(policy)
    invalid['parts']['control']['graph']['nodes'][0]['kind'] = 'python'
    with pytest.raises(ValueError):
        validate_harness(invalid)
    invalid = copy.deepcopy(policy)
    invalid['parts']['control']['graph']['nodes'][0]['next'] = '/etc/passwd'
    with pytest.raises(ValueError):
        validate_harness(invalid)


def test_composed_branch_and_role_addition_are_real_legal_targets(policy):
    graph = policy['parts']['control']['graph']
    policy['parts']['tools']['roles'].append({'name': 'auditor', 'instruction': 'Review {{candidate}}.', 'result': 'review'})
    graph['nodes'].insert(0, {'id': 'audit', 'kind': 'role', 'role': 'auditor', 'next': 'prepare',
                             'when': {'all': [{'path': 'state.has_candidate', 'op': 'eq', 'value': True},
                                              {'not': {'path': 'state.has_final', 'op': 'eq', 'value': True}}]}})
    graph['entry'] = 'audit'
    validate_harness(policy)


def test_declared_part_must_match_dynamic_target_owner(policy):
    request = {'component': 'HARNESS', 'operation': 'replace_config', 'target': 'parts.control.graph', 'harness_part': 'memory'}
    with pytest.raises(ValueError, match='ownership'):
        validate_task_harness_request(policy, [request])


def test_task_cannot_replace_internal_checks_with_official_scoring(policy):
    policy['parts']['submission']['checks'] = [{'name': 'cheat', 'kind': 'official_reward'}]
    with pytest.raises(ValueError, match='official scoring'):
        validate_harness(policy)


def test_input_cannot_omit_task_and_no_change_is_not_an_h_patch(policy):
    with pytest.raises(ValueError, match='unchanged task'):
        edit_value(policy, 'parts.input.task_template', 'Always output an answer.')
    with pytest.raises(ValueError, match='did not change'):
        edit_value(policy, 'parts.control.graph', policy['parts']['control']['graph'])


@pytest.mark.parametrize('target', ['parts.tools.recovery', 'parts.submission.repair_prompt',
                                   'parts.tools.roles', 'parts.input.task_template', 'parts.input.action_step',
                                   'parts.control.prompts.planning_task', 'parts.control.prompts.summary_post',
                                   'parts.submission.prompts.final_post'])
def test_new_templates_cannot_request_variables_their_runtime_does_not_bind(policy, target):
    text = '{{task}} {{previous_steps}}'
    value = {'max_retries': 1, 'prompt': text} if target.endswith('recovery') else (
        [{'name': 'planner', 'instruction': text, 'result': 'plan'}] if target.endswith('roles') else text)
    with pytest.raises(ValueError, match='undeclared template variable'):
        edit_value(policy, target, value)
