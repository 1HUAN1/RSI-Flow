"""Task-specific JSON state machine, contracts, migration and actual capabilities.

Policies cannot execute code or acquire models, files, tools, scores or datasets.
The graph only arranges the registered runtime primitives. Its mutable content is
separate from the fixed source identity captured by the existing run protocol.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

from sia.task_meta.meta_harness.bundle import reject_links, strict_json
from sia.task_meta.meta_harness.policies import validate_condition

PARTS = ('input', 'control', 'tools', 'memory', 'submission')
INTERFACE_VERSION = 'task-harness-v2'
MAX_TRANSITIONS = 512
NODE_KINDS = {'prepare', 'memory', 'act', 'tools', 'inspect', 'commit', 'finalize', 'role', 'branch', 'stop'}
STATE_FIELDS = {'has_final', 'has_candidate', 'can_continue', 'needs_repair', 'ready_to_submit',
                'step_number', 'last_tool_error', 'tool_success_count', 'evidence_count',
                'memory_count', 'last_check_passed', 'last_role', 'remaining_model_calls',
                'remaining_tool_calls', 'candidate', 'step', 'tool_success'}
ID = re.compile(r'^[a-z][a-z0-9_]{0,47}$')
PROTECTED = ('budget', 'initialization', 'reference', 'interface_version', 'schema_version',
             'name', 'adaptations', 'alignment_status')
FIELDS = {
    'input': {'action_system', 'task_template', 'action_step', 'section_order'},
    'control': {'planning', 'prompts', 'graph'},
    'tools': {'action', 'validate_arguments', 'recovery', 'roles'},
    'memory': {'context', 'prompts', 'history', 'assets'},
    'submission': {'prompts', 'strip_final_answer', 'checks', 'repair_prompt', 'answer_format'},
}
EDITABLE = {
    'input': ('action_system', 'task_template', 'action_step', 'section_order'),
    'control': ('planning.enabled', 'planning.summary_interval', 'planning.max_steps',
                'prompts.planning_initial', 'prompts.planning_task', 'prompts.summary_pre', 'prompts.summary_post', 'graph'),
    'tools': ('action.max_tools_per_step', 'action.parse_retries', 'validate_arguments', 'recovery', 'roles'),
    'memory': ('context.shortterm_enabled', 'context.shortterm_interval', 'context.max_shortterm_items',
               'context.use_artifacts', 'context.artifact_char_limit', 'context.generate_artifacts',
               'prompts.memory_extract', 'prompts.memory_prune', 'history', 'assets'),
    'submission': ('prompts.final_pre', 'prompts.final_post', 'strip_final_answer', 'checks', 'repair_prompt', 'answer_format'),
}
PART_DESCRIPTIONS = {
    'input': 'Serialize the unchanged task, genuine tool descriptions and memory-selected context into model requests.',
    'control': 'Connect bounded Task steps, conditional branches, planning, review and stopping in one state machine.',
    'tools': 'Dispatch existing tool interfaces and current-Qwen roles; validate and repair parameters within remaining budget.',
    'memory': 'Select visible history and currently active assets; maintain fresh per-rollout memory and natural output rules.',
    'submission': 'Check visible candidates, issue repair signals and serialize the actual final output; never score official rewards.',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _keys(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f'{label} must contain exactly its declared fields')


def _number(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}]')


ROLE_VARIABLES = {'task', 'candidate', 'observations', 'review', 'step', 'error', 'schema', 'history'}
PROMPT_VARIABLES = {
    'input.action_system': set(),
    'input.task_template': {'task'},
    'input.action_step': {'task', 'tool_functions_json'},
    'control.prompts.planning_initial': set(),
    'control.prompts.planning_task': {'task'},
    'control.prompts.summary_pre': {'task', 'step'},
    'control.prompts.summary_post': {'task', 'step'},
    'memory.prompts.memory_extract': {'task', 'previous_steps', 'current_memory', 'context_delta'},
    'memory.prompts.memory_prune': {'task', 'previous_steps', 'memory_items', 'max_shortterm_items'},
    'submission.prompts.final_pre': set(),
    'submission.prompts.final_post': {'task'},
}


def _text(value, label, *, empty=False, variables=None):
    if not isinstance(value, str) or (not value.strip() and not empty) or len(value) > 24000:
        raise ValueError(f'{label} must be bounded text')
    if '{%' in value:
        raise ValueError('Executable templates are not supported')
    placeholders = {'task', 'tool_functions_json', 'previous_steps', 'current_memory', 'context_delta',
                    'memory_items', 'max_shortterm_items', 'step', 'candidate', 'error', 'schema', 'history',
                    'observations', 'review'}
    if set(re.findall(r'{{\s*(.*?)\s*}}', value)) - (placeholders if variables is None else variables):
        raise ValueError(f'{label} uses an undeclared template variable')


def _condition(value, *, selection=False):
    validate_condition(value)
    def walk(expression):
        if isinstance(expression, dict):
            for key, item in expression.items():
                if key in {'path', 'value_path'}:
                    sections = item.split('.')
                    head = sections[0]
                    if (head not in {'state', 'item'} or (head == 'item' and not selection)
                            or (head == 'state' and (len(sections) < 2 or sections[1] not in STATE_FIELDS))):
                        raise ValueError('Task conditions can inspect only declared visible state')
                    if any(part in {'reward', 'terminal_reward', 'hidden', 'answers', 'references', 'task_id',
                                    'labels', 'private', 'test_cases', 'final_benchmark'} for part in item.split('.')):
                        raise ValueError('Task predicates cannot inspect answer keys, IDs or reward labels')
                elif isinstance(item, list):
                    for child in item:
                        walk(child)
                elif isinstance(item, dict):
                    walk(item)
    walk(value)


def _graph(graph, roles):
    _keys(graph, {'entry', 'nodes'}, 'control.graph')
    nodes = graph['nodes']
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32:
        raise ValueError('Task graph must contain 1 to 32 bounded nodes')
    ids = set()
    targets = []
    for node in nodes:
        if not isinstance(node, dict) or not {'id', 'kind', 'next'} <= set(node) or set(node) - {'id', 'kind', 'next', 'when', 'role'}:
            raise ValueError('Undeclared Task graph node field')
        if not isinstance(node['id'], str) or not ID.fullmatch(node['id']) or node['id'] in ids:
            raise ValueError('Task node IDs must be safe and unique')
        ids.add(node['id'])
        if node['kind'] not in NODE_KINDS:
            raise ValueError('Task graph contains an unregistered executable primitive')
        if 'when' in node:
            _condition(node['when'])
        if node['kind'] == 'role':
            if node.get('role') not in roles:
                raise ValueError('Task graph role is not declared in tools.roles')
        elif 'role' in node:
            raise ValueError('Only role nodes may select a Task role')
        edges = node['next']
        if isinstance(edges, str):
            targets.append(edges)
        elif isinstance(edges, list) and 1 <= len(edges) <= 8:
            for edge in edges:
                _keys(edge, {'when', 'to'}, 'control transition')
                _condition(edge['when'])
                if not isinstance(edge['to'], str):
                    raise ValueError('Transition destination must name a node')
                targets.append(edge['to'])
            if edges[-1]['when'] is not True:
                raise ValueError('Conditional transitions need one final unconditional destination')
        else:
            raise ValueError('Invalid Task graph transition')
    if graph['entry'] not in ids or not set(targets) <= ids:
        raise ValueError('Task graph references missing nodes')
    if not any(node['kind'] == 'stop' for node in nodes):
        raise ValueError('Task graph must declare a stop node')
    # Cycles are legitimate re-planning. Runtime hard-limits every transition and
    # every real model/tool attempt independently, including cycles without calls.


def legacy_view(spec):
    """Mechanical v1 parameter view; executable v2 rules are not discarded at run time."""
    parts = spec['parts']
    prompts = {'action_system': parts['input']['action_system'], 'action_step': parts['input']['action_step']}
    for part in ('control', 'memory', 'submission'):
        prompts.update(parts[part]['prompts'])
    return {**{key: copy.deepcopy(spec[key]) for key in PROTECTED if key not in {'interface_version', 'schema_version'}},
            'schema_version': 1, 'prompts': prompts, 'planning': copy.deepcopy(parts['control']['planning']),
            'context': copy.deepcopy(parts['memory']['context']),
            'action': {**copy.deepcopy(parts['tools']['action']), 'strip_final_answer': parts['submission']['strip_final_answer']}}


def migrate_seed(spec):
    """Explicit representation migration. Old seed files and old execution stay intact."""
    from sia.task_meta.seed import _validate_seed_v1
    _validate_seed_v1(spec)
    prompts = spec['prompts']
    chain = ['prepare', 'memory', 'act', 'tools', 'inspect', 'commit', 'branch']
    nodes = [{'id': name, 'kind': name, 'next': following} for name, following in pairwise(chain)]
    nodes.extend([
        {'id': 'branch', 'kind': 'branch', 'next': [
            {'when': {'path': 'state.has_final', 'op': 'eq', 'value': True}, 'to': 'stop'},
            {'when': {'path': 'state.can_continue', 'op': 'eq', 'value': True}, 'to': 'prepare'},
            {'when': True, 'to': 'finalize'}]},
        {'id': 'finalize', 'kind': 'finalize', 'next': 'stop'},
        {'id': 'stop', 'kind': 'stop', 'next': 'stop'},
    ])
    result = {key: copy.deepcopy(spec[key]) for key in PROTECTED if key not in {'interface_version', 'schema_version'}}
    result.update(schema_version=2, interface_version=INTERFACE_VERSION, parts={
        'input': {'action_system': prompts['action_system'], 'task_template': 'New task:\n{{task}}',
                  'action_step': prompts['action_step'], 'section_order': ['history', 'guidance', 'action']},
        'control': {'planning': copy.deepcopy(spec['planning']),
                    'prompts': {name: prompts[name] for name in ('planning_initial', 'planning_task', 'summary_pre', 'summary_post')},
                    'graph': {'entry': 'prepare', 'nodes': nodes}},
        'tools': {'action': {name: spec['action'][name] for name in ('max_tools_per_step', 'parse_retries')},
                  'validate_arguments': False,
                  'recovery': {'max_retries': 0, 'prompt': 'Repair the failed tool arguments from the genuine error and original tool schema. Return strict JSON with an arguments object.'},
                  'roles': []},
        'memory': {'context': copy.deepcopy(spec['context']),
                   'prompts': {name: prompts[name] for name in ('memory_extract', 'memory_prune')},
                   'history': {'filter': True, 'max_messages': None, 'max_chars': None, 'required_roles': ['system']},
                   'assets': {'filter': True, 'max_items': 100000}},
        'submission': {'prompts': {name: prompts[name] for name in ('final_pre', 'final_post')},
                       'strip_final_answer': spec['action']['strip_final_answer'], 'checks': [],
                       'repair_prompt': 'Repair the current candidate using only visible task information and the internal check errors.',
                       'answer_format': {'mode': 'identity', 'key': 'answer'}},
    })
    return validate_harness(result)


def validate_harness(spec):
    from sia.task_meta.seed import _validate_seed_v1
    if not isinstance(spec, dict):
        raise ValueError('Task Harness must be a JSON object')
    if spec.get('schema_version') == 1:
        return _validate_seed_v1(spec)
    _keys(spec, set(PROTECTED) | {'parts'}, 'Task Harness v2')
    if type(spec['schema_version']) is not int or spec['schema_version'] != 2 or spec['interface_version'] != INTERFACE_VERSION:
        raise ValueError('Unsupported Task Harness interface')
    if len(canonical(spec)) > 500000:
        raise ValueError('Task Harness exceeds the fixed policy size ceiling')
    _keys(spec['parts'], PARTS, 'Task Harness parts')
    for part in PARTS:
        _keys(spec['parts'][part], FIELDS[part], 'parts.' + part)
    parts = spec['parts']
    _validate_seed_v1(legacy_view(spec))
    for target, variables in PROMPT_VARIABLES.items():
        value = parts
        for section in target.split('.'):
            value = value[section]
        _text(value, target, variables=variables)
    if not re.search(r'{{\s*task\s*}}', parts['input']['task_template']):
        raise ValueError('Input task template must include the unchanged task')
    order = parts['input']['section_order']
    if not isinstance(order, list) or sorted(order) != ['action', 'guidance', 'history']:
        raise ValueError('Input must present history, selected guidance and action instructions exactly once')
    tools = parts['tools']
    if type(tools['validate_arguments']) is not bool:
        raise ValueError('validate_arguments must be boolean')
    _keys(tools['recovery'], {'max_retries', 'prompt'}, 'tools.recovery')
    _number(tools['recovery']['max_retries'], 0, 2, 'tools.recovery.max_retries')
    _text(tools['recovery']['prompt'], 'tools.recovery.prompt', variables=ROLE_VARIABLES)
    if not isinstance(tools['roles'], list) or len(tools['roles']) > 8:
        raise ValueError('At most eight current-Task-model roles are permitted')
    roles = set()
    for role in tools['roles']:
        _keys(role, {'name', 'instruction', 'result'}, 'Task role')
        if not isinstance(role['name'], str) or not ID.fullmatch(role['name']) or role['name'] in roles:
            raise ValueError('Role names must be safe and unique')
        roles.add(role['name'])
        _text(role['instruction'], 'Task role instruction', variables=ROLE_VARIABLES)
        if role['result'] not in {'plan', 'review', 'memory', 'context'}:
            raise ValueError('Task role result is not a declared state destination')
    _graph(parts['control']['graph'], roles)
    history = parts['memory']['history']
    _keys(history, {'filter', 'max_messages', 'max_chars', 'required_roles'}, 'memory.history')
    _condition(history['filter'], selection=True)
    if history['max_messages'] is not None:
        _number(history['max_messages'], 1, 4096, 'history.max_messages')
    if history['max_chars'] is not None:
        _number(history['max_chars'], 100, 1000000, 'history.max_chars')
    if (not isinstance(history['required_roles'], list) or len(history['required_roles']) != len(set(history['required_roles']))
            or not set(history['required_roles']) <= {'system', 'user', 'assistant', 'tool-response'}):
        raise ValueError('History required_roles contains an unknown role')
    assets = parts['memory']['assets']
    _keys(assets, {'filter', 'max_items'}, 'memory.assets')
    _condition(assets['filter'], selection=True)
    _number(assets['max_items'], 0, 100000, 'assets.max_items')
    submission = parts['submission']
    _text(submission['repair_prompt'], 'submission.repair_prompt', variables=ROLE_VARIABLES)
    _keys(submission['answer_format'], {'mode', 'key'}, 'submission.answer_format')
    if submission['answer_format']['mode'] not in {'identity', 'last_line', 'json_key'}:
        raise ValueError('Unknown actual-output parser')
    if not isinstance(submission['answer_format']['key'], str) or not ID.fullmatch(submission['answer_format']['key']):
        raise ValueError('Output JSON key must be an identifier')
    if not isinstance(submission['checks'], list) or len(submission['checks']) > 8:
        raise ValueError('At most eight internal checks are allowed')
    check_ids = set()
    for check in submission['checks']:
        if (not isinstance(check, dict) or not {'name', 'kind'} <= set(check)
                or set(check) - {'name', 'kind', 'minimum', 'role'}):
            raise ValueError('Each internal check requires name and kind; only name, kind, minimum, role are allowed. Supported kinds: nonempty, tool_success, evidence_count, model_review. Names must be safe and unique.')
        if not isinstance(check['name'], str) or not ID.fullmatch(check['name']) or check['name'] in check_ids:
            raise ValueError('Internal check names must be safe and unique')
        check_ids.add(check['name'])
        if check['kind'] not in {'nonempty', 'tool_success', 'evidence_count', 'model_review'}:
            raise ValueError('Internal checks cannot access official scoring')
        if check['kind'] == 'model_review':
            if check.get('role') not in roles or 'minimum' in check:
                raise ValueError('Model review requires a declared current-Qwen role')
        elif 'role' in check:
            raise ValueError('Only model_review can invoke a review role')
        if 'minimum' in check:
            if check['kind'] not in {'tool_success', 'evidence_count'}:
                raise ValueError('This check does not consume a count')
            _number(check['minimum'], 0, 100, 'internal check minimum')
    return spec


def load_harness(path):
    path = Path(path)
    reject_links(path)
    return validate_harness(strict_json(path.read_text(encoding='utf-8-sig')))


def runtime_dependencies():
    """Fixed Task/Meta tree plus its standalone package/entry dependencies.

    This is an explicit inventory, not a search of user directories or candidate
    siblings. Third-party packages, models and benchmark data stay external.
    """
    project = Path(__file__).resolve().parents[3]
    files = [p for p in (project / 'sia/task_meta').rglob('*.py') if '__pycache__' not in p.parts]
    files.extend(project / name for name in (
        'sia/__init__.py', 'sia/layout.py', 'sia/prompts.py', 'sia/config.py',
        'scripts/run_multidomain.py', 'scripts/train_task_meta_sft.py', 'pyproject.toml'))
    return {p.relative_to(project).as_posix(): p for p in sorted(files) if p.is_file()}


def harness_identity(path):
    spec = load_harness(path)
    dependencies = runtime_dependencies()
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in dependencies.items()}
    return {'interface_version': INTERFACE_VERSION if spec['schema_version'] == 2 else 'legacy-task-seed-v1',
            'schema_version': spec['schema_version'], 'config_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'semantic_hash': fingerprint(spec), 'fixed_runtime_hash': fingerprint(hashes), 'fixed_runtime_files': hashes}


def _targets():
    return {'parts.' + part + '.' + leaf: part for part, leaves in EDITABLE.items() for leaf in leaves}


def task_harness_capabilities(path):
    spec = load_harness(path)
    if spec['schema_version'] == 1:
        from sia.task_meta.seed import seed_capabilities
        return seed_capabilities(path)
    targets = _targets()
    return {'available': True, 'interface_version': INTERFACE_VERSION, 'seed_sha256': fingerprint(spec),
            'component_types': ['Context', 'Memory', 'Planning', 'Reasoning', 'Tools', 'Verification'],
            'legacy_part_mapping': {'input': ['Context'], 'control': ['Planning', 'Reasoning'],
                'tools': ['Tools'], 'memory': ['Context', 'Memory'], 'submission': ['Verification']},
            'mapping_mode': 'compatibility_labels_only_no_new_capabilities_or_model_calls',
            'operations': [{'operation': 'replace_config', 'targets': sorted(targets)}], 'target_parts': targets,
            'part_specs': {part: {'description': PART_DESCRIPTIONS[part], 'targets': [t for t, p in targets.items() if p == part]}
                           for part in PARTS},
            'value_contracts': {'parts.control.graph': {'node_kinds': sorted(NODE_KINDS), 'max_nodes': 32,
                                                       'max_transitions': MAX_TRANSITIONS, 'conditions': ['state.' + field for field in sorted(STATE_FIELDS)]},
                                'parts.tools.roles': {'max_roles': 8, 'model_binding': 'current Task checkpoint only',
                                                      'template_variables': sorted(ROLE_VARIABLES)},
                                'parts.submission.checks': {'kinds': ['nonempty', 'tool_success', 'evidence_count', 'model_review']},
                                'template_variables': {'parts.' + key: sorted(value) for key, value in PROMPT_VARIABLES.items()},
                                'complete_current_values': 'seed.json', 'fixed_validator_source': 'runtime/sia/task_meta/task_harness/policy.py'},
            'budget': copy.deepcopy(spec['budget']),
            'constraints': ['A HARNESS intervention may atomically change multiple declared targets and multiple parts',
                            'Every requested target must name its matching harness_part; unrequested changes are forbidden',
                            'Policies are bounded JSON; arbitrary code, model/provider changes and tools permissions are unavailable',
                            'Raw task/answers/IDs/rewards, evaluator, original observations, assets bytes and external budgets are protected',
                            'Expected semantic benefit is unverified; internal review is not an official score'],
            'protected_fields': list(PROTECTED)}


def task_harness_sources(path):
    spec = load_harness(path)
    sources = {'seed.json': json.dumps(spec, ensure_ascii=False, indent=2)}
    # Give the actual loaded strategy and its narrow fixed implementation context,
    # without granting the Meta tool access to the host, dataset or API secrets.
    project = Path(__file__).resolve().parents[3]
    for name in ('sia/task_meta/seed.py', 'sia/task_meta/task_harness/policy.py', 'sia/task_meta/task_harness/runtime.py'):
        source = project / name
        if source.is_file():
            sources['runtime/' + name] = source.read_text(encoding='utf-8')
    return sources


def validate_task_harness_request(spec_or_path, requested_changes):
    spec = load_harness(spec_or_path) if isinstance(spec_or_path, (str, Path)) else validate_harness(spec_or_path)
    from sia.task_meta.seed import EDITABLE_SEED_PATHS
    targets = _targets() if spec['schema_version'] == 2 else dict.fromkeys(EDITABLE_SEED_PATHS)
    seen = set()
    if not requested_changes:
        raise ValueError('A Task Harness request must declare concrete targets')
    for change in requested_changes:
        item = change.model_dump(mode='json') if hasattr(change, 'model_dump') else change
        if (item.get('component') != 'HARNESS' or item.get('operation') != 'replace_config'
                or item.get('target') not in targets or item['target'] in seen):
            raise ValueError('Task Harness request must declare distinct legal replace_config targets')
        if spec['schema_version'] == 2 and item.get('harness_part') != targets[item['target']]:
            raise ValueError('Task Harness request must declare the actual five-part ownership')
        seen.add(item['target'])


def edit_value(spec, target, value):
    if target not in _targets():
        raise ValueError('Undeclared Task Harness target')
    result = copy.deepcopy(spec)
    current = result
    sections = target.split('.')
    for section in sections[:-1]:
        current = current[section]
    if current[sections[-1]] == value:
        raise ValueError('Declared Task Harness target did not change')
    current[sections[-1]] = copy.deepcopy(value)
    return validate_harness(result)
