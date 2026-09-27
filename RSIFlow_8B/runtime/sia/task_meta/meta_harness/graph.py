"""Method 2: bounded control/data edges over the existing Meta handlers.

Workflows own handler content; graphs own connectivity and input contracts. No
new agent backend, Python evaluator, side-effect executor or global context bus.
"""
from __future__ import annotations
import copy
import hashlib
import json

TYPES = ('Context', 'Memory', 'Planning', 'Reasoning', 'Tools', 'Verification')
KINDS = {'evidence': 'Context', 'experience': 'Memory', 'propose': 'Planning',
         'analyze': 'Reasoning', 'repair': 'Reasoning', 'read_dependencies': 'Tools',
         'inspect_targets': 'Tools', 'check': 'Verification'}
FIELDS = ('evidence', 'experience_context', 'dependencies', 'checks', 'analyses',
          'candidate', 'candidate_valid', 'candidate_delivery_valid', 'last_check_passed')
ROOT_FIELDS = ('trusted_facts', 'task_state', 'decision', 'instruction', 'raw_trajectories',
               'experiences', 'current_files', 'outcome_review', 'native_candidate_repair')
CONTENT_KEYS = {'instruction', 'paths', 'max_chars', 'offset'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def migrate(policy):
    """Equivalent ordered paths; type labels add no calls or evidence."""
    result = copy.deepcopy(policy)
    if result['schema_version'] == 2:
        return result
    graphs = {}
    for operation, steps in result['workflows'].items():
        nodes = {}
        edges = []
        for index, step in enumerate(steps):
            nodes[step['id']] = {'type': KINDS[step['kind']], 'handler_version': 1,
                'input_schema': 'meta-state-v1', 'output_schema': 'meta-state-v1',
                'root_fields': list(ROOT_FIELDS), 'max_visits': 1}
            edges.append({'source': step['id'],
                'target': steps[index + 1]['id'] if index + 1 < len(steps) else None,
                'when': True, 'fields': list(FIELDS)})
        graphs[operation] = {'entry': steps[0]['id'], 'nodes': nodes, 'edges': edges,
                             'max_steps': len(steps)}
    result.update(schema_version=2, graphs=graphs,
                  method2={'graph_version': 0, 'last_slow_generation': -2,
                           'mismatches': [], 'edits': []})
    return result


def topology(policy):
    p = migrate(policy)
    return {'graphs': p['graphs'], 'handlers': {op: sorted(
        ({k: v for k, v in step.items() if k not in CONTENT_KEYS} for step in steps),
        key=lambda x: x['id']) for op, steps in p['workflows'].items()}}


def identity(policy):
    p = migrate(policy)
    return {'graph_version': p['method2']['graph_version'],
            'graph_hash': fingerprint(topology(p)),
            'node_content_hash': fingerprint({k: p[k] for k in
                ('evidence', 'experience', 'diagnosis', 'self_update', 'workflows')}),
            'component_types': list(TYPES)}


def validate(policy):
    from .policies import validate_condition
    if set(policy['graphs']) != set(policy['workflows']):
        raise ValueError('Graph operations must match the declared workflows')
    meta = policy['method2']
    if set(meta) != {'graph_version', 'last_slow_generation', 'mismatches', 'edits'}:
        raise ValueError('Invalid Method 2 controller metadata')
    if type(meta['graph_version']) is not int or meta['graph_version'] < 0:
        raise ValueError('Invalid graph version')
    if len(meta['mismatches']) > 64 or len(meta['edits']) > 32:
        raise ValueError('Method 2 history limit exceeded')
    for op, graph in policy['graphs'].items():
        if set(graph) != {'entry', 'nodes', 'edges', 'max_steps'}:
            raise ValueError('Invalid graph fields')
        steps = {step['id']: step for step in policy['workflows'][op]}
        if set(graph['nodes']) != set(steps) or graph['entry'] not in steps:
            raise ValueError('Graph nodes must own exactly the existing handler entries')
        if type(graph['max_steps']) is not int or not 1 <= graph['max_steps'] <= 48:
            raise ValueError('Graph step bound must be in [1,48]')
        if not 1 <= len(graph['edges']) <= 64:
            raise ValueError('Invalid bounded edges')
        for key, node in graph['nodes'].items():
            if set(node) != {'type', 'handler_version', 'input_schema', 'output_schema', 'root_fields', 'max_visits'}:
                raise ValueError('Invalid node interface')
            if (node['type'] != KINDS[steps[key]['kind']] or node['handler_version'] != 1
                or node['input_schema'] != 'meta-state-v1' or node['output_schema'] != 'meta-state-v1'):
                raise ValueError('Handler ownership/schema mismatch')
            if not isinstance(node['root_fields'], list) or not set(node['root_fields']) <= set(ROOT_FIELDS):
                raise ValueError('Node requested undeclared root input')
            if type(node['max_visits']) is not int or not 1 <= node['max_visits'] <= 3:
                raise ValueError('Node revisit bound exceeded')
            if steps[key]['kind'] == 'propose' and node['max_visits'] != 1:
                raise ValueError('Only one Task candidate proposal is permitted')
        outgoing = {key: [] for key in steps}
        for edge in graph['edges']:
            if set(edge) != {'source', 'target', 'when', 'fields'}:
                raise ValueError('Invalid control/data edge')
            if edge['source'] not in steps or edge['target'] not in {*steps, None}:
                raise ValueError('Unknown edge endpoint')
            validate_condition(edge['when'])
            if (not isinstance(edge['fields'], list) or len(set(edge['fields'])) != len(edge['fields'])
                or not set(edge['fields']) <= set(FIELDS)):
                raise ValueError('Invalid data projection')
            outgoing[edge['source']].append(edge)
        for edges in outgoing.values():
            if not edges or edges[-1]['when'] is not True or any(e['when'] is True for e in edges[:-1]):
                raise ValueError('Every node needs ordered guards and one final unconditional edge')
        seen, todo = set(), [graph['entry']]
        while todo:
            key = todo.pop()
            if key is None or key in seen:
                continue
            seen.add(key)
            todo.extend(edge['target'] for edge in outgoing[key])
        if seen != set(steps):
            raise ValueError('Unreachable node; prune it explicitly')
        # Protected candidate checks remain outside the graph. In addition,
        # every terminating structural path must encounter exactly one proposal.
        queue = [(graph['entry'], frozenset(), 0, frozenset(FIELDS))]
        explored = set()
        while queue:
            key, visited, proposals, fields = queue.pop()
            signature = (key, visited, proposals, fields)
            if signature in explored:
                continue
            explored.add(signature)
            if len(explored) > 4096:
                raise ValueError('Graph path analysis budget exceeded')
            if key is None:
                if proposals != 1:
                    raise ValueError('A terminal path bypasses the unique proposal')
                continue
            kind = steps[key]['kind']
            required = {'candidate', 'candidate_valid', 'candidate_delivery_valid', 'last_check_passed', 'checks'}
            if kind in {'propose', 'repair', 'analyze'}:
                required |= set(FIELDS)
            if not required <= fields:
                raise ValueError('Data edge omits required node inputs')
            if kind == 'repair' and not proposals:
                raise ValueError('Repair cannot precede its candidate')
            if kind == 'propose':
                proposals += 1
                if proposals > 1:
                    raise ValueError('A control path revisits candidate proposal')
            if key in visited:
                # Loops are bounded at runtime; traversing a non-proposal cycle
                # does not create a new candidate or expose future evidence.
                continue
            output_fields = set(fields)
            output_fields |= {'evidence'} if kind == 'evidence' else set()
            output_fields |= {'experience_context'} if kind == 'experience' else set()
            output_fields |= {'dependencies'} if kind in {'read_dependencies', 'inspect_targets'} else set()
            for edge in outgoing[key]:
                if not set(edge['fields']) <= output_fields:
                    raise ValueError('Data edge references unavailable output')
                queue.append((edge['target'], visited | {key}, proposals, frozenset(edge['fields'])))
    return policy


class Cursor:
    """One version-pinned operation. Edges carry only explicit, fresh fields."""
    def __init__(self, graph, steps, initial):
        self.graph, self.steps = graph, {s['id']: s for s in steps}
        self.node = graph['entry']
        self.initial = copy.deepcopy(initial)
        self.input = copy.deepcopy(initial)
        self.visits = {}
        self.count = 0
        self.proposals = 0

    def enter(self):
        if self.node is None:
            return None
        self.count += 1
        self.visits[self.node] = self.visits.get(self.node, 0) + 1
        if self.count > self.graph['max_steps'] or self.visits[self.node] > self.graph['nodes'][self.node]['max_visits']:
            raise ValueError('Meta graph loop/step limit exceeded')
        step = copy.deepcopy(self.steps[self.node])
        return step, copy.deepcopy(self.input), self.visits[self.node]

    def leave(self, output, condition, scope):
        scope = {**scope, 'graph': {'node_id': self.node, 'iteration': self.visits[self.node],
                                  'steps': self.count, 'visits': copy.deepcopy(self.visits)}}
        edges = [e for e in self.graph['edges'] if e['source'] == self.node]
        edge = next(e for e in edges if condition(e['when'], scope))
        missing = set(edge['fields']) - set(output)
        if missing:
            raise ValueError('Uninitialized graph outputs: ' + ','.join(sorted(missing)))
        self.input = {k: copy.deepcopy(output[k]) for k in edge['fields']}
        record = {'source': self.node, 'target': edge['target'], 'fields': edge['fields'],
                  'iteration': self.visits[self.node], 'input_hash': fingerprint(self.input),
                  'freshness': 'immediate_predecessor_output'}
        self.node = edge['target']
        return record


def apply_slow(policy, proposal, mismatches, *, experience_id):
    """Apply at most one structural edit on the already-validated fast version."""
    from .policies import validate_policy
    result = copy.deepcopy(policy)
    if not proposal:
        return result, {'status': 'not_proposed'}
    p = proposal.model_dump(mode='json') if hasattr(proposal, 'model_dump') else proposal
    if p['operation'] not in {'Insert', 'Prune', 'Rewire'}:
        raise ValueError('Unknown slow operation')
    try:
        generation = int(experience_id.rsplit('_', 1)[1])
    except (AttributeError, ValueError, IndexError):
        raise ValueError('Slow update requires a completed intervention experience')
    relevant = [x for x in mismatches if x['issue_id'] == p['issue_id'] and x['target_workflow'] == p['workflow']]
    independent = {x['experience_id'] for x in relevant}
    if not independent:
        raise ValueError('Slow edit requires recorded structural mismatch evidence after content review')
    if set(p['experience_ids']) != independent or not p['replay_requirements']:
        raise ValueError('Slow edit must cite accumulated mismatch experiences and replay requirements')
    op = p['workflow']
    if op not in result['graphs']:
        raise ValueError('Unknown target operation')
    previous = {s['id']: s for s in result['workflows'][op]}
    proposed = {s['id']: s for s in p['steps']}
    added, removed = set(proposed) - set(previous), set(previous) - set(proposed)
    kind = p['operation']
    if ((kind == 'Insert' and (len(added) != 1 or removed))
        or (kind == 'Prune' and (len(removed) != 1 or added))
        or (kind == 'Rewire' and (added or removed))):
        raise ValueError('Slow operation does not match its node delta')
    for key in previous.keys() & proposed.keys():
        if previous[key] != proposed[key]:
            raise ValueError('Slow editing cannot overwrite retained fast node content')
    result['workflows'][op] = copy.deepcopy(p['steps'])
    result['graphs'][op] = copy.deepcopy(p['graph'])
    validate_policy(result)
    before, after = fingerprint(topology(policy)), fingerprint(topology(result))
    if before == after:
        raise ValueError('Slow edit has no structural effect')
    result['method2']['graph_version'] += 1
    result['method2']['last_slow_generation'] = generation
    event = {'operation': kind, 'workflow': op, 'issue_id': p['issue_id'],
             'experience_ids': sorted(independent), 'parent_graph_hash': before,
             'graph_hash': after, 'status': 'validated',
             'replay_requirements': p['replay_requirements'],
             'validation_kind': 'bounded_control_data_path_analysis',
             'semantic_effect': 'not_yet_observed', 'model_replay_performed': False}
    result['method2']['edits'].append(event)
    return result, event
