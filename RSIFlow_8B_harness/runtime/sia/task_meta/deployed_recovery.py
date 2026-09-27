"""Explicit recovery of an evaluated, deployed round whose Meta commit is pending.

Original protocol bytes and evidence remain archived; no rollout/updater is called.
"""
import copy
import json
from dataclasses import asdict
from pathlib import Path

from sia.task_meta.durable import load_task, record_experience, task_hash
from sia.task_meta.storage import digest, save_json
from sia.task_meta.types import ImprovementExperience


def read(path):
    return json.loads(Path(path).read_text())


def authorize_revision(root, proposed):
    root = Path(root)
    authorization = read(root/'recovery/authorization.json')
    prior = root/'recovery/protocol_before.json'
    if digest(prior) != authorization['source_protocol_sha256']:
        raise ValueError('Recovery original protocol changed')
    old = read(prior)
    normalized = copy.deepcopy(proposed)
    from sia.task_meta.meta_backends.input_budget import MetaInputBudget
    if 'input_budget' not in old['config']['meta']:
        if normalized['config']['meta'].pop('input_budget', None) != MetaInputBudget().model_dump():
            raise ValueError('Recovery permits only the declared default Meta input budget')
    for key in set(old) | set(normalized):
        if key not in {'hash', 'controller', 'meta_identity'} and old.get(key) != normalized.get(key):
            raise ValueError('Recovery changed experiment inputs: '+key)
    if proposed['controller'] != authorization['controller']:
        raise ValueError('Recovery controller differs from explicitly authorized source')
    verify_evidence(root, authorization)
    receipt = root/'recovery/protocol_revision.json'
    revision = {'previous_protocol_sha256':digest(prior), 'new_protocol':proposed,
                'reason':'Meta evidence-storage repair; resume accepted Task without reevaluation'}
    if receipt.exists() and read(receipt) != revision:
        raise ValueError('Recovery protocol revision changed')
    save_json(receipt, revision)
    save_json(root/'protocol.json', proposed)


def verify_evidence(root, authorization=None):
    root = Path(root)
    authorization = authorization or read(root/'recovery/authorization.json')
    if authorization['round_index'] != 0 or authorization['operation'] != 'finish_deployed_round':
        raise ValueError('Unsupported recovery boundary')
    for relative, expected in authorization['evidence_sha256'].items():
        path = root/relative
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Recovery evidence changed: '+relative)
    return authorization


def finish_deployed_round(root, number, parent, meta, agent, history, protocol,
                          update_handler, resume, *, early_rollout=None):
    from sia.task_meta.loop import _accept_meta_update
    from sia.task_meta.sequential_loop import content_identity, positive_gain
    from sia.task_meta.round_checkpoint import commit_round_checkpoint
    root = Path(root)
    verify_evidence(root)
    directory = root/f'round_{number}'
    deployment = read(directory/'deployment.json')
    experience = ImprovementExperience(**read(directory/'experience.json'))
    after = load_task(deployment['task_after'])
    attempts = experience.candidate_attempts
    if (number != 0 or deployment['status'] != 'deployed' or len(attempts) != 1
            or not positive_gain(attempts[0]['paired_outcome'])
            or deployment['parent_content_hash'] != content_identity(parent)
            or deployment['output_content_hash'] != content_identity(after)
            or meta.bundle_hash != read(root/'initial_meta_state.json')['bundle_hash']):
        raise ValueError('Recovery does not match the accepted paired round')
    before = copy.deepcopy(meta)
    protocol.begin(number, meta, history)
    protocol.attempts = attempts
    protocol.recursive_summary = read(directory/'recursive_summary.json')
    protocol.boundary = read(directory/'outer_round_evidence.json') if (directory/'outer_round_evidence.json').exists() else None
    protocol.register_rows(directory/'before/gen_0/agent_execution.json', 1)
    protocol.prepare_meta(agent, experience, directory)
    record_experience(root/'meta/experiences.jsonl', experience)
    agent.client.bind_context(root.name, number+1, task_hash(after))
    print('[recovery] accepted Task restored; skipping rollout, SFT and paired retest; finishing Meta', flush=True)
    if early_rollout and number + 1 < protocol.total_stages and protocol.pause_after_round != number + 1:
        early_rollout.start(number + 1, after)
    learned = agent.learn_from_experience(meta, experience, [*history, experience], after)
    protocol.validate_meta(learned, meta)
    meta = _accept_meta_update(root, meta, learned, [experience], directory/'meta_self_update.json',
                              'learn_from_experience', update_handler, resume)
    record = {**deployment, 'input_content_hash':content_identity(parent),
              'meta_before_hash':before.bundle_hash, 'meta_after':asdict(meta),
              'score':{'generation':number, **experience.performance_after},
              'meta_content_changed':meta.bundle_hash != before.bundle_hash,
              'next_use':'pending_next_round'}
    protocol.freeze_system(record, directory)
    save_json(directory/'complete.json', record)
    commit_round_checkpoint(directory, number, after, meta)
    return after, meta, experience, record
