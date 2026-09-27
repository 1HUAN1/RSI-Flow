"""Evidence-bound routing and branch plans; uses the existing immutable Meta library."""
import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sia.task_meta.evolution_protocol import fingerprint, freeze, read, SFT_PRESET

COMPONENTS = ('HARNESS', 'MODEL', 'ARTIFACTS')
class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class FaultHypothesis(Strict):
    fault_id: str
    owner: Literal['WORKFLOW','SKILL','PLANNING','BUILDER','INTERFACE','MODEL','ARTIFACTS','ENVIRONMENT','UNKNOWN']
    mechanism: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    counter_evidence: list[str]
    uncertainty: str = Field(min_length=1)


class SkillUse(Strict):
    principle_id: str
    revision: int = Field(ge=1)
    record_hash: str
    match_reason: str = Field(min_length=1)
    applicability_limits: str = Field(min_length=1)


class ChangePlan(Strict):
    target: str
    fault_ids: list[str] = Field(min_length=1)
    selected_skill_ids: list[str]
    mechanism: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    trigger_condition: str = Field(min_length=1)
    preserve: list[str] = Field(min_length=1)


class InterventionPlan(Strict):
    fault_report_hash: str
    retrieval_hash: str
    faults: list[FaultHypothesis] = Field(min_length=1)
    skills: list[SkillUse]
    no_matching_skill_reason: str | None
    alternatives: dict[str, str]
    changes: list[ChangePlan] = Field(min_length=1)


def decision_sources(rows, representatives):
    """All-task index plus bounded real excerpts; never mistake the excerpts for full coverage."""
    if not rows or any(r.get('purpose')!='evolution_train' or r.get('collection_stage')!='parent_pre_update' for r in rows):
        return copy.deepcopy(representatives)
    from sia.task_meta.evolution_protocol import meta_routing_rows
    from sia.task_meta.runtime_extensions import hydrate_rollout
    from sia.task_meta.observations import action_failure
    selected=meta_routing_rows(rows,maximum=48,per_domain=16,per_outcome=8,seed=42)
    examples={(r['task_id'],r['rollout_id']) for r in selected}
    def excerpt(value, depth=0):
        if depth>6: return {'omitted_sha256':fingerprint(value)}
        if isinstance(value,str): return value[:800]
        if isinstance(value,list): return [excerpt(v,depth+1) for v in value[:8]]
        if isinstance(value,dict): return {k:excerpt(v,depth+1) for k,v in list(value.items())[:16]}
        return value
    keys=('task_id','question_id','rollout_id','trajectory_id','source_role','purpose','round_id','branch_id',
          'collection_stage','manifest_hash','source','domain','split','verification','terminal_reward',
          'execution_error_type','error_type','infrastructure_error','input_tokens','output_tokens',
          'wall_time_seconds','model_call_count','harness_hash','parent_policy_hash','submission_origin',
          'parse_failure','parse_failed','output_truncated','finish_reason','api_failure','api_error',
          'model_call_failed','error','valid_answer','_action_contract_failure')
    sources=[]
    for row in rows:
        example=(row['task_id'],row['rollout_id']) in examples
        value={}
        if example:
            full=hydrate_rollout(row)
            for name in ('events','tool_calls','model_calls','transport_calls'):
                source=full.get(name) or []
                indices=sorted({0,len(source)-1}) if source else []
                value[name]=[excerpt(source[i]) for i in indices]
                value[name+'_excerpt_indices']=indices
            value['final_answer']=excerpt(full.get('final_answer',full.get('model_answer','')))
        # Scoring/provenance never comes from clipped fields.
        value.update({k:copy.deepcopy(row[k]) for k in keys if k in row})
        if row.get('execution_error_type', row.get('error_type')) == 'parse_error':
            # Classify from the full recorded response BEFORE excerpting. Otherwise
            # a full-cohort empty-tools failure can disappear from Meta's statistics.
            source = row if '_action_contract_failure' in row else (full if example else hydrate_rollout(row))
            value['_action_contract_failure'] = action_failure(source)
        sha=fingerprint(row)
        value.update(source_id='pre:'+sha,original_trajectory_sha256=sha,
            evidence_detail='bounded_representative_excerpt' if example else 'task_summary_only',
            original_trajectory_id=row.get('trajectory_id'),
            excerpt_policy='48 detailed rows: domain 16 each; success/failure 8 each with within-domain backfill; failure error round-robin, success stable hash; domain deficits cross-domain backfill; first/last events, depth6/list8/string800')
        sources.append(value)
    return sources


def fault_report(envelope):
    """Full-coverage observations first; causal attribution is subsequently authored by Meta."""
    from sia.task_meta.meta_harness.runtime import _source_records
    groups = defaultdict(list)
    for record in _source_records(envelope['raw_trajectories'], 'trajectory'):
        row = record['item']
        if (row.get('source_role') != 'train_evolution' or row.get('purpose') != 'evolution_train'
                or row.get('collection_stage') != 'parent_pre_update' or row.get('submission_origin')):
            raise ValueError('Diagnosis accepts native current parent pre only')
        verified = row.get('verification', {})
        outcome = ('infrastructure' if row.get('infrastructure_error') else
                   'unscored' if verified.get('status') != 'completed' else
                   'success' if verified.get('success') is True else 'failure')
        error = row.get('execution_error_type') or row.get('error_type') or outcome
        groups[(row['domain'], outcome, str(error))].append((record['id'], row['task_id']))
    findings = []
    for key, rows in sorted(groups.items()):
        value = dict(domain=key[0], outcome=key[1], error=key[2], n_tasks=len(rows),
                     evidence_ids=[v[0] for v in rows], task_ids=[v[1] for v in rows])
        findings.append(dict(fault_id='fault:'+fingerprint(value), **value))
    value = dict(findings=findings, n_tasks=sum(v['n_tasks'] for v in findings),
                 omitted_tasks=0, attribution='unassigned; Meta must supply evidence-bound hypotheses',
                 statistics=envelope['trusted_facts']['trajectory_statistics'])
    return {**value, 'report_hash':fingerprint(value)}


def retrieve(memory, report, component=None):
    """Deterministic lexical shortlist, not a claim of semantic/causal relevance."""
    words = lambda s: set(re.findall(r'[a-z][a-z0-9]{2,}|[\u4e00-\u9fff]', s.lower().replace('_',' ')))
    query = words(' '.join(f['domain']+' '+f['error'] for f in report['findings']))
    selected = []; empty = []
    buckets = {**memory['component_skills'], 'GENERAL':memory['general_principles']}
    for bucket, records in buckets.items():
        if component and bucket not in (component, 'GENERAL'): continue
        matches = []
        for record in records:
            if not record.get('active', False): continue
            terms = words(' '.join(str(record.get(k,'')) for k in
                           ('applicability','statement','procedure','expected_next_behavior')))
            overlap = sorted(query & terms)
            if not overlap: continue
            matches.append(dict(principle_id=record['principle_id'], revision=record['revision'],
                record_hash=fingerprint(record), record=copy.deepcopy(record), matching_terms=overlap,
                match_reason='lexical overlap with observed training failure/domain; Meta must assess applicability',
                score=len(overlap), bucket=bucket))
        matches.sort(key=lambda v:(-v['score'],v['principle_id']))
        selected.extend(matches[:8])
        if not matches: empty.append(bucket)
    value = dict(library_hash=memory['library_hash'], fault_report_hash=report['report_hash'],
        searched_components=[component] if component else list(COMPONENTS), selected=selected,
        empty_buckets=empty, policy='active_only_lexical_top8_per_component_and_general',
        limitation='No matching terms can miss relevant skills; no-match permits a new explicitly untested hypothesis')
    return {**value, 'retrieval_hash':fingerprint(value)}


def prepare_routing(protocol, envelope):
    rows=envelope['raw_trajectories']
    if (len(rows)!=len(protocol.store.members) or {r['task_id'] for r in rows}!=set(protocol.store.members)
            or any(r.get('round_id')!=protocol.store.round_id or r.get('manifest_hash')!=protocol.store.current['manifest_hash'] for r in rows)):
        raise ValueError('Routing requires exactly the current B_r, with no future or missing tasks')
    report = fault_report(envelope)
    retrieval = retrieve(protocol.loaded_memory, report)
    index = envelope['trusted_facts']['external_budget'].get('candidate_index', 0)
    folder = protocol.root/f'round_{protocol.store.round_id-1}'/f'decision_{index}'
    freeze(folder/'01_fault_report.json', report)
    freeze(folder/'02_skill_retrieval.json', retrieval)
    protocol.routing_evidence = dict(report=report, retrieval=retrieval)
    envelope['trusted_facts']['intervention_evidence'] = protocol.routing_evidence
    envelope['trusted_facts']['component_decision_contract']['evidence_contract'] = (
        'Complete intervention_plan: diagnose observed fault IDs with evidence, counter-evidence and uncertainty; '
        'prior skills are optional decision support, not a selection prerequisite. Explain BOTH other components. '
        'No future outcomes. If HARNESS is selected, request only produce_harness/harness_bundle: do not preselect a '
        'file or module. HarnessForge Stage 1 performs authoritative module localization after routing, followed by '
        'Stage 2 directions, Stage 3 complete-bundle generation, and Stage 4 validation/repair. '
        'MODEL uses existing final-verifier-correct parent pre only, unchanged LoRA preset. '
        'ARTIFACTS edits existing submissions and rescores them, without Task inference or reusable asset injection.')


def validate_plan(decision, evidence):
    """Validate the component blueprint; HarnessForge owns HARNESS file targets."""
    from sia.task_meta.types import DecisionConstraintError
    try:
        plan = InterventionPlan.model_validate(decision.intervention_plan)
        if decision.action.value == 'HARNESS':
            if len(plan.changes) != 1 or plan.changes[0].target != 'harness_bundle':
                raise ValueError('HARNESS blueprint must target the complete harness_bundle without naming a module')
        elif len(plan.changes)!=len(decision.requested_changes) or {c.target for c in plan.changes}!={c.target for c in decision.requested_changes}:
            raise ValueError('Blueprint targets must exactly match requested changes')
        return plan
    except ValueError as exc:
        raise DecisionConstraintError(str(exc)) from exc


def citation_audit(decision, evidence):
    """Record unresolved labels without rejecting decisions or importing cited data."""
    plan=decision.intervention_plan;report=evidence['report'];retrieval=evidence['retrieval']
    known={f['fault_id'] for f in report['findings']}|{i for f in report['findings'] for i in f['evidence_ids']}
    cited={i for f in plan.faults for i in [f.fault_id,*f.evidence_ids,*f.counter_evidence]}
    library={s['principle_id']:s for s in retrieval['selected']}
    return dict(policy='advisory_only',unresolved_evidence_labels=sorted(cited-known),
        unresolved_skill_labels=[s.principle_id for s in plan.skills if s.principle_id not in library
            or s.revision!=library[s.principle_id]['revision'] or s.record_hash!=library[s.principle_id]['record_hash']],
        declared_report_matches=plan.fault_report_hash==report['report_hash'],
        declared_retrieval_matches=plan.retrieval_hash==retrieval['retrieval_hash'],
        source_boundary='current controller training evidence only',
        labels_grant_no_read_permissions=True)


def branch_binding(task, decision, context):
    from sia.task_meta.durable import task_hash
    plan = decision.intervention_plan
    if plan is None: return None  # Legacy receipts remain readable, never silently migrated.
    value = dict(parent_hash=task_hash(task), decision_id=decision.decision_id,
        component=decision.action.value, plan=plan.model_dump(mode='json'),
        parent_policy=task.checkpoint_manifest, behavior_harness=read(task.harness_path),
        meta_bundle_hash=context.meta_state.bundle_hash if context.meta_state else None)
    if decision.action.value=='MODEL':
        value.update(source='existing_current_parent_pre_final_verified_success',
                     sft_preset=SFT_PRESET, extra_collection=False)
    elif decision.action.value=='ARTIFACTS':
        value.update(evaluation='direct_submission',task_model_calls=0,eligible_for_sft=False)
    freeze(context.directory/'intervention_blueprint.json',value)
    return value
