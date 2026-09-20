"""Training-side candidate evidence; no evaluator or additional model execution."""
from collections import Counter, defaultdict
from pathlib import Path
import json
from sia.task_meta.evolution_protocol import read, freeze, fingerprint

SEARCH_POLICY = 'single_candidate_strict_positive_gain'


def aligned_differences(pre, post, pair):
    """Exact semantic anchors, chronological occurrence matching, never step-number zipping.

    Missing explicit subgoals stay missing. Repeated anchors are ambiguous; fragments
    are observations, not a causal claim. The complete pair remains in the report.
    """
    def anchors(row):
        found = defaultdict(list)
        for field in ('events', 'tool_calls'):
            for index, event in enumerate(row.get(field, [])):
                if not isinstance(event, dict): continue
                keys = []
                subgoal = event.get('subgoal_id') or event.get('subgoal')
                if subgoal: keys.append(('subgoal', str(subgoal)))
                tool = event.get('name') if field == 'tool_calls' else event.get('tool')
                if tool: keys.append(('tool', str(tool)))
                error = event.get('error_type') or event.get('error')
                if isinstance(error,str) and error: keys.append(('error', str(error)))
                for key in keys:
                    found[key].append(dict(field=field,index=index,event_hash=fingerprint(event),
                        event_preview=json.dumps(event,ensure_ascii=False,sort_keys=True)[:1600],
                        preview_truncated=len(json.dumps(event,ensure_ascii=False,sort_keys=True))>1600))
        return found
    left, right = anchors(pre), anchors(post)
    result = []
    for key in sorted(left.keys() & right.keys()):
        for a,b in zip(left[key],right[key]):
            same = a['event_hash'] == b['event_hash']
            fragment = dict(anchor_type=key[0],anchor=key[1],
                applicability={'domain':pair['domain'],'source':pair['source'],
                               'condition':'Recorded shared '+key[0]+' event'},
                pre=a,post=b,behavior_change='preserved' if same else 'changed',
                repeated_anchor_ambiguous=len(left[key])>1 or len(right[key])>1,
                task_id=pair['task_id'],transition=pair['transition'],
                parent_result=pair['parent'],child_result=pair['child'],
                mechanism_status='hypothesis_requires_Meta_review',
                persistent_or_shifted_failure=('same_reported_error' if pair['parent']['error']==pair['child']['error'] else 'changed_error_possible_later_bottleneck') if pair['transition']=='failure_to_failure' else None,
                pre_trajectory_path=pair['pre_trajectory_path'],post_trajectory_path=pair['post_trajectory_path'])
            fragment['source_id']='difference:'+fingerprint(fragment)
            result.append(fragment)
    return result, dict(task_id=pair['task_id'],n_fragments=len(result),
        unmatched_pre=sum(len(v) for k,v in left.items() if k not in right),
        unmatched_post=sum(len(v) for k,v in right.items() if k not in left),
        status='anchored' if result else 'no_shared_explicit_anchor',
        limitation='No semantic subgoal inference; repeated anchors use chronological occurrence, not step equality')


def candidate_feedback(protocol, experience, round_dir):
    from sia.task_meta.round_evolution import training_pairs
    from sia.task_meta.types import ImprovementExperience
    from dataclasses import asdict
    pre=read(experience.trajectory_before);reports=[];coverage=[];fragments=[]
    for attempt in experience.candidate_attempts:
        base=dict(attempt_id=attempt['attempt_id'],component=attempt['action'],
            decision_id=attempt['decision']['decision_id'],status=attempt['status'],
            generating_meta=attempt['generating_G'],actual_change=attempt.get('update',{}),
            committed=attempt.get('accepted_for_deployment',False),
            outcome_class=attempt.get('outcome_class','unknown'))
        if not attempt.get('candidate_executed'):
            reports.append({**base,'reason':attempt.get('reason'),'pairs':[],
                            'overall':{'complete':False,'n_expected':len(pre),'n_paired':0,'success_delta':None}})
            continue
        post=read(attempt['trajectory_after'])
        item=ImprovementExperience(**{**asdict(experience),'trajectory_after':attempt['trajectory_after'],
            'actual_change':attempt['update'],'decision':attempt['decision']})
        selected=dict(decision_id=base['decision_id'],parent_hash=attempt['parent_content_hash'],
                      child_hash=fingerprint(attempt['candidate_state']))
        report=training_pairs(protocol.store.current,pre,post,item,selected)
        reports.append({**report,**base})
        a={x['task_id']:x for x in pre};b={x['task_id']:x for x in post}
        for pair in report['pairs']:
            local,covered=aligned_differences(a[pair['task_id']],b[pair['task_id']],pair)
            for value in local:
                value.update(decision_id=base['decision_id'],component=base['component'],
                             actual_change=base['actual_change'],generating_meta=base['generating_meta'])
                value['source_id']='difference:'+fingerprint(value)
            fragments.extend(local);coverage.append({**covered,'attempt_id':base['attempt_id']})
    # Representatives bound prompt size; every task/candidate still has a pair and coverage entry.
    groups=defaultdict(list)
    for item in fragments:
        groups[(item['component'],item['transition'],str(item['parent_result']['error']),
                str(item['child_result']['error']))].append(item)
    representatives=[]
    for key in sorted(groups):
        representatives.extend(sorted(groups[key],key=lambda x:x['source_id'])[:2])
    full=dict(source_role='train_evolution',purpose='evolution_train',round_id=protocol.store.round_id,
        attempts=reports,local_differences=fragments,coverage=coverage,
        candidate_task_pairs=sum(len(r['pairs']) for r in reports),
        current_manifest=protocol.store.current['manifest_hash'],
        baseline_tasks=len(pre),deployment_status=experience.deployment_status,
        gain_attribution='generating_meta_only; next Meta assessed on subsequent interventions')
    freeze(Path(round_dir)/'all_candidate_effects.json',full)
    summaries=[{k:v for k,v in r.items() if k!='pairs'} for r in reports]
    return full,dict(source_id='candidate_effects:'+fingerprint(full),attempts=summaries,
        candidate_task_pairs=full['candidate_task_pairs'],coverage=coverage,
        representatives=representatives,all_fragment_count=len(fragments),
        representative_count=len(representatives),
        all_executed_pairs_scored=bool(coverage) and all(r['overall']['complete'] for r in reports if r['pairs']),
        all_executed_tasks_have_coverage=len(coverage)==sum(len(r['pairs']) for r in reports),
        full_evidence_path=str(Path(round_dir)/'all_candidate_effects.json'))


def boundary_evidence(protocol, current_summary):
    """Third domain's native Meta update also consolidates this outer round; no extra update."""
    stage=protocol.store.round_id
    if not protocol.store.sequential_domains or stage%3: return None
    summaries=[]
    for i in range(stage-3,stage):
        summary=current_summary if i==stage-1 else read(protocol.root/f'round_{i}/recursive_summary.json')
        summaries.append(dict(execution_stage_id=i+1,domain=('tool_use','code','searchqa')[i%3],
            experience_id=f'experience_{i}_{i+1}',summary=summary))
    return dict(source_id='outer_round:'+fingerprint(summaries),allocation_round_id=stage//3,
        evidence=summaries,domains=['tool_use','code','searchqa'],
        instruction='Derive a new conditional general principle from all three domain stages. '
        'Cite all three experience IDs. Separate domain-specific evidence from hypothesized transfer; '
        'do not claim cross-domain gains were verified when they were not. No additional Task modification.')


def validate_recursive_memory(candidate, meta, attempts, experience_id, boundary, local_ids):
    """Append-only library plus the native evidence-bound fast/slow Meta mechanism editor."""
    from sia.task_meta.types import MetaHarnessUpdate
    update=MetaHarnessUpdate.model_validate(candidate);review=update.five_stage
    if review is None: raise ValueError('Recursive Meta update requires native five-stage review')
    previous=read(Path(meta.bundle_path)/'principles.json')
    seen={r['principle_id'] for r in previous['records']}
    decisions={a['decision']['decision_id']:a['action'] for a in attempts}
    required=set(decisions.values());skills={};principles=[]
    by_component={component:[a for a in attempts if a['action']==component] for component in required}
    new_skill_ids=[]
    for op in review.principle_operations:
        r=op.record
        if op.operation!='ADD' or r is None or op.principle_id in seen or r.revision!=1:
            raise ValueError('Preserve all earlier skills/principles; revisions append new IDs')
        seen.add(op.principle_id)
        if r.principle_id!=op.principle_id:
            raise ValueError('New memory record must match its library key')
        if op.principle_id.startswith('skill.'):
            component=op.principle_id.split('.')[1]
            if principles or component not in required:
                raise ValueError('Append skills for actual attempted components before general principles')
            if not r.expected_next_behavior.strip() or not r.invalid_when.strip():
                raise ValueError('A skill needs mechanism revision and applicability limits')
            component_attempts=by_component[component]
            decision_ids={a['decision']['decision_id'] for a in component_attempts}
            attempt_ids={a.get('attempt_id') for a in component_attempts if a.get('attempt_id')}
            if not decision_ids <= set(r.source_decisions) or experience_id not in r.source_experiences:
                raise ValueError('Component skill must cite its actual decision and current experience')
            cited=set(op.evidence_ids)|set(r.supporting_evidence)|set(r.source_events)
            if not attempt_ids or not cited.intersection(attempt_ids):
                raise ValueError('Component skill must cite its actual candidate attempt/outcome evidence')
            outcomes={a.get('outcome_class') for a in component_attempts}
            unknown=outcomes-{'accepted_positive_gain','zero_gain','negative_gain',
                              'incomplete_comparison','candidate_unavailable'}
            if unknown:
                raise ValueError('Component skill has an unknown candidate outcome class')
            required_tag='success' if outcomes=={'accepted_positive_gain'} else 'failure'
            if required_tag not in r.library_tags:
                raise ValueError(f'Component skill for {sorted(outcomes)} requires library tag {required_tag}')
            skills[op.principle_id]=component
            new_skill_ids.append(op.principle_id)
        elif op.principle_id.startswith('principle.') and skills:
            if not any(skill_id in op.rationale for skill_id in new_skill_ids):
                raise ValueError('General principle rationale must derive from a newly appended component skill')
            if experience_id not in r.source_experiences or not set(decisions) <= set(r.source_decisions):
                raise ValueError('General principle must preserve this round decision and experience attribution')
            principles.append(r)
        else: raise ValueError('General principles must follow and cite newly appended component skills')
    if set(skills.values())!=required or not principles:
        raise ValueError('Every attempted component, including failures, needs an attributed skill')
    if not attempts: raise ValueError('No intervention evidence for recursive memory')
    # Native materialize validates fast bindings, unchanged interfaces, residual
    # mismatch history, bounded graph edits and schema. Never bypass it here.
    
    # Native Materialize 会验证快速绑定、未更改的接口、残留数据
    # 不匹配历史记录、有界图编辑和模式。切勿在此处绕过它。
    return dict(skills=list(skills),principles=[r.principle_id for r in principles])
