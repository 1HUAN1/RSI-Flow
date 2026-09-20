import copy
import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import test_contracts as native
import test_append_memory as memory_test
principle=memory_test.principle
from sia.task_meta.recursive_feedback import aligned_differences, validate_recursive_memory
from sia.task_meta.meta_harness.five_stage import materialize
from sia.task_meta.round_evolution import RoundProtocol, MEMORY_POLICY


class RecursiveFeedback(unittest.TestCase):
    def setUp(self):
        self.fixture=memory_test.AppendMemory();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.attempts=[dict(attempt_id='attempt1',action='MODEL',status='evaluated',outcome_class='zero_gain',decision={'decision_id':'decision1'})]

    def test_unequal_steps_align_on_tool_not_position_and_account_for_absence(self):
        pair=dict(task_id='fixture',source='nq_open',domain='searchqa',transition='failure_to_success',
            parent={'success':False,'error':'bad_args'},child={'success':True,'error':None},
            pre_trajectory_path='pre',post_trajectory_path='post')
        pre={'tool_calls':[{'name':'search','step':8,'arguments':{'q':'old'}}]}
        post={'events':[{'kind':'planning'}]*10,'tool_calls':[
            {'name':'calculator','step':1}, {'name':'search','step':12,'arguments':{'q':'revised'}}]}
        fragments,coverage=aligned_differences(pre,post,pair)
        self.assertEqual(len(fragments),1);self.assertEqual(fragments[0]['pre']['index'],0)
        self.assertEqual(fragments[0]['post']['index'],1)
        self.assertEqual(fragments[0]['anchor'],'search');self.assertEqual(coverage['unmatched_post'],1)
        self.assertEqual(aligned_differences({},post,pair)[1]['status'],'no_shared_explicit_anchor')

    def test_native_append_and_actual_Meta_instruction_revision_preserve_history(self):
        f=self.fixture;u=f.update.model_copy(deep=True)
        u.harness += '\nInspect tool contracts before choosing component changes.'
        u.five_stage.principle_operations[0].record.g_targets=['instructions.md']
        from sia.task_meta.meta_harness.five_stage import HarnessBinding
        u.five_stage.harness_bindings=[HarnessBinding(principle_ids=['skill.MODEL.r1'],target='instructions.md',
            purpose='Evidence-bound diagnosis revision',expected_event='Next routing checks tool schema',layer='instruction')]
        u.five_stage.principle_operations[0].evidence_ids.append('attempt1')
        validate_recursive_memory(u,f.meta,self.attempts,'experience1',None,[])
        updated,flags=materialize(u,f.files,allowed_evidence={'pair1','decision1','experience1'})
        self.assertTrue(flags['harness_policy_changed'])
        self.assertEqual(updated['instructions.md'],u.harness)
        self.assertEqual(json.loads(updated['principles.json'])['records'][0],principle('legacy'))
        u.five_stage.harness_bindings=[]
        with self.assertRaises(ValueError):materialize(u,f.files)

    def test_all_attempts_skills_and_three_domain_consolidation_required(self):
        f=self.fixture;u=f.update.model_copy(deep=True)
        attempts=self.attempts+[dict(attempt_id='attempt2',action='HARNESS',status='unavailable',outcome_class='candidate_unavailable',decision={'decision_id':'decision2'})]
        u.five_stage.principle_operations[0].evidence_ids.append('attempt1')
        u.five_stage.principle_operations[-1].record.source_decisions.append('decision2')
        with self.assertRaisesRegex(ValueError,'Every attempted component'):
            validate_recursive_memory(u,f.meta,attempts,'experience1',None,[])
        op=u.five_stage.principle_operations[0].model_copy(deep=True)
        op.principle_id=op.record.principle_id='skill.HARNESS.r1';op.record.source_decisions=['decision2'];op.evidence_ids=['attempt2']
        op.record.statement+=' distinct intervention'
        u.five_stage.principle_operations.insert(1,op)
        boundary={'evidence':[{'experience_id':x} for x in ['previous_tool','previous_code','experience1']]}
        validate_recursive_memory(u,f.meta,attempts,'experience1',boundary,[])
        u.five_stage.principle_operations[-1].record.source_experiences += ['previous_tool','previous_code']
        validate_recursive_memory(u,f.meta,attempts,'experience1',boundary,[])
        validate_recursive_memory(u,f.meta,attempts,'experience1',boundary,['difference:actual'])
        u.five_stage.principle_operations[0].evidence_ids.append('difference:actual')
        validate_recursive_memory(u,f.meta,attempts,'experience1',boundary,['difference:actual'])
        u.five_stage.principle_operations[0].operation='REVISE'
        with self.assertRaisesRegex(ValueError,'Preserve all earlier'):
            validate_recursive_memory(u,f.meta,attempts,'experience1',boundary,[])

    def test_component_and_general_memory_require_actual_attribution(self):
        f=self.fixture;u=f.update.model_copy(deep=True)
        skill=u.five_stage.principle_operations[0]
        skill.record.source_decisions=[];skill.record.source_experiences=[]
        skill.evidence_ids=['unresolved prose label']
        with self.assertRaisesRegex(ValueError,'actual decision and current experience'):
            validate_recursive_memory(u,f.meta,self.attempts,'experience1',None,['local:actual'])
        skill.record.source_decisions=['decision1'];skill.record.source_experiences=['experience1']
        with self.assertRaisesRegex(ValueError,'actual candidate attempt/outcome evidence'):
            validate_recursive_memory(u,f.meta,self.attempts,'experience1',None,['local:actual'])
        skill.evidence_ids=['attempt1']
        general=u.five_stage.principle_operations[-1]
        general.rationale='Unlinked generalization'
        with self.assertRaisesRegex(ValueError,'derive from a newly appended component skill'):
            validate_recursive_memory(u,f.meta,self.attempts,'experience1',None,['local:actual'])
        general.rationale='Derived from skill.MODEL.r1'
        validate_recursive_memory(u,f.meta,self.attempts,'experience1',None,['local:actual'])

    def test_unavailable_candidate_can_be_cited_without_trajectory_fragment(self):
        f=self.fixture;u=f.update.model_copy(deep=True)
        attempt=dict(attempt_id='attempt-unavailable',action='MODEL',status='unavailable',
                     outcome_class='candidate_unavailable',decision={'decision_id':'decision1'})
        u.five_stage.principle_operations[0].evidence_ids=['attempt-unavailable']
        validate_recursive_memory(u,f.meta,[attempt],'experience1',None,[])

    def test_component_skill_tag_matches_measured_candidate_outcome(self):
        f=self.fixture;u=f.update.model_copy(deep=True)
        attempt=dict(attempt_id='attempt-positive',action='MODEL',status='evaluated',
                     outcome_class='accepted_positive_gain',decision={'decision_id':'decision1'})
        skill=u.five_stage.principle_operations[0]
        skill.evidence_ids=['attempt-positive']
        with self.assertRaisesRegex(ValueError,'requires library tag success'):
            validate_recursive_memory(u,f.meta,[attempt],'experience1',None,[])
        skill.record.library_tags=['success']
        validate_recursive_memory(u,f.meta,[attempt],'experience1',None,[])

    def test_retry_guard_only_admits_controller_registered_completed_attempts(self):
        p=object.__new__(RoundProtocol)
        p.config=SimpleNamespace(round_protocol={'candidate_policy':'single_candidate_strict_positive_gain'})
        p.store=SimpleNamespace(round_id=1);p.meta_stage='decision';p.attempt_evidence=set()
        path=self.fixture.root/'candidate.json';path.write_text('[]')
        from sia.task_meta.evolution_protocol import file_hash
        p.registry={'x':dict(path=str(path),sha256=file_hash(path),round_id=1,source_role='train_evolution',
            purpose='evolution_train',derived_from=[],collection_stages=['child_post_update'])}
        with self.assertRaisesRegex(ValueError,'post/future'):p.guard()
        p.attempt_evidence.add(str(path));p.guard()
        p.registry['x']['source_role']='independent_validation'
        with self.assertRaises(ValueError):p.guard()

    def test_native_slow_edit_requires_recorded_residual_mismatch(self):
        from sia.task_meta.meta_harness.graph import apply_slow
        with self.assertRaisesRegex(ValueError,'recorded structural mismatch'):
            apply_slow({},dict(operation='Rewire',issue_id='missing_evidence',workflow='routing',
                experience_ids=['experience_0_1','experience_1_2'],replay_requirements=['schema']),
                [],experience_id='experience_1_2')
