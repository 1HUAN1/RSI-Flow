"""Native K materialization and next-decision wiring, with no model calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_contracts as native
from sia.task_meta.types import MetaHarnessUpdate
from sia.task_meta.meta_harness.five_stage import materialize, canonical
from sia.task_meta.round_evolution import (validate_memory_append, memory_view,
    RoundProtocol, MEMORY_POLICY)


def principle(identifier):
    return dict(principle_id=identifier,revision=1,active=True,evidence_state='tentative',
        library_tags=['failure'],applicability='Reusable contexts '+identifier,
        statement='Verify the available tool contract '+identifier,procedure='Inspect schema before invoking tools',
        scope='local_behavior',supporting_evidence=['pair1'],counter_evidence=[],unknowns=['Small sample'],
        source_experiences=['experience1'],source_decisions=['decision1'],source_events=[],
        invalid_when='Interface unavailable',g_targets=[],expected_next_behavior='Validate arguments')


class AppendMemory(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.files={'instructions.md':'Existing rules\n','context.json':'{}','workflow.json':'{}',
            'evolution.json':'{"schema_version":1}',
            'principles.json':canonical(dict(schema_version=1,records=[principle('legacy')],revisions=[]))}
        for name,value in self.files.items():(self.root/name).write_text(value,encoding='utf-8')
        self.meta=SimpleNamespace(bundle_path=str(self.root),harness_path=str(self.root/'instructions.md'))
        finding=dict(status='unknown',explanation='Measured comparison',evidence_ids=['pair1'],unknowns=[])
        operations=[]
        for ident,reason in [('skill.MODEL.r1','Measured model intervention'),('principle.r1','Derived from skill.MODEL.r1')]:
            operations.append(dict(operation='ADD',principle_id=ident,record=principle(ident),
                rationale=reason,compared_ids=['legacy'],evidence_ids=['pair1']))
        self.update=MetaHarnessUpdate(harness=self.files['instructions.md'],rationale='Training comparison',
            changed_rules=['new skill and principle'],experience_id='experience1',five_stage=dict(
                protocol='five-stage-principle-update-v1',expectation_id=None,outcome_hash='outcome1',
                implementation=finding,activation=finding,outcome=finding,principle_operations=operations,
                harness_bindings=[],no_change_reason=None,subsequent_verification='Next decision uses Memory'))

    def validate(self,value):
        return validate_memory_append(value,self.meta,'MODEL','decision1','experience1')

    def test_native_materialization_preserves_history_and_next_request_loads_both(self):
        self.validate(self.update)
        after,audit=materialize(self.update,self.files,allowed_evidence={'pair1','decision1','experience1'})
        library=json.loads(after['principles.json'])
        self.assertEqual(library['records'][0],principle('legacy'))
        self.assertEqual(len(library['records']),3)
        self.assertEqual([r['operation'] for r in library['revisions']],['ADD','ADD'])
        self.assertTrue(audit['principle_memory_updated']);self.assertFalse(audit['harness_policy_changed'])
        for name,value in after.items():(self.root/name).write_text(value,encoding='utf-8')
        view=memory_view(self.meta)
        self.assertEqual(view['component_skills']['MODEL'][0]['principle_id'],'skill.MODEL.r1')
        self.assertIn('principle.r1',[r['principle_id'] for r in view['general_principles']])
        protocol=object.__new__(RoundProtocol)
        protocol.config=SimpleNamespace(round_protocol={'memory_policy':MEMORY_POLICY})
        protocol.loaded_memory=view;protocol.guard=lambda:None
        from sia.task_meta.meta import MetaAgent
        calls=[]
        agent=SimpleNamespace(round_protocol=protocol,_five_stage=lambda state:False,
            client=SimpleNamespace(supports_evolution=True,complete=lambda *a,**k:calls.append(k)))
        with patch('sia.task_meta.meta.operation_input',return_value={'trusted_facts':{},'current_files':{}}):
            MetaAgent.diagnose_and_route(agent,self.meta,SimpleNamespace(generation=1))
        self.assertEqual(calls[0]['operation_input']['trusted_facts']['meta_memory'],view)
        self.assertEqual(calls[0]['operation_input']['trusted_facts']['component_decision_contract']['candidate_limit'],1)
        # Replaying the same additions against committed K cannot append twice.
        with self.assertRaises(ValueError):self.validate(self.update)

    def test_rewrite_retire_wrong_component_and_missing_attribution_rejected(self):
        original=self.update.model_dump()
        bad=[]
        for operation in ('REVISE','MERGE','RETIRE'):
            value=copy.deepcopy(original);value['five_stage']['principle_operations'][0]['operation']=operation;bad.append(value)
        value=copy.deepcopy(original);value['harness']='changed rules';bad.append(value)
        value=copy.deepcopy(original);value['bundle_files']={'../outside':'x'};bad.append(value)
        value=copy.deepcopy(original);value['five_stage']['principle_operations'].reverse();bad.append(value)
        value=copy.deepcopy(original);value['five_stage']['principle_operations'][0]['record']['source_decisions']=[];bad.append(value)
        value=copy.deepcopy(original);value['five_stage']['principle_operations'][1]['rationale']='unlinked';bad.append(value)
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):self.validate(value)
        with self.assertRaises(ValueError):validate_memory_append(self.update,self.meta,'HARNESS','decision1','experience1')

    def test_only_one_native_component_decision(self):
        import inspect
        from sia.task_meta.sequential_loop import run_sequential_task_meta
        self.assertIn('for index in range(1)',inspect.getsource(run_sequential_task_meta))

    def test_reduced_release_copies_only_selected_records(self):
        from prepare_revision import repack
        from evolution_protocol import make_manifest, file_hash
        source=self.root/'original.jsonl';rows=[]
        with source.open('wb') as f:
            for i in range(4):
                data=(json.dumps({'question':str(i)})+'\n').encode();offset=f.tell();f.write(data)
                rows.append(dict(source='nq_open',task_id=str(i),group_id=str(i),content_hash=str(i),
                    record_file=str(source),record_offset=offset,record_length=len(data)))
        for row in rows:row['record_sha256']=file_hash(source)
        old=make_manifest('train_evolution',1,rows,split_seed=42)
        reduced=repack(old,[rows[1],rows[3]],self.root/'reduced')
        self.assertEqual([r['task_id'] for r in reduced['tasks']],['1','3'])
        self.assertEqual(len((self.root/'reduced/nq_open.jsonl').read_text().splitlines()),2)
        self.assertEqual(file_hash(source),rows[0]['record_sha256'])
        with self.assertRaises(FileExistsError):repack(old,[rows[1]],self.root/'reduced')


if __name__=='__main__':unittest.main()
