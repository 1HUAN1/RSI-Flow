import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sia.task_meta.deployed_recovery import authorize_revision, verify_evidence
from sia.task_meta.storage import digest, save_json
from sia.task_meta.meta_backends.input_budget import MetaInputBudget


def fixture(root):
    previous = {'config':{'meta':{'model':'fixed'},'max_generations':3},
                'controller':{'old':'code'}, 'meta_identity':{'old':'identity'},
                'data_hash':'unchanged', 'hash':'old'}
    save_json(root/'protocol.json',previous)
    save_json(root/'recovery/protocol_before.json',previous)
    save_json(root/'evidence.json',{'score':27})
    proposed = copy.deepcopy(previous)
    proposed.update(controller={'new':'code'},meta_identity={'new':'identity'},hash='new')
    proposed['config']['meta']['input_budget']=MetaInputBudget().model_dump()
    save_json(root/'recovery/authorization.json',{'round_index':0,'operation':'finish_deployed_round',
        'source_protocol_sha256':digest(root/'recovery/protocol_before.json'),
        'controller':proposed['controller'], 'evidence_sha256':{'evidence.json':digest(root/'evidence.json')}})
    return proposed


def test_revision_preserves_original_and_accepts_only_repaired_execution(tmp_path):
    proposed=fixture(tmp_path)
    original=(tmp_path/'recovery/protocol_before.json').read_bytes()
    authorize_revision(tmp_path,proposed)
    authorize_revision(tmp_path,proposed)
    assert (tmp_path/'recovery/protocol_before.json').read_bytes()==original
    assert (tmp_path/'recovery/protocol_revision.json').exists()


@pytest.mark.parametrize('field',['data_hash','controller','rounds'])
def test_revision_rejects_changed_inputs(tmp_path,field):
    proposed=fixture(tmp_path)
    if field=='rounds': proposed['config']['max_generations']=2
    else: proposed[field]='tampered'
    with pytest.raises(ValueError): authorize_revision(tmp_path,proposed)


def test_changed_score_cannot_be_inherited(tmp_path):
    fixture(tmp_path)
    save_json(tmp_path/'evidence.json',{'score':100})
    with pytest.raises(ValueError,match='evidence changed'): verify_evidence(tmp_path)


def test_resume_skips_parent_rollout_and_updaters_then_runs_validation(tmp_path,monkeypatch):
    from sia.task_meta.sequential_loop import run_sequential_task_meta
    from sia.task_meta.types import TaskAgentState,MetaAgentState
    import sia.task_meta.deployed_recovery as recovery
    save_json(tmp_path/'recovery/authorization.json',{})
    task=TaskAgentState(1,'model','harness')
    meta=MetaAgentState('meta','instructions')
    record={'score':{'generation':0,'macro_success':.15}}
    finish=Mock(return_value=(task,meta,SimpleNamespace(),record))
    monkeypatch.setattr(recovery,'finish_deployed_round',finish)
    executor=Mock()
    executor.execute.side_effect=AssertionError('Must not repeat rollout')
    before,validation=Mock(),Mock()
    protocol=SimpleNamespace(total_stages=1,positive_search=False,pause_after_round=None,
                             store=SimpleNamespace(sequential_domains=False))
    result=run_sequential_task_meta(tmp_path,task,meta,executor,SimpleNamespace(client=Mock()),{},
        max_generations=1,resume=True,round_protocol=protocol,before_evaluation=before,after_round=validation)
    assert result['rounds_completed']==1
    executor.execute.assert_not_called()
    finish.assert_called_once()
    before.assert_called_once_with(task)
    validation.assert_called_once_with(0,record)
