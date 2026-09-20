import json

import pytest

from sia.task_meta.report_predictions import public_final_task
from sia.task_meta.reporting import ReportBlocked


@pytest.mark.parametrize('identifier', ['hotpotqa_dev', '2wiki_dev'])
def test_final_search_excludes_gold_and_oracle_passages(identifier):
    task = public_final_task(identifier, {'_id': 'q', 'question': 'Question', 'answer': 'GOLD',
                             'context': 'ORACLE_CONTEXT', 'supporting_facts': 'PRIVATE'})
    assert task.prompt == 'Question' and task.split == 'report_eval'
    assert 'GOLD' not in json.dumps(task.public_payload()) and task.payload == {}


def test_final_code_exposes_public_interface_only():
    task = public_final_task('humaneval_plus', {'task_id': 'HumanEval/0', 'prompt': 'public signature',
                                              'entry_point': 'f', 'plus_input': ['HIDDEN'], 'canonical_solution': 'GOLD'})
    assert task.payload == {'tests': {'fn_name': 'f'}}
    assert 'HIDDEN' not in json.dumps(task.public_payload())


def test_tool_benchmarks_do_not_silently_use_code_or_search_protocol():
    with pytest.raises(ReportBlocked, match='multi-turn'):
        public_final_task('bfcl_v3', {})
