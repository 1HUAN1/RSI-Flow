"""Frozen reporting must retain the execution semantics that were evaluated."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sia.task_meta import pipeline
from sia.task_meta.storage import checkpoint_manifest, save_json
from sia.task_meta.task_harness import harness_identity
from sia.task_meta.types import TaskAgentState


def test_freeze_binds_runtime_and_rejects_post_evaluation_source_changes(tmp_path, monkeypatch):
    directory = tmp_path / 'run'
    generation = directory / 'gen_0'
    generation.mkdir(parents=True)
    checkpoint = tmp_path / 'weights'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'offline fixture only')
    seed = generation / 'seed.json'
    seed.write_bytes((Path(__file__).resolve().parents[1] / 'seed_harness/v2/seed.json').read_bytes())
    state = TaskAgentState(0, 'test_override', str(seed), checkpoint_path=str(checkpoint),
                           checkpoint_manifest=checkpoint_manifest(checkpoint))
    save_json(generation / 'evaluated_state.json', asdict(state))
    save_json(directory / 'final_state.json', {'status': 'completed', 'primary_metric': 'macro_success',
        'performance_history': [{'generation': 0, 'macro_success': 1, 'probe_identity': 'test_override_probe'}]})
    identity = pipeline.source_identity()
    save_json(directory / 'protocol.json', {'hash': 'test_override_protocol', 'controller': identity})
    frozen = pipeline.freeze(directory)
    assert frozen['task_harness_identity'] == harness_identity(seed)
    assert json.loads((directory / 'frozen_task.json').read_text()) == frozen
    monkeypatch.setattr(pipeline, 'source_identity', lambda: {**identity, 'task_runtime.py': 'changed'})
    with pytest.raises(ValueError, match='source changed'):
        pipeline.freeze(directory)
