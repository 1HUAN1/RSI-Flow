import json
from pathlib import Path
from unittest.mock import patch

from launch_meta import wait_for_idle_gpus, process_still_running
from task_adapter import TaskAdapter


def test_all_eight_cards_and_training_exit_are_required(tmp_path):
    idle = [{"gpu": i, "memory_used_mib": 0, "utilization_percent": 0} for i in range(8)]
    busy = [dict(row, memory_used_mib=6000 if row["gpu"] == 7 else 0) for row in idle]
    samples = iter([idle[:4], busy, idle, idle, idle])
    pauses = []
    with patch('launch_meta.process_still_running', side_effect=[True, True, True, False, False]):
        result = wait_for_idle_gpus(tmp_path, gpu_ids=range(8),
                                   wait_processes=[{'pid': 123, 'start_ticks': '1'}],
                                   probe=lambda: next(samples), pause=pauses.append)
    assert len(pauses) == 4
    assert result['gpu_ids'] == list(range(8))
    assert result['waiting_processes'] == []
    assert result['consecutive_idle_checks'] == 2


def test_pid_reuse_does_not_delay_start():
    import os
    pid = os.getpid()
    start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    assert process_still_running({'pid': pid, 'start_ticks': start})
    assert not process_still_running({'pid': pid, 'start_ticks': str(int(start) + 1)})


def test_runtime_eight_gpu_configuration_preserves_global_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    adapter = TaskAdapter(root, tmp_path)
    config = adapter._config({'config_path': str(root / 'configs/train_180_a0_v1.json')})
    assert config.gpu_execution == 'phased_eight'
    assert [r['gpu'] for r in config.task_replicas] == list(range(8))
    assert len(set(r['base_url'] for r in config.task_replicas)) == 8
    assert config.training['gradient_accumulation_steps'] == 8
    assert config.max_generations == 5
    from sia.task_meta.pipeline import PipelineConfig
    import pytest
    value = config.model_dump()
    value['task_replicas'] = value['task_replicas'][:4]
    with pytest.raises(ValueError, match='8 inference replicas'):
        PipelineConfig.model_validate(value).checked()


def test_service_verification_reads_every_eight_gpu_binding(tmp_path):
    from contextlib import contextmanager
    from sia.task_meta.gpu_phases import verify_services
    from sia.task_meta.storage import checkpoint_manifest
    root = Path(__file__).resolve().parents[1]
    config = TaskAdapter(root, tmp_path)._config({'config_path': str(root / 'configs/train_180_a0_v1.json')})
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"mock-weight")
    expected = {'checkpoint_path': str(checkpoint), 'weights': checkpoint_manifest(checkpoint)}
    called = []
    class Opener:
        @contextmanager
        def open(self, url, timeout):
            import io
            gpu = next(r['gpu'] for r in config.task_replicas if url == r['base_url'].removesuffix('/v1') + '/health')
            called.append(gpu)
            yield io.StringIO(json.dumps({'ready': True, 'visible_devices': str(gpu), 'bindings': {str(checkpoint): expected}}))
    with patch('sia.task_meta.gpu_phases.build_opener', return_value=Opener()):
        records = verify_services(config, checkpoint)
    assert called == list(range(8))
    assert len(records) == 8
