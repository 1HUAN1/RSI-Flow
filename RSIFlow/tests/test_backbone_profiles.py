"""Both backbones must execute one skill algorithm with identical hyperparameters."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]

from task_adapter import TaskAdapter
import launcher


def test_effective_runtime_differs_only_in_checkpoint(tmp_path):
    adapter = TaskAdapter(ROOT, tmp_path)
    configs = [adapter._config({'config_path': str(ROOT / f'configs/train_{size}.json')}).model_dump()
               for size in ('4b', '8b')]
    assert configs[0].pop('task_checkpoint').endswith('Qwen3-4B')
    assert configs[1].pop('task_checkpoint').endswith('Qwen3-8B')
    assert configs[0] == configs[1]
    profiles = [json.loads((ROOT / f'configs/train_{size}.json').read_text()) for size in ('4b', '8b')]
    for profile in profiles:
        profile.pop('base_config')
        profile.pop('run_name')
    assert profiles[0] == profiles[1]


def test_backbones_dispatch_same_launcher_and_preparation(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda argv, **kw: (calls.append((argv, kw)) or SimpleNamespace(returncode=0)))
    for size in ('4b', '8b'):
        assert launcher.main(size, ['--prepare-data', '--run-dir', '/tmp/test-run']) == 0
    assert len(calls) == 4
    for index, size in ((0, '4b'), (2, '8b')):
        assert calls[index][0][1] == str(ROOT / 'prepare_600_744.py')
        assert calls[index+1][0][1] == str(ROOT / 'launch_meta.py')
        assert calls[index][0][-1] == str(ROOT / f'configs/train_{size}.json')
        assert calls[index+1][0][3] == str(ROOT / f'configs/train_{size}.json')


def test_preflight_never_prepares_data(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda argv, **kw: (calls.append(argv) or SimpleNamespace(returncode=0)))
    assert launcher.main('8b', ['--prepare-data', '--check']) == 0
    assert len(calls) == 1
    assert calls[0][1] == str(ROOT / 'launch_meta.py')


def test_seed_manifest_digest_and_vendored_runtime_resolve():
    from rsiflow.task_meta.rsiflow_manifest import load_manifest
    from rsiflow.task_meta.rsiflow_runtime import _upstream_root
    from rsiflow.task_meta.rsiflow_production import PRODUCTION_ROOT
    manifest = load_manifest(ROOT / 'seed_harness/rsiflow_base_manifest.json')
    assert manifest.to_dict()['bundle_sha256'] == manifest.bundle_sha256
    assert _upstream_root() == ROOT / 'upstream/RSIFlow_task'
    assert PRODUCTION_ROOT.is_dir()
