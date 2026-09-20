"""Regression for the real validator import that created an undeclared .pyc."""
import subprocess
import sys
import tomllib
from pathlib import Path

from sia.task_meta.meta_backends.codex_openrouter import render_codex_config
from sia.task_meta.meta_backends.contracts import MetaBackendConfig


def test_validator_import_uses_fixed_no_bytecode_environment(tmp_path):
    config = tomllib.loads(render_codex_config(MetaBackendConfig(), isolated=True))
    policy = config["shell_environment_policy"]
    assert policy["inherit"] == "none"
    assert policy["set"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    # Pinned Codex applies include_only AFTER set. Retain the explicit setting.
    environment = {k: v for k, v in policy["set"].items() if k in policy["include_only"]}
    assert environment == {"PYTHONDONTWRITEBYTECODE": "1"}
    root = Path(__file__).resolve().parents[1]
    validator = tmp_path / "meta_policy_validator.py"
    validator.write_bytes((root / "sia/task_meta/meta_harness/policies.py").read_bytes())
    code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('fixed',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.validate_policy); print(sys.dont_write_bytecode)"
    result = subprocess.run([sys.executable, "-c", code, str(validator)], env=environment,
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"
    assert list(tmp_path.iterdir()) == [validator]
