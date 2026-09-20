"""Use the normalized in-tree RSIFlow runtime without source rewriting.

The previous project installed an older runtime and then applied a stack of
string-replacement patches.  RSIFlow_4B owns its runtime directly: the code
that is tested is exactly the code that is executed.
"""

from pathlib import Path

from common import ROOT


def install(config):
    runtime = (ROOT / "runtime").resolve()
    configured = Path(config.get("runtime_source", runtime))
    if not configured.is_absolute():
        configured = ROOT / configured
    if configured.resolve() != runtime:
        raise ValueError(
            "RSIFlow_4B executes only its in-tree runtime; runtime_source must point to "
            + str(runtime)
        )
    required = (
        runtime / "sia/task_meta/pipeline.py",
        runtime / "sia/task_meta/harnessforge_manifest.py",
        runtime / "scripts/train_task_meta_sft.py",
        runtime / "pyproject.toml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Incomplete RSIFlow_4B runtime: " + ", ".join(missing))
    return runtime


__all__ = ["install"]
