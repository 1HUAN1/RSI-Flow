"""Pure HarnessForge Task-Harness facade for the RSIFlow_4B loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sia.task_meta.harnessforge_manifest import HarnessBundleManifest, load_manifest
from sia.task_meta.harnessforge_production import harnessforge_capabilities
from sia.task_meta.harnessforge_runtime import HarnessForgeBudgetExceeded, run_harnessforge


# Existing executors catch this name at their outer budget boundary.
_BudgetExhausted = HarnessForgeBudgetExceeded


def validate_seed(spec: dict[str, Any] | HarnessBundleManifest) -> dict[str, Any]:
    manifest = spec if isinstance(spec, HarnessBundleManifest) else HarnessBundleManifest.from_dict(spec)
    return manifest.to_dict()


def load_seed(path: str | Path) -> dict[str, Any]:
    """Load only a complete, content-addressed HarnessForge bundle manifest."""
    return load_manifest(path).to_dict()


def seed_capabilities(path: str | Path) -> dict[str, Any]:
    return harnessforge_capabilities(path)


def run_seed(
    seed_spec: dict[str, Any] | HarnessBundleManifest,
    model_callable,
    environment,
    task_prompt: str,
    artifacts_text: str = "",
    seed: int = 42,
    *,
    artifact_sources=None,
    memory_storage_root: str | Path | None = None,
    bench_type: str | None = None,
    max_model_calls: int = 128,
    max_tool_calls: int = 128,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_steps: int = 40,
):
    manifest = (
        seed_spec
        if isinstance(seed_spec, HarnessBundleManifest)
        else HarnessBundleManifest.from_dict(seed_spec)
    )
    return run_harnessforge(
        manifest,
        model_callable,
        environment,
        task_prompt,
        artifacts_text,
        seed,
        artifact_sources=artifact_sources,
        memory_storage_root=memory_storage_root,
        bench_type=bench_type,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_tokens=max_tokens,
        temperature=temperature,
        max_steps=max_steps,
    )


__all__ = [
    "_BudgetExhausted",
    "load_seed",
    "run_seed",
    "seed_capabilities",
    "validate_seed",
]
