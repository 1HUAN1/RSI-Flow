#!/usr/bin/env python3
"""Export a new, immutable Task H initialization including its runtime dependencies.

No API, model inference, GPU, training or baseline implementation is invoked.
Model/data bytes are verified and fingerprinted, not copied into this export.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sia.task_meta.pipeline import load_config, project_path  # noqa: E402
from sia.task_meta.task_harness import (  # noqa: E402
    harness_identity,
    legacy_view,
    load_harness,
    runtime_dependencies,
    task_harness_capabilities,
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Initialization sources/destinations cannot traverse links or junctions")
    return path


def _relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Initialization manifest paths must be contained relative paths")
    return Path(name)


def _copy(source, destination):
    source = _no_links(source)
    if not source.is_file():
        raise ValueError(f"Missing declared initialization dependency: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _model_files(model, checkpoint):
    checkpoint = _no_links(checkpoint)
    if Path(model["path"]).resolve() != checkpoint.resolve():
        raise ValueError("Prepared model identity refers to another checkpoint")
    declared = {}
    for item in model["weights"]:
        name = item["path"]
        path = checkpoint / _relative(name)
        _no_links(path)
        if not path.is_file() or path.stat().st_size != item["bytes"] or _digest(path) != item["sha256"]:
            raise ValueError("Checkpoint bytes differ from the prepared model identity")
        declared[name] = {"sha256": item["sha256"], "bytes": item["bytes"]}
    actual_weights = {p.name for p in checkpoint.iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin"}}
    if not declared or set(declared) != actual_weights:
        raise ValueError("Checkpoint weight inventory differs from the prepared model identity")
    for name, expected in model["tokenizer_and_config"].items():
        path = checkpoint / _relative(name)
        _no_links(path)
        if not path.is_file() or _digest(path) != expected:
            raise ValueError("Tokenizer/config bytes differ from the prepared model identity")
        declared[name] = {"sha256": expected, "bytes": path.stat().st_size}
    if model.get("model_type") != "qwen3" or not model.get("chat_template_sha256"):
        raise ValueError("A prepared Qwen3 chat-template identity is required")
    return declared


def export(config, destination):
    """Copy executable H and its exact dependency inventory into a NEW folder."""
    destination = _no_links(destination)
    if destination.exists():
        raise FileExistsError("Initialization export requires a new destination")
    seed_path = _no_links(project_path(config.seed_harness))
    harness = load_harness(seed_path)
    if harness.get("schema_version") != 2:
        raise ValueError("New initialization export requires a v2 Task Harness")
    legacy = legacy_view(harness)
    data = _no_links(project_path(config.data_dir))
    model_path = _no_links(data / "model_identity.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model_inventory = _model_files(model, config.task_checkpoint)
    tasks = _no_links(data / "tasks.sqlite")
    retrieval_path = _no_links(data / "search.sqlite.manifest.json")
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    corpus = _no_links(data / "search.sqlite")
    if not corpus.is_file() or _digest(corpus) != retrieval["sha256"]:
        raise ValueError("Retrieval corpus differs from its immutable manifest")
    dependencies = runtime_dependencies()
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Task runtime must declare its actual dependency inventory")
    required = {"sia/__init__.py", "sia/task_meta/__init__.py", "sia/task_meta/seed.py",
                "sia/task_meta/task_harness/__init__.py", "sia/task_meta/task_harness/policy.py",
                "sia/task_meta/task_harness/runtime.py"}
    if not required <= set(dependencies):
        raise ValueError("Declared Task dependencies omit a required execution module")
    source_map = {}
    for name, source in dependencies.items():
        _relative(name)
        source = _no_links(source)
        if not source.resolve().is_relative_to(ROOT):
            raise ValueError("Task runtime dependency escaped the fixed project source")
        source_map[name] = source
    reference = legacy["reference"]
    reference_root = ROOT / "seed_harness/reference"
    for name, expected in reference["files_sha256"].items():
        path = _no_links(reference_root / _relative(name))
        if not path.is_file() or _digest(path) != expected:
            raise ValueError("Archived HarnessForge reference differs from its registered fingerprint")
    seed_hash = _digest(seed_path)
    source_hashes = {name: _digest(path) for name, path in source_map.items()}
    destination.mkdir(parents=True, exist_ok=False)
    _copy(seed_path, destination / "seed.json")
    _copy(ROOT / "seed_harness/seed.json", destination / "legacy/seed.json")
    _copy(ROOT / "seed_harness/README.md", destination / "legacy/README.md")
    for name, path in source_map.items():
        _copy(path, destination / "runtime" / _relative(name))
    for name in reference["files_sha256"]:
        _copy(reference_root / name, destination / "reference" / _relative(name))
    _copy(ROOT / "pyproject.toml", destination / "dependencies/pyproject.toml")
    versions = {}
    for distribution in ("pydantic", "jsonschema", "transformers", "torch", "httpx", "requests"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    _write(destination / "dependencies/environment.json", {"python": platform.python_version(),
        "implementation": platform.python_implementation(), "packages": versions,
        "scope": "Recorded installed dependency versions; external packages are not vendored or installed by export"})
    _write(destination / "task_capabilities.json", task_harness_capabilities(seed_path))
    _write(destination / "model_identity.json", model)
    (destination / "artifacts").mkdir()
    (destination / "README.md").write_text(
        "# Shared five-part Task initialization\n\n"
        "This export snapshots the project Task Harness and every runtime file declared by runtime_dependencies(). "
        "Model checkpoint/data files remain external and are bound by verified hashes; they are not embedded. "
        "Third-party Python packages remain external with versions recorded.\n\n"
        "Load runtime/sia/task_meta/seed.py through the copied runtime package and call run_seed with the trusted current-Qwen "
        "model callback and permitted environment. This snapshot does not create an unrestricted model/tool client. "
        "Use the parent export script's --verify command before loading.\n\n"
        "Initial artifacts and working memory are empty. The seven historical reference cold-start rules are registered "
        "as Harness preloads with native long-term provision disabled; old run memory is excluded. "
        "Legacy adaptations remain documented under legacy/README.md.\n\n"
        "Future baselines must receive separate copies of the same snapshot and keep their own search algorithms. "
        "Exporting does not establish that any baseline has been executed or aligned. No API/GPU/training is started.\n",
        encoding="utf-8")
    files = {p.relative_to(destination).as_posix(): {"sha256": _digest(p), "bytes": p.stat().st_size}
             for p in sorted(destination.rglob("*")) if p.is_file()}
    record = {"schema_version": "task-harness-initialization-v2", "state": "CODE_COMPLETE",
        "status_scope": "Artifact construction and fingerprint validation only; runtime/pilot/report readiness is not inferred",
        "task_harness": harness_identity(seed_path), "harness_sha256": seed_hash,
        "runtime_entrypoint": "sia.task_meta.seed.run_seed", "runtime_root": "runtime",
        "runtime_dependencies": source_hashes, "files": files,
        "task_model": model, "external_model_files": model_inventory,
        "model_verification": "Current checkpoint/tokenizer/config bytes matched the prepared identity; no model was loaded",
        "external_data": {"tasks": {"path": str(tasks), "sha256": _digest(tasks)},
                          "retrieval": {"path": str(corpus), "sha256": retrieval["sha256"]},
                          "retrieval_manifest": retrieval},
        "initialization": legacy["initialization"], "artifact_initial_manifest": [],
        "artifact_initial_state": "empty", "budget": legacy["budget"],
        "protocol": {"random_seed": config.seed, "probe_per_domain": config.probe_per_domain,
            "window_quotas": config.window_quotas, "rollouts_per_task": config.rollouts_per_task,
            "probe_rollouts": config.probe_rollouts, "task_enable_thinking": config.task_enable_thinking,
            "task_checkpoint": config.task_checkpoint},
        "reference": reference, "legacy_seed_sha256": _digest(ROOT / "seed_harness/seed.json"),
        "alignment_status": "five_part_common_initialization_exported; baseline execution and real runtime readiness unverified",
        "real_api_calls": 0, "gpu_calls": 0, "training_runs": 0}
    record["initialization_hash"] = _identity(record)
    if _digest(seed_path) != seed_hash or any(_digest(path) != source_hashes[name] for name, path in source_map.items()):
        raise ValueError("Initialization source changed during export; this candidate export must not be used")
    _write(destination / "initialization.json", record)
    verify_export(destination)
    return record


def verify_export(destination, *, verify_external=False):
    """Reject modified/extra files without importing exported executable code."""
    destination = _no_links(destination)
    record = json.loads((destination / "initialization.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in record.items() if key != "initialization_hash"}
    if record.get("schema_version") != "task-harness-initialization-v2" or _identity(payload) != record.get("initialization_hash"):
        raise ValueError("Initialization manifest identity changed")
    files = record["files"]
    actual = set()
    for path in destination.rglob("*"):
        _no_links(path)
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if actual != set(files) | {"initialization.json"}:
        raise ValueError("Initialization contains an undeclared or missing file")
    for name, entry in files.items():
        path = destination / _relative(name)
        if path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
            raise ValueError("Initialization payload bytes changed")
    if not (destination / "artifacts").is_dir() or any((destination / "artifacts").iterdir()):
        raise ValueError("Initial artifacts are not empty")
    if verify_external:
        _model_files(record["task_model"], record["protocol"]["task_checkpoint"])
        for key in ("tasks", "retrieval"):
            external = record["external_data"][key]
            if _digest(_no_links(external["path"])) != external["sha256"]:
                raise ValueError("External data fingerprint changed")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multidomain-dev.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="New directory only; never overwrite old exports")
    group.add_argument("--verify", type=Path, help="Verify an existing exported snapshot without loading its code")
    parser.add_argument("--verify-external", action="store_true", help="Also rehash external checkpoint/data on --verify")
    args = parser.parse_args(argv)
    if args.verify_external and args.verify is None:
        parser.error("--verify-external requires --verify")
    try:
        result = verify_export(args.verify, verify_external=args.verify_external) if args.verify else export(load_config(args.config), args.output)
    except FileExistsError:
        parser.error("--output must be a new directory")
    print(json.dumps({"state": result["state"], "initialization_hash": result["initialization_hash"],
                      "runtime_ready": False, "pilot_verified": False, "report_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
