"""Pinned upstream HarnessForge production for one complete Task candidate.

The three prompts, Stage-3 parser, and Stage-4 validator are loaded from the
vendored upstream commit. This module adapts only Meta calls and Task-state
commit; it does not preselect files or translate candidates into the retired
event/configuration interface.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sia.task_meta.harnessforge_manifest import (
    HarnessBundleManifest,
    MANIFEST_KIND,
    PINNED_HARNESSFORGE_COMMIT,
    build_manifest,
    load_manifest,
    save_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "HarnessForge_4B"
PRODUCTION_ROOT = UPSTREAM_ROOT / "harness_production"
BASE_HARNESS_ROOT = UPSTREAM_ROOT / "harness_factory" / "base_harness"
EVOLVED_HARNESS_ROOT = UPSTREAM_ROOT / "evolved_pairs" / "harness_factory" / "rounds"
MAX_FIX_ATTEMPTS = 3


class StageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1)


class RepairFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    content: str


class RepairResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    files: list[RepairFile]


def _load_upstream(name: str, filename: str):
    path = PRODUCTION_ROOT / filename
    if not path.is_file():
        raise RuntimeError(f"Pinned HarnessForge production file is missing: {path}")
    module_name = f"_rsi_pinned_harnessforge_{name}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned HarnessForge production file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def upstream_generation_module():
    return _load_upstream("generation", "run_harness_production.py")


def upstream_validation_module():
    return _load_upstream("validation", "04_validation_retry.py")


def initialize_base_manifest(destination: str | Path) -> HarnessBundleManifest:
    """Create an initial Task Harness from the pinned upstream base bundle."""
    if not BASE_HARNESS_ROOT.is_dir():
        raise RuntimeError(f"Pinned upstream base harness is missing: {BASE_HARNESS_ROOT}")
    manifest = build_manifest(BASE_HARNESS_ROOT, harness_name="base_harness")
    save_manifest(destination, manifest)
    return manifest


def is_harnessforge_manifest(path: str | Path) -> bool:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("kind") == MANIFEST_KIND


def harnessforge_capabilities(path: str | Path | None = None) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    if path is not None:
        manifest = load_manifest(path)
        identity = {"harness_name": manifest.harness_name, "bundle_sha256": manifest.bundle_sha256}
    return {
        "available": True,
        **identity,
        "upstream_commit": PINNED_HARNESSFORGE_COMMIT,
        "operations": [{"operation": "produce_harness", "target": "harness_bundle"}],
        "constraints": [
            "HarnessForge Stage 1 owns module localization after component routing",
            "Stage 2 uses Meta experience only as optional advice",
            "Stage 3 emits one complete independent Harness bundle",
            "The upstream validator may perform at most three small repairs",
            "Acceptance is decided later only by strict positive same-task score gain",
        ],
    }


def harnessforge_sources(path: str | Path) -> dict[str, str]:
    manifest = load_manifest(path)
    return {"harness_manifest.json": json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), **{
        "harness/" + name: content for name, content in manifest.files.items()
    }}


def harnessforge_identity(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(path)
    return {
        "schema_version": "harnessforge_bundle_v1",
        "kind": MANIFEST_KIND,
        "harness_name": manifest.harness_name,
        "bundle_sha256": manifest.bundle_sha256,
        "upstream_commit": manifest.upstream_commit,
        "source_hashes": {
            name: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for name, content in manifest.files.items()
        },
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_once(path: Path, content: str) -> None:
    normalized = content.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != normalized:
            raise ValueError(f"Recovered HarnessForge stage changed: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")


def _snapshot(manifest: HarnessBundleManifest, max_chars: int = 120_000) -> str:
    upstream = upstream_generation_module()
    blocks = [upstream.format_file_block(name, content) for name, content in manifest.files.items()]
    return upstream.clip("\n\n".join(blocks), max_chars, manifest.harness_name)


def _archive_examples() -> tuple[str, str, str]:
    upstream = upstream_generation_module()
    names = upstream.collect_existing_harness_names(UPSTREAM_ROOT, "harness_factory")
    selected = sorted(path.parent for path in EVOLVED_HARNESS_ROOT.rglob("builder.py"))
    if not selected:
        selected = [BASE_HARNESS_ROOT]
    examples = upstream.snapshot_many(
        [str(path) for path in selected], UPSTREAM_ROOT, max_chars=160_000
    )
    overview = {
        "source": "pinned upstream HarnessForge archive",
        "upstream_commit": PINNED_HARNESSFORGE_COMMIT,
        "available_harnesses": names.splitlines(),
        "selected_evolved_pair_examples": [path.name for path in selected],
        "selection_note": "Published evolved-pair winners are bounded few-shot references.",
    }
    return _json(overview), examples, names


def _optional_meta_experience(context: Any, decision: Any) -> str:
    observation = getattr(context, "observation", None)
    payload: dict[str, Any] = {
        "policy": "optional advisory reference; empty or unmatched memory never blocks production",
        "decision_used_principle_ids": list(getattr(decision, "used_principle_ids", []) or []),
        "improvement_history": list(getattr(observation, "improvement_history", []) or [])[-24:],
        "experience_ledger": list(getattr(observation, "experience_ledger", []) or [])[-24:],
    }
    meta = getattr(context, "meta_state", None)
    bundle = Path(meta.bundle_path) if getattr(meta, "bundle_path", None) else None
    principle_path = bundle / "principles.json" if bundle else None
    if principle_path and principle_path.is_file():
        library = json.loads(principle_path.read_text(encoding="utf-8"))
        records = library.get("records", []) if isinstance(library, dict) else []
        payload["active_harness_and_general_principles"] = [
            record for record in records
            if record.get("active") and (
                str(record.get("principle_id", "")).startswith("skill.HARNESS.")
                or str(record.get("principle_id", "")).startswith("principle.")
            )
        ]
    return _json(payload)


def _trajectory_values(context: Any) -> dict[str, str]:
    observation = getattr(context, "observation", None)
    if observation is None:
        return {
            "metrics_summary": "{}", "answer_type_breakdown": "{}", "failure_summary": "[]",
            "trajectory_overview": "{}", "failure_trajectory_samples": "[]",
            "success_trajectory_samples": "[]",
        }
    raw = list(getattr(observation, "raw_trajectories", []) or getattr(observation, "trajectories", []) or [])
    failures = [row for row in raw if (row.get("verification") or {}).get("success") is False]
    successes = [row for row in raw if (row.get("verification") or {}).get("success") is True]
    summary = getattr(observation, "trajectory_summary", {}) or {}
    return {
        "metrics_summary": _json(getattr(observation, "current_performance", {})),
        "answer_type_breakdown": _json(summary.get("answer_type_breakdown", {})),
        "failure_summary": _json({"count": len(failures), "trajectory_summary": summary}),
        "trajectory_overview": _json({
            "coverage": getattr(observation, "observation_coverage", {}),
            "summary": summary, "total_raw_trajectories": len(raw),
        }),
        "failure_trajectory_samples": _json(getattr(observation, "failure_examples", None) or failures[:24]),
        "success_trajectory_samples": _json(getattr(observation, "success_examples", None) or successes[:24]),
    }


def _candidate_name(generation: int, decision_id: str) -> str:
    suffix = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:10]
    return f"harness_g{generation:04d}_{suffix}"


def _render_stage(stage: str, values: dict[str, str]) -> str:
    upstream = upstream_generation_module()
    template = upstream.load_prompt_template(
        PRODUCTION_ROOT / upstream.STAGE_FILE_CANDIDATES[stage][0]
    )
    prompt = upstream.render_template(template, values)
    return prompt + (upstream.STAGE3_FILE_CONTRACT if stage == "stage3" else "")


def _parse_candidate(
    response: str, candidate_dir: Path, target_round: str, candidate_name: str
) -> HarnessBundleManifest:
    upstream_generation_module().extract_stage3_files(
        response,
        candidate_dir=candidate_dir,
        target_round=target_round,
        candidate_name=candidate_name,
        overwrite=True,
    )
    return build_manifest(candidate_dir, harness_name=candidate_name)


def _pure_generation_check(value: Any, target_round: str, candidate_name: str) -> dict[str, Any]:
    response = StageResponse.model_validate(value)
    with tempfile.TemporaryDirectory(prefix="rsi_harnessforge_stage3_") as temporary:
        manifest = _parse_candidate(
            response.content, Path(temporary) / candidate_name, target_round, candidate_name
        )
    return {"passed": True, "bundle_sha256": manifest.bundle_sha256}


class _RepairModelAdapter:
    def __init__(self, complete: Callable[[str, type[BaseModel]], BaseModel]):
        self.complete = complete

    def __call__(self, messages: list[dict[str, Any]]):
        prompt = messages[0]["content"][0]["text"]
        result = RepairResponse.model_validate(self.complete(prompt, RepairResponse))
        return SimpleNamespace(content=json.dumps(result.model_dump(mode="json"), ensure_ascii=False))


def _validate_and_repair(
    candidate_dir: Path,
    workflow: Path,
    complete_repair: Callable[[str, type[BaseModel]], BaseModel],
) -> dict[str, Any]:
    from sia.task_meta.types import DecisionConstraintError

    upstream = upstream_validation_module()
    from sia.task_meta.harnessforge_validation import validate_candidate

    history: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []
    report = validate_candidate(candidate_dir)
    history.append(report.to_dict())
    attempts = 0
    while (
        report.verdict != "passed"
        and report.verdict in upstream.FIXABLE_STATUSES
        and attempts < MAX_FIX_ATTEMPTS
    ):
        attempts += 1
        before = upstream.file_hashes(candidate_dir)
        payload = upstream.request_fix(
            model=_RepairModelAdapter(complete_repair), candidate_path=candidate_dir,
            report=report, attempt=attempts, max_attempts=MAX_FIX_ATTEMPTS,
        )
        declared = upstream.apply_fix_payload(candidate_dir, payload, dry_run=False)
        actual = upstream.changed_files(before, upstream.file_hashes(candidate_dir))
        fixes.append({
            "repair_agent": "meta_json_model", "attempt": attempts,
            "summary": payload.get("summary"), "declared_changed_files": declared,
            "changed_files": actual,
        })
        if not actual:
            break
        report = validate_candidate(candidate_dir)
        history.append(report.to_dict())
    final_verdict = report.verdict
    if report.verdict == "passed" and attempts:
        final_verdict = "fixed_after_retry"
    elif report.verdict != "passed" and report.verdict != "failed_environment" and attempts >= MAX_FIX_ATTEMPTS:
        final_verdict = "needs_regeneration"
    result = {
        "success": report.verdict == "passed", "final_verdict": final_verdict,
        "upstream_validator": str(PRODUCTION_ROOT / "04_validation_retry.py"),
        "max_fix_attempts": MAX_FIX_ATTEMPTS, "fixes_used": attempts,
        "history": history, "fixes": fixes,
    }
    _write_once(workflow / "04_validation_retry.json", _json(result))
    if not result["success"]:
        detail = report.errors[0] if report.errors else report.verdict
        message = f"HarnessForge candidate failed validation ({report.verdict}): {detail}"
        if report.verdict == "failed_environment":
            raise RuntimeError(message)
        raise DecisionConstraintError(message)
    return result


class HarnessForgeProductionUpdater:
    """Run one upstream production and commit exactly one child manifest."""

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _validate_request(decision: Any) -> None:
        from sia.task_meta.types import DecisionConstraintError, TaskUpdateAction
        if decision.action != TaskUpdateAction.HARNESS or decision.target_components != [TaskUpdateAction.HARNESS]:
            raise DecisionConstraintError("HarnessForge production requires the selected HARNESS component")
        if len(decision.requested_changes) != 1:
            raise DecisionConstraintError("HarnessForge production accepts one complete-bundle request")
        change = decision.requested_changes[0]
        if (change.operation, change.target) != ("produce_harness", "harness_bundle"):
            raise DecisionConstraintError(
                "HARNESS must request produce_harness/harness_bundle without preselecting a module"
            )

    def apply(self, task_state: Any, decision: Any, context: Any):
        from sia.task_meta.meta import evolution_kwargs
        from sia.task_meta.storage import clone_task, digest
        from sia.task_meta.types import DecisionConstraintError, TaskUpdate, TaskUpdateAction

        started = time.monotonic()
        self._validate_request(decision)
        parent_path = Path(task_state.harness_path)
        parent_digest = digest(parent_path)
        try:
            parent = load_manifest(parent_path)
        except (OSError, TypeError, ValueError) as exc:
            raise DecisionConstraintError(
                "HARNESS production requires a pinned HarnessForge bundle manifest"
            ) from exc

        workflow = Path(context.directory) / "harnessforge_production"
        workflow.mkdir(parents=True, exist_ok=True)
        target_round = f"round_rsi_{context.generation:04d}"
        candidate_name = _candidate_name(
            context.generation, decision.decision_id or f"generation-{context.generation}"
        )
        candidate_dir = (
            workflow / "validation_project" / "generated_harnesses" / "rounds"
            / target_round / candidate_name
        )
        winner_snapshot = _snapshot(parent)
        archive_overview, archive_examples, existing_names = _archive_examples()
        values = {
            **_trajectory_values(context),
            "winner_harness_name": parent.harness_name,
            "winner_harness_snapshot": winner_snapshot,
            "winner_harness_template": winner_snapshot,
            "module_files_info": "\n".join(f"- {name}" for name in parent.files),
            "harness_pool_overview": archive_overview
            + "\n\n## Optional Meta experience (advisory)\n"
            + _optional_meta_experience(context, decision),
            "harness_examples": archive_examples,
            "selected_harness_examples": archive_examples,
            "existing_harness_names": existing_names,
            "target_round": target_round,
            "candidate_name": candidate_name,
            "example_name": "pinned evolved-pair winner",
            "module_localization_report": "",
            "improvement_direction_brief": "",
        }
        sources = {f"winner/{name}": content for name, content in parent.files.items()}

        def call_stage(stage: str, validator: Callable[[Any], Any]) -> StageResponse:
            prompt = _render_stage(stage, values)
            stem = {
                "stage1": "01_module_localization", "stage2": "02_improvement_directions",
                "stage3": "03_harness_generation",
            }[stage]
            _write_once(workflow / f"{stem}.prompt.md", prompt)
            receipt = workflow / f"{stem}.response.json"
            if receipt.exists():
                try:
                    response = StageResponse.model_validate(json.loads(receipt.read_text(encoding="utf-8")))
                    validator(response)
                except (OSError, ValueError, SyntaxError, ValidationError) as exc:
                    raise DecisionConstraintError(
                        f"HarnessForge {stage} candidate response is invalid: {exc}"
                    ) from exc
                sources[f"workflow/{stem}.md"] = response.content
                return response
            kwargs = evolution_kwargs(
                self.client, context, task_state, decision, sources, validator
            )
            raw_response = self.client.complete(
                prompt, StageResponse, meta_state=context.meta_state,
                operation="harness_patch", decision_id=decision.decision_id, **kwargs,
            )
            try:
                response = StageResponse.model_validate(raw_response)
                validator(response)
            except (OSError, ValueError, SyntaxError, ValidationError) as exc:
                raise DecisionConstraintError(
                    f"HarnessForge {stage} candidate response is invalid: {exc}"
                ) from exc
            _write_once(receipt, _json(response.model_dump(mode="json")))
            _write_once(workflow / f"{stem}.md", response.content)
            sources[f"workflow/{stem}.md"] = response.content
            return response

        def nonempty(value: Any) -> dict[str, bool]:
            return {"passed": bool(StageResponse.model_validate(value).content.strip())}

        localization = call_stage("stage1", nonempty)
        values["module_localization_report"] = localization.content
        directions = call_stage("stage2", nonempty)
        values["improvement_direction_brief"] = directions.content
        generated = call_stage(
            "stage3", lambda value: _pure_generation_check(value, target_round, candidate_name)
        )
        try:
            candidate = _parse_candidate(
                generated.content, candidate_dir, target_round, candidate_name
            )
        except (OSError, ValueError, SyntaxError) as exc:
            raise DecisionConstraintError(
                f"HarnessForge Stage 3 did not produce a complete bundle: {exc}"
            ) from exc

        def complete_repair(prompt: str, schema: type[BaseModel]):
            def validate_repair(value: Any):
                repair = RepairResponse.model_validate(value)
                for item in repair.files:
                    raw = item.path.replace("\\", "/")
                    path = PurePosixPath(raw)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe HarnessForge repair path: {item.path}")
                return {"passed": True}

            kwargs = evolution_kwargs(
                self.client, context, task_state, decision, sources, validate_repair
            )
            return self.client.complete(
                prompt, schema, meta_state=context.meta_state, operation="harness_patch",
                decision_id=decision.decision_id, **kwargs,
            )

        validation = _validate_and_repair(candidate_dir, workflow, complete_repair)
        candidate = build_manifest(candidate_dir, harness_name=candidate_name)
        if dict(candidate.files) == dict(parent.files):
            raise DecisionConstraintError("HarnessForge generated an unchanged candidate bundle")
        if digest(parent_path) != parent_digest:
            raise DecisionConstraintError("Parent HarnessForge bundle changed during production")

        successor = clone_task(task_state, context.generation, context.directory)
        save_manifest(successor.harness_path, candidate)
        if load_manifest(successor.harness_path).to_dict() != candidate.to_dict():
            raise RuntimeError("Committed HarnessForge bundle did not reload exactly")

        sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        names = sorted(set(parent.files) | set(candidate.files))
        files = [{
            "path": name,
            "before_sha256": sha(parent.files[name]) if name in parent.files else None,
            "after_sha256": sha(candidate.files[name]) if name in candidate.files else None,
        } for name in names if parent.files.get(name) != candidate.files.get(name)]
        applied = [{
            "id": decision.requested_changes[0].id, "operation": "produce_harness",
            "target": "harness_bundle", "before_sha256": parent.bundle_sha256,
            "after_sha256": candidate.bundle_sha256,
            "changed_files": [item["path"] for item in files],
            "semantic_status": "unverified_until_same_task_evaluation",
        }]
        return successor, TaskUpdate(
            TaskUpdateAction.HARNESS,
            f"Produced one complete HarnessForge candidate: {candidate_name}",
            {
                "mechanism": "pinned_upstream_harnessforge_production",
                "upstream_commit": PINNED_HARNESSFORGE_COMMIT,
                "candidate_name": candidate_name,
                "parent_bundle_sha256": parent.bundle_sha256,
                "bundle_sha256": candidate.bundle_sha256,
                "module_localization": localization.content,
                "improvement_direction": directions.content,
                "validation": validation, "candidate_count": 1,
            },
            {"wall_time_seconds": time.monotonic() - started, "api_cost_usd": None, "gpu_hours": None},
            requested_changes=[change.model_dump(mode="json") for change in decision.requested_changes],
            applied_changes=applied, files=files,
            checks=[
                {"name": "upstream_stage1_module_localization", "passed": True},
                {"name": "upstream_stage2_improvement_directions", "passed": True},
                {"name": "upstream_stage3_complete_bundle", "passed": True},
                {"name": "upstream_static_import_build_validation", "passed": True},
            ],
            semantic_status="unverified_until_same_task_evaluation",
        )


__all__ = [
    "HarnessForgeProductionUpdater", "harnessforge_capabilities", "harnessforge_identity",
    "harnessforge_sources", "initialize_base_manifest", "is_harnessforge_manifest",
]
