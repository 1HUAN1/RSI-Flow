"""Immutable G snapshots; v1 compatibility and executable v2 rules.

Only the trusted controller publishes snapshots. The v2 rule interpreter is a
project extension, not a Codex kernel edit or an arbitrary Python execution hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

LEGACY_EDITABLE_FILES = frozenset({"instructions.md", "context.json", "workflow.json"})
EDITABLE_FILES = LEGACY_EDITABLE_FILES | {"evolution.json"}
PRINCIPLE_FILES = EDITABLE_FILES | {"principles.json", "self_update_protocol.md"}
SCHEMA_FILES = {"meta-bundle-v1": LEGACY_EDITABLE_FILES, "meta-bundle-v2": EDITABLE_FILES, "meta-bundle-v3": PRINCIPLE_FILES}
CONTRACT_VERSION = "rsi-meta-contract-v2"
OPERATIONS = ("routing", "harness_patch", "artifact_patch", "model_request", "meta_self_update", "final_consolidation", "compatibility_smoke")
ENTRYPOINTS = {name: "sia.task_meta.meta_harness.runtime.execute" for name in OPERATIONS}
MECHANISM_LAYERS = {name: "rsiH_project_executable_configuration" for name in
                    ("evidence", "diagnosis", "modification", "experience", "self_update")}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def strict_json(text: str):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def reject_links(path: Path) -> None:
    """Inspect lexical paths, including the root, before resolving them."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Symlink or junction is forbidden in Bundle paths: {part}")


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(path: Path, value) -> None:
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


class ContextPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_evidence_chars: int = Field(default=64000, ge=4000, le=120000)
    selection: str = "head_and_tail"


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section_order: list[str] = Field(default_factory=lambda: ["instructions", "evidence", "response_contract"])
    checklist: list[str] = Field(default_factory=lambda: ["Inspect concrete evidence", "Select one available intervention", "Validate the requested output schema"])


def validate_files(files):
    schema = next((name for name, allowed in SCHEMA_FILES.items() if set(files) == allowed), None)
    if schema is None:
        raise ValueError("Bundle must have exactly the declared v1 or v2 editable files")
    from .evidence_delivery import SKILL_LIBRARY_BYTES
    for name, content in files.items():
        limit = SKILL_LIBRARY_BYTES if name == 'principles.json' else 128000
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > limit:
            raise ValueError(f"Invalid bounded Bundle file: {name}; limit={limit}")
    context = ContextPolicy.model_validate(strict_json(files["context.json"]), strict=schema != "meta-bundle-v1")
    if context.selection not in {"head", "tail", "head_and_tail"}:
        raise ValueError("Unsupported context selection")
    workflow = WorkflowPolicy.model_validate(strict_json(files["workflow.json"]), strict=schema != "meta-bundle-v1")
    if sorted(workflow.section_order) != ["evidence", "instructions", "response_contract"]:
        raise ValueError("Workflow must include each required section exactly once")
    if not 1 <= len(workflow.checklist) <= 12 or any(not s.strip() or len(s) > 1000 for s in workflow.checklist):
        raise ValueError("Workflow checklist exceeds its immutable bounds")
    if schema in {"meta-bundle-v2", "meta-bundle-v3"}:
        from .policies import validate_evolution
        validate_evolution(strict_json(files["evolution.json"]))
    if schema == "meta-bundle-v3":
        from .five_stage import validate_library, PROTOCOL_HASH, LEGACY_PROTOCOL_HASH, PREVIOUS_PROTOCOL_HASH
        validate_library(strict_json(files["principles.json"]))
        if sha256(files["self_update_protocol.md"].encode()) not in {PROTOCOL_HASH, LEGACY_PROTOCOL_HASH, PREVIOUS_PROTOCOL_HASH}:
            raise ValueError("Five-stage responsibilities and protocol file are fixed")
    return schema


def semantic_files(files):
    """JSON layout/key order and trailing prose whitespace are not a G change."""
    return {name: strict_json(value) if name.endswith(".json") else value.rstrip() for name, value in files.items()}


def _read_declared(path: Path, names):
    reject_links(path)
    for name in names:
        reject_links(path / name)
    return {name: (path / name).read_text(encoding="utf-8") for name in sorted(names)}


@dataclass(frozen=True)
class MetaHarnessBundle:
    path: Path
    manifest: dict

    @property
    def hash(self):
        return self.manifest["bundle_hash"]

    @property
    def version(self):
        return self.manifest["version"]

    @property
    def schema_version(self):
        return self.manifest["schema_version"]

    @property
    def content_hash(self):
        return sha256(canonical(semantic_files(self.read_files())))

    @property
    def compatibility_mode(self):
        return "legacy_v1_prompt_assembly" if self.schema_version == "meta-bundle-v1" else "executable_v2"

    def verify(self):
        reject_links(self.path)
        names = SCHEMA_FILES.get(self.schema_version)
        if names is None or self.manifest.get("editable_files") != sorted(names):
            raise ValueError("Unsupported Meta Bundle schema or editable file contract")
        for item in self.path.rglob("*"):
            reject_links(item)
        actual = {p.relative_to(self.path).as_posix() for p in self.path.rglob("*")}
        if actual != names | {"manifest.json"}:
            raise ValueError("Bundle contains an undeclared file, directory or symlink")
        on_disk = strict_json((self.path / "manifest.json").read_text(encoding="utf-8"))
        if on_disk != self.manifest:
            raise ValueError("Immutable Meta Bundle manifest was modified")
        files = _read_declared(self.path, names)
        hashes = {name: sha256((self.path / name).read_bytes()) for name in sorted(names)}
        if hashes != self.manifest["files"]:
            raise ValueError("Immutable Meta Bundle was modified")
        payload = {k: v for k, v in self.manifest.items() if k != "bundle_hash"}
        if sha256(canonical(payload)) != self.hash:
            raise ValueError("Meta Bundle manifest hash mismatch")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Invalid Meta Bundle version")
        if validate_files(files) != self.schema_version:
            raise ValueError("Meta Bundle file/schema mismatch")
        if self.schema_version in {"meta-bundle-v2", "meta-bundle-v3"}:
            if (self.manifest.get("contract_version") != CONTRACT_VERSION or
                    self.manifest.get("entrypoints") != ENTRYPOINTS or
                    self.manifest.get("mechanism_layers") != MECHANISM_LAYERS):
                raise ValueError("Meta Bundle fixed execution contract mismatch")
            if self.manifest.get("content_hash") != sha256(canonical(semantic_files(files))):
                raise ValueError("Meta Bundle semantic content hash mismatch")
        return self

    def read_files(self):
        self.verify()
        return _read_declared(self.path, SCHEMA_FILES[self.schema_version])

    def evolution_policy(self):
        files = self.read_files()
        if self.schema_version == "meta-bundle-v1":
            return None
        from .policies import validate_evolution
        return validate_evolution(strict_json(files["evolution.json"]))

    def execution_spec(self):
        return self.evolution_policy()

    def render(self, evidence: str, response_contract: str, *, preselected=False):
        """Legacy prompt assembly. V2 execution also uses runtime.execute."""
        files = self.read_files()
        context = ContextPolicy.model_validate(strict_json(files["context.json"]), strict=self.schema_version != "meta-bundle-v1")
        workflow = WorkflowPolicy.model_validate(strict_json(files["workflow.json"]), strict=self.schema_version != "meta-bundle-v1")
        limit = context.max_evidence_chars
        selected = evidence
        if not preselected and len(evidence) > limit:
            if context.selection == "head":
                selected = evidence[:limit]
            elif context.selection == "tail":
                selected = evidence[-limit:]
            else:
                selected = evidence[:limit // 2] + "\n[bounded context: middle omitted]\n" + evidence[-limit // 2:]
        sections = {"instructions": files["instructions.md"], "evidence": selected, "response_contract": response_contract}
        prompt = "Workflow checklist:\n" + "\n".join(f"- {x}" for x in workflow.checklist)
        prompt += "\n\n" + "\n\n".join(f"[{key}]\n{sections[key]}" for key in workflow.section_order)
        return prompt, {"bundle_hash": self.hash, "version": self.version,
                        "schema_version": self.schema_version, "compatibility_mode": self.compatibility_mode,
                        "preselected_evidence_preserved": preselected,
                        "context_policy": context.model_dump(), "workflow": workflow.model_dump(),
                        "input_chars": len(evidence), "selected_chars": len(selected),
                        "input_sha256": sha256(evidence.encode()), "rendered_sha256": sha256(prompt.encode()),
                        "load_status": "prepared", "runtime_verified": False}


class MetaHarnessStore:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        reject_links(self.root)

    def active(self):
        reject_links(self.root / "active_harness.json")
        value = strict_json((self.root / "active_harness.json").read_text(encoding="utf-8"))
        directory = value.get("directory")
        if not isinstance(directory, str) or not re.fullmatch(r"v[0-9]{3,}_[a-f0-9]{12}", directory):
            raise ValueError("Invalid active Bundle directory")
        path = self.root / "harness_versions" / directory
        reject_links(path)
        bundle = MetaHarnessBundle(path, strict_json((path / "manifest.json").read_text(encoding="utf-8"))).verify()
        if bundle.hash != value["bundle_hash"] or directory != f"v{bundle.version:03d}_{bundle.hash[:12]}":
            raise ValueError("Active pointer hash/version mismatch")
        return bundle

    def initialize(self, seed_path: Path, source_commit: str | None = None, binary_sha256: str | None = None):
        if (self.root / "active_harness.json").exists():
            return self.active()  # Existing histories are never silently migrated.
        seed_path = Path(seed_path)
        reject_links(seed_path)
        names = {p.name for p in seed_path.iterdir()}
        if names not in (LEGACY_EDITABLE_FILES, EDITABLE_FILES, PRINCIPLE_FILES):
            raise ValueError("Seed must contain exactly the declared v1 or v2 files")
        files = _read_declared(seed_path, names)
        return self._commit(files, None, source_commit, binary_sha256, provenance={"origin": "seed"})

    def initialize_from_bundle(self, path: Path, expected_hash: str):
        """Import an accepted snapshot without rewriting its version or identity."""
        source = MetaHarnessBundle(path, strict_json((path / "manifest.json").read_text(encoding="utf-8"))).verify()
        if source.hash != expected_hash:
            raise ValueError("Initial Meta Bundle does not match the explicitly pinned hash")
        receipt = self.root / "initial_bundle_import.json"
        binding = {"source": str(path.resolve()), "bundle_hash": source.hash, "version": source.version}
        if receipt.exists() and strict_json(receipt.read_text(encoding="utf-8")) != binding:
            raise ValueError("Initial Meta Bundle import binding changed")
        active_exists = (self.root / "active_harness.json").exists()
        if active_exists and not receipt.exists():
            raise ValueError("Existing Meta store was not initialized from this Bundle")
        # Real commits traverse ancestors to reject duplicate request IDs. Verify
        # the entire source chain before importing any snapshot or active pointer.
        ancestry = [source]
        while ancestry[-1].manifest.get("parent_hash") is not None:
            child = ancestry[-1]
            parent_hash = child.manifest["parent_hash"]
            if child.version < 1 or not isinstance(parent_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", parent_hash):
                raise ValueError("Invalid initial Meta Bundle ancestry")
            parent_path = source.path.parent / f"v{child.version - 1:03d}_{parent_hash[:12]}"
            parent = MetaHarnessBundle(parent_path, strict_json((parent_path / "manifest.json").read_text(encoding="utf-8"))).verify()
            if parent.hash != parent_hash or parent.version != child.version - 1:
                raise ValueError("Initial Meta Bundle parent identity mismatch")
            ancestry.append(parent)
        if ancestry[-1].version != 0:
            raise ValueError("Initial Meta Bundle ancestry does not reach version zero")
        for snapshot in reversed(ancestry):
            target = self.root / "harness_versions" / f"v{snapshot.version:03d}_{snapshot.hash[:12]}"
            reject_links(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copytree(snapshot.path, target)
            copied = MetaHarnessBundle(target, strict_json((target / "manifest.json").read_text(encoding="utf-8"))).verify()
            if copied.hash != snapshot.hash:
                raise ValueError("Imported Meta Bundle differs from its source ancestor")
        atomic_json(receipt, binding)
        if not active_exists:
            atomic_json(self.root / "active_harness.json", {
                "directory": f"v{source.version:03d}_{source.hash[:12]}", "bundle_hash": source.hash})
        return self.active()

    def initialize_from_markdown(self, markdown_path: Path, evolution=None, *, source_commit=None, binary_sha256=None):
        """Explicit import to a new store; the legacy Markdown stays untouched."""
        if (self.root / "active_harness.json").exists():
            raise ValueError("Markdown migration requires a new empty store")
        reject_links(markdown_path)
        raw = Path(markdown_path).read_bytes()
        from .policies import default_evolution
        files = {"instructions.md": raw.decode("utf-8"),
                 "context.json": canonical(ContextPolicy().model_dump()).decode(),
                 "workflow.json": canonical(WorkflowPolicy().model_dump()).decode(),
                 "evolution.json": canonical(default_evolution() if evolution is None else evolution).decode()}
        return self._commit(files, None, source_commit, binary_sha256,
                            provenance={"origin": "explicit_migration", "source_schema": "legacy_markdown", "source_hash": sha256(raw)})

    def initialize_from_v1(self, source: MetaHarnessBundle, evolution=None):
        """Import an existing v1 snapshot into a new store without editing it."""
        if (self.root / "active_harness.json").exists():
            raise ValueError("Bundle migration requires a new empty store")
        source.verify()
        if source.schema_version != "meta-bundle-v1":
            raise ValueError("Explicit import requires a v1 Bundle")
        from .policies import default_evolution
        files = source.read_files()
        files["evolution.json"] = canonical(default_evolution() if evolution is None else evolution).decode()
        return self._commit(files, None, source.manifest["source_commit"], source.manifest["binary_sha256"],
                            provenance={"origin": "explicit_migration", "source_schema": source.schema_version,
                                        "source_hash": source.hash})

    def migrate_to_v2(self, expected_hash: str, evolution=None, *, request_id=None, experience_id=None):
        current = self.active()
        if current.hash != expected_hash:
            raise ValueError("Stale Meta Bundle migration")
        if current.schema_version != "meta-bundle-v1":
            raise ValueError("Explicit migration requires a v1 Bundle")
        from .policies import default_evolution
        files = current.read_files()
        files["evolution.json"] = canonical(default_evolution() if evolution is None else evolution).decode()
        return self._commit(files, expected_hash, current.manifest["source_commit"], current.manifest["binary_sha256"],
                            provenance={"origin": "explicit_migration", "source_schema": current.schema_version,
                                        "source_hash": current.hash, "request_id": request_id, "experience_id": experience_id})

    def commit_update(self, expected_hash: str, *, instruction_text: str | None = None,
                      file_updates: dict | None = None, request_id=None, experience_id=None,
                      phase=None, changed_mechanisms=None):
        current = self.active()
        if current.hash != expected_hash:
            raise ValueError("Stale Meta Bundle update")
        changes = dict(file_updates or {})
        if instruction_text is not None:
            if "instructions.md" in changes and changes["instructions.md"] != instruction_text:
                raise ValueError("Conflicting legacy and Bundle instructions")
            changes["instructions.md"] = instruction_text
        if not set(changes) <= SCHEMA_FILES[current.schema_version]:
            raise ValueError("Attempt to change protected Meta runtime/configuration; v1 requires explicit migration")
        files = current.read_files()
        files.update(changes)
        return self._commit(files, expected_hash, current.manifest["source_commit"], current.manifest["binary_sha256"],
                            provenance={"origin": "meta_update" if request_id else "compatibility_call",
                                        "request_id": request_id, "experience_id": experience_id, "phase": phase,
                                        "claimed_changed_mechanisms": [] if changed_mechanisms is None else changed_mechanisms})

    def _receipt_path(self, request_id):
        if request_id is None:
            return None
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 256:
            raise ValueError("Invalid Meta request identity")
        return self.root / "commit_events" / (sha256(request_id.encode()) + ".json")

    def reconcile_update(self, expected_hash: str, *, instruction_text=None, file_updates=None,
                         request_id=None, experience_id=None, phase=None):
        """Read an already committed request without committing or counting it again.

        This bridges publication of the store's event/pointer and publication of
        the outer controller acceptance receipt. Content alone never identifies
        a request: parent, request, experience and phase must also match.
        """
        current = self.active()
        receipt_path = self._receipt_path(request_id)
        if receipt_path is not None:
            reject_links(receipt_path)
        event = (strict_json(receipt_path.read_text(encoding="utf-8"))
                 if receipt_path is not None and receipt_path.is_file() else None)
        if current.hash == expected_hash:
            if event is None:
                if request_id is not None and current.manifest.get("provenance", {}).get("request_id") == request_id:
                    raise ValueError("Committed Meta request has a different parent state")
                return None
            prior = current
            status = "NO_CHANGE"
        else:
            if current.manifest.get("parent_hash") != expected_hash:
                raise ValueError("Active Meta Bundle is unrelated to the pending request")
            if not request_id:
                raise ValueError("Committed Meta child cannot be reconciled without request identity")
            provenance = current.manifest.get("provenance", {})
            if any(provenance.get(key) != value for key, value in {
                "request_id": request_id, "experience_id": experience_id, "phase": phase,
            }.items()):
                raise ValueError("Committed Meta child request/experience/phase differs")
            prior_path = self.root / "harness_versions" / f"v{current.version - 1:03d}_{expected_hash[:12]}"
            reject_links(prior_path)
            prior = MetaHarnessBundle(prior_path, strict_json((prior_path / "manifest.json").read_text(encoding="utf-8"))).verify()
            if prior.hash != expected_hash:
                raise ValueError("Committed Meta child has an invalid parent snapshot")
            status = "CHANGED"
        if event is not None:
            expected = {"schema_version": "meta-commit-event-v1", "status": status,
                        "request_id": request_id, "experience_id": experience_id, "phase": phase,
                        "input_bundle_hash": expected_hash, "output_bundle_hash": current.hash,
                        "content_hash": current.content_hash, "version": current.version}
            if event != expected:
                raise ValueError("Meta commit event does not match the pending request and state")
        changes = dict(file_updates or {})
        if instruction_text is not None:
            if "instructions.md" in changes and changes["instructions.md"] != instruction_text:
                raise ValueError("Conflicting legacy and Bundle instructions")
            changes["instructions.md"] = instruction_text
        if not set(changes) <= SCHEMA_FILES[prior.schema_version]:
            raise ValueError("Recovery candidate attempts to change protected Bundle paths")
        candidate = {**prior.read_files(), **changes}
        validate_files(candidate)
        if semantic_files(candidate) != semantic_files(current.read_files()):
            raise ValueError("Committed Meta content differs from the pending candidate")
        if current.hash != self.active().hash:
            raise ValueError("Active Meta Bundle changed during reconciliation")
        return current

    def _reject_duplicate(self, request_id, existing):
        receipt = self._receipt_path(request_id)
        if receipt is None:
            return
        reject_links(receipt)
        if receipt.exists():
            raise ValueError("Duplicate Meta request commit")
        # Covers a crash after pointer publication but before event publication.
        current = existing
        while current is not None:
            if current.manifest.get("provenance", {}).get("request_id") == request_id:
                raise ValueError("Duplicate Meta request commit")
            parent = current.manifest.get("parent_hash")
            if parent is None:
                break
            path = self.root / "harness_versions" / f"v{current.version - 1:03d}_{parent[:12]}"
            reject_links(path)
            if not path.is_dir():
                raise ValueError("Broken Meta Bundle history")
            current = MetaHarnessBundle(path, strict_json((path / "manifest.json").read_text(encoding="utf-8"))).verify()
            if current.hash != parent:
                raise ValueError("Broken Meta Bundle parent identity")

    def _record_commit(self, provenance, old_hash, bundle, status):
        receipt = self._receipt_path(provenance.get("request_id"))
        if receipt is not None:
            atomic_json(receipt, {"schema_version": "meta-commit-event-v1", "status": status,
                                 "request_id": provenance["request_id"], "experience_id": provenance.get("experience_id"),
                                 "phase": provenance.get("phase"),
                                 "input_bundle_hash": old_hash, "output_bundle_hash": bundle.hash,
                                 "content_hash": bundle.content_hash, "version": bundle.version})

    def _commit(self, files, expected_hash, source_commit, binary_sha256, *, provenance=None):
        from sia.task_meta.file_lock import exclusive_lock
        reject_links(self.root / ".commit.lock")
        with exclusive_lock(self.root / ".commit.lock"):
            return self._commit_locked(files, expected_hash, source_commit, binary_sha256, provenance=provenance or {})

    def _commit_locked(self, files, expected_hash, source_commit, binary_sha256, *, provenance=None):
        provenance = dict(provenance or {})
        schema = validate_files(files)
        self._receipt_path(provenance.get("request_id"))
        experience_id = provenance.get("experience_id")
        if experience_id is not None and (not isinstance(experience_id, str) or not experience_id.strip() or len(experience_id) > 256):
            raise ValueError("Invalid Meta experience identity")
        if provenance.get("phase") not in (None, "meta_self_update", "final_consolidation", "migration"):
            raise ValueError("Invalid Meta commit phase")
        claimed = provenance.get("claimed_changed_mechanisms", [])
        if not isinstance(claimed, list) or any(name not in MECHANISM_LAYERS for name in claimed):
            raise ValueError("Invalid claimed Meta mechanism scope")
        reject_links(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / (".staging_" + uuid.uuid4().hex)
        try:
            existing = self.active() if (self.root / "active_harness.json").exists() else None
            if (existing.hash if existing else None) != expected_hash:
                raise ValueError("Concurrent/stale Meta Bundle commit")
            self._reject_duplicate(provenance.get("request_id"), existing)
            previous_files = existing.read_files() if existing else {}
            if existing and semantic_files(previous_files) == semantic_files(files):
                self._record_commit(provenance, expected_hash, existing, "NO_CHANGE")
                return existing
            changed = sorted(name for name in files if name not in previous_files or
                             semantic_files({name: previous_files[name]}) != semantic_files({name: files[name]}))
            manifest = {"schema_version": schema, "version": existing.version + 1 if existing else 0,
                        "parent_hash": expected_hash, "source_commit": source_commit, "binary_sha256": binary_sha256,
                        "files": {name: sha256(files[name].encode()) for name in sorted(files)},
                        "editable_files": sorted(files),
                        "capabilities": {"instructions": True, "context_selection": True, "workflow_assembly": True,
                                         "runtime_source_edit": False, "runtime_build": False},
                        "provenance": provenance,
                        "runtime_validation": "PENDING_NEXT_OPERATION"}
            if schema in {"meta-bundle-v2", "meta-bundle-v3"}:
                manifest.update(contract_version=CONTRACT_VERSION, entrypoints=ENTRYPOINTS,
                                mechanism_layers=MECHANISM_LAYERS, content_hash=sha256(canonical(semantic_files(files))),
                                provenance=provenance, change_scope=changed)
                manifest["capabilities"].update(evidence_rules=True, experience_rules=True,
                                                conditional_workflows=True, self_update_rules=True)
            if schema == "meta-bundle-v3":
                manifest.update(protocol_sha256=manifest["files"]["self_update_protocol.md"],
                                principles_sha256=manifest["files"]["principles.json"],
                                meta_memory_is_task_artifact=False)
            manifest["bundle_hash"] = sha256(canonical(manifest))
            staging.mkdir()
            for name, content in files.items():
                with (staging / name).open("xb") as handle:
                    handle.write(content.encode())
                    handle.flush()
                    os.fsync(handle.fileno())
            atomic_json(staging / "manifest.json", manifest)
            bundle = MetaHarnessBundle(staging, manifest).verify()
            bundle.render("load probe", "schema probe")
            directory = f"v{manifest['version']:03d}_{manifest['bundle_hash'][:12]}"
            destination = self.root / "harness_versions" / directory
            reject_links(destination)
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                recovered = MetaHarnessBundle(destination, strict_json((destination / "manifest.json").read_text(encoding="utf-8"))).verify()
                if recovered.manifest != manifest:
                    raise ValueError("Orphan Meta version differs from the pending commit")
            else:
                os.replace(staging, destination)
                _fsync_directory(destination.parent)
            atomic_json(self.root / "active_harness.json", {"directory": directory, "bundle_hash": manifest["bundle_hash"]})
            committed = self.active()
            self._record_commit(provenance, expected_hash, committed, "CHANGED")
            return committed
        finally:
            if staging.exists():
                reject_links(staging)
                shutil.rmtree(staging)
