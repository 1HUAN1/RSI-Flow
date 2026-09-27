"""Fixed loader for three-file Meta programs. It never chooses a Task component.

Packages are complete, round-pinned copies; skills/context live outside them.
Executable checks run in a child, and failed hooks are returned as observations.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

FILES = ("workflow.py", "planning.py", "memory.py")
PROJECT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


class _PackageSourceLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        # Always read source; edits in the same candidate directory cannot reuse a stale pyc.
        exec(compile(self.path.read_text(), str(self.path), "exec"), module.__dict__)


class _PackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespace, folder):
        self.namespace, self.folder = namespace, folder

    def find_spec(self, fullname, path=None, target=None):
        if fullname.rpartition(".")[0] != self.namespace:
            return None
        filename = fullname.rpartition(".")[2] + ".py"
        if filename in FILES:
            return importlib.util.spec_from_loader(fullname, _PackageSourceLoader(self.folder / filename))
        return None


@contextlib.contextmanager
def load_package(folder):
    """Normal Python sibling imports, with modules alive until the actual hook returns."""
    folder = Path(folder).resolve()
    namespace = "_meta_program_" + uuid.uuid4().hex
    package = types.ModuleType(namespace)
    package.__path__ = [str(folder)]
    package.__package__ = namespace
    package.__spec__ = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
    sys.modules[namespace] = package
    finder = _PackageFinder(namespace, folder)
    sys.meta_path.insert(0, finder)
    try:
        modules = {filename[:-3]: importlib.import_module(namespace + "." + filename[:-3])
                   for filename in FILES}
        yield modules
    finally:
        sys.meta_path.remove(finder)
        for key in list(sys.modules):
            if key == namespace or key.startswith(namespace + "."):
                sys.modules.pop(key, None)


def wiring_probe(folder):
    """Call the same entrypoints used by the live bridge/tool library, with tiny fixtures."""
    calls = []
    with load_package(folder) as modules, tempfile.TemporaryDirectory(prefix="meta-probe-") as temporary:
        ledger = str(Path(temporary) / "skills.jsonl")
        Path(ledger).touch()
        context = {"round": 1, "phase": "route", "evidence_paths": [], "selections": []}
        for module, function, args in (
            ("workflow", "prepare", (context,)), ("workflow", "review", (context,)),
            ("planning", "prepare", (context,)),
            ("planning", "prepare", ({**context, "phase": "meta_review"},)),
            ("memory", "append", (ledger, [{"id": "skill.HARNESS.wiring_probe", "kind": "case", "component": "HARNESS"}])),
            ("memory", "maintain", (ledger, [])),
            ("memory", "retrieve", (ledger, {}, {"max_chars": 4096, "per_category": 1})),
        ):
            value = getattr(modules[module], function)(*args)
            if not isinstance(value, dict):
                raise TypeError(f"{module}.{function} must return a factual dictionary")
            json.dumps(value)
            calls.append(f"{module}.{function}")
    return {"status": "checked", "files": list(FILES), "executed_hooks": calls,
            "checks": "compile, isolated import and real entrypoint execution; no model/API/GPU",
            "capability_improvement_tested": False}


class MetaHarnessRuntime:
    def __init__(self, project, run):
        self.project, self.run = Path(project).resolve(), Path(run).resolve()
        self.root = self.run / "meta_harness"
        self.state_path = self.root / "state.json"
        self.config_path = self.run / "meta/harness_config.json"

    @contextlib.contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".version.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def initialize(self):
        with self.locked():
            if self.state_path.exists():
                return read_json(self.state_path)
            config = read_json(self.config_path) if self.config_path.exists() else {}
            seed = Path(config.get("seed_path") or self.project / "meta_harness/G000")
            if not seed.is_dir():
                seed = PROJECT / "meta_harness/G000"
            destination = self.root / "G000"
            destination.mkdir(exist_ok=True)
            for filename in FILES:
                shutil.copy2(seed / filename, destination / filename)
            state = {"active": "G000", "pending": None, "bindings": {},
                     "mode": config.get("mode", "evolving"), "fixed_skills": config.get("fixed_skills", False),
                     "versions": {"G000": {"version": "G000", "bundle_path": str(destination),
                                             "source": str(seed), "parent": None}}}
            write_json(self.state_path, state)
            return state

    def binding(self, round_number=0):
        self.initialize()
        with self.locked():
            state = read_json(self.state_path)
            key = str(int(round_number))
            if key not in state["bindings"]:
                pending = state.get("pending")
                if pending and int(round_number) >= pending["effective_from_round"]:
                    state["active"] = pending["version"]
                    state["pending"] = None
                state["bindings"][key] = state["active"]
                write_json(self.state_path, state)
            version = state["bindings"][key]
            return {**state["versions"][version], "round": int(round_number)}

    def status(self, round_number=0):
        binding = self.binding(round_number)
        state = read_json(self.state_path)
        return {"status": "loaded", "used": binding, "pending": state.get("pending"),
                "mode": state["mode"], "fixed_skills": state.get("fixed_skills", False),
                "state_path": str(self.state_path)}

    def invoke(self, module, hook, *args, round_number=0):
        binding = self.binding(round_number)
        used, warning = binding, None
        try:
            with load_package(binding["bundle_path"]) as modules:
                value = getattr(modules[module], hook)(*args)
            if not isinstance(value, dict):
                raise TypeError(f"{module}.{hook} must return a dictionary")
            json.dumps(value)
        except Exception as exc:
            warning = f"{type(exc).__name__}: {exc}"
            state = read_json(self.state_path)
            fallback = binding.get("parent") or "G000"
            used = state["versions"][fallback]
            with load_package(used["bundle_path"]) as modules:
                value = getattr(modules[module], hook)(*args)
        trace = {"round": int(round_number), "requested_version": binding["version"],
                 "executed_version": used["version"], "module": module, "hook": hook,
                 "bundle_path": used["bundle_path"], "warning": warning}
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({**trace, "time": time.time(), "output_fields": list(value)}, ensure_ascii=False) + "\n")
        return {**value, "meta_harness_call": trace}

    def phase(self, context):
        number = int(context.get("round", 0))
        return {"workflow": self.invoke("workflow", "prepare", context, round_number=number),
                "planning": self.invoke("planning", "prepare", context, round_number=number)}

    def materialize(self, destination, round_number=0):
        source = self.binding(round_number)
        destination = Path(destination).resolve()
        if destination.exists() and any(destination.iterdir()):
            return {"status": "destination_exists", "candidate_dir": str(destination), "used": source}
        destination.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            shutil.copy2(Path(source["bundle_path"]) / filename, destination / filename)
        return {"status": "materialized", "candidate_dir": str(destination), "used": source,
                "files": list(FILES), "api": {"workflow": ["prepare(context)", "review(context)"],
                    "planning": ["prepare(context)"], "memory": ["retrieve(path, fingerprint, options)",
                    "append(path, entries)", "maintain(path, operations)"]},
                "note": "Edit only evidence-identified behavior; provide all three files. No Meta score gate."}

    def check(self, candidate, timeout=30):
        report = self.root / "checks" / (uuid.uuid4().hex + ".json")
        report.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run([sys.executable, str(PROJECT / "meta_harness_runtime.py"),
                                     "--probe", str(Path(candidate).resolve()), "--result", str(report)],
                                    capture_output=True, text=True, timeout=timeout,
                                    cwd=self.project if self.project.is_dir() else PROJECT)
            receipt = read_json(report) if report.exists() else {
                "status": "validation_failed", "error": result.stderr[-3000:]}
        except Exception as exc:
            receipt = {"status": "validation_failed", "error": f"{type(exc).__name__}: {exc}"}
        receipt.update(check_path=str(report), candidate_dir=str(Path(candidate).resolve()))
        write_json(report, receipt)
        return receipt

    def update(self, *, decision, round_number, candidate=None, reason=None, gaps=None, evidence_refs=None):
        used = self.binding(round_number)
        state = read_json(self.state_path)
        if state["mode"] == "fixed":
            return {"status": "meta_fixed", "used": used, "note": "Comparison mode keeps the program fixed; the experiment continues."}
        if decision not in {"keep", "replace"}:
            return {"status": "needs_decision", "note": "Meta chooses keep or replace; no automatic trigger."}
        check = None
        if decision == "replace":
            if not candidate:
                return {"status": "validation_failed", "error": "Provide a complete candidate folder."}
            check = self.check(candidate)
            if check["status"] != "checked":
                return {**check, "retained_version": used["version"], "continue_same_meta": True}
        with self.locked():
            state = read_json(self.state_path)
            review = {"round": int(round_number), "decision": decision, "used": used,
                      "reason": reason, "gaps": gaps or [], "evidence_refs": evidence_refs or [], "check": check}
            if decision == "replace":
                pending = state.get("pending")
                if pending and pending["effective_from_round"] == int(round_number) + 1:
                    old = Path(state["versions"][pending["version"]]["bundle_path"])
                    if all((old / name).read_bytes() == (Path(candidate) / name).read_bytes() for name in FILES):
                        return {"status": "queued", "used": used, "pending": pending, "reused": True}
                version = f"G{len(state['versions']):03d}"
                destination = self.root / version
                destination.mkdir()
                for filename in FILES:
                    shutil.copy2(Path(candidate) / filename, destination / filename)
                record = {"version": version, "bundle_path": str(destination), "parent": used["version"],
                          "created_after_round": int(round_number), "check_path": check["check_path"]}
                state["versions"][version] = record
                state["pending"] = {**record, "effective_from_round": int(round_number) + 1}
                review["selected_next"] = record
            else:
                # A changed mind before the boundary cancels this round's pending choice.
                pending = state.get("pending")
                if pending and pending["effective_from_round"] == int(round_number) + 1:
                    state["pending"] = None
                review["selected_next"] = used
            review_path = self.root / "reviews" / f"round_{int(round_number)}.json"
            write_json(review_path, review)
            write_json(self.state_path, state)
            return {"status": "queued" if decision == "replace" else "retained", "used": used,
                    "pending": state.get("pending"), "review_path": str(review_path),
                    "capability_gain_required": False}

    def snapshot_reference(self, round_number=0):
        info = self.status(round_number)
        state = read_json(self.state_path)
        pending = info["pending"]
        next_version = state['bindings'].get(str(int(round_number) + 1)) or (
            pending['version'] if pending and pending['effective_from_round'] <= int(round_number) + 1 else info['used']['version'])
        return {"used": info["used"], "selected_next": state["versions"][next_version],
                "mode": state["mode"], "calls_path": str(self.root / "calls.jsonl"),
                "review_path": str(self.root / "reviews" / f"round_{int(round_number)}.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = wiring_probe(args.probe)
    except Exception as exc:
        receipt = {"status": "validation_failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.result, receipt)


if __name__ == "__main__":
    main()
