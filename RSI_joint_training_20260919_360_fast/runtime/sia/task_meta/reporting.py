"""Final-only official evaluation, fixed denominators and transparent tables.

This module has no Meta client and returns no training observation. Search QA
executes the pinned official scripts. Code/Tool native evaluator command adapters
require a separately verified worker-isolation runner before execution.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

from .meta_harness.bundle import atomic_json, canonical, sha256
from .storage import artifact_manifest, checkpoint_manifest, digest

BENCHMARKS = ("BFCL-v3", "ACEBench", "LiveCodeBench", "HumanEval+", "MBPP+", "HotpotQA-dev", "2Wiki-dev")
CODE_BENCHMARKS = {"LiveCodeBench", "HumanEval+", "MBPP+"}
BENCHMARK_IDS = dict(zip(("bfcl_v3", "acebench", "livecodebench", "humaneval_plus", "mbpp_plus", "hotpotqa_dev", "2wiki_dev"), BENCHMARKS, strict=True))


class ReportBlocked(RuntimeError):
    pass


def task_fingerprint(state):
    value = asdict(state) if is_dataclass(state) else dict(state)
    harness = Path(value["harness_path"])
    checkpoint = value.get("checkpoint_path")
    artifacts = value.get("artifacts") or {}
    files = {"harness": {"path": str(harness.resolve()), "sha256": digest(harness)},
             "artifacts": artifact_manifest(artifacts.get("directory")),
             "checkpoint": checkpoint_manifest(checkpoint) if checkpoint else []}
    return {"state": value, "files": files, "state_hash": sha256(canonical({"state": value, "files": files}))}


def freeze_best_on_dev(candidates: list[dict], states: dict, *, probe_manifest_hash: str,
                       selection_metric: str, mode: str, destination: Path, protocol_hash: str):
    """Only predeclared fixed-dev scores select; ties use earliest generation."""
    if mode not in {"min", "max"} or not candidates:
        raise ValueError("Selection requires candidates and a registered direction")
    for record in candidates:
        if record.get("split") != "search_dev" or record.get("probe_manifest_hash") != probe_manifest_hash:
            raise ValueError("Final selection can use only the fixed search_dev probe")
        if record.get("status") != "completed" or record.get("metric") != selection_metric:
            raise ValueError("Incomplete or incompatible development evaluation")
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Selection score must be finite")
    selected = sorted(candidates, key=lambda r: (r["score"] if mode == "min" else -r["score"], r["generation"]))[0]
    snapshot = task_fingerprint(states[selected["state_id"]])
    record = {"schema_version": "frozen-report-task-v1", "split": "report_eval", "protocol_hash": protocol_hash,
              "selection": {"metric": selection_metric, "mode": mode, "probe_manifest_hash": probe_manifest_hash,
                            "selected": selected, "tie_policy": "earliest_generation"},
              "last_evaluated": max(candidates, key=lambda r: r["generation"]), **snapshot}
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("A frozen report Task cannot be reselected using later results")
    else:
        atomic_json(destination, record)
    return record


@dataclass(frozen=True)
class OfficialEvaluatorSpec:
    benchmark: str
    repository: str
    commit: str
    entrypoint: str
    entrypoint_sha256: str
    python_executable: str
    data_path: str
    data_sha256: str
    task_ids_path: str
    task_ids_sha256: str
    protocol: str
    alias_path: str | None = None
    alias_sha256: str | None = None
    release_version: str | None = None
    subset: str | None = None
    timeout_seconds: int = 3600
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "benchmark", BENCHMARK_IDS.get(self.benchmark, self.benchmark))

    def validate(self):
        if self.benchmark not in BENCHMARKS or not re.fullmatch(r"[a-f0-9]{40}", self.commit):
            raise ValueError("Unregistered benchmark or unpinned official evaluator")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 86400:
            raise ValueError("Invalid official evaluator time budget")
        for filename, expected in ((self.entrypoint, self.entrypoint_sha256), (self.data_path, self.data_sha256),
                                   (self.task_ids_path, self.task_ids_sha256)):
            if not Path(filename).is_file() or digest(Path(filename)) != expected:
                raise ReportBlocked(f"Pinned evaluator/data identity unavailable: {Path(filename).name}")
        if (self.benchmark == "2Wiki-dev" and self.protocol != 'official_2wiki_original_v1'
                and (not self.alias_path or not Path(self.alias_path).is_file() or digest(Path(self.alias_path)) != self.alias_sha256)):
            raise ReportBlocked("2Wiki official aliases are required and must be pinned")
        ids = json.loads(Path(self.task_ids_path).read_text())
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
            raise ValueError("Official benchmark requires a nonempty unique ordered ID manifest")
        if self.benchmark in {"HotpotQA-dev", "2Wiki-dev"}:
            native = json.loads(Path(self.data_path).read_text())
            if not isinstance(native, list) or len(native) != len(ids) or {row.get("_id") for row in native} != set(ids):
                raise ReportBlocked("Official QA gold IDs do not match the frozen denominator")
        return ids


def export_predictions(benchmark: str, rows: list[dict], expected_ids: list[str], directory: Path, frozen_hash: str):
    """Exactly one final answer per task; incomplete runs never silently drop IDs."""
    if benchmark not in BENCHMARKS:
        raise ValueError("Unregistered final benchmark")
    seen = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id in seen or task_id not in expected_ids:
            raise ValueError("Duplicate or unexpected final prediction")
        if row.get("state_hash") != frozen_hash or row.get("split") != "report_eval":
            raise ValueError("Prediction is not from the frozen final-evaluation Task")
        if row.get("infrastructure_error"):
            raise ReportBlocked("Infrastructure failure leaves the fixed-denominator report incomplete")
        if row.get("final_submission_count") != 1 or not isinstance(row.get("final_answer"), str):
            raise ValueError("System-level pass@1 requires one final string submission per task")
        seen[task_id] = row
    if set(seen) != set(expected_ids):
        raise ReportBlocked(f"partial predictions: {len(seen)}/{len(expected_ids)} completed")
    directory.mkdir(parents=True, exist_ok=True)
    ordered = [seen[task_id] for task_id in expected_ids]
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if benchmark in {"HumanEval+", "MBPP+"}:
        target = directory / "official_samples.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write(json.dumps({"task_id": row["task_id"], "solution": row["final_answer"]}) + "\n")
    elif benchmark == "LiveCodeBench":
        target = directory / "official_samples.json"
        atomic_json(target, [{"question_id": row["task_id"], "code_list": [row["final_answer"]]} for row in ordered])
    elif benchmark in {"HotpotQA-dev", "2Wiki-dev"}:
        target = directory / "official_predictions.json"
        payload = {"answer": {r["task_id"]: r["final_answer"] for r in ordered},
                   "sp": {r["task_id"]: r.get("supporting_facts", []) for r in ordered}}
        if benchmark == "2Wiki-dev":
            payload["evidence"] = {r["task_id"]: r.get("evidence", []) for r in ordered}
        atomic_json(target, payload)
    else:
        target = directory / "official_predictions.jsonl"
        # Tool evaluators need their official subset/output handler; never guess a
        # conversation or AST format from a generic final string.
        raise ReportBlocked(f"{benchmark} official subset/output protocol adapter is not yet validated")
    return target


def official_command(spec: OfficialEvaluatorSpec, predictions: Path, directory: Path):
    python = str(Path(spec.python_executable).resolve())
    if spec.benchmark in {"HotpotQA-dev", "2Wiki-dev"}:
        command = [python, spec.entrypoint, str(predictions), spec.data_path]
        if spec.benchmark == "2Wiki-dev" and spec.protocol != 'official_2wiki_original_v1':
            command.append(spec.alias_path)
        return command
    if spec.benchmark in {"HumanEval+", "MBPP+"}:
        if not spec.release_version:
            raise ReportBlocked("EvalPlus dataset version must be explicit")
        return [python, "-m", "evalplus.evaluate", "--dataset", "humaneval" if spec.benchmark == "HumanEval+" else "mbpp",
                "--samples", str(predictions), "--parallel", "1", "--version", spec.release_version,
                "--output-file", str(directory / "official_results.json"), "--test-details"]
    if spec.benchmark == "LiveCodeBench":
        if not spec.release_version:
            raise ReportBlocked("LiveCodeBench release/date coverage must be explicit")
        return [python, "-m", "lcb_runner.runner.custom_evaluator", "--scenario", "codegeneration",
                "--custom_output_file", str(predictions), "--release_version", spec.release_version,
                "--num_process_evaluate", "1"]
    raise ReportBlocked("Official Tool evaluator mode/subset has not been validated")


def parse_official_metrics(spec, directory, stdout):
    if spec.benchmark in {"HotpotQA-dev", "2Wiki-dev"}:
        start = stdout.rfind("{")
        if start < 0:
            raise ReportBlocked("Official QA evaluator did not produce its metric dictionary")
        raw = ast.literal_eval(stdout[start:].strip()) if spec.benchmark == "HotpotQA-dev" else json.loads(stdout[start:])
        scale = 1.0 if spec.benchmark == "HotpotQA-dev" else 100.0
        metrics = {"EM": raw["em"] / scale, "F1": raw["f1"] / scale}
    elif spec.benchmark in {"HumanEval+", "MBPP+"}:
        raw = json.loads((directory / "official_results.json").read_text())
        task_results = raw["eval"]
        passes = []
        for values in task_results.values():
            if not isinstance(values, list) or len(values) != 1:
                raise ReportBlocked("EvalPlus results are not one final sample per task")
            passes.append(values[0]["base_status"] == "pass" and values[0]["plus_status"] == "pass")
        metrics = {"system_pass@1": sum(passes) / len(passes)}
    else:
        path = directory / "official_samples_codegeneration_output_eval.json"
        raw = json.loads(path.read_text())
        metrics = {"system_pass@1": raw[0]["pass@1"]}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
        raise ReportBlocked("Official metrics missing, invalid, or using an unregistered scale")
    return metrics, raw


def evaluate_official(spec: OfficialEvaluatorSpec, rows: list[dict], frozen_record: dict, output_dir: Path,
                      *, isolated_code_runner=None):
    """Execute real official evaluators; no shell interpolation or API environment."""
    ids = spec.validate()
    if "task_state" in frozen_record:
        from .durable import load_task, task_hash
        current_hash = task_hash(load_task(frozen_record["task_state"]))
        if frozen_record.get("status") != "frozen_for_report_eval":
            raise ReportBlocked("The controller has not frozen this Task for final evaluation")
        if frozen_record.get('checkpoint_files') is not None and checkpoint_manifest(frozen_record['task_state']['checkpoint_path']) != frozen_record['checkpoint_files']:
            raise ReportBlocked('Frozen checkpoint files were modified after selection')
    else:
        current_hash = task_fingerprint(frozen_record["state"])["state_hash"]
    if current_hash != frozen_record["state_hash"]:
        raise ReportBlocked("Frozen Task was modified after development-only selection")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        previous = json.loads((output_dir / "result.json").read_text())
        if previous.get("status") == "completed":
            raise ValueError("Final evaluation already completed; do not rerun to select favorable scores")
    base = {"benchmark": spec.benchmark, "state_hash": frozen_record["state_hash"], "split": "report_eval",
            "expected": len(ids), "completed": len(rows), "evaluator": asdict(spec), "metrics": None,
            "metric_unit": "fraction", "feedback_to_meta": False}
    try:
        predictions = export_predictions(spec.benchmark, rows, ids, output_dir, frozen_record["state_hash"])
        command = official_command(spec, predictions, output_dir)
        native_ipc = spec.benchmark in CODE_BENCHMARKS and isolated_code_runner is None
        atomic_json(output_dir / "invocation.json", {"command": None if native_ipc else command,
                    "python_entrypoint": ("sia.task_meta.lcb_isolated.evaluate_lcb_isolated" if spec.benchmark == 'LiveCodeBench' else "sia.task_meta.evalplus_isolated.evaluate_evalplus_isolated") if native_ipc else None, "cwd": str(output_dir),
                    "predictions_sha256": digest(predictions), "candidate_credential_environment": "none",
                    "trusted_evaluator_execution": "in_process" if native_ipc else "subprocess_clean_environment"})
        if native_ipc:
            from .evalplus_isolated import evaluate_evalplus_isolated
            from .lcb_isolated import evaluate_lcb_isolated
            try:
                runner = evaluate_lcb_isolated if spec.benchmark == 'LiveCodeBench' else evaluate_evalplus_isolated
                result = runner(spec, predictions, output_dir)
            except (RuntimeError, ImportError, OSError, subprocess.SubprocessError) as exc:
                raise ReportBlocked(f"Official EvalPlus candidate worker blocked: {exc}") from exc
        elif spec.benchmark in CODE_BENCHMARKS:
            if isolated_code_runner is None:
                raise ReportBlocked("Official code evaluator requires verified isolation at each untrusted candidate worker; no root subprocess fallback")
            # The only accepted extension is a trusted object supplied by the
            # controller, never a Task/Meta-generated command or boolean claim.
            proof = isolated_code_runner.validate(spec)
            if proof.get("candidate_worker_isolated") is not True or not proof.get("probe_evidence"):
                raise ReportBlocked("Official candidate isolation has no verified worker probe")
            result = isolated_code_runner.run(command, cwd=output_dir, spec=spec)
        else:
            clean_env = {"PATH": str(Path(spec.python_executable).parent), "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1"}
            with (output_dir / "stdout.txt").open("wb") as stdout, (output_dir / "stderr.txt").open("wb") as stderr:
                process = subprocess.run(command, cwd=output_dir, env=clean_env, stdout=stdout, stderr=stderr,
                                         timeout=spec.timeout_seconds, check=False)
            result = {"returncode": process.returncode}
        if result["returncode"] != 0:
            raise ReportBlocked("Official evaluator infrastructure/process failure")
        stdout = (output_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace")
        metrics, raw = parse_official_metrics(spec, output_dir, stdout)
        if spec.benchmark in {"HumanEval+", "MBPP+"} and set(raw["eval"]) != set(ids):
            raise ReportBlocked("Official result IDs do not match the fixed denominator")
        atomic_json(output_dir / "official_raw.json", raw)
        record = {**base, "status": "completed", "metrics": metrics, "predictions_sha256": digest(predictions)}
    except (ReportBlocked, subprocess.TimeoutExpired, OSError, ValueError, KeyError, ZeroDivisionError) as exc:
        record = {**base, "status": "partial" if len(rows) < len(ids) else "blocked", "reason": str(exc)}
    atomic_json(output_dir / "result.json", record)
    return record


def write_report_tables(output_dir: Path, results: list[dict], experiment_metadata: dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(row["benchmark"] not in BENCHMARKS for row in results) or len({row["benchmark"] for row in results}) != len(results):
        raise ValueError("Report contains an unregistered or duplicate benchmark")
    indexed = {row["benchmark"]: row for row in results}
    csv_rows, latex_values = [], []
    for name in BENCHMARKS:
        row = indexed.get(name, {"benchmark": name, "status": "pending", "reason": "Not executed", "metrics": None, "completed": 0, "expected": None})
        metrics = row.get("metrics") if row.get("status") == "completed" else None
        if not metrics:
            metrics = {"score": None}
        for metric, value in metrics.items():
            csv_rows.append({"benchmark": name, "metric": metric, "value": "N/A" if value is None else value,
                             "unit": "fraction", "status": row["status"], "completed": row.get("completed"),
                             "expected": row.get("expected"), "reason": row.get("reason", "")})
        values = list(metrics.values())
        latex_values.append("N/A" if any(value is None for value in values) else " / ".join(f"{value * 100:.2f}" for value in values))
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    method = str(experiment_metadata.get("method", "RSI")).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    (output_dir / "table_row.tex").write_text(method + " & " + " & ".join(latex_values) + r" \\" + "\n", encoding="utf-8")
    atomic_json(output_dir / "report_manifest.json", {"benchmarks": list(BENCHMARKS), "experiment": experiment_metadata,
                "results": results, "overall": None, "overall_reason": "No overall aggregation formula is registered",
                "latex_unit": "percent", "csv_unit": "fraction", "meta_feedback_permitted": False})
    return output_dir / "results.csv", output_dir / "table_row.tex"


def run_report_eval(specs_path, frozen_task_path, predictions_dir, output_dir, experiment_metadata=None):
    """Final-only controller entrypoint; inputs are completed frozen predictions.

No prediction generation, Meta model, or search-dev callback is reachable here.
Missing evaluator configurations produce explicit N/A rows for all seven IDs.
"""
    specs_path, predictions_dir, output_dir = Path(specs_path), Path(predictions_dir), Path(output_dir)
    frozen = json.loads(Path(frozen_task_path).read_text(encoding="utf-8"))
    config = json.loads(specs_path.read_text(encoding="utf-8"))
    configured = config.get("evaluators", config)
    if not isinstance(configured, dict) or set(configured) - set(BENCHMARK_IDS):
        raise ValueError("Evaluator configuration must use the seven registered benchmark IDs")
    results = []
    for identifier, name in BENCHMARK_IDS.items():
        pending = {"benchmark": name, "status": "pending", "metrics": None, "completed": 0, "expected": None}
        if identifier not in configured:
            blocker = config.get('blocked', {}).get(identifier, {})
            results.append({**pending, 'status': 'blocked' if blocker else 'pending',
                            "reason": blocker.get('reason', "Official evaluator configuration is not provided")})
            continue
        predictions = predictions_dir / (identifier + ".jsonl")
        try:
            spec = OfficialEvaluatorSpec(**{**configured[identifier], "benchmark": identifier})
            pending['expected'] = len(spec.validate())
            if not predictions.is_file():
                results.append({**pending, "reason": "Frozen Task predictions are not available"})
                continue
            rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
            results.append(evaluate_official(spec, rows, frozen, output_dir / identifier))
        except (ReportBlocked, ValueError, OSError, TypeError) as exc:
            results.append({**pending, "status": "blocked", "reason": str(exc)})
    write_report_tables(output_dir, results, {**(experiment_metadata or {}), "state_hash": frozen["state_hash"],
                        "protocol_hash": frozen["protocol_hash"], "evaluator_config_sha256": digest(specs_path)})
    return results
