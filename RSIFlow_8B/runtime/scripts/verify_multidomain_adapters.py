#!/usr/bin/env python3
"""CPU-only integration against real prepared data; never creates Task training rows.

The Code fixture is the first stored source solution, explicitly test_override.
No Task model, Meta model, GPU training or final benchmark evaluation is invoked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.data import TaskRecord, audit_code_tool_leakage, content_hash, jsonl_rows, sha256_file
from sia.task_meta.environments import EnvScalerAdapter, SearchQAAdapter, TACOAdapter, audit_taco_formats
from sia.task_meta.retrieval import FrozenSearchIndex
from sia.task_meta.sandbox import LinuxSandbox, SandboxLimits


def _first(path: Path, domain: str, source: str) -> TaskRecord:
    _, _, row = next(jsonl_rows(path))
    prompt = row[{"tool_use": "task", "code": "problem", "searchqa": "question"}[domain]]
    return TaskRecord(f"infrastructure_fixture:{source}:first_row", domain, source, "evolve_train", prompt,
                      row, sha256_file(path), content_hash(prompt))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/RSI_iclr2027/dataset"))
    parser.add_argument("--search-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envscaler-utils", type=Path)
    parser.add_argument("--envscaler-commit", default="96ae8b02dc0187c911b8e2101e7bb6904271597b")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--max-code-cases", type=int, default=128,
                        help="Fail explicitly before running if the fixed first fixture exceeds this budget; never truncate tests")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a fresh infrastructure evidence directory")
    args.output.mkdir(parents=True)
    root = args.data_root
    source_manifest = args.source_manifest or root / "manifests/dataset_manifest.json"
    official_utils = args.envscaler_utils or root / "evaluators/envscaler/EnvScaler/interact_with_env/envscaler_env/utils/env_util.py"
    results = {"schema_version": 1, "decision_source": "test_override", "development_only": True,
               "eligible_for_training": False, "is_formal_benchmark": False, "model_calls": 0,
               "fixtures": {}, "failures": {}, "started_unix": time.time()}

    def save():
        (args.output / "integration_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        audit = audit_code_tool_leakage(source_manifest, args.output / "independent_data_audit.json")
        results["data_audit_status"] = audit["status"]
        if audit["status"] == "failed_overlap":
            results["failures"]["data_audit"] = "Observed exact overlap; inspect independent_data_audit.json"
    except Exception as exc:
        results["failures"]["data_audit"] = f"{type(exc).__name__}: {exc}"
    save()
    print("Independent data audit recorded", flush=True)
    try:
        formats = audit_taco_formats([root / "train/code/deepcoder_taco/train.jsonl", root / "train/code/deepcoder_taco/val.jsonl"])
        (args.output / "taco_format_audit.json").write_text(json.dumps(formats, indent=2) + "\n", encoding="utf-8")
        results["taco_format_status"] = formats["status"]
        if formats["unsupported"]:
            results["failures"]["taco_formats"] = f"{len(formats['unsupported'])} tasks contain unsupported cases"
    except Exception as exc:
        results["failures"]["taco_formats"] = f"{type(exc).__name__}: {exc}"
    save()
    print("All TACO case schemas audited", flush=True)
    sandbox = LinuxSandbox(limits=SandboxLimits(wall_seconds=10, cpu_seconds=5, memory_bytes=1024**3))
    env = None
    try:
        record = _first(root / "train/tool_use/envscaler/train/scenarios.jsonl", "tool_use", "envscaler")
        env = EnvScalerAdapter([root / "train/tool_use/envscaler/train/environments.jsonl"], official_utils,
                              runtime_commit=args.envscaler_commit, sandbox=sandbox,
                              official_utils_sha256=sha256_file(official_utils))
        initial = env.reset(record, "cpu_fixture_env_0", 42)
        choices = [tool for tool in env.tools if not tool["function"]["parameters"].get("required")]
        choices.sort(key=lambda tool: (not tool["function"]["name"].startswith(("get_", "list_", "query_", "show_")), tool["function"]["name"]))
        if choices:
            action = {"name": choices[0]["function"]["name"], "arguments": {}}
        elif 'USR001' in record.prompt and any(t['function']['name'] == 'get_user_by_id' for t in env.tools):
            # Reviewed fixed first fixture: the user ID comes from its public request.
            action = {'name': 'get_user_by_id', 'arguments': {'user_id': 'USR001'}}
        else:
            raise RuntimeError("Fixed environment has no reviewed integration action")
        observation = env.step(**action)
        verdict = env.evaluate("")
        results["fixtures"]["tool_use"] = {"task_id": record.task_id, "source_hash": record.source_hash,
            "initial_public_observation": initial, "actual_tool_action": action, "actual_tool_observation": observation,
            "official_verification": asdict(verdict), "task_success_is_not_required_for_getter_fixture": True}
        if observation.get("runtime_error") or verdict.infrastructure_error:
            raise RuntimeError("Real environment operation or official feedback failed")
    except Exception as exc:
        results["failures"]["tool_use"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env:
            env.close()
    save()
    print("Real EnvScaler tool and official feedback recorded", flush=True)
    try:
        record = _first(root / "train/code/deepcoder_taco/train.jsonl", "code", "deepcoder_taco")
        tests = record.payload["tests"]
        tests = json.loads(tests) if isinstance(tests, str) else tests
        if len(tests["inputs"]) > args.max_code_cases:
            raise RuntimeError(f"Fixed fixture requires {len(tests['inputs'])} cases, above declared {args.max_code_cases}; no tests were truncated")
        solutions = record.payload["solutions"]
        if not solutions or not isinstance(solutions[0], str):
            raise RuntimeError("No first trusted source solution available")
        adapter = TACOAdapter(sandbox)
        adapter.reset(record, "cpu_fixture_code_0", 42)
        verdict = adapter.evaluate(solutions[0])
        results["fixtures"]["code"] = {"task_id": record.task_id, "source_hash": record.source_hash,
            "candidate_source": "first_dataset_solution_integration_fixture_only", "decision_source": "test_override",
            "candidate_sha256": __import__("hashlib").sha256(solutions[0].encode()).hexdigest(),
            "verification": asdict(verdict)}
        if verdict.infrastructure_error or not verdict.verification.get("full_verifier") or not verdict.verification.get("success"):
            raise RuntimeError("Fixed first source solution did not pass the complete declared verifier; inspect actual outcomes")
    except Exception as exc:
        results["failures"]["code"] = f"{type(exc).__name__}: {exc}"
    save()
    print("Real TACO first-source-solution fixture recorded", flush=True)
    index = None
    try:
        record = _first(root / "train/searchqa/hotpotqa/train.jsonl", "searchqa", "hotpotqa")
        sidecar = json.loads(args.search_index.with_suffix(args.search_index.suffix + ".manifest.json").read_text(encoding="utf-8"))
        index = FrozenSearchIndex(args.search_index, expected_sha256=sidecar["sha256"])
        adapter = SearchQAAdapter(index)
        initial = adapter.reset(record, "cpu_fixture_search_0", 42)
        observed = adapter.step("search", {"query": record.prompt})
        verdict = adapter.evaluate("")
        results["fixtures"]["searchqa"] = {"task_id": record.task_id, "initial_public_observation": initial,
            "actual_query": record.prompt, "actual_retrieval": observed, "empty_answer_verification": asdict(verdict),
            "corpus_manifest": index.manifest}
        if not observed["results"] or verdict.infrastructure_error:
            raise RuntimeError("Real retrieval returned no evidence or trusted answer verifier failed")
    except Exception as exc:
        results["failures"]["searchqa"] = f"{type(exc).__name__}: {exc}"
    finally:
        if index:
            index.close()
    results["finished_unix"] = time.time()
    results["status"] = "CPU_INTEGRATION_PASSED" if not results["failures"] else "CPU_INTEGRATION_FAILED"
    save()
    print(json.dumps({"status": results["status"], "failures": results["failures"], "evidence": str(args.output)}, ensure_ascii=False))
    return 0 if not results["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
