#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
HARNESS_ROOT = PROJECT_ROOT / "harness_factory"
LEGACY_FILES = ("toolhop_round1_report.md", "toolhop_round1_evolution_prompt.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic round1 ToolHop metrics.json and report.md files "
            "for harness1..7 under harness_factory/."
        )
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--harness-root", type=Path, default=HARNESS_ROOT)
    parser.add_argument("--metrics-filename", default="metrics.json")
    parser.add_argument("--report-filename", default="report.md")
    parser.add_argument(
        "--keep-legacy-files",
        action="store_true",
        help="Do not remove legacy toolhop_round1_report.md and toolhop_round1_evolution_prompt.md files.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def find_result_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("toolhop_*_results.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No ToolHop result file found under {run_dir}")
    return matches[0]


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def extract_framework_name(result_file: Path) -> str:
    name = result_file.name
    prefix = "toolhop_flash_searcher_flash_searcher_"
    suffix = "_local_qwen3-aevolve_closed_results.jsonl"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def extract_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("agent_trajectory", "trajectory"):
        value = row.get(key)
        if isinstance(value, list):
            return [step for step in value if isinstance(step, dict)]
    return []


def extract_tool_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for step in extract_steps(row):
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name"):
                names.append(str(call["name"]))
    return names


def extract_obs_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for step in extract_steps(row):
        obs = step.get("obs")
        if obs:
            parts.append(str(obs))
    return "\n".join(parts)


def parse_structure(description_text: str) -> dict[str, str]:
    structure = {"planning": "", "action": "", "memory": ""}
    for line in description_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Planning:"):
            structure["planning"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- Execution:"):
            structure["action"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- Memory:"):
            structure["memory"] = stripped.split(":", 1)[1].strip()
    return structure


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round4(value: float) -> float:
    return round(value, 4)


def round2(value: float) -> float:
    return round(value, 2)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_metrics(rows: list[dict[str, Any]], harness_name: str, framework_name: str, result_file: Path) -> dict[str, Any]:
    prompt_tokens = [float((row.get("metrics") or {}).get("prompt_tokens", 0) or 0) for row in rows]
    completion_tokens = [float((row.get("metrics") or {}).get("completion_tokens", 0) or 0) for row in rows]
    total_tokens = [float((row.get("metrics") or {}).get("total_tokens", 0) or 0) for row in rows]
    runtime_sec = [float((row.get("metrics") or {}).get("elapsed_time", 0) or 0) for row in rows]
    actions = [float(row.get("toolhop_action_count", 0) or 0) for row in rows]
    tool_calls = [float(row.get("tool_call_count", 0) or 0) for row in rows]
    answer_correct = [float(row.get("answer_correct", 0) or 0) for row in rows]
    has_valid_answer = [float(row.get("has_valid_answer", 0) or 0) for row in rows]
    path_score = [float(row.get("path_score", 0) or 0) for row in rows]

    return {
        "num_tasks": len(rows),
        "answer_correct": round4(safe_mean(answer_correct)),
        "has_valid_answer": round4(safe_mean(has_valid_answer)),
        "path_score": round4(safe_mean(path_score)),
        "average_actions": round4(safe_mean(actions)),
        "average_tool_calls": round4(safe_mean(tool_calls)),
        "prompt_tokens_total": int(sum(prompt_tokens)),
        "completion_tokens_total": int(sum(completion_tokens)),
        "tokens_total": int(sum(total_tokens)),
        "prompt_tokens_avg": round2(safe_mean(prompt_tokens)),
        "completion_tokens_avg": round2(safe_mean(completion_tokens)),
        "tokens_avg": round2(safe_mean(total_tokens)),
        "runtime_total_min": round2(sum(runtime_sec) / 60.0),
        "runtime_avg_sec": round2(safe_mean(runtime_sec)),
        "harness_name": harness_name,
        "framework_name": framework_name,
        "source_result_file": str(result_file),
    }


def compute_type_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("answer_type") or "unknown")].append(row)
    breakdown: dict[str, dict[str, float | int]] = {}
    for answer_type, items in sorted(grouped.items()):
        breakdown[answer_type] = {
            "count": len(items),
            "accuracy": round4(safe_mean([float(item.get("answer_correct", 0) or 0) for item in items])),
            "path_score": round4(safe_mean([float(item.get("path_score", 0) or 0) for item in items])),
        }
    return breakdown


def compute_failure_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if int(row.get("answer_correct", 0)) != 1]
    successes = [row for row in rows if int(row.get("answer_correct", 0)) == 1]
    all_tools = Counter()
    first_tools = Counter()
    failure_tools = Counter()
    error_markers = Counter()
    answer_type_failures = Counter()
    short_action_failures = 0
    long_loop_failures = 0
    high_path_gap_failures = 0
    parallel_failures = 0
    worker_failures = 0

    for row in rows:
        tools = extract_tool_names(row)
        all_tools.update(tools)
        if tools:
            first_tools[tools[0]] += 1

    for row in failures:
        tools = extract_tool_names(row)
        failure_tools.update(tools)
        answer_type_failures[str(row.get("answer_type") or "unknown")] += 1
        obs = extract_obs_text(row)
        if "Error for tool call" in obs:
            error_markers["tool_call_error"] += 1
        if "missing 1 required positional argument" in obs:
            error_markers["missing_required_argument"] += 1
        if "You should only use this tool with a correct input" in obs:
            error_markers["schema_mismatch_hint"] += 1
        if float(row.get("toolhop_action_count", 0) or 0) <= 1.2:
            short_action_failures += 1
        if float(row.get("tool_call_count", 0) or 0) >= 8.0:
            long_loop_failures += 1
        if float(row.get("path_score", 0) or 0) >= 0.5:
            high_path_gap_failures += 1
        if any(name.startswith("parallel_agent_") or name == "expert_parallel" for name in tools):
            parallel_failures += 1
        if any(name.startswith("worker_") for name in tools):
            worker_failures += 1

    return {
        "top_tools": all_tools.most_common(10),
        "top_first_tools": first_tools.most_common(8),
        "top_failure_tools": failure_tools.most_common(10),
        "error_markers": dict(error_markers),
        "answer_type_failures": dict(answer_type_failures),
        "failure_count": len(failures),
        "success_count": len(successes),
        "short_action_failures": short_action_failures,
        "long_loop_failures": long_loop_failures,
        "high_path_gap_failures": high_path_gap_failures,
        "parallel_failures": parallel_failures,
        "worker_failures": worker_failures,
        "avg_failed_tool_calls": round2(safe_mean([float(row.get("tool_call_count", 0) or 0) for row in failures])),
        "avg_success_tool_calls": round2(safe_mean([float(row.get("tool_call_count", 0) or 0) for row in successes])),
        "avg_failed_actions": round2(safe_mean([float(row.get("toolhop_action_count", 0) or 0) for row in failures])),
        "avg_success_actions": round2(safe_mean([float(row.get("toolhop_action_count", 0) or 0) for row in successes])),
    }


def build_overall_assessment(metrics: dict[str, Any], type_breakdown: dict[str, dict[str, float | int]], signals: dict[str, Any]) -> str:
    acc = metrics["answer_correct"]
    path = metrics["path_score"]
    avg_actions = metrics["average_actions"]
    token_avg = metrics["tokens_avg"]
    framework_name = metrics["framework_name"]

    if acc >= 0.48 and token_avg <= 45000:
        opening = "Overall, this harness balances improved quality with controlled cost."
    elif acc >= 0.45 and token_avg > 60000:
        opening = "Overall, quality improves slightly but cost rises substantially; the tradeoff is unfavorable."
    elif acc < 0.35 and avg_actions <= 1.5:
        opening = "Overall, this harness keeps costs low but achieves limited gains, with insufficient reasoning depth."
    else:
        opening = "Overall, this harness performs moderately, with useful features and clear structural bottlenecks."

    stronger_types = [name for name, stats in type_breakdown.items() if float(stats["accuracy"]) >= max(acc, 0.45)]
    weaker_types = [name for name, stats in type_breakdown.items() if float(stats["accuracy"]) <= min(acc - 0.15, 0.15)]

    if stronger_types:
        fit_text = f"It better suits tasks such as {', '.join(stronger_types[:3])}, with clear evidence chains and little postprocessing."
    else:
        fit_text = "It better suits short entity chains followed by a simple final transformation."

    if weaker_types:
        weak_text = f"It currently struggles with tasks such as {', '.join(weaker_types[:3])}, requiring consistent strings, times, or output formats."
    else:
        weak_text = "It remains fragile under strict formatting, complex string processing, or time conversion."

    if signals["short_action_failures"] >= max(8, int(signals["failure_count"] * 0.45)):
        close_text = (
            f"Behaviorally, {framework_name} averages only {avg_actions:.2f} actions; many failures suggest submission before completing the hop chain."
        )
    else:
        close_text = (
            f"Behaviorally, path score ({path:.4f}) exceeds exact accuracy ({acc:.4f}), suggesting partially correct intermediate reasoning but unreliable final synthesis and submission."
        )

    return " ".join([opening, fit_text, weak_text, close_text])


def build_failure_patterns(metrics: dict[str, Any], type_breakdown: dict[str, dict[str, float | int]], signals: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    if metrics["path_score"] - metrics["answer_correct"] >= 0.12:
        bullets.append("Many failures retain partially correct hop chains but fail at final synthesis or submission; answer canonicalization may be weaker than retrieval.")
    if signals["short_action_failures"] >= max(8, int(signals["failure_count"] * 0.45)):
        bullets.append("Many failures have low action depth, suggesting early termination without enough verification or repair turns.")
    if signals["avg_failed_tool_calls"] >= signals["avg_success_tool_calls"] + 1.0:
        bullets.append("Other failures involve excessive ineffective exploration: failed samples use more tool calls than successful ones, suggesting weak convergence under uncertainty.")
    if signals["error_markers"]:
        rendered = ", ".join(f"{name}={count}" for name, count in signals["error_markers"].items())
        bullets.append(f"Tool-contract errors also recur: observed markers include {rendered}, suggesting unreliable argument names, required fields, or schema interpretation.")
    weakest = sorted(type_breakdown.items(), key=lambda item: float(item[1]["accuracy"]))[:2]
    if weakest:
        rendered = ", ".join(f"{name}({float(stats['accuracy']):.2f})" for name, stats in weakest)
        bullets.append(f"The weakest task types are usually {rendered}, which rely more on string processing, time conversion, or final-answer formatting.")
    if signals["parallel_failures"] >= max(8, int(signals["failure_count"] * 0.35)):
        bullets.append("Parallel branches appear in many failures, suggesting premature dispatch before dependencies are ready and incorrect aggregation by the coordinator.")
    if signals["worker_failures"] >= max(8, int(signals["failure_count"] * 0.35)):
        bullets.append("Worker failures are also common; unclear subtask boundaries may force workers to guess missing upstream entities.")
    if not bullets:
        bullets.append("No single failure mode dominates. Several smaller issues likely reduce accuracy; incremental revision may be appropriate.")
    return bullets[:5]


def build_module_diagnosis(structure: dict[str, str], metrics: dict[str, Any], signals: dict[str, Any]) -> dict[str, dict[str, str]]:
    path_gap = metrics["path_score"] - metrics["answer_correct"]
    diagnosis: dict[str, dict[str, str]] = {}

    planning_help = structure["planning"] or "The harness has an explicit planning layer."
    if path_gap >= 0.1:
        planning_hurt = "Planning often decomposes intermediate paths but fails to constrain the final answer, leaving a path-correct / final-wrong gap."
    elif metrics["average_actions"] <= 1.5:
        planning_hurt = "Planning is present, but execution is too shallow to carry out the plan step by step."
    else:
        planning_hurt = "Planning provides an initial framework without consistently guiding subsequent execution."

    action_help = structure["action"] or "The harness has an explicit action layer."
    if signals["short_action_failures"] >= max(8, int(signals["failure_count"] * 0.45)):
        action_hurt = "The action layer often terminates early, submitting after one pass without verification or repair."
    elif signals["avg_failed_tool_calls"] >= signals["avg_success_tool_calls"] + 1.0:
        action_hurt = "Under uncertainty, the action layer explores low-value alternatives with excessive calls and weak stopping rules."
    else:
        action_hurt = "The action layer handles basic execution but needs stronger evidence checks across multihop dependencies."

    memory_help = structure["memory"] or "The harness includes a memory layer."
    if signals["parallel_failures"] > 0 or signals["worker_failures"] > 0:
        memory_hurt = "Memory may support coordinator narration more than grounded tool contracts or entity caching, amplifying errors during parallel or delegated execution."
    elif path_gap >= 0.1:
        memory_hurt = "Memory may aid decomposition without reliably improving final submission, suggesting weak operational constraints."
    else:
        memory_hurt = "Memory gains are limited and have not clearly changed the main failure modes."

    diagnosis["Planning"] = {"help": planning_help, "hurt": planning_hurt}
    diagnosis["Action"] = {"help": action_help, "hurt": action_hurt}
    diagnosis["Memory"] = {"help": memory_help, "hurt": memory_hurt}
    return diagnosis


def render_metrics_section(metrics: dict[str, Any]) -> list[str]:
    return [
        f"- Evaluated tasks: {metrics['num_tasks']}",
        f"- Exact accuracy: {pct(metrics['answer_correct'])}",
        f"- Valid answer rate: {pct(metrics['has_valid_answer'])}",
        f"- Average path score: {metrics['path_score']:.4f}",
        f"- Average actions: {metrics['average_actions']:.2f}",
        f"- Average tool calls: {metrics['average_tool_calls']:.2f}",
        f"- Prompt / completion / total tokens: {metrics['prompt_tokens_total']} / {metrics['completion_tokens_total']} / {metrics['tokens_total']}",
        f"- Average prompt / completion / total tokens: {metrics['prompt_tokens_avg']:.2f} / {metrics['completion_tokens_avg']:.2f} / {metrics['tokens_avg']:.2f}",
        f"- Total runtime: {metrics['runtime_total_min']:.2f} min",
        f"- Average runtime per task: {metrics['runtime_avg_sec']:.2f} sec",
    ]


def build_report(harness_name: str, structure: dict[str, str], metrics: dict[str, Any], type_breakdown: dict[str, dict[str, float | int]], signals: dict[str, Any]) -> str:
    overall = build_overall_assessment(metrics, type_breakdown, signals)
    failure_patterns = build_failure_patterns(metrics, type_breakdown, signals)
    module_diagnosis = build_module_diagnosis(structure, metrics, signals)

    lines = [
        f"# {harness_name} Analysis",
        "",
        "## Structure",
        f"- Planning: {structure['planning'] or 'No explicit planning summary found.'}",
        f"- Action: {structure['action'] or 'No explicit action summary found.'}",
        f"- Memory: {structure['memory'] or 'No explicit memory summary found.'}",
        "",
        "## Aggregate Metrics",
        *render_metrics_section(metrics),
        "",
        "## Overall Assessment",
        overall,
        "",
        "## Failure Pattern Analysis",
    ]
    for bullet in failure_patterns:
        lines.append(f"- {bullet}")
    lines.extend(["", "## Module-level Diagnosis"])
    for module_name in ("Planning", "Action", "Memory"):
        lines.append(f"### {module_name}")
        lines.append(f"- What Helps: {module_diagnosis[module_name]['help']}")
        lines.append(f"- What Hurts: {module_diagnosis[module_name]['hurt']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def remove_legacy_files(harness_dir: Path) -> None:
    for filename in LEGACY_FILES:
        path = harness_dir / filename
        if path.exists():
            path.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze_harness(harness_index: int, args: argparse.Namespace) -> None:
    harness_name = f"harness{harness_index}"
    run_dir = args.output_root / f"toolhop_round1_{harness_name}"
    harness_dir = args.harness_root / harness_name

    result_file = find_result_file(run_dir)
    rows = load_jsonl(result_file)
    framework_name = extract_framework_name(result_file)
    description = read_text_if_exists(harness_dir / "Description.md")
    structure = parse_structure(description)
    metrics = build_metrics(rows, harness_name, framework_name, result_file)
    type_breakdown = compute_type_breakdown(rows)
    signals = compute_failure_signals(rows)
    report = build_report(harness_name, structure, metrics, type_breakdown, signals)

    if not args.keep_legacy_files:
        remove_legacy_files(harness_dir)

    write_json(harness_dir / args.metrics_filename, metrics)
    (harness_dir / args.report_filename).write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for harness_index in range(1, 8):
        analyze_harness(harness_index, args)


if __name__ == "__main__":
    main()
