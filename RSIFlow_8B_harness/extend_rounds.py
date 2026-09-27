"""Extend frozen evolution batches without resampling prior rounds or validation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_protocol import allocate, catalog, materialize, native_index, scan
from evolution_protocol import assert_disjoint, freeze, make_manifest, read, training_quotas


def extend_release(config, evaluation, output, rounds):
    original = Path(config["frozen_data_dir"]).resolve()
    output = Path(output).resolve()
    existing = [read(path) for path in sorted(original.glob("B*/manifest.json"),
                                             key=lambda p: int(p.parent.name[1:]))]
    previous = len(existing)
    if rounds <= previous:
        raise ValueError("Target round count must exceed the existing frozen rounds")
    if output.exists():
        raise FileExistsError("Use a new release directory; existing releases stay unchanged")
    revised = {**config, "rounds": rounds,
               "allocated_tasks": rounds * config["training_tasks_per_pass"],
               "frozen_data_dir": str(output), "data_release": output.name,
               "cross_round_tool_environment_reuse": True}
    training_quotas(revised)
    print(json.dumps({"status": "scanning", "preserved_rounds": previous,
                      "additional_rounds": rounds - previous}), flush=True)
    items, audit = scan(catalog(config, evaluation), config["grouping"])
    excluded = {field: {row[field] for batch in existing for row in batch["tasks"]}
                for field in ("task_id", "group_id", "content_hash")}
    eligible = [row for row in items if row["kind"] != "train" or
                not any(row[field] in excluded[field] for field in excluded
                        if field != "group_id" or row["source"] != "envscaler")]
    allocation_config = {**config, "rounds": rounds - previous}
    allocated, _ = allocate(eligible, allocation_config, {"coverage": {}, "events": []})
    additional = []
    for batch in allocated[:rounds - previous]:
        number = batch["round_id"] + previous
        additional.append(make_manifest(
            "train_evolution", number,
            [{**row, "round_id": number} for row in batch["tasks"]],
            split_seed=config["split_seed"], feedback_scope="all_B_r"))
    # Only task allocation consistency is checked; no model routing/acceptance gate.
    # All 51 EnvScaler environments occur in B1-B3. Preserve task-level disjointness;
    # a distinct scenario may reuse an environment definition, never an old task.
    assert_task_disjoint(existing + additional)
    assert_disjoint([{**batch, "tasks": [row for row in batch["tasks"]
                                       if row["source"] != "envscaler"]}
                     for batch in existing + additional])
    output.mkdir(parents=True)
    for batch in existing:
        (output / f'B{batch["round_id"]}').symlink_to(
            original / f'B{batch["round_id"]}', target_is_directory=True)
    packed = []
    for batch in additional:
        directory = output / f'B{batch["round_id"]}'
        result = materialize(batch, directory)
        freeze(directory / "manifest.json", result)
        packed.append(result)
    published = existing + packed
    freeze(output / "protocol.json", revised)
    index_config = {**revised, "retrieval_index": str(original / "search.sqlite")}
    native_index(output, published, index_config, link_retrieval=True)
    result = {
        "status": "prepared_not_executed", "preserved_data_dir": str(original),
        "rounds": rounds, "allocated_tasks": sum(len(b["tasks"]) for b in published),
        "unchanged_rounds": [b["round_id"] for b in existing],
        "new_rounds": [b["round_id"] for b in packed],
        "cross_round_tool_environment_reuse": True,
        "manifests": [{"round_id": b["round_id"], "tasks": len(b["tasks"]),
                       "by_domain": dict(Counter(t["domain"] for t in b["tasks"])),
                       "manifest_hash": b["manifest_hash"]} for b in published],
        "source_audit": audit,
    }
    freeze(output / "preparation_audit.json", result)
    print(json.dumps(result["manifests"], ensure_ascii=False), flush=True)
    return result


def assert_task_disjoint(batches):
    for field in ("task_id", "content_hash"):
        values = [row[field] for batch in batches for row in batch["tasks"]]
        if len(values) != len(set(values)):
            raise ValueError(f"Repeated task across batches: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", required=True, type=int)
    args = parser.parse_args()
    result = extend_release(read(args.config), read(args.evaluation_config), args.output, args.rounds)
    print(json.dumps({key: value for key, value in result.items() if key != "source_audit"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
