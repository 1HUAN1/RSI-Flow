"""Explicitly import old G into a NEW v2 store; never mutate a historical run.

python scripts/migrate_meta_bundle.py markdown --source old/harness_v0.md --destination new/meta
python scripts/migrate_meta_bundle.py v1 --source old/meta --destination new/meta
Optional --policy supplies a validated evolution.json instead of the shipped G0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.meta_harness import MetaHarnessStore
from sia.task_meta.meta_harness.bundle import reject_links, strict_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("markdown", "v1"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    reject_links(args.source)
    reject_links(args.destination)
    if args.destination.exists():
        parser.error("Destination must not exist; historical stores are never overwritten")
    source = args.source.resolve()
    destination = args.destination.absolute()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Migration destination must be separate from the historical source")
    policy = None
    if args.policy:
        reject_links(args.policy)
        policy = strict_json(args.policy.read_text(encoding="utf-8"))
    store = MetaHarnessStore(destination)
    if args.format == "markdown":
        bundle = store.initialize_from_markdown(source, policy)
    else:
        bundle = store.initialize_from_v1(MetaHarnessStore(source).active(), policy)
    print(json.dumps({"status": "EXPLICIT_MIGRATION", "source": str(source),
                      "destination": str(destination), "bundle_hash": bundle.hash,
                      "schema_version": bundle.schema_version,
                      "provenance": bundle.manifest["provenance"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
