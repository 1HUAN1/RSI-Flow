#!/usr/bin/env python3
"""Offline native catalog generation from pinned source and public metadata."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.meta_backends.catalog import build_catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--codex-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build_catalog(args.metadata, args.codex_source, args.output), indent=2))
