"""Select a Task backbone for the shared skill-evolution implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(backbone: str | None = None, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    if backbone is None:
        parser.add_argument('--backbone', choices=('4b', '8b'), default='4b')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--prepare-data', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    size = backbone or args.backbone
    config = (args.config or ROOT / f'configs/train_{size}.json').resolve()
    if args.prepare_data and '--check' not in remaining and '--help' not in remaining and '-h' not in remaining:
        result = subprocess.run([sys.executable, str(ROOT / 'prepare_600_744.py'),
                                 '--config', str(config)], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return subprocess.run([sys.executable, str(ROOT / 'launch_meta.py'),
                           '--config', str(config), *remaining], cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
