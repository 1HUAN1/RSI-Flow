"""Run RSIFlow skill evolution with the Qwen3-8B Task backbone."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'RSIFlow'))
from launcher import main

if __name__ == '__main__':
    raise SystemExit(main('8b'))
