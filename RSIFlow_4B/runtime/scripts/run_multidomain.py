"""Project entry point; real API/training modes must be selected explicitly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sia.task_meta.pipeline import main

if __name__ == '__main__':
    main()
