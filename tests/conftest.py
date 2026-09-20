"""Pytest configuration for the SIA test suite.

Ensures the tests directory is importable so shared helpers (e.g. golden_master)
can be imported as top-level modules regardless of pytest's import mode.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
