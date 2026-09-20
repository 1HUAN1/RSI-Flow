"""Backward-compatible shim.

Agent impls moved to the ``sia.agent_impls`` package. This module is kept so
existing imports (``from sia.util import run_agent``) continue to work.
"""

from sia.agent_impls import available_agent_impls, get_agent_impl, run_agent

__all__ = ["available_agent_impls", "get_agent_impl", "run_agent"]
