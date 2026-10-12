"""Backward-compatible shim.

Agent impls moved to the ``rsiflow.agent_impls`` package. This module is kept so
existing imports (``from rsiflow.util import run_agent``) continue to work.
"""

from rsiflow.agent_impls import available_agent_impls, get_agent_impl, run_agent

__all__ = ["available_agent_impls", "get_agent_impl", "run_agent"]
