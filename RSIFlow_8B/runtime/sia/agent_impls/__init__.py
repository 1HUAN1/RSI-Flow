"""Agent-implementation registry package.

Importing this package registers all built-in agent impls (claude, openhands,
pydantic-ai) without importing their optional SDKs.
"""

# Import agent-impl modules for their registration side effects.
from sia.agent_impls import claude, openhands, pydantic_ai  # noqa: F401  (registers impls)
from sia.agent_impls.base import available_agent_impls, get_agent_impl, register, run_agent

__all__ = ["available_agent_impls", "get_agent_impl", "register", "run_agent"]
