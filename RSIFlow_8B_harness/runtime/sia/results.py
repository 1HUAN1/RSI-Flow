"""Result dataclasses replacing positional tuple returns.

Internally the orchestrator builds these for clarity; at the call boundary it
still returns ``.as_tuple()`` to preserve the existing wire contract that tests
and callers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetAgentResult:
    """Outcome of running a target agent generation."""

    success: bool
    stdout: str
    stderr: str
    error_msg: str

    def as_tuple(self) -> tuple[bool, str, str, str]:
        return (self.success, self.stdout, self.stderr, self.error_msg)


@dataclass
class FeedbackContext:
    """The two text blocks the feedback prompt is built from."""

    execution_status: str
    execution_section: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.execution_status, self.execution_section)
