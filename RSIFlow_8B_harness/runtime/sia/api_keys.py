"""Model-name → provider API key resolution."""

from __future__ import annotations

import os


def resolve_api_key(model_name: str) -> str | None:
    """Return the provider-specific API key for ``model_name`` from the environment.

    Precedence (matches the original run_agent_openhands logic):
      - claude / anthropic → ANTHROPIC_API_KEY
      - gemini / google    → GOOGLE_API_KEY or GEMINI_API_KEY
      - gpt / openai       → OPENAI_API_KEY
      - anything else      → LLM_API_KEY

    Returns None when the matched variable is unset (the caller may then fall back).
    """
    name = model_name.lower()
    if "claude" in name or "anthropic" in name:
        return os.getenv("ANTHROPIC_API_KEY")
    if "gemini" in name or "google" in name:
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if "gpt" in name or "openai" in name:
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("LLM_API_KEY")
