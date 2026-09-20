"""Claude Code SDK agent impl."""

from __future__ import annotations

from datetime import datetime

from sia.agent_impls.base import register
from sia.logging_setup import get_logger

logger = get_logger(__name__)


async def run_agent_claude(model_name, max_turns, prompt, agent_working_directory, provider=None):
    """Run agent using Claude Code SDK

    The ``provider`` argument is accepted for a uniform agent-impl signature but ignored:
    the Claude Code SDK authenticates against Anthropic natively (ANTHROPIC_API_KEY).

    Note: Claude Code automatically saves trajectories to ~/.claude/projects/
    """
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    logger.info(f"Starting agent execution with {model_name} model (max turns: {max_turns})")
    logger.debug("=" * 80)
    logger.debug(f"Working directory: {agent_working_directory}")
    logger.debug("=" * 80)

    turn_count = 0
    start_time = datetime.now()

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=agent_working_directory,
                allowed_tools=["Bash", "Read", "Write", "Edit", "Glob"],
                permission_mode="bypassPermissions",
                max_turns=max_turns,
                model=model_name,
            ),
        ):
            logged_content = False

            if hasattr(message, "content") and message.content:
                for block in message.content:
                    # Log agent text responses
                    if hasattr(block, "text") and block.text:
                        if not logged_content:
                            turn_count += 1
                            logger.debug(f"\n{'─' * 80}")
                            logger.debug(f"TURN {turn_count}: Agent Response")
                            logger.debug(f"{'─' * 80}")
                            logged_content = True
                        logger.debug(f"{block.text}")

                    # Log tool calls
                    elif hasattr(block, "name"):
                        if not logged_content:
                            turn_count += 1
                            logger.debug(f"\n{'─' * 80}")
                            logger.debug(f"TURN {turn_count}: Tool Execution")
                            logger.debug(f"{'─' * 80}")
                            logged_content = True

                        logger.debug(f"🔧 Tool: {block.name}")
                        if hasattr(block, "input") and block.input:
                            # Pretty print tool input
                            import json

                            try:
                                input_str = json.dumps(block.input, indent=2)
                                logger.debug(f"   Input: {input_str}")
                            except (TypeError, ValueError):
                                logger.debug(f"   Input: {block.input}")

                    # Log tool results
                    elif hasattr(block, "type") and block.type == "tool_result":
                        if hasattr(block, "content"):
                            result = block.content if isinstance(block.content, str) else str(block.content)
                            # Truncate very long outputs
                            if len(result) > 500:
                                result = result[:500] + f"\n... (truncated, {len(result)} total chars)"
                            logger.debug(f"   ✓ Result: {result}")

            # Log final result
            if isinstance(message, ResultMessage):
                elapsed_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"\n{'=' * 80}")
                logger.debug(f"Final result: {message.result}")
                logger.debug(f"{'=' * 80}")
                logger.info(f"Execution complete: {turn_count} turns in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {e!s}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


register("claude", run_agent_claude)
