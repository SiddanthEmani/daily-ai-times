"""Lifecycle hooks for the agent loop. Logs tool traffic and enforces invariants."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("daily_ai.agent")


async def pre_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    context: Any,
) -> dict[str, Any] | None:
    log.info("tool:call %s input=%s", tool_name, _truncate(tool_input))
    if tool_name == "mcp__daily_ai__update_source_config":
        return {
            "decision": "block",
            "message": "update_source_config requires explicit human approval; skipping.",
        }
    return None


async def post_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    context: Any,
) -> None:
    is_error = False
    if isinstance(tool_output, dict):
        is_error = bool(tool_output.get("is_error") or tool_output.get("isError"))
    log.info("tool:done %s ok=%s", tool_name, not is_error)


async def on_stop(reason: str, context: Any) -> None:
    log.info("agent:stop reason=%s", reason)


def _truncate(payload: Any, *, limit: int = 200) -> str:
    text = repr(payload)
    return text if len(text) <= limit else text[:limit] + "…"


HOOKS = {
    "PreToolUse": [pre_tool_use],
    "PostToolUse": [post_tool_use],
    "Stop": [on_stop],
}
