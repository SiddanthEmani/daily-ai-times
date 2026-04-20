"""Build the ClaudeAgentOptions wiring tools, skills, subagents, and hooks."""
from __future__ import annotations

import os
from typing import Any

try:
    from claude_agent_sdk import ClaudeAgentOptions
except ImportError:  # pragma: no cover
    ClaudeAgentOptions = None  # type: ignore[assignment]

from src.agent.hooks import HOOKS
from src.agent.schemas import PIPELINE_RUN_JSON_SCHEMA
from src.agent.subagents import AGENTS
from src.tools.server import ALLOWED_TOOL_NAMES, build_server


SYSTEM_PROMPT = """You are the Daily AI Times v2 orchestrator.

Your job is to run a news aggregation pipeline end-to-end by invoking the
Markdown Skills in .claude/skills/ in this order:
  1. news-collection
  2. content-classification
  3. headline-ranking
  4. publish-verify

If any step fails, surface the error and stop. Never publish partial results.

Delegate heavy scoring or classification work to the `scorer` and `classifier`
subagents when the article pool is large (>60 items).

Return a final JSON payload that conforms to the PipelineRunResult schema:
{
  "run_id": "<iso timestamp>",
  "started_at": "<iso>",
  "finished_at": "<iso>",
  "skills_run": [{"skill": "<name>", "status": "ok|warning|error", "summary": "<one line>", "metrics": {...}}],
  "published_artifacts": ["latest.json", ...],
  "manifest_sha": "<16 hex>",
  "parity_overlap": <float or null>,
  "notes": "<one paragraph>"
}
"""


def build_options(
    *,
    extra_tools: list[str] | None = None,
    read_only: bool = False,
) -> Any:
    if ClaudeAgentOptions is None:
        raise RuntimeError("claude-agent-sdk not installed; `make bootstrap` first")

    server = build_server()
    allowed = list(ALLOWED_TOOL_NAMES)
    if extra_tools:
        allowed.extend(extra_tools)

    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"daily_ai": server},
        allowed_tools=allowed + ["Skill", "Read", "Glob", "Grep"],
        agents=AGENTS,
        setting_sources=["project"],
        hooks=HOOKS,
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        max_turns=int(os.getenv("DAT_MAX_TURNS", "40")),
        permission_mode=("default" if not read_only else "acceptEdits"),
        cwd=os.getcwd(),
    )
