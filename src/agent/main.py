"""Daily AI Times v2 entrypoint.

- `python -m src.agent.main` runs the full pipeline via the Claude Agent SDK.
- Set `DAT_FIXTURE=tests/fixtures/articles.jsonl DAT_MOCK_PROVIDER=1` for an
  offline fixture run that bypasses the Claude loop and exercises tools directly
  against the pipeline (used by `make fixture` and CI).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=os.getenv("DAT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("daily_ai.main")


async def _fixture_run() -> int:
    """Offline deterministic run. Uses heuristic scoring and fixture articles."""
    from src.pipeline.dedupe import dedupe
    from src.pipeline.models import RawArticle
    from src.pipeline.normalize import normalize
    from src.tools.publishing import publish_v2, validate_api
    from src.tools.ranking import pick_headlines, rank_category
    from src.tools.scoring import classify_articles, score_batch
    from src.tools.state import get_state, reset_state

    fixture = Path(os.environ["DAT_FIXTURE"])
    log.info("fixture run: loading %s", fixture)
    raw = [
        RawArticle(**json.loads(line))
        for line in fixture.read_text().splitlines()
        if line.strip()
    ]
    articles = dedupe(normalize(raw))
    reset_state()
    get_state().articles = articles
    log.info("fixture: %d deduped articles", len(articles))

    def _handler(tool):
        return getattr(tool, "handler", None) or getattr(tool, "func", None) or tool

    await _handler(score_batch)({})
    await _handler(classify_articles)({})
    categories = {s.article.category for s in get_state().classified}
    for category in categories:
        await _handler(rank_category)({"category": category, "limit": 15})
    await _handler(pick_headlines)({"latest_limit": 30, "widget_limit": 8})
    await _handler(publish_v2)({})
    verify = await _handler(validate_api)({})
    if verify.get("is_error"):
        log.error("fixture verify failed: %s", verify["content"][0]["text"])
        return 1
    log.info("fixture run ok")
    print(verify["content"][0]["text"])
    return 0


async def _agent_run() -> int:
    """Full Claude Agent SDK run. Requires ANTHROPIC_API_KEY."""
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
    except ImportError:
        log.error("claude-agent-sdk not installed. Run `make bootstrap`.")
        return 1

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY missing; aborting live run")
        return 1

    from src.agent.options import build_options

    options = build_options()
    run_id = datetime.now(timezone.utc).isoformat()
    prompt = (
        f"Run the /run-pipeline command. Tag this run as {run_id}. "
        "Return the final JSON payload on the last line with no prose before or after it."
    )

    final_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in getattr(message, "content", []):
                if isinstance(block, TextBlock):
                    final_text = block.text
                elif isinstance(block, ToolUseBlock):
                    log.debug("claude:tool_use %s", block.name)
        elif isinstance(message, ResultMessage):
            if message.subtype == "success":
                log.info("agent run ok (%.2fs)", getattr(message, "duration_seconds", 0.0))
            else:
                log.error("agent run failed: %s", message.subtype)

    print(final_text or "(no final text)")
    return 0


def cli() -> None:
    if os.getenv("DAT_FIXTURE"):
        sys.exit(asyncio.run(_fixture_run()))
    sys.exit(asyncio.run(_agent_run()))


if __name__ == "__main__":
    cli()
