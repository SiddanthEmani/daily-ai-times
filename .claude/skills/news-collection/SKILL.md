---
name: news-collection
description: Run the collection phase of the Daily AI Times v2 pipeline. Use when you need to fetch the latest articles from configured RSS / API sources and get a clean, deduplicated working set ready for scoring. Triggers whenever the user says "run collection", "fetch news", "refresh the article pool", or starts a full pipeline run.
allowed-tools:
  - mcp__daily_ai__list_sources
  - mcp__daily_ai__collect_sources
---

# News Collection

This skill owns the collection phase of the v2 pipeline: polling every configured source, normalizing, deduplicating, and leaving a clean article pool in agent state.

## When to use

- A new pipeline run is starting.
- The user asks to refresh or top-up articles before ranking.
- A downstream step reports "no articles in state".

## Steps

1. **Inspect sources (optional).** If the user mentions a specific category (research, industry, government, media, open_source), call `list_sources` first to confirm which feeds will be polled. Otherwise skip.
2. **Collect.** Call `collect_sources`. If the user asked to limit to a category, pass `{"category": "<name>"}`; otherwise call with no arguments.
3. **Read the summary.** Confirm `deduped_count > 0`. If it is zero, treat this as a hard failure and report which category was requested so the user can decide whether to broaden.
4. **Stop here.** Do not score, classify, or publish from this skill. Those belong to other skills.

## Output contract

Return a short status sentence to the user with:

- Number of sources polled.
- Raw vs deduplicated article counts.
- Two or three sample titles to show freshness.

## Things to avoid

- Do not write to disk. Publishing belongs to the `publish-verify` skill.
- Do not call `update_source_config`; that requires explicit permission approval from the user.
