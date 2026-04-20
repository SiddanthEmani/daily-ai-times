---
name: headline-ranking
description: Rank classified articles within each category, then pick the daily headlines and widget set. Triggers on "rank headlines", "pick top stories", "generate the front page", or when the user wants to see the final story order before publishing.
allowed-tools:
  - mcp__daily_ai__rank_category
  - mcp__daily_ai__pick_headlines
---

# Headline Ranking

This skill turns scored+classified articles into the ordered sets the frontend consumes: a per-category ranking, a global `latest` feed, and a compact `widget` teaser.

## When to use

- `content-classification` has run and state contains classified articles.
- The user wants the final ordering but has not yet published.

## Steps

1. **Rank each category.** For each category present in state (typically research, industry, government, media, open_source), call `rank_category` with `{"category": "<name>", "limit": 15}`.
2. **Pick headlines.** Call `pick_headlines` with `{"latest_limit": 30, "widget_limit": 8}` unless the user specified different limits.
3. **Report top headlines.** Print the widget titles and note the global top story.

## Output contract

- Per-category counts after ranking.
- Final widget titles in order.
- Total size of the `latest` feed.

## Things to avoid

- Do not publish. That belongs to `publish-verify`.
- Do not invent categories. Use only what is present in state; if a category has no articles, skip silently.
