---
name: content-classification
description: Score and classify collected articles into canonical categories. Use after articles have been collected and before ranking. Triggers on "score articles", "classify the batch", "rate quality", or whenever the user wants the bulk tier to prioritize the article pool.
allowed-tools:
  - mcp__daily_ai__score_batch
  - mcp__daily_ai__classify_articles
---

# Content Classification

This skill runs the bulk-tier scoring and category normalization for the current article pool in state.

## When to use

- The `news-collection` skill has already run and left a non-empty article pool.
- The user wants to refresh scores without re-collecting.

## Steps

1. **Score.** Call `score_batch` (no args, or `{"limit": N}` if the user specified a cap). Bulk tier models (Cerebras / Groq / Gemini / OpenAI) fan out across batches.
2. **Classify.** Call `classify_articles` with no arguments. This normalizes aliases (tech→industry, ai→industry, policy→government, …) against `src/config/app.yaml` categories.
3. **Summarize.** Report: total count, mean score, per-category counts, and the three highest-scoring titles.

## Output contract

- Confirm the bulk provider/model actually used (present in `score_batch` response).
- Flag anything unusual — all-zero scores, unknown categories, empty result.

## Things to avoid

- Do not re-collect. If state is empty, defer to the `news-collection` skill.
- Do not rank by category yet; that is `headline-ranking`.
