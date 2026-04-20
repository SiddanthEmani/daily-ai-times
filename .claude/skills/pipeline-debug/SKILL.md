---
name: pipeline-debug
description: Diagnose why a pipeline run produced unexpected output (empty feed, missing category, wrong ordering, parity regression). Use when the user reports an issue with a recent run or when CI parity fails. Triggers on "debug the pipeline", "why is <category> empty", "explain the parity failure", "something looks wrong".
allowed-tools:
  - mcp__daily_ai__list_sources
  - mcp__daily_ai__validate_api
  - mcp__daily_ai__parity_check
  - Read
  - Glob
  - Grep
---

# Pipeline Debug

This skill helps investigate issues in a completed run without re-running the whole pipeline.

## When to use

- The user reports a missing or wrong result in `src/frontend/api/v2/`.
- `make parity` or the CI parity job fails.
- `validate_api` returns errors.

## Steps

1. **Read the manifest.** Open `src/frontend/api/v2/manifest.json` and cross-reference the declared counts against actual files.
2. **Validate.** Call `validate_api`. The exact violations from step 1 usually show up here.
3. **Parity.** Call `parity_check`. If overlap is unexpectedly low, inspect the most recent pipeline log (look under `.cache/` or `logs/` if present, otherwise the GitHub Actions artifact).
4. **Check sources.** If a whole category is empty, call `list_sources` scoped to that category to confirm feeds exist; then check recent collection logs for source failures.
5. **Cross-reference fixtures.** If the run was a fixture run (`DAT_FIXTURE` env var set), read `tests/fixtures/articles.jsonl` to confirm the seed data matches expectations.

## Output contract

- A short list of concrete findings, each tied to a specific artifact or config file.
- A recommended next action (re-collect, fix a source, update a skill prompt, adjust `providers.yaml`).

## Things to avoid

- Do not rewrite v2 output files from this skill; fix the root cause and let a fresh run regenerate them.
- Do not disable sources from inside debug; that is a user decision requiring approval.
