---
description: Compare the current v2 output against the v1 API for parity
---

Use the `mcp__daily_ai__parity_check` tool to compare `src/frontend/api/latest.json` (v1) against `src/frontend/api/v2/latest.json` (v2).

Report:

- URL overlap count and percentage.
- Articles unique to v1 and unique to v2.
- An overlap percentage below 20% on a real production run is a regression signal; flag it clearly.

If either file is missing, explain which pipeline needs to run first (v1 via `make` in legacy land, v2 via `/run-pipeline`).
