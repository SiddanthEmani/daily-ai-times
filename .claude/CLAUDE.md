# Daily AI Times — Project Memory

Claude Agent SDK orchestrates a four-hour news aggregation pipeline. Claude reasons and dispatches; a provider-neutral compute pool (Groq / Cerebras / OpenAI / Gemini) does bulk inference. Pure Python stages do I/O and file writes.

## Canonical layout

- `src/agent/` — Claude Agent SDK entrypoint, options, hooks, Pydantic schemas, subagent definitions.
- `src/tools/` — `@tool` functions wrapped by `create_sdk_mcp_server(name="daily_ai")`.
- `src/providers/` — provider-neutral `LLMClient` protocol + Groq / Cerebras / OpenAI / Gemini adapters with TPM buckets.
- `src/pipeline/` — pure stages (collect, normalize, dedupe, publish, audio). No LLM calls directly.
- `src/config/` — YAML config (`app.yaml`, `providers.yaml`, `sources/*.yaml`) + loader.
- `src/frontend/` — static site. Outputs land in `src/frontend/api/v2/**.json`.
- `src/legacy/` — v1 orchestrator (runs in parallel until cutover).
- `.claude/skills/` — model-invoked Markdown capabilities. Loaded via `setting_sources=["project"]`.
- `tests/` — pytest suite plus fixtures (articles, LLM replay, golden outputs).

## Tools exposed via MCP

All namespaced `mcp__daily_ai__*`:

- `collect_sources`, `list_sources`, `update_source_config` (approval required).
- `score_batch`, `classify_articles`, `rank_category`, `pick_headlines`.
- `publish_v2`, `write_manifest`, `validate_api`, `parity_check`.

## Subagents

- `scorer` — isolated context for scoring 100-article batches via the provider pool.
- `classifier` — parallel per-category classification.

## Commands

- `make bootstrap` — create `.venv`, install deps.
- `make fixture` — offline pipeline run with mock provider.
- `make test` — pytest suite.
- `make publish-v2` — live pipeline, writes `src/frontend/api/v2/`.
- `make parity` — diff v1 vs v2 on fixture corpus.

## Rules for agents working in this repo

- Do not modify `src/legacy/` without an explicit ticket; it keeps prod v1 alive.
- Every LLM response consumed by Python code must go through a Pydantic schema via `output_format={"type": "json_schema", ...}` or a provider client that validates against a Pydantic model.
- New LLM-touching behaviour goes through `src/tools/` (as a `@tool`) or a Skill in `.claude/skills/`, never ad-hoc in pipeline code.
- Provider keys are env-driven: `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`. `ANTHROPIC_API_KEY` drives the Claude agent loop.
- Never commit generated content under `src/frontend/api/v2/` unless it came from a real pipeline run.
