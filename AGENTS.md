# Repository Guidelines

Daily AI Times v3 is a Claude Agent SDK orchestrator with a provider-neutral compute pool. This file is the shared contract for AI coding agents (Codex, Claude Code, Cursor).

## Project structure

- `src/agent/` — Claude Agent SDK entrypoint (`main.py`, `options.py`, `hooks.py`, `schemas.py`, `subagents.py`).
- `src/tools/` — `@tool`-decorated MCP tools, wrapped by `create_sdk_mcp_server(name="daily_ai")` in `server.py`.
- `src/providers/` — `LLMClient` protocol plus Groq, Cerebras, OpenAI, Gemini adapters. Routing lives in `registry.py`.
- `src/pipeline/` — pure async stages (collect, normalize, dedupe, publish, audio). No LLM calls directly.
- `src/config/` — YAML (`app.yaml`, `providers.yaml`, `sources/*.yaml`) plus loader and prompts.
- `src/frontend/` — static newspaper-style site. v2 outputs land in `src/frontend/api/v2/`.
- `src/legacy/` — v1 orchestrator. Keeps prod alive until cutover. Do not modify casually.
- `.claude/skills/` — Markdown Skills discovered via `setting_sources=["project"]`.
- `tests/` — pytest suite plus `fixtures/articles.jsonl`, `fixtures/llm_replay/`, `fixtures/golden/api/v2/`.

## Build, test, dev

| Command | Purpose |
|---|---|
| `make bootstrap` | Create `.venv` and install `pyproject.toml` deps (editable) |
| `make fixture` | Offline pipeline run with mock provider, writes local `src/frontend/api/v2/` |
| `make test` | pytest suite |
| `make e2e` | Playwright against served fixture site |
| `make mcp` | Boot in-process MCP server on stdio |
| `make publish-v2` | Real pipeline run (needs `ANTHROPIC_API_KEY` plus at least one provider key) |
| `make parity` | Diff v1 vs v2 on fixture corpus |

## Coding conventions

- Python: 4-space indent, snake_case, explicit `src.*` imports, Pydantic v2 models for all stage I/O.
- LLM interactions only in `src/tools/` or `src/providers/`. Pipeline stages must be pure.
- Every LLM response consumed by Python code must validate against a Pydantic schema (via `output_format` or provider client).
- JavaScript: 4-space indent, kebab-case filenames, named exports.
- Config in YAML, never hardcoded in Python.

## Testing

- Unit tests mock LLM calls via `tests/mock_provider.py` (record/replay keyed on `sha256(prompt)`).
- Golden fixture parity: `tests/test_pipeline.py` diffs `src/frontend/api/v2/` against `tests/fixtures/golden/api/v2/`.
- Add `test_*.py` next to the behaviour under test.

## Commit and PR conventions

- Conventional commits: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`.
- Automated content commits stay prefixed with `🤖 AI Pipeline Update:`.
- PRs describe pipeline or user-visible impact, link issues, attach screenshots for frontend changes.

## Security

- Secrets via `.env.local` or GitHub Actions. At minimum `ANTHROPIC_API_KEY` plus one compute provider key.
- Generated content in `src/frontend/api/` and `src/frontend/api/v2/` is build output; treat accordingly.

## For AI agents

- Always use the OpenAI Docs MCP or Claude Code docs MCP when answering SDK or MCP questions; both are registered in `.mcp.json`.
- Prefer Skills for model-invoked workflows, `@tool` for deterministic callable behaviour, subagents for isolated context.
- Do not introduce ad-hoc HTTP calls to LLM providers outside `src/providers/`.
