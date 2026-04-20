# CLAUDE.md

See [`.claude/CLAUDE.md`](./.claude/CLAUDE.md) for authoritative project memory.

## Quick orientation

- v2 pipeline (Claude Agent SDK) lives in `src/{agent,tools,providers,pipeline}/`.
- v1 pipeline (Groq-only) lives under `src/legacy/` and runs via `.github/workflows/pipeline.yml` until cutover.
- Outputs dual-publish: v1 → `src/frontend/api/`, v2 → `src/frontend/api/v2/`. Frontend toggles via `?v2=1`.

## Commands

| Command | Purpose |
|---|---|
| `make bootstrap` | Create `.venv` and install deps |
| `make fixture` | Offline pipeline run against `tests/fixtures/` with mock provider |
| `make test` | pytest suite |
| `make publish-v2` | Live v2 pipeline run |
| `make parity` | Diff v1 vs v2 on fixture corpus |
| `make dev` | Serve `src/frontend/` on :8000 |

## Secrets

`ANTHROPIC_API_KEY` drives the Claude agent loop. Bulk compute providers: `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`. Any subset can be enabled; registry skips providers without keys.
