# Provider limits (reference)

Source of truth for TPM numbers is `src/config/providers.yaml`. This document
captures human-facing context and links.

## Groq

Consult your Groq dashboard: https://console.groq.com/dashboard/limits

v2 uses:

- `llama-3.1-8b-instant` (bulk)
- `gemma2-9b-it` (bulk)
- `llama3-8b-8192` (bulk)
- `meta-llama/llama-4-scout-17b-16e-instruct` (deep)
- `llama-3.3-70b-versatile` (deep)

## Cerebras

Reference rate table:

| Model | Ctx | RPM | TPM |
|-------|-----|-----|-----|
| `llama3.1-8b` | 8,192 | 30 | 60,000 |
| `llama-3.3-70b` | 65,536 | 30 | 64,000 |
| `qwen-3-32b` | 65,536 | 30 | 64,000 |
| `qwen-3-235b-a22b-instruct-2507` | 64,000 | 30 | 60,000 |

v2 currently uses `llama3.1-8b` (bulk) and `llama-3.3-70b` (deep).

## OpenAI

Consult your org's rate limit dashboard. v2 uses:

- `gpt-4o-mini` (bulk) — 200k TPM typical tier 3.
- `gpt-4o` (deep) — 30k TPM typical tier 3.

## Gemini

Via Google AI Studio / Vertex AI. v2 uses:

- `gemini-2.5-flash` (bulk) — 250k TPM free tier.
- `gemini-2.5-pro` (deep) — 32k TPM free tier.

## Changing limits

Edit `src/config/providers.yaml`. The `TPMBucket` in `src/providers/base.py`
enforces a sliding 60 s window per model.
