# GROQ API Rate Limits Reference

Current limits for the account backing `GROQ_API_KEY`. Every model the pipeline
references must appear in this list — a model that is not here has been
decommissioned, and calls against it fail silently, causing the pipeline to
reject every article that agent was assigned.

## Chat Completions

| Model | RPM | Requests/day | TPM | Tokens/day |
|---|---|---|---|---|
| `allam-2-7b` | 30 | 7K | 6K | 500K |
| `groq/compound` | 30 | 250 | 70K | No limit |
| `groq/compound-mini` | 30 | 250 | 70K | No limit |
| `meta-llama/llama-prompt-guard-2-22m` | 30 | 14.4K | 15K | 500K |
| `meta-llama/llama-prompt-guard-2-86m` | 30 | 14.4K | 15K | 500K |
| `openai/gpt-oss-120b` | 30 | 1K | 8K | 200K |
| `openai/gpt-oss-20b` | 30 | 1K | 8K | 200K |
| `openai/gpt-oss-safeguard-20b` | 30 | 1K | 8K | 200K |
| `qwen/qwen3.6-27b` | 30 | 1K | 8K | 200K |

### Which of these the pipeline can actually use

- **`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.6-27b`** — general
  chat models with JSON mode. These carry the bulk and deep intelligence swarms.
- **`openai/gpt-oss-safeguard-20b`** — safety classification, not general
  scoring. Not used.
- **`meta-llama/llama-prompt-guard-2-*`** — prompt-injection classifiers, not
  chat models. Cannot produce scoring JSON.
- **`allam-2-7b`** — Arabic-centric. Large daily budget (500K) but a quality
  risk for English AI news scoring. Held in reserve.
- **`groq/compound` / `compound-mini`** — agentic systems with built-in tool
  use. Unlimited daily tokens would suit bulk scoring, but JSON-mode support is
  unverified. Not used.

## Daily token budget

Every usable model is capped at **200K tokens/day**. That is the binding
constraint on how often the pipeline can run.

Measured cost: ~43 input tokens per article, a ~460-token system prompt per
batch of 12, and ~45 output tokens per article. With ~630 articles split three
ways, that is **~27K tokens per bulk agent per run**. The two agents that also
serve the deep intelligence swarm add **~12K per run** each.

| Runs/day | Cron | gpt-oss-20b | qwen3.6-27b | gpt-oss-120b | |
|---|---|---|---|---|---|
| 6 | `0 */4 * * *` | 160K | 230K | 230K | over cap |
| 4 | `0 */6 * * *` | 107K | 153K | 153K | current |
| 3 | `0 */8 * * *` | 80K | 115K | 115K | extra headroom |

The workflow runs every 6 hours for this reason. Raising the frequency without
either shrinking the scored corpus or adding daily token capacity will exhaust
the two dual-role agents part-way through the day — at which point they reject
every article they are handed and the paper publishes empty.

If more frequent runs are needed, the options are: pre-filter the corpus before
bulk scoring, add `allam-2-7b` (500K/day) as a fourth bulk agent, or move bulk
scoring to `groq/compound-mini` (no daily token cap) if its JSON-mode support
proves adequate.

## Decommissioned Models

Retired from this account. Do not reference them in `swarm.yaml` or `app.yaml`.

| Retired model | Replacement used here |
|---|---|
| `gemma2-9b-it` | `openai/gpt-oss-20b` |
| `llama3-8b-8192` | `openai/gpt-oss-20b` |
| `llama-3.1-8b-instant` | `qwen/qwen3.6-27b` |
| `llama3-70b-8192` | `openai/gpt-oss-120b` |
| `llama-3.3-70b-versatile` | `openai/gpt-oss-120b` |
| `meta-llama/llama-4-scout-17b-16e-instruct` | `openai/gpt-oss-120b` |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | `openai/gpt-oss-120b` |

See https://console.groq.com/docs/deprecations for the current list.

## Speech Services

### Speech To Text
- **Requests**: 20 per minute, 2,000 per day
- **Audio Processing**: 7,200 seconds per hour, 28,800 seconds per day
- Models: `distil-whisper-large-v3-en`, `whisper-large-v3`, `whisper-large-v3-turbo`

### Text To Speech
- **Requests**: 10 per minute, 100 per day
- **Tokens**: 1,200 per minute, 3,600 per day
- Models: `playai-tts`, `playai-tts-arabic`

---
Last updated : August 18 2026
