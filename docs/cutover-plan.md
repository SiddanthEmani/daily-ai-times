# v1 → v2 Cutover Plan

This document describes how to move Daily AI Times production from the legacy
Groq-only orchestrator (`src/legacy/`) to the Claude Agent SDK pipeline
(`src/agent/`) safely.

## Current state

- `.github/workflows/pipeline.yml` runs v1 every 4 hours → writes
  `src/frontend/api/*.json`.
- `.github/workflows/pipeline-v2.yml` runs v2 offset 30 minutes → writes
  `src/frontend/api/v2/*.json`.
- Frontend serves v1 by default; `?v2=1` opts a visitor into v2.
- `tests/parity_runner.py` and the `parity` CI job ensure v2 fixture runs are
  deterministic and match the golden manifest at
  `tests/fixtures/golden/api/v2/manifest.json`.

## Cutover gates

Before flipping the default:

1. **Fixture parity green for seven consecutive runs.** CI `parity` job stays
   green on every PR and nightly.
2. **Live parity ≥ 80 %.** The `parity_check` tool reports URL overlap above
   0.8 between v1 and v2 `latest.json` for a week of scheduled runs.
3. **Validate green for seven consecutive runs.** `validate_api` never returns
   `is_error` in the pipeline-v2 workflow artifact.
4. **Cost ceiling confirmed.** Monthly Anthropic + provider bill under the
   agreed budget based on `src/config/providers.yaml` TPM bookings.
5. **Frontend canary.** `?v2=1` beta users report no regressions for seven days.

## Flip procedure

Once all five gates pass:

1. Update `src/frontend/utils/api-base.js`:
   - Invert the default: `return isV2Enabled() ? './api' : './api/v2';`
   - Rename the query param to `v1` so opt-out is explicit.
2. Drop the banner for v2, add one for v1 legacy mode.
3. Keep `pipeline.yml` running for one additional week as a safety net.
4. After the safety week:
   - Disable `pipeline.yml` (archive, do not delete).
   - Delete `src/legacy/` in a standalone PR.
   - Collapse `src/frontend/api/v2/` back to `src/frontend/api/`.

## Rollback

At any point before the delete-legacy PR, rollback is one commit:

- Re-invert `api-base.js` default to v1.
- Re-enable `pipeline.yml` if disabled.
- v2 artifacts remain on disk for forensic analysis.

## Post-cutover cleanup

- Delete `.github/workflows/pipeline.yml` and `.github/workflows/pipeline-v2.yml`,
  replace with a single `.github/workflows/pipeline.yml` targeting v2.
- Rename `src/tools/publishing.py::parity_check` to `legacy_parity_check` and
  mark deprecated.
- Remove dual-publish helpers from `api-base.js`.
