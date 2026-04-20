---
name: publish-verify
description: Publish the v2 API JSON files and verify the output satisfies the contract. Use as the final step of any pipeline run. Triggers on "publish", "write the API", "ship v2", "generate the feed", or "finalize the run".
allowed-tools:
  - mcp__daily_ai__publish_v2
  - mcp__daily_ai__validate_api
  - mcp__daily_ai__parity_check
---

# Publish and Verify

This skill atomically writes `src/frontend/api/v2/**.json`, then validates the output, then optionally compares against v1 for parity.

## When to use

- `headline-ranking` has run and state contains populated `latest`, `widget`, and `category_rankings`.
- User explicitly asks to publish or finalize.

## Steps

1. **Publish.** Call `publish_v2` with no arguments. Inspect the returned sha256 digests to confirm every artifact was written.
2. **Validate.** Call `validate_api`. If it returns `is_error=true`, stop and report the exact violation - do not proceed to parity.
3. **Parity (optional).** If `src/frontend/api/latest.json` (v1) exists, call `parity_check`. Report overlap counts. An overlap below 20% on a real production run should be flagged as a regression signal.

## Output contract

- Short status: "Published N artifacts; manifest sha=<16 hex>".
- Validation result.
- Parity numbers when applicable.

## Things to avoid

- Do not rerun collection or scoring here.
- Do not delete or rewrite v1 files. Dual publishing keeps both alive until cutover.
- If the user requested a dry run, call `validate_api` alone after reminding them that nothing will be written.
