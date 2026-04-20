---
description: Execute the full v2 pipeline end-to-end and publish
---

Run the complete Daily AI Times v2 pipeline:

1. Invoke the `news-collection` skill to fetch and dedupe articles.
2. Invoke the `content-classification` skill to score and normalize categories.
3. Invoke the `headline-ranking` skill for per-category ranking and headline selection.
4. Invoke the `publish-verify` skill to atomically write `src/frontend/api/v2/` and validate the output.

Stop and surface any `is_error` result immediately - do not proceed past a failed step.

Report final summary: number of articles in `latest`, widget titles, publish digests, and parity overlap with v1.
