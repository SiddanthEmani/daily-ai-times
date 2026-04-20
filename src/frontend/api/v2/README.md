# v2 API outputs

Files under this directory are written by the Claude Agent SDK pipeline in
`src/agent/main.py`. Before the first live v2 run they are absent; the
frontend falls back silently to v1 unless the user opts in with `?v2=1`.

Schema version: `2.0`. The canonical contract is described by
`PipelineRunResult` in `src/agent/schemas.py` and asserted by the
`validate_api` tool in `src/tools/publishing.py`.

Artifacts:

- `latest.json` — global headline feed.
- `widget.json` — compact top-N used by the embeddable widget.
- `stats.json` — run metadata.
- `categories/<name>.json` — per-category rankings.
- `manifest.json` — schema version, artifact list, sha256 digests, counts.
