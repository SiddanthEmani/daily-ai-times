"""Offline parity runner. Invoked by `make parity`.

Runs the fixture pipeline, then compares the generated manifest against
the snapshot at tests/fixtures/golden/api/v2/manifest.json (if present).
Emits a short JSON report on stdout; exits non-zero on drift.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _snapshot_paths() -> tuple[Path, Path]:
    return (
        ROOT / "src" / "frontend" / "api" / "v2" / "manifest.json",
        ROOT / "tests" / "fixtures" / "golden" / "api" / "v2" / "manifest.json",
    )


async def _run_fixture() -> None:
    os.environ["DAT_FIXTURE"] = str(ROOT / "tests" / "fixtures" / "articles.jsonl")
    from src.agent.main import _fixture_run

    rc = await _fixture_run()
    if rc != 0:
        raise SystemExit(rc)


def _compare() -> dict:
    live_path, golden_path = _snapshot_paths()
    live = json.loads(live_path.read_text())
    report: dict[str, object] = {"live_counts": live.get("counts", {})}
    if not golden_path.exists():
        report["golden_present"] = False
        return report
    golden = json.loads(golden_path.read_text())
    drift: list[str] = []
    if golden.get("schema_version") != live.get("schema_version"):
        drift.append(
            f"schema_version drift: golden={golden.get('schema_version')} live={live.get('schema_version')}"
        )
    for key, expected in golden.get("counts", {}).items():
        actual = live.get("counts", {}).get(key)
        if actual != expected:
            drift.append(f"counts.{key}: golden={expected} live={actual}")
    report.update(
        {
            "golden_present": True,
            "drift": drift,
            "status": "ok" if not drift else "drift",
        }
    )
    return report


def main() -> int:
    asyncio.run(_run_fixture())
    report = _compare()
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") != "drift" else 2


if __name__ == "__main__":
    sys.exit(main())
