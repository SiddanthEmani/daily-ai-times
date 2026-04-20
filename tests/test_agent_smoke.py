"""Smoke tests for the agent wiring (no live Claude calls)."""
from __future__ import annotations

import asyncio
import importlib
import os

import pytest


def test_schemas_roundtrip():
    from src.agent.schemas import PIPELINE_RUN_JSON_SCHEMA, PipelineRunResult, SkillResult

    skill = SkillResult(skill="news-collection", status="ok", summary="fetched 42", metrics={"count": 42})
    run = PipelineRunResult(
        run_id="2026-04-19T00:00:00Z",
        started_at="2026-04-19T00:00:00+00:00",
        skills_run=[skill],
    )
    blob = run.model_dump(mode="json")
    restored = PipelineRunResult.model_validate(blob)
    assert restored.skills_run[0].status == "ok"
    assert PIPELINE_RUN_JSON_SCHEMA["schema"]["type"] == "object"


def test_build_options_returns_options_with_tools(monkeypatch):
    # Stub SDK so this test runs without the full claude-agent-sdk installed.
    import src.agent.options as options_mod

    class _StubOptions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    monkeypatch.setattr(options_mod, "ClaudeAgentOptions", _StubOptions)
    monkeypatch.setattr(options_mod, "build_server", lambda: "stub-server")

    opts = options_mod.build_options()
    assert "Skill" in opts.allowed_tools
    assert any(name.startswith("mcp__daily_ai__") for name in opts.allowed_tools)
    assert opts.setting_sources == ["project"]
    assert "scorer" in opts.agents
    assert opts.mcp_servers["daily_ai"] == "stub-server"


@pytest.mark.asyncio
async def test_fixture_run_produces_v2_artifacts(tmp_path, monkeypatch):
    fixture_src = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/tests/fixtures/articles.jsonl"
    )
    monkeypatch.setenv("DAT_FIXTURE", fixture_src)
    output = tmp_path / "v2"

    publish_mod = importlib.import_module("src.pipeline.publish")
    publishing_mod = importlib.import_module("src.tools.publishing")

    monkeypatch.setattr(publish_mod, "_FRONTEND_V2", output)
    monkeypatch.setattr(publishing_mod, "_FRONTEND_ROOT", tmp_path)

    from src.agent.main import _fixture_run

    rc = await _fixture_run()
    assert rc == 0
    assert (output / "latest.json").exists()
    assert (output / "widget.json").exists()
    assert (output / "manifest.json").exists()
