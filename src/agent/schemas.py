"""Pydantic schemas used as structured outputs for the top-level agent loop."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SkillResult(BaseModel):
    skill: str
    status: str = Field(pattern="^(ok|warning|error)$")
    summary: str
    metrics: dict[str, int | float | str] = Field(default_factory=dict)


class PipelineRunResult(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    skills_run: list[SkillResult]
    published_artifacts: list[str] = Field(default_factory=list)
    manifest_sha: str = ""
    parity_overlap: float | None = None
    notes: str = ""


PIPELINE_RUN_JSON_SCHEMA: dict = {
    "name": "pipeline_run",
    "strict": True,
    "schema": PipelineRunResult.model_json_schema(),
}
