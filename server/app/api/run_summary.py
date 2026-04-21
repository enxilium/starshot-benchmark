"""`run.json` summary — persisted terminal state for each run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core.config import get_settings


class RunSummary(BaseModel):
    run_id: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    glb_url: str | None = None
    error: str | None = None
    retry_summary: dict[str, int] = {}


def summary_path(run_id: str) -> Path:
    return get_settings().runs_dir / run_id / "run.json"


def write_summary(summary: RunSummary) -> Path:
    path = summary_path(summary.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2))
    return path


def read_summary(run_id: str) -> RunSummary | None:
    path = summary_path(run_id)
    if not path.exists():
        return None
    return RunSummary.model_validate(json.loads(path.read_text()))
