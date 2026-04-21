"""Shared fixtures for pipeline integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.context import RunContext, set_current
from app.core.events import EventLog
from app.llm.recorded import RecordedLLMClient
from app.mesh_gen.stub import StubMeshGenerator
from app.state_repo import InMemoryStateRepository


def write_fixture(fixtures_dir: Path, step_id: str, idx: int, payload: dict) -> None:
    step_dir = fixtures_dir / step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"{idx:03d}.json").write_text(json.dumps(payload))


def install_run_context(
    *,
    run_id: str,
    runs_dir: Path,
    fixtures_dir: Path,
    model_id: str = "claude-opus-4-7",
    max_retries: int = 3,
) -> RunContext:
    events = EventLog(run_id, runs_dir)
    llm = RecordedLLMClient(fixtures_dir)
    mesh_gen = StubMeshGenerator()
    state_repo = InMemoryStateRepository()
    ctx = RunContext(
        run_id=run_id,
        model_id=model_id,
        max_retries=max_retries,
        state_repo=state_repo,
        events=events,
        mesh_generator=mesh_gen,
        llm=llm,
    )
    set_current(ctx)
    return ctx


@pytest.fixture
def ctx_factory(tmp_path: Path):
    """Returns a callable that builds a RunContext with tmp dirs + a unique run_id."""

    counter = {"n": 0}

    def _make(
        *,
        fixtures_dir: Path | None = None,
        max_retries: int = 3,
    ) -> RunContext:
        counter["n"] += 1
        run_id = f"run_{counter['n']:03d}"
        fd = fixtures_dir if fixtures_dir is not None else tmp_path / "fixtures"
        fd.mkdir(parents=True, exist_ok=True)
        return install_run_context(
            run_id=run_id,
            runs_dir=tmp_path / "runs",
            fixtures_dir=fd,
            max_retries=max_retries,
        )

    return _make
