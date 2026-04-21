"""
PIPELINE.md step 1 — HTTP entry point (`POST /generate`).

Builds a per-run `RunContext`, spawns the pipeline on a detached
`asyncio.Task` via `execute_run` (step 2), and returns `run_id` +
URLs the client can poll / subscribe to.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter

from app.api import factories
from app.api.registry import get_registry
from app.api.run_summary import RunSummary, write_summary
from app.api.schemas import GenerateRequest, GenerateResponse
from app.core.config import get_settings
from app.core.context import RunContext
from app.core.events import EventLog
from app.core.terminal_log import TerminalLogger
from app.pipeline.step02_execute_run import execute_run
from app.state_repo import InMemoryStateRepository

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    settings = get_settings()
    run_id = uuid.uuid4().hex
    max_retries = (
        request.max_retries if request.max_retries is not None else settings.max_retries
    )

    events = EventLog(run_id, settings.runs_dir)
    state_repo = InMemoryStateRepository()
    mesh_generator = factories.make_mesh_generator()
    llm = factories.make_llm(request.model)

    ctx = RunContext(
        run_id=run_id,
        model_id=request.model,
        max_retries=max_retries,
        state_repo=state_repo,
        events=events,
        mesh_generator=mesh_generator,
        llm=llm,
    )

    registry = get_registry()
    registry.register(ctx)
    # Initial `pending` summary so GET /runs/{id} works before the task
    # even spins up. execute_run will overwrite to `running` and then to
    # a terminal state.
    write_summary(
        RunSummary(run_id=run_id, status="pending", started_at=datetime.now(UTC))
    )

    terminal = TerminalLogger(events, verbose=settings.log_level == "DEBUG")
    task: asyncio.Task[object] = asyncio.create_task(
        execute_run(ctx=ctx, prompt=request.prompt, terminal=terminal),
        name=f"run:{run_id}",
    )
    registry.attach_task(run_id, task)

    return GenerateResponse(
        run_id=run_id,
        events_url=f"/runs/{run_id}/events",
        status_url=f"/runs/{run_id}",
    )
