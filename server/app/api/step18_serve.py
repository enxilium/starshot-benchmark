"""
PIPELINE.md step 18 — post-run HTTP surface.

Four endpoints, all independent of the pipeline task:

  * `GET /runs/{run_id}`          — run summary (status + timings + glb_url)
  * `DELETE /runs/{run_id}`       — cancel an in-flight run
  * `GET /runs/{run_id}/events`   — SSE stream: live for running tasks,
                                    replay from disk for completed ones
  * `GET /glb/{run_id}`           — serve the published `.glb`
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.api.registry import get_registry
from app.api.run_summary import RunSummary, read_summary
from app.core.config import get_settings
from app.core.events import EventLog, read_events_jsonl
from app.pipeline.step17_publish_glb import glb_path_for

router = APIRouter()


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str) -> RunSummary:
    summary = read_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No such run {run_id!r}")
    return summary


@router.delete("/runs/{run_id}", status_code=204)
async def cancel_run(run_id: str) -> Response:
    state = get_registry().get(run_id)
    if state is None or state.task is None:
        # Allow idempotent cancellation of already-completed runs (no-op).
        summary = read_summary(run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No such run {run_id!r}")
        return Response(status_code=204)
    if not state.task.done():
        state.task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await state.task
    return Response(status_code=204)


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> EventSourceResponse:
    settings = get_settings()
    state = get_registry().get(run_id)

    last_event_id_header = request.headers.get("Last-Event-ID", "0")
    try:
        after_seq = int(last_event_id_header)
    except ValueError:
        after_seq = 0

    events_path = settings.runs_dir / run_id / "events.jsonl"
    if state is None and not events_path.exists():
        raise HTTPException(status_code=404, detail=f"No such run {run_id!r}")

    async def live_stream(log: EventLog) -> AsyncIterator[dict[str, str]]:
        async for event in log.subscribe(after_seq=after_seq):
            yield {"id": str(event.seq), "data": event.model_dump_json()}

    async def replay_stream() -> AsyncIterator[dict[str, str]]:
        for event in read_events_jsonl(events_path, after_seq=after_seq):
            yield {"id": str(event.seq), "data": event.model_dump_json()}

    if state is not None:
        return EventSourceResponse(live_stream(state.ctx.events))
    return EventSourceResponse(replay_stream())


@router.get("/glb/{run_id}")
async def get_glb(run_id: str) -> FileResponse:
    path = glb_path_for(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No GLB for run {run_id!r}")
    return FileResponse(
        path, media_type="model/gltf-binary", filename=f"{run_id}.glb"
    )
