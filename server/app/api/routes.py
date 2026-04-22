"""HTTP API — POST /generate, GET /run (SSE)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline import divider
from app.utils import logging as rlog

RUNS_DIR = Path("./runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_task: asyncio.Task[None] | None = None


class GenerateRequest(BaseModel):
    prompt: str
    model: str


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/artifacts", StaticFiles(directory=RUNS_DIR), name="artifacts")

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        global _task
        run_id = uuid.uuid4().hex
        rlog.start_run(prompt=req.prompt, model=req.model)
        _task = asyncio.create_task(_run(run_id, req.prompt, req.model))
        return {"run_id": run_id, "events_url": "/run"}

    @app.get("/run")
    async def run_stream() -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        # Subscribe and snapshot synchronously — no await between them, so no
        # log() call can land in both the snapshot and the live queue.
        q = rlog.subscribe()
        snapshot = list(rlog.STATE["events"])
        return StreamingResponse(_sse(q, snapshot), media_type="text/event-stream")

    return app


async def _sse(
    q: asyncio.Queue[dict[str, object]],
    snapshot: list[dict[str, object]],
) -> AsyncIterator[str]:
    # All events ride the default SSE "message" type; the client dispatches
    # by `event.kind` internally. Keeps the client listener table flat.
    try:
        for event in snapshot:
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] == "run.done":
                return
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] == "run.done":
                return
    finally:
        rlog.unsubscribe(q)


async def _run(run_id: str, prompt: str, model: str) -> None:
    try:
        await divider.run(run_id=run_id, prompt=prompt, model=model, runs_dir=RUNS_DIR)
    except Exception as e:  # noqa: BLE001
        rlog.log("run.error", message=f"{type(e).__name__}: {e}")
    finally:
        rlog.finish_run()
