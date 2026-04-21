"""
PIPELINE.md step 2 — per-run execution wrapper.

Owns the lifecycle of a single pipeline run:

  * activates `RunContext` via contextvars
  * spawns the terminal logger task
  * invokes the orchestrator (step 3)
  * writes `run.json` on exit (success OR failure OR cancellation)
  * best-effort emits `RunFailed` on cancellation so SSE subscribers unwind

Any exception (including `asyncio.CancelledError`) propagates out after
bookkeeping so the detached task's eventual `task.exception()` surfaces
it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from datetime import UTC, datetime

from app.api.run_summary import RunSummary, write_summary
from app.core.context import RunContext, set_current
from app.core.events import (
    Event,
    EventLog,
    RunFailed,
    StepRetried,
    parse_event_line,
)
from app.core.terminal_log import TerminalLogger
from app.pipeline.step03_orchestrator import run as orchestrator_run


def _retry_summary(log: EventLog) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not log.path.exists():
        return {}
    with open(log.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event: Event = parse_event_line(line)
            if isinstance(event, StepRetried):
                counts[event.step_id] += 1
    return dict(counts)


async def execute_run(
    *,
    ctx: RunContext,
    prompt: str,
    terminal: TerminalLogger,
) -> None:
    set_current(ctx)
    started_at = datetime.now(UTC)
    write_summary(
        RunSummary(run_id=ctx.run_id, status="running", started_at=started_at)
    )

    terminal_task = asyncio.create_task(terminal.run())
    status: str = "failed"
    glb_url: str | None = None
    error: str | None = None

    try:
        try:
            glb_url = await orchestrator_run(prompt, ctx.model_id)
            status = "completed"
        except asyncio.CancelledError:
            error = "cancelled"
            if not ctx.events.closed:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        ctx.events.emit(RunFailed(error="cancelled"))
                    )
            raise
        except Exception as exc:
            error = repr(exc)
            raise
    finally:
        finished_at = datetime.now(UTC)
        summary = RunSummary(
            run_id=ctx.run_id,
            status=status,  # type: ignore[arg-type]
            started_at=started_at,
            finished_at=finished_at,
            glb_url=glb_url,
            error=error,
            retry_summary=_retry_summary(ctx.events),
        )
        write_summary(summary)
        # Give the terminal logger a chance to drain, bounded so a misbehaving
        # logger can't keep the task alive forever.
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(terminal_task, timeout=5.0)
