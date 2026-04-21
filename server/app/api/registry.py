"""
Per-process registry of live runs. Each `POST /generate` creates an entry; the
SSE and DELETE endpoints look up a run here. Completed runs live on disk
(runs/{run_id}/events.jsonl + run.json + scene.glb) and do not need registry
entries for replay.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.context import RunContext


@dataclass
class RunState:
    ctx: RunContext
    task: asyncio.Task[object] | None = None


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}

    def register(self, ctx: RunContext) -> RunState:
        state = RunState(ctx=ctx)
        self._runs[ctx.run_id] = state
        return state

    def attach_task(self, run_id: str, task: asyncio.Task[object]) -> None:
        self._runs[run_id].task = task

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def forget(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


_registry = RunRegistry()


def get_registry() -> RunRegistry:
    return _registry


def reset_registry_for_tests() -> None:
    global _registry
    _registry = RunRegistry()
