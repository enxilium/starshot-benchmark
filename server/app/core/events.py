from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.core.errors import ValidationConflict

type Phase = Literal["phase1", "phase2"]


class _BaseEvent(BaseModel):
    """Fields stamped by EventLog.emit()."""

    run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    seq: int = 0


class RunStarted(_BaseEvent):
    type: Literal["run_started"] = "run_started"
    prompt: str
    model_id: str


class PhaseStarted(_BaseEvent):
    type: Literal["phase_started"] = "phase_started"
    phase: Phase
    scope_id: str


class StepStarted(_BaseEvent):
    type: Literal["step_started"] = "step_started"
    step_id: str
    inputs_summary: dict[str, Any] = {}


class StepCompleted(_BaseEvent):
    type: Literal["step_completed"] = "step_completed"
    step_id: str
    duration_ms: float
    output_summary: dict[str, Any] = {}


class StepRetried(_BaseEvent):
    type: Literal["step_retried"] = "step_retried"
    step_id: str
    attempt: int
    conflict: ValidationConflict


class StateRepoWrite(_BaseEvent):
    type: Literal["state_repo_write"] = "state_repo_write"
    entry_type: Literal["plan", "realized"]
    scope_id: str


class MeshGenerated(_BaseEvent):
    type: Literal["mesh_generated"] = "mesh_generated"
    object_id: str
    duration_ms: float
    backend: str


class RunCompleted(_BaseEvent):
    type: Literal["run_completed"] = "run_completed"
    glb_url: str
    total_duration_ms: float
    retry_summary: dict[str, int] = {}


class RunFailed(_BaseEvent):
    type: Literal["run_failed"] = "run_failed"
    error: str
    step_id: str | None = None


Event = Annotated[
    RunStarted
    | PhaseStarted
    | StepStarted
    | StepCompleted
    | StepRetried
    | StateRepoWrite
    | MeshGenerated
    | RunCompleted
    | RunFailed,
    Field(discriminator="type"),
]

_TERMINAL_TYPES = ("run_completed", "run_failed")
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event_line(line: str) -> Event:
    return _event_adapter.validate_json(line)


def read_events_jsonl(path: Path, after_seq: int = 0) -> Iterator[Event]:
    """Read events from a JSONL file, filtering by seq > after_seq."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            event = parse_event_line(line)
            if event.seq > after_seq:
                yield event


class EventLog:
    """
    Run-scoped event bus. Every emitted event is:
      1. Appended to {runs_dir}/{run_id}/events.jsonl (canonical, persisted)
      2. Broadcast to every live subscriber via asyncio.Queue
      3. (The terminal logger and SSE endpoint are both subscribers — registered
         by whoever owns the run.)

    Exactly-once ordered delivery: a subscriber sees every event with
    seq > after_seq, drawing on the JSONL for history and the queue for live.

    Closes automatically when a terminal event (RunCompleted / RunFailed) fires.
    """

    def __init__(self, run_id: str, runs_dir: Path) -> None:
        self.run_id = run_id
        self.runs_dir = runs_dir
        self.path = runs_dir / run_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[Event | None]] = []
        self._closed = False

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def closed(self) -> bool:
        return self._closed

    async def emit(self, event: Event) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"EventLog for run {self.run_id} is closed; cannot emit {event.type}"
                )
            self._seq += 1
            event.seq = self._seq
            event.run_id = self.run_id
            event.timestamp = datetime.now(UTC)
            line = event.model_dump_json() + "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
            for q in list(self._subscribers):
                await q.put(event)
            if event.type in _TERMINAL_TYPES:
                await self._close_locked()

    async def subscribe(self, after_seq: int = 0) -> AsyncIterator[Event]:
        """
        Yield every event with seq > after_seq. Replays from JSONL up to the
        current seq, then streams live events. Returns when the log closes.
        """
        async with self._lock:
            snapshot_seq = self._seq
            closed_at_subscribe = self._closed
            q: asyncio.Queue[Event | None] | None = None
            if not closed_at_subscribe:
                q = asyncio.Queue()
                self._subscribers.append(q)

        if after_seq < snapshot_seq:
            for event in read_events_jsonl(self.path, after_seq=after_seq):
                if event.seq > snapshot_seq:
                    continue  # live-queue will deliver this
                yield event

        if q is None:
            return

        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                if event.seq <= max(after_seq, snapshot_seq):
                    continue  # already delivered via replay
                yield event
        finally:
            async with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q in list(self._subscribers):
            await q.put(None)
