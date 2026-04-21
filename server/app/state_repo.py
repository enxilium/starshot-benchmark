"""
Per-run state repository.

Holds two kinds of entries, both keyed by `scope_id`:

  * `PlanEntry`     — written by PIPELINE.md step 7 when step 6 has
                      decided a non-atomic breakdown. Carries the
                      high-level plan the LLM emitted for each child.
  * `RealizedEntry` — written by PIPELINE.md step 15 when a leaf
                      subscene finishes its full object pipeline.
                      Carries the realized object list + prompts +
                      coords.

Reads (`read_visible`) are performed by PIPELINE.md step 6 on every
recursion and return *everything* in the repository — both entry kinds
across the whole run — so the LLM can maintain stylistic consistency
with plans-in-flight *and* with leaves that have already fully
generated elsewhere in the recursion.

The repository is **run-scoped** — a new instance lives on each `RunContext`.
Global singletons would cross-contaminate concurrent benchmark runs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import AnchorObject, BoundingBox, Relationship


class PlanEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_id: str
    prompt: str
    bbox: BoundingBox
    high_level_plan: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RealizedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope_id: str
    prompt: str
    bbox: BoundingBox
    objects: list[AnchorObject]
    relationships: list[Relationship] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VisibleState(BaseModel):
    """Snapshot returned from `read_visible` — the union of everything the
    reader should see at that point in the run."""

    plans: list[PlanEntry]
    realized: list[RealizedEntry]


@runtime_checkable
class StateRepository(Protocol):
    async def write_plan(self, entry: PlanEntry) -> None: ...
    async def write_realized(self, entry: RealizedEntry) -> None: ...
    async def read_visible(self) -> VisibleState: ...


class InMemoryStateRepository:
    """
    Async-safe in-memory implementation. Writes are serialized through an
    `asyncio.Lock`; reads take a quick lock to snapshot the dict views into
    lists (safe against concurrent writes when we flip phase 2 to parallel
    later).
    """

    def __init__(self) -> None:
        self._plans: dict[str, PlanEntry] = {}
        self._realized: dict[str, RealizedEntry] = {}
        self._lock = asyncio.Lock()

    async def write_plan(self, entry: PlanEntry) -> None:
        async with self._lock:
            self._plans[entry.scope_id] = entry

    async def write_realized(self, entry: RealizedEntry) -> None:
        async with self._lock:
            self._realized[entry.scope_id] = entry

    async def read_visible(self) -> VisibleState:
        async with self._lock:
            plans = list(self._plans.values())
            realized = list(self._realized.values())
        return VisibleState(plans=plans, realized=realized)
