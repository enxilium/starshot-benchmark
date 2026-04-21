"""
PIPELINE.md step 7 — write plan entries to the state repository.

For every subscene spec emitted by step 6 (scene breakdown) on the
current node, persist a `PlanEntry` *before* recursing. A sibling
subscene that later runs step 6 will see these plans in the
visible-state summary it reads (see `step06_scene_breakdown.py`).

This is run even if the phase-1 and phase-2 phases interleave in a
future refactor — the write/read contract is in place today.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.context import current
from app.core.events import StateRepoWrite
from app.llm.prompts.phase1_step06_scene_breakdown import SubsceneSpec
from app.state_repo import PlanEntry


async def write_plans(specs: Iterable[SubsceneSpec]) -> None:
    """Persist one `PlanEntry` per spec and emit `StateRepoWrite` events."""
    ctx = current()
    for spec in specs:
        await ctx.state_repo.write_plan(
            PlanEntry(
                scope_id=spec.scope_id,
                prompt=spec.prompt,
                bbox=spec.bbox,
                high_level_plan=spec.high_level_plan,
            )
        )
        await ctx.events.emit(
            StateRepoWrite(entry_type="plan", scope_id=spec.scope_id)
        )
