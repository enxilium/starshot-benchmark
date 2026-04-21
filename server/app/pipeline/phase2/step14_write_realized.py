"""
PIPELINE.md step 14 — write a realized leaf to the state repository.

Called by step 13 after the completion loop exits. Persists a
`RealizedEntry` capturing the leaf's final object list + relationships
+ bbox so later step-6 `read_visible` calls can surface it as a
completed reference for stylistic consistency.
"""

from __future__ import annotations

from app.core.context import current
from app.core.events import StateRepoWrite
from app.core.types import AnchorObject, Relationship, SubsceneNode
from app.state_repo import RealizedEntry


async def write_realized(
    *,
    leaf: SubsceneNode,
    placed: list[AnchorObject],
    relationships: list[Relationship],
) -> None:
    """Persist a `RealizedEntry` for `leaf` and emit `StateRepoWrite`."""
    ctx = current()
    await ctx.state_repo.write_realized(
        RealizedEntry(
            scope_id=leaf.scope_id,
            prompt=leaf.prompt,
            bbox=leaf.bbox,
            objects=placed,
            relationships=relationships,
        )
    )
    await ctx.events.emit(
        StateRepoWrite(entry_type="realized", scope_id=leaf.scope_id)
    )
