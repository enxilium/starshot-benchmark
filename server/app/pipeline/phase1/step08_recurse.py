"""
PIPELINE.md step 8 — the recursive divide driver.

Phase-1 public entry: `divide(user_prompt)`. It runs step 4 (overall
bbox, root only) to construct the root `SubsceneNode`, then kicks off
the recursion.

Per-node recursion (`_recurse`):
  * step 5  — frame decider (may return `[]`)
  * step 6  — scene breakdown + bbox validator retry loop
  * if atomic: terminate recursion at this node
  * else:
      step 7  — write plan entries for each child (before descending)
      step 8  — recurse into each child
"""

from __future__ import annotations

from app.core.context import current
from app.core.events import PhaseStarted
from app.core.types import SubsceneNode

from . import (
    step04_overall_bbox,
    step05_frame_decider,
    step06_scene_breakdown,
    step07_write_plans,
)


async def divide(user_prompt: str) -> SubsceneNode:
    """Top-level phase-1 entry. Returns the fully-divided subscene tree."""
    ctx = current()
    await ctx.events.emit(PhaseStarted(phase="phase1", scope_id="root"))

    overall_bbox = await step04_overall_bbox.run(user_prompt)
    root = SubsceneNode(
        scope_id="root", prompt=user_prompt, bbox=overall_bbox, high_level_plan=""
    )
    await _recurse(root)
    return root


async def _recurse(node: SubsceneNode) -> None:
    """Populate `node.frames` and either `node.children` or `node.is_atomic`."""
    # Step 5: frames for this subscene (may be empty for outdoor scenes).
    node.frames = await step05_frame_decider.run(prompt=node.prompt, bbox=node.bbox)

    # Step 6: break down this scene (retries live inside step 6).
    breakdown = await step06_scene_breakdown.run(
        scope_id=node.scope_id, prompt=node.prompt, bbox=node.bbox
    )

    if breakdown.is_atomic:
        node.is_atomic = True
        return

    # Step 7: write plan entries before recursing so siblings see them.
    await step07_write_plans.write_plans(breakdown.subscenes)

    # Step 8: recurse into each child sequentially. Sequential recursion
    # keeps the State Repository view read by later siblings deterministic.
    for spec in breakdown.subscenes:
        child = SubsceneNode(
            scope_id=spec.scope_id,
            prompt=spec.prompt,
            bbox=spec.bbox,
            high_level_plan=spec.high_level_plan,
        )
        await _recurse(child)
        node.children.append(child)
