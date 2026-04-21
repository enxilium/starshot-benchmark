"""
PIPELINE.md step 8 — the recursive divide driver, with interleaved
phase 2 on leaves.

`divide(user_prompt)` is the phase-1 public entry. It runs step 4
(overall bbox, root only) to construct the root `SubsceneNode`, then
kicks off `_recurse`.

Per-node recursion (`_recurse`):
  * step 5  — frame decider (may return `[]`)
  * step 6  — scene breakdown + bbox validator retry loop
  * if atomic:
      * mark the node atomic
      * if `realize_leaves=True` (the live-pipeline default), run the
        whole of phase 2 on this leaf **before returning up**: step 13
        drives steps 9 → 10 → 11 → 12 → 13 loop → 14, which writes a
        `RealizedEntry` to the state repo.
      * return
  * else:
      * step 7  — write plan entries for each child (before descending)
      * step 8  — recurse into each child

The interleaving is what makes the state repo useful: a sibling's later
step-6 `read_visible()` call sees realized leaves from siblings that
already completed their full phase 2, not just plans in flight.

`ancestor_frames` accumulates down the call chain so each leaf receives
its full inherited frame set (every ancestor's + its own).
"""

from __future__ import annotations

from app.core.context import current
from app.core.events import PhaseStarted
from app.core.types import Frame, SubsceneNode
from app.pipeline.phase2 import step13_completion_loop
from app.pipeline.phase2.step13_completion_loop import GeneratedLeaf

from . import (
    step04_overall_bbox,
    step05_frame_decider,
    step06_scene_breakdown,
    step07_write_plans,
)


async def divide(
    user_prompt: str,
    *,
    realize_leaves: bool = True,
) -> tuple[SubsceneNode, list[GeneratedLeaf]]:
    """Phase-1 entry point. Returns the divided tree plus the generated
    leaves (in DFS order) when `realize_leaves=True`.

    Pass `realize_leaves=False` to get just the phase-1 tree with no
    phase-2 calls — useful for tests that only care about division
    behavior. `generated_leaves` is empty in that mode.
    """
    ctx = current()
    await ctx.events.emit(PhaseStarted(phase="phase1", scope_id="root"))

    overall_bbox = await step04_overall_bbox.run(user_prompt)
    root = SubsceneNode(
        scope_id="root", prompt=user_prompt, bbox=overall_bbox, high_level_plan=""
    )
    generated: list[GeneratedLeaf] = []
    await _recurse(
        root,
        ancestor_frames=[],
        realize_leaves=realize_leaves,
        generated=generated,
    )
    return root, generated


async def _recurse(
    node: SubsceneNode,
    *,
    ancestor_frames: list[Frame],
    realize_leaves: bool,
    generated: list[GeneratedLeaf],
) -> None:
    """Populate `node.frames` and either `node.children` or `node.is_atomic`.
    When a leaf is hit and `realize_leaves` is True, runs phase 2 on it
    and appends the result to `generated` before returning.
    """
    # Step 5: frames for this subscene (may be empty for outdoor scenes).
    node.frames = await step05_frame_decider.run(prompt=node.prompt, bbox=node.bbox)

    # Step 6: break down this scene (retries live inside step 6).
    breakdown = await step06_scene_breakdown.run(
        scope_id=node.scope_id, prompt=node.prompt, bbox=node.bbox
    )

    # Frames inherited by anything under this node (includes `node.frames`).
    inherited_frames = [*ancestor_frames, *node.frames]

    if breakdown.is_atomic:
        node.is_atomic = True
        if realize_leaves:
            # Phase 2 for this leaf runs here, BEFORE unwinding. step 13's
            # generate_leaf internally sequences steps 9 → 10 → 11 → 12 →
            # 13 loop → 14. Step 14 writes the RealizedEntry, so siblings'
            # later step-6 read_visible() calls will see this leaf.
            leaf_result = await step13_completion_loop.generate_leaf(
                leaf=node, frames=inherited_frames
            )
            generated.append(leaf_result)
        return

    # Step 7: write plan entries before recursing so siblings see them.
    await step07_write_plans.write_plans(breakdown.subscenes)

    # Step 8: recurse into each child sequentially. Sequential recursion
    # keeps the state repo view deterministic: a later sibling's step-6
    # read_visible() always sees the fully-realized state of earlier ones.
    for spec in breakdown.subscenes:
        child = SubsceneNode(
            scope_id=spec.scope_id,
            prompt=spec.prompt,
            bbox=spec.bbox,
            high_level_plan=spec.high_level_plan,
        )
        await _recurse(
            child,
            ancestor_frames=inherited_frames,
            realize_leaves=realize_leaves,
            generated=generated,
        )
        node.children.append(child)
