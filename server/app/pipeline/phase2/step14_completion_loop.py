"""
PIPELINE.md step 14 — the completion loop + phase-2 per-leaf driver.

Owns two things:

  * `_propose_one` — one structured LLM call that asks the model for
    EXACTLY ONE more object (or `stop=True`). No retry.
  * `generate_leaf` — the per-leaf driver that sequences phase-2 steps
    10 → 11 → 12 → 13, then loops `_propose_one` + incremental 11/12/13
    (step 14 body proper), and finally calls step 15 to persist the
    realized entry.

The LLM propose call itself does NOT retry under a validator. If the
proposal is structurally broken (cycle, contradiction, unknown target,
...), the loop breaks rather than re-asking. If the incremental bbox
step fails repeatedly under `ctx.max_retries`, that surfaces a
`RetryExhausted` from inside step 11, which fails the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import trimesh

from app.core.context import current
from app.core.events import PhaseStarted, StepCompleted, StepStarted
from app.core.types import AnchorObject, Frame, Relationship, SubsceneNode
from app.geometry.relationship_graph import GraphResult, validate_and_sort
from app.llm.client import PromptPayload
from app.llm.prompts.phase2_step14_completion_loop import (
    STEP_ID,
    SYSTEM_PROMPT,
    Output,
    render,
)

from . import (
    step10_anchor_objects,
    step11_object_bboxes,
    step12_mesh_generation,
    step13_rescale,
    step15_write_realized,
)
from .frame_summary import summarize_frames


@dataclass
class GeneratedLeaf:
    """Output of `generate_leaf`. Meshes are already rescaled and placed."""

    scope_id: str
    leaf: SubsceneNode
    objects: list[AnchorObject]
    relationships: list[Relationship]
    frames: list[Frame]
    meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)


def _summarize_realized(objects: list[AnchorObject]) -> str:
    if not objects:
        return "  (none placed yet)"
    lines: list[str] = []
    for o in objects:
        bbox_str = o.bbox.model_dump() if o.bbox is not None else "<unresolved>"
        lines.append(f"  - id={o.id!r} prompt={o.prompt!r} bbox={bbox_str}")
    return "\n".join(lines)


async def _propose_one(
    *,
    leaf: SubsceneNode,
    frames: list[Frame],
    placed_objects: list[AnchorObject],
) -> Output:
    """One LLM call — either `stop=True` or a single new object + rels."""
    ctx = current()
    payload = PromptPayload(
        system=SYSTEM_PROMPT,
        user=render(
            leaf_prompt=leaf.prompt,
            leaf_bbox=leaf.bbox,
            frames_summary=summarize_frames(frames),
            realized_summary=_summarize_realized(placed_objects),
        ),
    )
    return await ctx.llm.call_structured(
        step_id=STEP_ID, prompt=payload, output_schema=Output
    )


async def generate_leaf(*, leaf: SubsceneNode, frames: list[Frame]) -> GeneratedLeaf:
    """Run phase 2 for a single leaf. Returns a fully-populated `GeneratedLeaf`."""
    ctx = current()
    await ctx.events.emit(PhaseStarted(phase="phase2", scope_id=leaf.scope_id))

    # ---- Step 10: anchor objects + graph validator -------------------------
    await ctx.events.emit(StepStarted(step_id="step10"))
    step10 = await step10_anchor_objects.run(leaf=leaf, frames=frames)
    await ctx.events.emit(
        StepCompleted(
            step_id="step10",
            duration_ms=0.0,
            output_summary={
                "n_objects": len(step10.objects),
                "n_relationships": len(step10.relationships),
            },
        )
    )

    # ---- Step 11: object bbox generation (initial) -------------------------
    await ctx.events.emit(StepStarted(step_id="step11"))
    bboxes = await step11_object_bboxes.run(
        leaf=leaf,
        frames=frames,
        objects=step10.objects,
        relationships=step10.relationships,
        topo_order=step10.graph.order,
        already_resolved={},
        to_resolve=step10.graph.order,
    )
    await ctx.events.emit(
        StepCompleted(
            step_id="step11",
            duration_ms=0.0,
            output_summary={"n_bboxes": len(bboxes)},
        )
    )

    objects_with_bboxes = [
        obj.model_copy(update={"bbox": bboxes[obj.id]}) for obj in step10.objects
    ]

    # ---- Steps 12 + 13: mesh gen and rescale -------------------------------
    meshes_raw = await step12_mesh_generation.run(objects=objects_with_bboxes)
    meshes: dict[str, trimesh.Trimesh] = {
        oid: step13_rescale.rescale_mesh_to_bbox(mesh, bboxes[oid])
        for oid, mesh in meshes_raw.items()
    }

    # ---- Step 14: completion loop ------------------------------------------
    placed = list(objects_with_bboxes)
    all_rels = list(step10.relationships)

    while True:
        proposal = await _propose_one(
            leaf=leaf, frames=frames, placed_objects=placed
        )
        if proposal.stop or proposal.object is None:
            break

        new_obj = AnchorObject(id=proposal.object.id, prompt=proposal.object.prompt)
        candidate_objects = [*placed, new_obj]
        candidate_rels = [*all_rels, *proposal.new_relationships]

        # Re-run step 10's graph validator on the candidate graph. A rejected
        # proposal means the LLM produced something structurally wrong; break
        # rather than loop (we'd just get the same bad proposal again).
        frame_ids = {f.id for f in frames}
        graph_check = validate_and_sort(
            objects=candidate_objects,
            frame_ids=frame_ids,
            relationships=candidate_rels,
        )
        if not isinstance(graph_check, GraphResult):
            break

        # Step 11 in incremental mode: resolve only the new object's bbox.
        incr = await step11_object_bboxes.run(
            leaf=leaf,
            frames=frames,
            objects=candidate_objects,
            relationships=candidate_rels,
            topo_order=graph_check.order,
            already_resolved=bboxes,
            to_resolve=[new_obj.id],
        )
        bboxes = {**bboxes, **incr}
        new_obj_bbox = new_obj.model_copy(update={"bbox": bboxes[new_obj.id]})

        # Steps 12 + 13 for just the new object.
        new_meshes = await step12_mesh_generation.run(objects=[new_obj_bbox])
        meshes[new_obj.id] = step13_rescale.rescale_mesh_to_bbox(
            new_meshes[new_obj.id], bboxes[new_obj.id]
        )

        placed.append(new_obj_bbox)
        all_rels = candidate_rels

    # ---- Step 15: persist realized + return --------------------------------
    await step15_write_realized.write_realized(
        leaf=leaf, placed=placed, relationships=all_rels
    )

    return GeneratedLeaf(
        scope_id=leaf.scope_id,
        leaf=leaf,
        objects=placed,
        relationships=all_rels,
        frames=frames,
        meshes=meshes,
    )
