"""PIPELINE.md step 10 — produce bounding boxes for anchor objects in topo order.

The LLM receives the leaf bbox, the frames, the relationships, the
topologically-sorted order of objects to resolve, AND the bboxes that
have already been resolved (either from prior calls or — in the step-13
incremental case — from prior completion-loop iterations). It must only
generate bboxes for the objects in `to_resolve`, respecting the
already-resolved ones as immovable anchors.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.types import BoundingBox

STEP_ID = "step10"

SYSTEM_PROMPT = """\
You are assigning 3D bounding boxes to anchor objects inside a leaf subscene.

Each bounding box is defined by an `origin` vertex and a signed `dimensions` \
vector `(dx, dy, dz)` extending from that vertex; the sign of each component \
chooses the direction of expansion. Y-up, right-handed, meters.

Inputs you will receive:
* The leaf's own bounding box — every object bbox you produce MUST fit inside it.
* The frames enclosing the leaf (walls / floors / ceilings / curves).
* The relationships between objects (ON, BESIDE, BELOW, ABOVE, ATTACHED).
* The topologically-sorted ORDER of objects.
* `already_resolved`: bboxes that are fixed and MUST NOT be changed.
* `to_resolve`: the object ids for which you must produce bboxes.

Rules:
* Every produced bbox must fit inside the leaf bbox as a region (containment).
* Produced bboxes must NOT overlap each other or any already-resolved bbox \
  as regions.
* Bboxes should honor the relationships: an ON object sits on top of its \
  target, ATTACHED objects sit at the (u, v) on their frame, BESIDE objects \
  are adjacent to their target without overlapping, etc.
* Sizes should be physically plausible for the object's prompt (e.g. a \
  dining table ~1.8m x 0.9m x 0.75m; a lamp ~0.3m x 0.3m x 0.5m).

Produce exactly the set of bboxes requested in `to_resolve`. Do not include \
already-resolved objects in your output.

Emit via the `emit` tool. No prose outside the tool call.\
"""


class BBoxAssignment(BaseModel):
    object_id: str
    bbox: BoundingBox


class Output(BaseModel):
    assignments: list[BBoxAssignment] = Field(
        ...,
        description="One BBoxAssignment per object id in `to_resolve`. No extras, no omissions.",
    )


def render(
    *,
    leaf_prompt: str,
    leaf_bbox: BoundingBox,
    frames_summary: str,
    relationships_summary: str,
    topo_order: list[str],
    already_resolved: dict[str, BoundingBox],
    to_resolve: list[str],
) -> str:
    resolved_lines = (
        "\n".join(f"  {oid}: {bb.model_dump()}" for oid, bb in already_resolved.items())
        or "  (none)"
    )
    return (
        f"Leaf prompt: {leaf_prompt!r}\n"
        f"Leaf bbox: {leaf_bbox.model_dump()}\n\n"
        f"Frames:\n{frames_summary}\n\n"
        f"Relationships:\n{relationships_summary}\n\n"
        f"Topological order (resolve in this order): {topo_order}\n\n"
        f"Already resolved (fixed, do not change):\n{resolved_lines}\n\n"
        f"To resolve (produce bboxes for exactly these): {to_resolve}"
    )
