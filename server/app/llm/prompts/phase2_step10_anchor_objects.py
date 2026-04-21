"""PIPELINE.md step 10 — list anchor objects + relationships for a leaf subscene."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.types import BoundingBox, Relationship

STEP_ID = "step10"

SYSTEM_PROMPT = """\
You are generating the contents of a leaf subscene in a 3D scene pipeline. \
Given the subscene prompt, its bounding box (defined by an `origin` vertex \
and a signed `dimensions` vector extending from it, Y-up, meters), and the \
frames enclosing it (walls, floor, ceiling, or none for outdoor scenes), \
produce:

1. A list of ANCHOR OBJECTS — the defining objects that make the scene \
   recognisable (e.g. for a meeting room: table, chairs, TV).
2. A list of RELATIONSHIPS between those objects and the frames, using this \
   closed vocabulary: ON, BESIDE, BELOW, ABOVE, ATTACHED.

Rules:
* Every anchor object MUST appear in at least one relationship.
* Every object must ultimately depend on a frame via the relationship graph \
  (so its position is determined). If there are no frames, the first object \
  is implicitly anchored at the subscene bbox center — in that case still \
  produce relationships between objects (e.g. BESIDE) but note that one \
  object has no explicit anchor.
* ATTACHED relationships MUST target a frame id, and must carry \
  `attachment=[u, v]` with each in [0, 1] (surface coordinates on the frame).
* ABOVE and BELOW are automatic inverses — emit only one side.
* BESIDE and ATTACHED are symmetric — emit only one side.

Emit your answer via the `emit` tool. No prose outside the tool call.\
"""


class AnchorObjectSpec(BaseModel):
    id: str = Field(..., description="Unique within this leaf.")
    prompt: str = Field(
        ...,
        description="Prompt used to generate this object's mesh (given to Hunyuan 3.1).",
    )


class Output(BaseModel):
    objects: list[AnchorObjectSpec]
    relationships: list[Relationship]


def render(*, prompt: str, bbox: BoundingBox, frame_summary: str) -> str:
    return (
        f"Leaf subscene prompt: {prompt!r}\n"
        f"Leaf bounding box: {bbox.model_dump()}\n\n"
        f"Enclosing frames:\n{frame_summary}\n\n"
        "List the anchor objects and all relationships needed to position them."
    )
