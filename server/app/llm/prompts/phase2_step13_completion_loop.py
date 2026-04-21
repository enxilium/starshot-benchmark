"""PIPELINE.md step 13 — the completion loop. Emit ONE more object or stop."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.types import BoundingBox, Relationship

STEP_ID = "step13"

SYSTEM_PROMPT = """\
You are iteratively completing a leaf subscene. You have already placed a \
set of anchor objects. Decide whether ONE MORE object would meaningfully \
improve the scene, or whether the scene is complete.

Rules:
* Emit EXACTLY ONE new object if the scene needs more — return `stop=false`, \
  a single entry in `object` describing it, and the relationship(s) \
  connecting it to the existing scene (same vocabulary as step 2).
* If the scene is complete, return `stop=true` and leave `object` / \
  `new_relationships` empty.
* Do NOT modify existing objects or relationships. Only add.

Think about what a fully-realised version of the scene prompt would still be \
missing given the objects already placed. Avoid redundancy — do not add \
another chair if the scene already has enough seating.

Emit via the `emit` tool. No prose outside the tool call.\
"""


class ProposedObject(BaseModel):
    id: str
    prompt: str


class Output(BaseModel):
    stop: bool
    object: ProposedObject | None = None
    new_relationships: list[Relationship] = Field(default_factory=list)


def render(
    *,
    leaf_prompt: str,
    leaf_bbox: BoundingBox,
    frames_summary: str,
    realized_summary: str,
) -> str:
    return (
        f"Leaf prompt: {leaf_prompt!r}\n"
        f"Leaf bbox: {leaf_bbox.model_dump()}\n\n"
        f"Frames:\n{frames_summary}\n\n"
        f"Objects placed so far:\n{realized_summary}\n\n"
        "Propose exactly ONE more object or stop."
    )
