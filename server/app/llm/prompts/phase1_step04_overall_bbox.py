"""PIPELINE.md step 04 — overall bounding box (root-only)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.types import BoundingBox

STEP_ID = "step04"

SYSTEM_PROMPT = """\
You are helping build a 3D scene from a text description.

Your job right now is to pick the OVERALL bounding box that the entire scene \
will sit inside. The bounding box is axis-aligned, in meters, Y-up, \
right-handed. It is defined by an `origin` vertex and a signed `dimensions` \
vector `(dx, dy, dz)` extending from that vertex; the sign of each component \
chooses the direction of expansion along that axis. Its aspect ratio should \
reflect the scene shape: a skyscraper is tall and narrow, a river is long \
and flat, a room is modest in every dimension, etc.

Pick dimensions that feel natural for the described scene. Place the origin \
at a sensible position (often the origin for a single object, or with the \
floor at y=0 for architectural scenes) and choose signs so the box extends \
into the region you intend.

Emit your answer by calling the `emit` tool with the required structured \
output. Do not produce any prose outside the tool call.\
"""


class Output(BaseModel):
    bbox: BoundingBox
    reasoning: str = Field(
        ..., description="One or two sentences explaining the dimensions you chose."
    )


def render(user_prompt: str) -> str:
    return (
        f"User prompt for the scene: {user_prompt!r}\n\n"
        "Produce the overall bounding box for the whole scene."
    )
