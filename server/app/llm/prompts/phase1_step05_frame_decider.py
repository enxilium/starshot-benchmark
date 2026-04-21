"""PIPELINE.md step 05 — does this subscene need a frame? If yes, specify it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.types import BoundingBox, Vec3Tuple

STEP_ID = "step05"

SYSTEM_PROMPT = """\
You are building a 3D scene. Given a subscene (its prompt and bounding box), \
decide whether it needs FRAME geometry — the walls, floor, ceiling, or other \
structural surfaces that define its container — and if so, specify the \
frames.

Guidance:
* Indoor subscenes (rooms, bathrooms, hallways) typically need frames: \
  floor + walls + ceiling.
* Outdoor subscenes (fields, rivers, open areas) typically DO NOT need frames.
* Architectural curves (river walls, arena rims) use type `curve` with \
  control points describing the polyline in world space and a `height`.
* Standard flat surfaces (walls, floors, ceilings) use type `plane` with an \
  origin, u_axis, and v_axis in world space. u_axis and v_axis should be \
  perpendicular and lie within the plane.

All coordinates are Y-up, right-handed, meters. The subscene bbox is given \
as an `origin` vertex plus signed `dimensions` (the sign chooses the \
direction of expansion on each axis). Use its implied extents as the \
reference: a floor is a plane at the lower y extent with u/v spanning the \
x-z extents; ceilings at the upper y extent; walls at the x / z bounds.

Emit your answer via the `emit` tool. No prose outside the tool call.\
"""


class PlaneFrameSpec(BaseModel):
    kind: Literal["plane"] = "plane"
    id: str
    origin: Vec3Tuple
    u_axis: Vec3Tuple
    v_axis: Vec3Tuple


class CurveFrameSpec(BaseModel):
    kind: Literal["curve"] = "curve"
    id: str
    control_points: list[Vec3Tuple] = Field(min_length=2)
    height: float = Field(gt=0.0)


class Output(BaseModel):
    needs_frame: bool
    frames: list[PlaneFrameSpec | CurveFrameSpec] = Field(
        default_factory=list,
        description="Populated only when needs_frame is true.",
    )
    reasoning: str = Field(..., description="Brief rationale.")


def render(*, prompt: str, bbox: BoundingBox) -> str:
    return (
        f"Subscene prompt: {prompt!r}\n"
        f"Subscene bbox: {bbox.model_dump()}\n\n"
        "Decide whether this subscene needs frame geometry. If so, specify every frame."
    )
