"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.types import BoundingBox, Relationship, Vec3Tuple


# ---------- Step 2: overall bbox --------------------------------------------


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_OVERALL_BBOX = """\
You are helping build a 3D scene from a text description.

Pick the OVERALL bounding box that the entire scene will sit inside. The \
bounding box is axis-aligned, in meters, interpreted under the CANONICAL \
FRONT VIEW: +X = right, +Y = up, +Z = toward the viewer (front), -Z = back. \
It is defined by an `origin` vertex and a signed `dimensions` vector \
`(dx, dy, dz)` extending from that vertex; the sign of each component \
chooses the direction of expansion along that axis. Its aspect ratio should \
reflect the scene shape: a skyscraper is tall and narrow, a river is long \
and flat, a room is modest in every dimension.

Emit all coordinates to centimeter precision — two decimal places, exact \
multiples of 0.01 m. Place the origin sensibly (often the world origin; \
floor at y=0 for architectural scenes) and choose signs so the box extends \
into the region you intend.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_overall_bbox(user_prompt: str) -> str:
    return (
        f"User prompt for the scene: {user_prompt!r}\n\n"
        "Produce the overall bounding box for the whole scene."
    )


# ---------- Step 3: children decomposition ----------------------------------


class ChildNodeSpec(BaseModel):
    id: str
    prompt: str
    relationships: list[Relationship] = Field(default_factory=list)


class ChildrenDecompOutput(BaseModel):
    is_atomic: bool
    children: list[ChildNodeSpec] = Field(default_factory=list)


SYSTEM_CHILDREN_DECOMP = """\
You are decomposing a 3D scene ZONE (a Node) into its direct child zones.

EVERY node in this pipeline is a spatial zone — a bounded region of the \
scene (a room, a sub-area of a room, an outdoor quadrant). Zones are NEVER \
individual objects. Furniture, fixtures, props, plants, and other concrete \
objects are NOT child nodes — they are materialized later by the generation \
pipeline, which fills leaf zones with objects.

Given the current zone's prompt, bbox, and id, produce either:
  (a) `is_atomic = true, children = []` — this zone is a LEAF: its next \
      level of detail is individual objects, not further sub-zones. STOP \
      here. Good leaf-zone names describe a region, not a thing: \
      "toilet area", "shower area", "vanity area", "desk area", \
      "reading nook", "seating area", "garden bed". BAD leaf-zone names \
      describe a single object: "toilet", "showerhead", "sink", "chair", \
      "tree" — never emit these; the generation pipeline handles \
      individual objects.
  (b) `is_atomic = false, children = [...]` — this zone contains distinct \
      spatial sub-zones. Each child MUST itself be a zone (a bounded \
      region), never a single object. A bathroom decomposes into \
      "toilet area", "shower area", "vanity area" — NOT into "toilet", \
      "showerhead", "sink".

For each child zone, emit:
  * `id` — unique within this zone.
  * `prompt` — a detailed description of the child zone.
  * `relationships` — how the child is anchored inside this zone. Every \
    child MUST have at least one relationship.

A Relationship has:
  * `target` — either the parent id (provided below as PARENT_ID) or the \
    `id` of an earlier sibling already listed in this call's `children`.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — which CORNER of the TARGET's bbox this relationship \
    anchors against, under the canonical front view (+X right, +Y up, +Z \
    front). One of: TOP_LEFT_FRONT, TOP_LEFT_BACK, TOP_RIGHT_FRONT, \
    TOP_RIGHT_BACK, BOTTOM_LEFT_FRONT, BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, \
    BOTTOM_RIGHT_BACK.

Do NOT pick concrete coordinates or dimensions for children here — a \
downstream step resolves each child's bbox from its relationships and \
prompt.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_children_decomp(*, prompt: str, bbox: BoundingBox, parent_id: str) -> str:
    return (
        f"PARENT_ID: {parent_id!r}\n"
        f"Parent prompt: {prompt!r}\n"
        f"Parent bbox: {bbox.model_dump()}\n\n"
        "Decompose this node. Either mark it atomic, or list its children "
        "and each child's relationships to the parent / earlier siblings."
    )


# ---------- Step 4: bbox resolution (one call per child) --------------------


class BboxResolveOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_BBOX_RESOLVE = """\
You are placing a single child node inside a parent zone. Produce the \
child's axis-aligned bounding box.

Inputs:
  * Parent bbox (the enclosing zone).
  * Siblings already placed (id + bbox). You MUST NOT overlap any of them.
  * Child prompt (what the child is).
  * Relationships — each references either the parent or an already-placed \
    sibling by id, carries a `kind` (ON, BESIDE, BELOW, ABOVE, ATTACHED), \
    and a `reference_point` naming a corner of the TARGET's bbox under the \
    canonical front view (+X right, +Y up, +Z front).

Produce a bbox that:
  * Lies fully inside the parent bbox.
  * Does not overlap any sibling bbox.
  * Respects every relationship: for each one, the child should be \
    anchored near the named corner of the target, in the direction \
    implied by `kind` (e.g. ABOVE → higher y; BESIDE → adjacent on x or z; \
    ON → resting on the target's top face; ATTACHED → touching the target).
  * Has dimensions appropriate to the child's prompt (size a chair like a \
    chair, a wardrobe like a wardrobe).

Coordinates in meters, centimeter precision (multiples of 0.01). Use a \
signed `dimensions` vector from an `origin` vertex; sign chooses \
expansion direction.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_bbox_resolve(
    *,
    parent_id: str,
    parent_bbox: BoundingBox,
    child_id: str,
    child_prompt: str,
    siblings: list[tuple[str, BoundingBox]],
    relationships: list[Relationship],
) -> str:
    sibling_lines = (
        "\n".join(
            f"  - {sid}: {sbbox.model_dump()}" for sid, sbbox in siblings
        )
        if siblings
        else "  (none)"
    )
    rel_lines = "\n".join(
        f"  - target={r.target!r} kind={r.kind.value} reference_point={r.reference_point.value}"
        for r in relationships
    ) or "  (none)"
    return (
        f"Parent id: {parent_id!r}\n"
        f"Parent bbox: {parent_bbox.model_dump()}\n\n"
        f"Siblings already placed:\n{sibling_lines}\n\n"
        f"Child id: {child_id!r}\n"
        f"Child prompt: {child_prompt!r}\n"
        f"Relationships:\n{rel_lines}\n\n"
        "Produce this child's bounding box."
    )


# ---------- Step 5: frame decider -------------------------------------------


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


class GeneratedFrameSpec(BaseModel):
    kind: Literal["generated"] = "generated"
    id: str
    prompt: str = Field(min_length=1)


FrameSpec = PlaneFrameSpec | CurveFrameSpec | GeneratedFrameSpec


class FrameDeciderOutput(BaseModel):
    needs_frame: bool
    frames: list[FrameSpec] = Field(default_factory=list)


SYSTEM_FRAME_DECIDER = """\
You are deciding the architectural boundary geometry for a zone: its \
WALLS, CEILING, and FLOOR. Nothing else.

Frames define the containing surfaces of a zone. They are NOT used to \
generate objects (furniture, fixtures, decorations, roofs, overhangs, \
columns, railings, planters) — those belong to the generation pipeline, \
not here. Scope of frames is strictly walls, ceilings, floors.

Frame types:
* `plane` — a flat rectangular surface defined by `origin`, `u_axis`, and \
  `v_axis` in world space. Use for rectilinear walls, floors, and ceilings. \
  `u_axis` and `v_axis` must be perpendicular and lie within the plane.
* `curve` — a vertically-extruded surface whose horizontal footprint is a \
  SMOOTH Catmull-Rom spline through `control_points` (so a full circle \
  needs only ~6-8 points, not a dense polyline). `height` is the vertical \
  extrusion above the footprint. Use for curved walls (circular chambers, \
  cylindrical towers, arcs).
* `generated` — LAST-RESORT hatch for walls/ceilings/floors whose shape \
  is too complex for planes or curves (e.g. a vaulted cathedral ceiling, \
  an organic cave wall, a domed roof). A text-to-3D model produces a \
  single hollow shell from your `prompt`, rescaled into the zone bbox. \
  USE SPARINGLY — planes and curves are crisper, cheaper, and more \
  predictable. The `prompt` is sent verbatim to the 3D model and MUST \
  describe a HOLLOW SHELL only — no furniture, no props, no decorations, \
  no free-standing architectural features. Start it with "hollow" and \
  describe openings as rectangular/arched CUTOUTS on specific faces, not \
  as "windows" or "doors". Use abstract geometric primitives (hollow box, \
  cylinder, dome, vaulted arch, cone, slab) — never named objects \
  ("house", "chapel", "cave"). Do NOT use `generated` for anything that \
  is not a wall, ceiling, or floor.

CLOSED vs OPEN curves:
* If a curve's polyline closes on itself (repeat the first control point \
  as the last), the renderer treats it as a fully enclosed chamber and \
  automatically caps floor and ceiling — so a single closed curve describes \
  the entire wall + floor + ceiling of the zone. Do NOT emit extra `plane` \
  frames for its floor or ceiling.
* If the polyline is open (does not close), no caps are added; use this \
  for a partial curved wall (river bank, arena rim).

Guidance:
* Indoor rectilinear rooms: emit `plane` frames for floor + walls + \
  ceiling (typically 6 planes for a box room).
* Indoor curved/round rooms: emit ONE closed `curve` frame.
* Genuinely complex shells where planes + curves cannot express the \
  wall/ceiling/floor geometry: ONE `generated` frame. Do NOT mix it with \
  plane/curve frames — the generated mesh IS the whole shell.
* Outdoor open areas (fields, courtyards, gardens): no frames \
  (`needs_frame = false`).

All coordinates are Y-up, right-handed, meters, under the canonical front \
view (+X right, +Y up, +Z front).

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_frame_decider(*, prompt: str, bbox: BoundingBox) -> str:
    return (
        f"Zone prompt: {prompt!r}\n"
        f"Zone bbox: {bbox.model_dump()}\n\n"
        "Decide whether this zone needs frame geometry. If so, specify every frame."
    )
