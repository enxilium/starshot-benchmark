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
You are decomposing a 3D scene (a Node) into its direct children.

You receive the current node's prompt, its bounding box, and its id. You \
produce either:
  (a) `is_atomic = true, children = []` — this node should NOT be broken \
      down further (it is an indivisible object, e.g. a single chair, a \
      toilet area, a tree). Atomic nodes flow directly into the generation \
      pipeline downstream.
  (b) `is_atomic = false, children = [...]` — list the child subzones / \
      objects that compose this node.

For each child, emit:
  * `id` — unique within this node.
  * `prompt` — a detailed description of that child subzone / object.
  * `relationships` — how the child is anchored inside this node. Every \
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
You are building a 3D scene. Given a zone (its prompt and bounding box), \
decide whether it needs FRAME geometry — the walls, floor, ceiling, or \
other structural surfaces that define its container — and if so, specify \
the frames.

Frame types:
* `plane` — a flat rectangular surface defined by `origin`, `u_axis`, and \
  `v_axis` in world space. Use for rectilinear walls, floors, and ceilings. \
  u_axis and v_axis should be perpendicular and lie within the plane.
* `curve` — a vertically-extruded surface whose horizontal footprint is a \
  SMOOTH SPLINE. The `control_points` you supply are spline control points \
  (NOT a literal polyline) — the renderer passes a smooth Catmull-Rom curve \
  through them, so a circle only needs ~6-8 points, an ellipse ~8-10, a \
  gentle bend ~3-4. `height` is the vertical extrusion above the polyline.
* `generated` — escape hatch: a text-to-3D model (Hunyuan) produces the \
  enclosure as a single mesh, which we rescale into this zone's bbox. \
  The `prompt` is sent verbatim to Hunyuan; it MUST be crafted to produce \
  ONLY A HOLLOW SHELL, not a furnished scene:

  (1) ABSTRACT SHAPES ONLY — describe the shell as a composition of \
  geometric primitives (hollow box, cylinder, dome, triangular prism, \
  overhang, slab, arch, ridge, cone, rectangular cutout). DO NOT use \
  named objects like "house", "cave", "church".

  (2) START with the word "hollow" and repeat emptiness constraints: \
  "empty interior", "shell only", "no contents", "no furniture".

  (3) Describe openings as RECTANGULAR CUTOUTS or ARCHED CUTOUTS on \
  specific faces, not as "windows" or "doors".

  (4) No materials, colors, textures, styles, decorations, or surroundings.

CLOSED vs OPEN curves:
* If a curve's polyline CLOSES on itself (repeat the first control point \
  as the last), the renderer treats it as a fully enclosed chamber and \
  automatically caps floor and ceiling. Do NOT emit additional `plane` \
  frames for its floor or ceiling.
* If open, no caps are added; use this for a river wall, arena rim, etc.

Guidance:
* Indoor rectilinear rooms: emit `plane` frames for floor + walls + ceiling.
* Indoor curved/round rooms: emit ONE closed `curve` frame.
* Architecturally complex or organic enclosures: emit ONE `generated` \
  frame; do NOT mix it with plane/curve frames.
* Outdoor open areas (fields, rivers, courtyards): no frames.

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
