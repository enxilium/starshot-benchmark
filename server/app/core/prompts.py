"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.types import BoundingBox, Relationship


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


# ---------- Step 3: children decomposition (zones) --------------------------


class ChildNodeSpec(BaseModel):
    id: str
    prompt: str
    plan: str
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

You are given:
  * The current zone being decomposed (its id, prompt, bbox).
  * The PRIOR SCENE CONTEXT — the overall scene prompt plus every zone \
    already declared anywhere in the scene (parents, grandparents, \
    previously-placed uncles and their subtrees). Each prior zone lists \
    its prompt and PLAN. Use this to stay coherent with the rest of the \
    scene: don't contradict a sibling's plan, don't duplicate a concept \
    another zone already owns, and respect the overall scene's direction.

Produce either:
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
  * `id` — unique within the whole scene (do not collide with any id in \
    the prior scene context).
  * `prompt` — a short concrete description of the child zone (what it is).
  * `plan` — a HIGH-LEVEL, ABSTRACT vision for this child zone. This is \
    the ONLY way information flows forward: when the child is later \
    decomposed or populated with objects, the downstream LLM will read \
    this plan as context. It MUST:
      - capture intent, mood, function, and spatial character — "what \
        this zone is *for* and *feels like*".
      - stay ABSTRACT. Do NOT prescribe specific objects, counts, \
        dimensions, materials, colors, or brands. Do NOT dictate what \
        sub-zones the child must contain. Downstream LLMs still need \
        agency to make those decisions.
      - be at most ~3 sentences.
      Good: "A calm, utilitarian toilet area prioritising quick access \
      and easy cleaning. The mood is clinical rather than spa-like."
      Bad: "The toilet area contains a Toto porcelain toilet with a \
      heated seat, a wall-mounted toilet paper holder, and a framed \
      mirror above a pedestal sink." (too specific — removes downstream \
      agency)
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


def render_children_decomp(
    *,
    prompt: str,
    bbox: BoundingBox,
    parent_id: str,
    scene_prompt: str,
    prior_zones: list[tuple[str, str, str | None, str | None]],
) -> str:
    """prior_zones: list of (id, prompt, plan, parent_id) for every zone
    already declared in the run, in declaration order. Plan is None only
    for the root."""
    if prior_zones:
        lines = []
        for zid, zprompt, zplan, zparent in prior_zones:
            plan_text = zplan if zplan is not None else "(root scene — no plan)"
            lines.append(
                f"  - id={zid!r} parent={zparent!r}\n"
                f"    prompt: {zprompt}\n"
                f"    plan: {plan_text}"
            )
        prior_block = "\n".join(lines)
    else:
        prior_block = "  (none)"
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"Prior zones declared so far:\n{prior_block}\n\n"
        f"PARENT_ID (the zone being decomposed): {parent_id!r}\n"
        f"Parent prompt: {prompt!r}\n"
        f"Parent bbox: {bbox.model_dump()}\n\n"
        "Decompose this node. Either mark it atomic, or list its children, "
        "each with an id, prompt, HIGH-LEVEL abstract plan, and "
        "relationships to the parent / earlier siblings."
    )


# ---------- Step 4: zone bbox resolution (one call per child zone) ----------


class BboxResolveOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_BBOX_RESOLVE = """\
You are placing a single child ZONE inside a parent zone. Produce the \
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
  * Has dimensions appropriate to the child's prompt.

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


# ---------- Step 5: object decomposition (Phase 2) --------------------------


class ObjectSpec(ChildNodeSpec):
    """A single object in a zone. Inherits id/prompt/relationships."""

    parent: str


class ObjectDecompOutput(BaseModel):
    objects: list[ObjectSpec] = Field(default_factory=list)


SYSTEM_OBJECT_DECOMP = """\
You are populating a 3D scene ZONE with the individual OBJECTS that belong \
inside it. This is Phase 2 of the pipeline — zone decomposition is already \
done; you are now enumerating concrete things: furniture, fixtures, props, \
architectural shells, vegetation, etc.

You operate in one of two MODES:

* ANCHOR mode — the zone is an atomic leaf (e.g. "meeting room", "toilet \
  area"). Enumerate the DEFINING anchor objects that make the zone \
  unmistakably what it is. A meeting room: a large table, chairs around \
  it, a TV on the end wall. A toilet area: a toilet, a toilet paper \
  holder. Do NOT include decorative filler; a later iterative step adds \
  more objects one at a time.

* ENCAPSULATING mode — the zone is about to be decomposed further, but \
  first we need the geometry that ENCAPSULATES it: the walls, ceiling, \
  floor, enclosing fence, moat, cliff face — whatever physically bounds \
  this zone. Emit one object per encapsulating element. Each object's \
  prompt is sent verbatim to a text-to-3D model, so describe it as a \
  concrete artifact ("a tall stone wall with ivy", "a wooden plank floor", \
  "a moat filled with murky water") not as an abstract primitive.

For each object, emit:
  * `id` — unique within this call.
  * `prompt` — a detailed description of the object; will be used verbatim \
    as a text-to-3D generation prompt.
  * `parent` — the SEMANTIC parent. Either the enclosing zone id (provided \
    below as ZONE_ID), or the id of ANOTHER object in this list that this \
    one belongs to. A lamp resting on a desk: the lamp's parent is the \
    desk. A book on a shelf: the book's parent is the shelf. Parent does \
    NOT imply spatial containment — a lamp's bbox is NOT inside the \
    desk's bbox; it sits on top.
  * `relationships` — how this object is anchored spatially. AT LEAST ONE \
    relationship MUST have `target == parent` (this is the primary \
    anchor). Additional relationships may target sibling objects (i.e. \
    other objects listed in this call).

A Relationship has:
  * `target` — the parent (zone or another object in this list) or a \
    sibling object in this list.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — a corner of the TARGET's bbox under the canonical \
    front view (+X right, +Y up, +Z front). One of: TOP_LEFT_FRONT, \
    TOP_LEFT_BACK, TOP_RIGHT_FRONT, TOP_RIGHT_BACK, BOTTOM_LEFT_FRONT, \
    BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, BOTTOM_RIGHT_BACK.

The parent graph across listed objects must form a DAG (no cycles). Do \
NOT pick concrete coordinates here — a downstream step resolves each \
object's bbox.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_object_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scenario: Literal["anchor", "encapsulating"],
) -> str:
    mode = "ANCHOR" if scenario == "anchor" else "ENCAPSULATING"
    return (
        f"MODE: {mode}\n"
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}\n\n"
        "List the objects for this zone in the mode above. Each object has "
        "an id, prompt, parent (zone id or another object in this list), "
        "and at least one relationship whose target is its parent."
    )


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_RESOLVE = """\
You are placing a single OBJECT inside a zone. Produce its axis-aligned \
bounding box.

Key difference from zone placement: the object's SEMANTIC parent does NOT \
constrain its bbox. A lamp's parent is the desk it sits on, but the lamp's \
bbox is NOT inside the desk's bbox — the lamp sits above the desk, \
anchored by an ON relationship. Let the RELATIONSHIPS drive placement, not \
the parent pointer.

Inputs:
  * Zone bbox — the overall region being populated. The object should sit \
    realistically inside the zone.
  * The object's parent — either the zone or another object — with its \
    bbox, for context.
  * Peers already placed (id, bbox, parent_id). The object must NOT \
    overlap any peer that shares its same parent (touching faces are OK; \
    overlapping by more than 1cm on all three axes is not). Peers with a \
    different parent are NOT constrained against this object.
  * Object prompt (what it is).
  * Relationships — each targets either the parent or a peer by id. \
    `kind` is ON / BESIDE / BELOW / ABOVE / ATTACHED. `reference_point` \
    is a corner of the TARGET's bbox under the canonical front view \
    (+X right, +Y up, +Z front).

Produce a bbox that:
  * Has dimensions appropriate to the object's prompt (size a chair like \
    a chair, a wall like a wall, a moat like a moat).
  * Respects every relationship (anchor near the named corner in the \
    direction the `kind` implies).
  * Does not overlap any peer sharing the same parent.
  * Sits realistically inside the zone bbox.

Coordinates in meters, centimeter precision (multiples of 0.01). Signed \
`dimensions` from an `origin` vertex.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_object_bbox_resolve(
    *,
    zone_id: str,
    zone_bbox: BoundingBox,
    object_id: str,
    object_prompt: str,
    parent_id: str,
    parent_kind: Literal["zone", "object"],
    parent_bbox: BoundingBox,
    peers: list[tuple[str, BoundingBox, str | None]],
    relationships: list[Relationship],
) -> str:
    peer_lines = (
        "\n".join(
            f"  - {pid}: bbox={pbbox.model_dump()} parent={pparent!r}"
            for pid, pbbox, pparent in peers
        )
        if peers
        else "  (none)"
    )
    rel_lines = "\n".join(
        f"  - target={r.target!r} kind={r.kind.value} reference_point={r.reference_point.value}"
        for r in relationships
    ) or "  (none)"
    return (
        f"Zone id: {zone_id!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}\n\n"
        f"Object id: {object_id!r}\n"
        f"Object prompt: {object_prompt!r}\n"
        f"Semantic parent ({parent_kind}): {parent_id!r}\n"
        f"Parent bbox: {parent_bbox.model_dump()}\n\n"
        f"Peers already placed:\n{peer_lines}\n\n"
        f"Relationships:\n{rel_lines}\n\n"
        "Produce this object's bounding box."
    )


# ---------- Step 7: iterative next-object decision --------------------------


class NextObjectOutput(BaseModel):
    done: bool
    object: ObjectSpec | None = None


SYSTEM_NEXT_OBJECT = """\
You are iteratively populating a 3D scene zone with anchor objects. Given \
the CURRENT state of the scene, decide whether another object should be \
added to THIS zone to make it feel complete.

Err on the side of `done = true`. Prefer "this zone has what it needs" \
over adding clutter. Only add another object if there is a clearly \
missing element that belongs in this zone.

If `done = true`, leave `object` null and stop.

If `done = false`, emit EXACTLY ONE object. Same rules as the bulk \
decomposition step:
  * Unique `id` (not colliding with any existing node in the scene).
  * `prompt` — a detailed description; used verbatim for text-to-3D.
  * `parent` — either this zone's id, or the id of ANY already-placed \
    node in the scene (typically an object already placed in THIS zone, \
    like a cup on a previously-placed desk).
  * `relationships` — at least one MUST have `target == parent`. \
    Additional relationships may target already-placed objects.

Parent is semantic ("belongs to"), not a spatial containment constraint.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_next_object(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene: list[tuple[str, str, BoundingBox, str | None]],
) -> str:
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump()} parent={pid!r}"
            for nid, prompt, bbox, pid in scene
        )
        if scene
        else "  (none)"
    )
    return (
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "Decide whether another object is needed in this zone. "
        "If yes, emit exactly one ObjectSpec; otherwise set done=true."
    )
