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


# ---------- Step 2.5: zone plan (runs before every decomposition) -----------


class ZonePlanOutput(BaseModel):
    plan: str


SYSTEM_ZONE_PLAN = """\
You are authoring the PLAN for a 3D scene ZONE, BEFORE it gets decomposed \
into sub-zones. This plan is the zone's north-star: it will be fetched \
verbatim at the very next decomposition step, so the downstream LLM stays \
coherent with a single upfront vision rather than inventing one ad-hoc.

At the ROOT level (the whole-scene zone), this plan sets the overall scene \
direction — it is then fetched at every descendant decomposition step too.
At NON-ROOT levels, this plan is specific to the zone being planned, but it \
MUST stay consistent with the SCENE PLAN and with every prior zone's plan.

You are given:
  * The zone being planned: its id, prompt, and axis-aligned bounding box \
    (in meters, under the canonical front view: +X right, +Y up, +Z front).
  * For non-root zones only: the overall scene prompt, the SCENE PLAN (the \
    root zone's plan), and the PRIOR SCENE CONTEXT — every zone already \
    declared in the run, with their plans.

Write one cohesive plan for the zone that:
  * Captures the zone's intent, mood, and spatial character — what it *is* \
    and *feels like*.
  * Sketches the major sub-regions the zone should contain and how they \
    sit relative to one another (e.g. "a walled grounds split into a \
    formal front garden, a central residence, and a wilder rear orchard"). \
    Speak in REGIONS, not individual objects.
  * Stays ABSTRACT. Do NOT prescribe concrete coordinates, dimensions, \
    counts, materials, brands, or specific objects. Do NOT dictate an \
    exact sub-zone tree — the decomposition step still needs agency.
  * Is concise: a short paragraph, at most ~6 sentences.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_zone_plan(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene_prompt: str | None,
    scene_plan: str | None,
    prior_zones: list[tuple[str, str, str, str]],
) -> str:
    """For the root, pass scene_prompt=None, scene_plan=None, prior_zones=[].
    For non-root zones, pass the scene prompt, the root's plan, and every
    already-planned zone in the run (id, prompt, plan, parent_id)."""
    zone_block = (
        f"ZONE_ID (being planned): {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}"
    )
    if scene_plan is None:
        return (
            f"{zone_block}\n\n"
            "This is the ROOT zone — the whole scene. Produce the PLAN that "
            "will guide every downstream zone decomposition."
        )
    if prior_zones:
        lines = [
            f"  - id={zid!r} parent={zparent!r}\n"
            f"    prompt: {zprompt}\n"
            f"    plan: {zplan}"
            for zid, zprompt, zplan, zparent in prior_zones
        ]
        prior_block = "\n".join(lines)
    else:
        prior_block = "  (none)"
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"SCENE PLAN (the north-star for the whole scene):\n{scene_plan}\n\n"
        f"Prior zones declared so far:\n{prior_block}\n\n"
        f"{zone_block}\n\n"
        "Produce the PLAN for this zone — the north-star for its upcoming "
        "decomposition."
    )


# ---------- Step 3: children decomposition (zones) --------------------------


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

You are given:
  * The current zone being decomposed (its id, prompt, bbox, and its own \
    PLAN — written by the planning step immediately before this call). \
    Your decomposition MUST execute the zone's plan: the children you emit \
    should realise the regions it sketches.
  * The SCENE PLAN — the root zone's plan, authored before any \
    decomposition began. It is the north-star for every step. Your \
    decomposition MUST stay consistent with it: do not contradict its \
    intent, mood, or regional sketch.
  * The PRIOR SCENE CONTEXT — every zone already declared anywhere in the \
    scene (parents, grandparents, previously-placed uncles and their \
    subtrees). Each prior zone lists its prompt and PLAN. Use this to stay \
    coherent with the rest of the scene: don't contradict a sibling's \
    plan, don't duplicate a concept another zone already owns, and respect \
    the overall scene's direction.

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
  * `relationships` — how the child is anchored inside this zone. Every \
    child MUST have at least one relationship.

Do NOT author a plan for each child here — a dedicated planning step runs \
for every child zone right before it is itself decomposed.

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
    plan: str,
    parent_id: str,
    scene_prompt: str,
    scene_plan: str,
    prior_zones: list[tuple[str, str, str, str]],
) -> str:
    """prior_zones: list of (id, prompt, plan, parent_id) for every non-root
    zone already declared in the run, in declaration order. The root is
    communicated separately via scene_prompt + scene_plan. `plan` is the
    zone's own plan, written by the planning step right before this call."""
    if prior_zones:
        lines = []
        for zid, zprompt, zplan, zparent in prior_zones:
            lines.append(
                f"  - id={zid!r} parent={zparent!r}\n"
                f"    prompt: {zprompt}\n"
                f"    plan: {zplan}"
            )
        prior_block = "\n".join(lines)
    else:
        prior_block = "  (none)"
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"SCENE PLAN (the north-star for the whole scene):\n{scene_plan}\n\n"
        f"Prior zones declared so far:\n{prior_block}\n\n"
        f"PARENT_ID (the zone being decomposed): {parent_id!r}\n"
        f"Parent prompt: {prompt!r}\n"
        f"Parent bbox: {bbox.model_dump()}\n"
        f"Parent PLAN (this zone's own plan — execute it):\n{plan}\n\n"
        "Decompose this node. Either mark it atomic, or list its children, "
        "each with an id, prompt, and relationships to the parent / earlier "
        "siblings."
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
  * `relationships` — how this object is anchored spatially. EVERY object \
    is REQUIRED to include at least one relationship whose `target` is \
    EXACTLY EQUAL to that same object's `parent` field. This is the \
    primary anchor, it is NOT optional, and any object that lacks it is \
    malformed and will be rejected by the validator. Encapsulating \
    elements are no exception: a wall, floor, ceiling, moat, or fence \
    whose `parent` is the zone id must list the zone id as the target of \
    at least one of its relationships. Additional relationships may \
    target sibling objects (i.e. other objects listed in this call).

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
    previous_error: str | None = None,
) -> str:
    mode = "ANCHOR" if scenario == "anchor" else "ENCAPSULATING"
    retry_block = (
        (
            "\n\nPRIOR ATTEMPT FAILED VALIDATION:\n"
            f"  {previous_error}\n\n"
            "Fix the specific problem above. In particular, ensure every "
            "object's `relationships` list contains at least one item whose "
            "`target` is EXACTLY EQUAL to that same object's `parent` field."
        )
        if previous_error
        else ""
    )
    return (
        f"MODE: {mode}\n"
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}\n\n"
        "List the objects for this zone in the mode above. Each object has "
        "an id, prompt, parent (zone id or another object in this list), "
        "and at least one relationship whose target is its parent."
        f"{retry_block}"
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
  * `relationships` — REQUIRED to include at least one relationship whose \
    `target` is EXACTLY EQUAL to this emitted object's `parent` field. \
    This is the primary anchor, it is NOT optional, and any object that \
    lacks it is malformed and will be rejected by the validator. \
    Additional relationships may target already-placed objects.

Parent is semantic ("belongs to"), not a spatial containment constraint.

Emit via the `emit` tool. No prose outside the tool call.\
"""


class ImagePromptOutput(BaseModel):
    prompt: str


SYSTEM_IMAGE_PROMPT = """\
You are rewriting an object description into a text-to-image prompt for \
Google's Nano Banana Pro. The resulting image feeds an image-to-3D model, \
so the image must depict the object cleanly, completely, and in correct \
proportions — it is a reference photo, not a scene.

You are given:
  * The original object prompt.
  * The object's resolved axis-aligned bounding box, in meters (width +X, \
    height +Y, depth +Z) under the canonical front view.

Produce a single image-generation prompt that:
  * Depicts ONE instance of the object, centered, filling most of the \
    frame, fully visible with no cropping.
  * Sits on a plain neutral studio background (clean white or very light \
    grey seamless backdrop). No scenery, no context objects, no people, \
    no text, no labels, no watermarks.
  * Uses even, soft, diffuse lighting. No dramatic shadows, no coloured \
    gels, no environmental reflections that imply a setting.
  * Camera framing matches the object's true aspect:
      - Genuinely flat pieces (walls, floors, ceilings, thin panels — one \
        axis << the other two): show a near-orthographic head-on or \
        slightly angled view that makes the piece read as a thin flat \
        panel at the correct aspect ratio. Do NOT render them as boxes.
      - Volumetric objects (chairs, tables, trees, lamps): use a \
        three-quarter product-shot angle that reveals depth.
      - Long extruded features (moats, fences, cliff faces): show the \
        full length at a slight angle so both length and cross-section \
        read correctly.
  * Communicates the proportions verbally too — e.g. "a long narrow \
    16m-wide by 3m-tall flat wall panel, extremely thin, viewed head-on". \
    The image model uses this as a strong prior.
  * Preserves every material, colour, and stylistic detail from the \
    original prompt. Do NOT invent new features.
  * Reads as a single concrete prompt, 1-3 sentences, phrased as \
    photography / product-render direction rather than 3D modelling \
    instructions.

Emit via the `emit` tool. No prose outside the tool call.\
"""


def render_image_prompt(*, prompt: str, bbox: BoundingBox) -> str:
    w, h, d = bbox.size
    return (
        f"Original object prompt: {prompt!r}\n"
        f"Bounding box dimensions: width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m\n\n"
        "Rewrite as a Nano Banana Pro image prompt that produces a clean "
        "reference image of this object for an image-to-3D pipeline."
    )


def render_next_object(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene: list[tuple[str, str, BoundingBox, str | None]],
    previous_error: str | None = None,
) -> str:
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump()} parent={pid!r}"
            for nid, prompt, bbox, pid in scene
        )
        if scene
        else "  (none)"
    )
    retry_block = (
        (
            "\n\nPRIOR ATTEMPT FAILED VALIDATION:\n"
            f"  {previous_error}\n\n"
            "Fix the specific problem above. If you emit an object, its "
            "`relationships` list MUST contain at least one item whose "
            "`target` is EXACTLY EQUAL to that object's `parent` field."
        )
        if previous_error
        else ""
    )
    return (
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "Decide whether another object is needed in this zone. "
        "If yes, emit exactly one ObjectSpec; otherwise set done=true."
        f"{retry_block}"
    )
