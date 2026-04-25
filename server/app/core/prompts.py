"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.types import BoundingBox, ProxyShape, Relationship


# Shared proxy-shape documentation injected into every prompt that lets
# the LLM emit or reason about proxies. Keep the vocabulary and the math
# identical across decomposition and bbox-resolution steps so there is
# no drift — the bbox-resolution step needs the full formulas to place
# ON-anchored children correctly, so they live here alongside the
# vocabulary.
PROXY_SHAPE_DOC = """\
A `proxy_shape` describes the silhouette of the node's mesh INSIDE its \
axis-aligned bbox. The proxy is always inscribed in the AABB; you do \
NOT emit radii or cap sizes — they derive from the bbox dimensions. \
Emit `proxy_shape` ONLY when the mesh is noticeably non-boxy; omit it \
(i.e. null / absent) when the bbox itself is a good collision proxy.

NOTATION used below. For an AABB with min corner (x_min, y_min, z_min) \
and max corner (x_max, y_max, z_max): center (cx, cy, cz), \
half-extents (hx, hy, hz) = ((x_max-x_min)/2, (y_max-y_min)/2, \
(z_max-z_min)/2), full extents (sx, sy, sz) = (2·hx, 2·hy, 2·hz). \
Every proxy below defines a FOOTPRINT (the XZ region the shape covers) \
and a TOP-SURFACE FUNCTION Y_top(x, z) that returns the proxy's upper \
surface height at a given XZ inside that footprint. There is no \
automatic correction — when you anchor another node ON this one, YOU \
must compute Y_top at the anchor's XZ and place its bbox so its \
bottom face sits there.

Valid values:

  * null / omitted — BOX. The AABB is the proxy. Default.
      WHEN TO USE: walls, floors, ceilings, furniture, crates, \
      buildings, signs, rectangular terrain slabs — anything \
      rectilinear.
      FOOTPRINT: the full AABB rectangle, x ∈ [x_min, x_max], \
      z ∈ [z_min, z_max].
      Y_top(x, z) = y_max   (flat top face everywhere in the \
      footprint).

  * SPHERE — ellipsoid inscribed in the AABB, centered at (cx, cy, \
    cz) with semi-axes (hx, hy, hz).
      WHEN TO USE: boulders, planets, balls, orbs, fruits, pumpkins, \
      beach balls.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 in the XZ \
      plane through the centre.
      Y_top(x, z) = cy + hy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Apex: (cx, y_max, cz).

  * CAPSULE — Y-axis capsule inscribed in the AABB. Let r = \
    min(hx, hz). Axis is the vertical line through (cx, cz).
      WHEN TO USE: tree trunks, humans, pillars, lamp posts, \
      bottles — anything columnar.
      Top cap centre: (cx, y_max − r, cz). Bottom cap centre: \
      (cx, y_min + r, cz). Cylindrical section of height sy − 2r \
      between the cap centres (degenerates to a sphere when \
      sy ≤ 2r, in which case r = sy/2 instead of min(hx, hz)).
      FOOTPRINT: the disk (x-cx)² + (z-cz)² ≤ r² centered on the \
      axis — note this is generally smaller than the AABB footprint.
      Y_top(x, z) = (y_max − r) + √(r² − (x-cx)² − (z-cz)²).
      Apex: (cx, y_max, cz).

  * HEMISPHERE — upper half of an ellipsoid with its equatorial \
    disk resting on the AABB's bottom face (y = y_min) and its apex \
    at (cx, y_max, cz). Semi-axes are (hx, sy, hz) — the VERTICAL \
    half-extent is the FULL AABB height sy, NOT hy, because the \
    equator sits at y_min, not at cy.
      WHEN TO USE: DOMED TERRAIN — low islands rising from the \
      waterline, grassy mounds, half-buried boulders, snow hills, \
      cathedral domes.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 at \
      y = y_min.
      Y_top(x, z) = y_min + sy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Drops from the apex y_max at the centre to y_min at the \
      footprint boundary.

ON-RELATIONSHIP CONSEQUENCE. When you anchor a node ON a target with \
a non-BOX proxy, the AABB's top face is NOT the resting surface. \
Compute the target's Y_top at the anchored node's XZ centre using \
the target's AABB and the formula above, then set the anchored \
node's bbox so its bottom face Y equals Y_top. Example: a 0.8m tree \
placed ON a HEMISPHERE island whose AABB is (x_min=-5, y_min=0, \
z_min=-5) → (x_max=5, y_max=1.2, z_max=5), at XZ = (3, 0), rests at \
Y_top = 0 + 1.2·√(1 − 0.36) = 0.96, so its bbox spans y ∈ [0.96, \
1.76] — NOT [1.20, 2.00]. Getting this wrong leaves the tree visibly \
floating or sunk. For BOX targets the rule collapses to the familiar \
"bottom face at y_max".\
"""


def _render_proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


# ---------- Step 2: overall bbox --------------------------------------------


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_OVERALL_BBOX = """\
You are picking the OVERALL bounding box for a 3D scene — the SCENE'S \
CANVAS that every zone, object, and ambient element will be placed \
inside. This pipeline is part of StarshotBench, a head-to-head LLM \
benchmark, and this is the first authored decision of the run: the \
aspect ratio you pick here shapes how every downstream step composes. \
Match the scene's actual silhouette so later steps aren't fighting the \
canvas — a skyscraper is tall and narrow, a river is long and flat, a \
room is modest in every dimension.

The bounding box is axis-aligned, in meters, interpreted under the \
CANONICAL FRONT VIEW: +X = right, +Y = up, +Z = toward the viewer \
(front), -Z = back. It is defined by an `origin` vertex and a signed \
`dimensions` vector `(dx, dy, dz)` extending from that vertex; the \
sign of each component chooses the direction of expansion along that \
axis.

Emit all coordinates to centimeter precision — two decimal places, \
exact multiples of 0.01 m. Place the origin sensibly (often the world \
origin; floor at y=0 for architectural scenes) and choose signs so the \
box extends into the region you intend.

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_overall_bbox(user_prompt: str) -> str:
    return (
        f"User prompt for the scene: {user_prompt!r}\n\n"
        "Produce the overall bounding box for the whole scene."
    )


# ---------- Step 2.5: zone plan (runs before every decomposition) -----------


class SubzoneSeed(BaseModel):
    """A planned subzone authored alongside its parent's plan. The seed's
    `plan` becomes the child Node's own zone plan when the child is placed,
    so it must be a complete character/intent plan in its own right."""

    id: str
    prompt: str
    plan: str


class ZonePlanOutput(BaseModel):
    plan: str
    is_atomic: bool
    subzones: list[SubzoneSeed] = Field(default_factory=list)


SYSTEM_ZONE_PLAN = """\
You are authoring the PLAN for a ZONE of a 3D scene being built for \
StarshotBench — a head-to-head competitive benchmark where your scene \
will be rendered and judged against another LLM's rendering of the \
same user prompt. Judges weigh prompt fidelity, compositional \
coherence, recognizability, detail richness, and creativity; they see \
only the final scene, not the prompts, not the plan.

THIS step is where your aesthetic judgment for this zone locks in. \
The plan you write here is fetched VERBATIM by every downstream step \
that touches this zone — decomposition, object choice, bbox \
resolution, image authoring. A vivid, specific plan pushes every \
downstream decision toward a coherent authored vision. A vague, \
template-y plan produces a zone that reads like stock assets. The \
difference shows up in the final render, and the judges see it.

Don't play it safe. Commit to a point of view. If the scene prompt \
says "a swamp with islands", an adequate plan calls this zone \
"a medium muddy island"; a winning plan makes it "a low hummock \
fringed with twisted cypress roots, dominated by a lone lightning- \
split stump that reads as the island's character from any angle". \
Details like that propagate all the way to the rendered mesh.

At the ROOT level (the whole-scene zone), this plan sets the overall \
scene direction and is fetched at every descendant planning and \
decomposition step.
At NON-ROOT levels, this plan is specific to the zone being planned \
and MUST stay consistent with the SCENE PLAN and with every prior \
zone's plan.

<zone_definition>
A zone is a REGION OF THE SCENE — a subscene, an area, a place large \
enough to contain multiple distinct objects arranged inside it (a \
master bedroom, the left audience stand of an arena, the formal front \
garden of a mansion). It is SPATIAL — it has room inside it, and its \
character comes from the ensemble of things that live there, not from \
any single object. A single landmark, monument, trophy, centerpiece, \
or hero prop — no matter how important — is an OBJECT inside a zone, \
NOT a zone of its own: the trophy obelisk in an arena's imperial box \
is an object within the imperial-box zone, not its own zone. Your \
plan should make unmistakably clear what gives this region its \
identity as a place.
</zone_definition>

<this_step_does_three_things>
In a single output, you must:
  1. Author this zone's character/intent PLAN.
  2. Decide whether this zone is ATOMIC or SUBDIVIDES into child zones.
  3. If it subdivides, AUTHOR THE PLAN for each child zone inline — \
     each one a complete character/intent plan equivalent in quality \
     to your own zone's plan, since the child Node will inherit it \
     verbatim. If it is atomic, leave subzones empty.

The structural decision (atomic vs subdivides) and the per-child \
plans live in this same step deliberately: the parent's character \
and the child set's identities are deeply linked, and authoring them \
together produces a more coherent partition than splitting the work \
across two LLM calls. The next pipeline step is purely structural \
materialization — it takes your subzone seeds and assigns each one \
a proxy_shape and the relationships anchoring it to the parent or \
its siblings. It does NOT re-decide atomicity, re-author plans, or \
add/remove children.
</this_step_does_three_things>

<atomic_vs_subdivides>
This is the single most consequential structural decision in the \
pipeline. You are answering: which regions of the scene deserve their \
own dedicated plan and generation pass, and which melt into ambient \
background?

Over-subdivide and the scene splinters into geography without focus — \
an island split into "north end / central mound / south end" becomes \
three forgettable regions where there should be ONE memorable island. \
Judges see a scene that has no place to look.

Under-subdivide and loci of genuine interest disappear into \
undifferentiated masses — a palace treated as one atomic zone becomes \
a blob where there should be a throne room, a great hall, and a \
garden. Judges see a scene that's all exterior, no authored spaces.

Default to ATOMIC. Subdivision is opt-in. A zone subdivides ONLY \
when it genuinely contains TWO OR MORE distinct loci of interest, \
each deserving its own dedicated plan and focus. If you cannot give \
each proposed child a unique, named reason to exist beyond "this is \
the left part of the parent" or "this is the denser part of the \
parent", STOP — emit is_atomic=true with subzones=[].

The sharpest test: if the "thing" you would carve out is essentially \
ONE SINGLE ARTIFACT — a statue, a monument, an obelisk, a trophy, a \
throne, a fountain, a hearth, a chandelier, a single tree, a single \
vehicle — it is an OBJECT that lives INSIDE some zone, NOT a zone of \
its own. This is true even when the artifact is the most important, \
most visually prominent thing in its area. An arena's trophy obelisk \
is an anchor object of the imperial-box zone, not its own zone. A \
palace's throne is an anchor object of the throne-room zone, not its \
own zone. The focal landmark always belongs to the surrounding \
region; it does not replace it.

A zone is NEVER any of the following:
  * An individual OBJECT — furniture, a flag, a prop, a single tree, \
    a statue, a monument, a fountain, any single artifact. Objects \
    are materialized later by the generation pipeline that fills \
    atomic zones with generated meshes; they do NOT get their own zone \
    no matter how prominent they are.
  * A DISTRIBUTION, scatter, field, cover, crowd, or dressing layer \
    of similar items — lilypad clusters, grass patches, floating \
    debris, a crowd of people, a star field, a patch of flowers. \
    These are populations of instanced objects and live INSIDE an \
    atomic zone as individual objects, not as a subtree of zones.
  * A SURFACE or connective medium — the water skin of a pond, the \
    floor of a plaza, the sky, the asphalt of a street, a patch of \
    mist. Same rule: not a zone.
  * A BAND, RING, CORE, FRINGE, or REGION of something homogeneous, \
    defined only by density or proximity — dense core vs. sparse \
    edge, inner ring vs. outer ring, north half vs. south half. \
    Homogeneous content does not split into zones.
  * NEGATIVE SPACE — the ambient, connective, interstitial medium \
    BETWEEN the zones of interest: the swamp water expanse sprawling \
    between islands, the open sky around a cluster of towers, the \
    grassy meadow filling the gaps between named landmarks, the \
    ocean between ships. Negative space is whatever remains once the \
    loci of interest are carved out. Do NOT emit it as a child zone. \
    A separate negative-space pass at the end of the pipeline \
    enumerates the ambient content that fills it. If you find \
    yourself reaching for a child whose prompt is "the water", "the \
    sky", "the space between", or "the expanse", you are describing \
    negative space — leave it out.

    EXCEPTION: a named feature located WITHIN what would otherwise \
    be negative space IS a zone ONLY IF it is itself a small region \
    with multiple distinct objects inside it — a named whirlpool \
    with its debris ring, a shipwreck with its scattered cargo and \
    spars, a village of stilt huts in open swamp. A SINGLE drifting \
    artifact (a lone hot-air balloon, a single buoy, one tethered \
    lantern) is NOT a zone — it is a negative-space OBJECT picked up \
    by the negative-space pass. The zone covers only the feature's \
    extent; the surrounding medium remains negative space.

Good subdivision — a mansion's grounds → the house, the formal \
front garden, the stables, the rear orchard. Four distinct places, \
four distinct characters.
Good subdivision — a hotel room → bathroom, bedroom. Two rooms with \
different functions and fixtures.
Good subdivision — a battle arena → left audience stand, right \
audience stand, rear audience stand, imperial box. Four distinct \
seating regions with different characters (commoner stands vs. the \
ornate royal box). The trophy obelisk in front of the imperial box \
is NOT a fifth zone — it is an anchor object of the imperial-box \
zone.
Bad subdivision — a swamp → open water, lilypad distribution, log \
debris. The swamp-water expanse is NEGATIVE SPACE; the lilypad and \
log scatters are populations of instanced objects that live INSIDE \
that ambient layer, not separate zones.
Bad subdivision — an island → north end, central mound, south end. \
Pure geographic slicing with no distinct intent per piece.
Bad subdivision — a bedroom → bed area, dresser area, reading nook. \
Micro-regions of a single room — the bed, dresser, and chair are \
anchor OBJECTS inside the room, not sub-zones.
Bad subdivision — an arena → audience stands, imperial box, trophy \
obelisk. The first two are regions; the trophy obelisk is a single \
artifact and must live as an anchor object inside the imperial-box \
zone, not as a peer zone alongside it.
Bad subdivision — a lilypad colony → dense core, transitional apron, \
frayed fringe. Density bands of a homogeneous scatter — always atomic.
</atomic_vs_subdivides>

<inputs>
  * The zone being planned: its id, prompt, and axis-aligned bounding \
    box (in meters, under the canonical front view: +X right, +Y up, \
    +Z front).
  * For non-root zones only: the overall scene prompt, the SCENE PLAN \
    (the root zone's plan), and the PRIOR SCENE CONTEXT — every zone \
    already declared in the run, with their plans.
</inputs>

<output>
Emit a single ZonePlanOutput with three fields:

  * `plan` — the zone's own character/intent plan. Required.
    If an INHERITED PLAN is shown in the inputs (this zone was \
    pre-planned by its parent's planning step), that plan is \
    authoritative — re-emit it here verbatim, or with only minor \
    polish if you spot a clear improvement. Do NOT drift from its \
    character or invent new features. If no inherited plan is shown, \
    write one cohesive paragraph from scratch that:
      - Captures the zone's intent, mood, and character — what it IS \
        and what makes it distinctive in the scene. Lean into \
        specificity; vague plans cost you detail downstream.
      - Names the FEATURES OF INTEREST that give the zone its \
        identity — specific focal points, landmarks, or anchor \
        objects a viewer would point at (a red objective flag, a \
        stone hearth, a central fountain, a writing desk). Be \
        concrete about what makes each distinct.
      - Stays ABSTRACT about concrete layout. Do NOT prescribe \
        coordinates, dimensions, counts, materials, brands, or \
        specific object instances.
      - Is concise: a short paragraph, at most ~5 sentences.

  * `is_atomic` — bool. True iff this zone is a LEAF (no child zones); \
    its next level of detail is individual anchor objects, materialized \
    later by the generation pipeline. Most zones end up atomic.

  * `subzones` — list of SubzoneSeed. EMPTY when is_atomic=true. \
    REQUIRED when is_atomic=false: at least two seeds, one per child \
    zone you are committing to. Each seed has:
      - `id` — unique within the whole scene (do not collide with any \
        id in the prior scene context).
      - `prompt` — a short concrete description of the child zone.
      - `plan` — the CHILD'S character/intent plan, written to the \
        same standard as your own `plan` field above (vivid, \
        specific, names features of interest, stays abstract about \
        layout, ~5 sentences). The child Node will inherit this \
        verbatim — so it must read as a fully-authored zone plan in \
        its own right, not as a one-line label. Make each child's \
        plan distinct from its siblings; do NOT reuse phrasing across \
        siblings.

Do NOT pick coordinates, dimensions, proxy shapes, or relationships \
for subzones — those are the next step's job.
</output>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_zone_plan(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene_prompt: str | None,
    scene_plan: str | None,
    prior_zones: list[tuple[str, str, str, str]],
    inherited_plan: str | None = None,
) -> str:
    """For the root, pass scene_prompt=None, scene_plan=None, prior_zones=[].
    For non-root zones, pass the scene prompt, the root's plan, and every
    already-planned zone in the run (id, prompt, plan, parent_id).
    `inherited_plan` is set when this zone was pre-planned by its parent's
    planning step (every non-root zone in normal flow); the planner should
    re-emit it as-is and focus its energy on the structural decisions."""
    zone_block = (
        f"ZONE_ID (being planned): {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}"
    )
    if scene_plan is None:
        return (
            f"{zone_block}\n\n"
            "This is the ROOT zone — the whole scene. Produce the PLAN that "
            "will guide every downstream zone decomposition, decide whether "
            "the scene subdivides, and if so author each subzone's plan."
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
    inherited_block = (
        f"INHERITED PLAN (authored by this zone's parent — re-emit as the "
        f"`plan` field, do not drift from its character):\n{inherited_plan}"
        if inherited_plan is not None
        else "INHERITED PLAN: (none — author this zone's plan from scratch)"
    )
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"SCENE PLAN (the north-star for the whole scene):\n{scene_plan}\n\n"
        f"Prior zones declared so far:\n{prior_block}\n\n"
        f"{zone_block}\n\n"
        f"{inherited_block}\n\n"
        "Re-emit (or author) this zone's PLAN, decide whether it subdivides, "
        "and if so author every subzone's plan inline."
    )


# ---------- Step 3: children decomposition (zones) --------------------------


class ChildNodeSpec(BaseModel):
    id: str
    prompt: str
    proxy_shape: ProxyShape | None = None
    relationships: list[Relationship] = Field(default_factory=list)

    @field_validator("proxy_shape", mode="before")
    @classmethod
    def _box_means_none(cls, v: object) -> object:
        # The prompt describes BOX as "null/omitted" — no enum value — but
        # some models emit the literal string "BOX" anyway. Treat it as None.
        if isinstance(v, str) and v.upper() == "BOX":
            return None
        return v


class ChildrenDecompOutput(BaseModel):
    children: list[ChildNodeSpec] = Field(default_factory=list)


SYSTEM_CHILDREN_DECOMP = """\
You are the STRUCTURAL MATERIALIZATION step for a 3D scene zone in \
StarshotBench. The atomic-vs-subdivides decision and the per-subzone \
plans were ALREADY made by the upstream planning step. Your job is \
narrow: take the planning step's subzone seeds and turn each one into \
a concrete child spec by assigning a `proxy_shape` and the \
`relationships` that anchor it inside the parent.

You do NOT decide whether the zone subdivides. You do NOT add, remove, \
rename, or rewrite subzones. You do NOT author plans. The seeds you \
receive are authoritative — emit one ChildNodeSpec per seed, in the \
same order, with the seed's `id` and `prompt` copied through verbatim. \
Your only authored fields are `proxy_shape` and `relationships`.

<inputs>
  * The parent zone being decomposed: its id, prompt, bbox, and its \
    own PLAN.
  * Its SUBZONE SEEDS — one per child the planning step committed to. \
    Each seed has `id`, `prompt`, and a full character/intent `plan`. \
    Read each seed's plan; the silhouette implied by the plan is what \
    drives your `proxy_shape` choice and your relationship anchoring.
  * The SCENE PLAN — the root zone's plan, the north-star for every \
    step.
  * The PRIOR SCENE CONTEXT — every zone already declared in the run, \
    with their plans. Use this to stay coherent.
</inputs>

<output>
Emit one `ChildNodeSpec` per subzone seed, in the same order. For each:
  * `id` — copy the seed's id verbatim.
  * `prompt` — copy the seed's prompt verbatim.
  * `proxy_shape` — OPTIONAL. The child zone's collision-proxy shape, \
    chosen from the silhouette implied by the seed's plan (a domed \
    island → HEMISPHERE; a column or trunk-shaped region → CAPSULE; \
    most architectural zones → omit). See the PROXY SHAPE section.
  * `relationships` — REQUIRED, at least one per child. Anchors the \
    child inside the parent.

A Relationship has:
  * `target` — either the parent id (provided below as PARENT_ID) or \
    the `id` of an earlier sibling already listed in this call's \
    `children`.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — which CORNER of the TARGET's bbox this \
    relationship anchors against, under the canonical front view (+X \
    right, +Y up, +Z front). One of: TOP_LEFT_FRONT, TOP_LEFT_BACK, \
    TOP_RIGHT_FRONT, TOP_RIGHT_BACK, BOTTOM_LEFT_FRONT, \
    BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, BOTTOM_RIGHT_BACK.

Do NOT pick concrete coordinates or dimensions — a downstream batch \
step resolves each child's bbox from its relationships and prompt.

Number of `children` MUST equal the number of subzone seeds. Do not \
omit a seed and do not invent extra children.
</output>

<proxy_shape>
""" + PROXY_SHAPE_DOC + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_children_decomp(
    *,
    prompt: str,
    bbox: BoundingBox,
    plan: str,
    subzones: list["SubzoneSeed"],
    parent_id: str,
    scene_prompt: str,
    scene_plan: str,
    prior_zones: list[tuple[str, str, str, str]],
) -> str:
    """prior_zones: list of (id, prompt, plan, parent_id) for every non-root
    zone already declared in the run, in declaration order. The root is
    communicated separately via scene_prompt + scene_plan. `plan` is the
    zone's own plan, written by the planning step. `subzones` is the seed
    list authored by that same planning step — id+prompt+plan per child."""
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
    seed_lines = "\n\n".join(
        f"  - id={s.id!r}\n"
        f"    prompt: {s.prompt}\n"
        f"    plan: {s.plan}"
        for s in subzones
    )
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"SCENE PLAN (the north-star for the whole scene):\n{scene_plan}\n\n"
        f"Prior zones declared so far:\n{prior_block}\n\n"
        f"PARENT_ID (the zone being decomposed): {parent_id!r}\n"
        f"Parent prompt: {prompt!r}\n"
        f"Parent bbox: {bbox.model_dump_json()}\n"
        f"Parent PLAN:\n{plan}\n\n"
        f"Subzone seeds to materialize ({len(subzones)}):\n{seed_lines}\n\n"
        "Emit one ChildNodeSpec per seed (same order, copy id+prompt "
        "verbatim). Author proxy_shape (optional) and relationships."
    )


# ---------- Step 4: zone bbox batch resolution (all siblings at once) -------


class BboxAssignment(BaseModel):
    id: str
    bbox: BoundingBox


class BboxBatchOutput(BaseModel):
    assignments: list[BboxAssignment] = Field(default_factory=list)


SYSTEM_ZONE_BBOX_BATCH = """\
You are a constraint solver. Place ALL sibling child ZONES inside a \
parent zone in one shot, deriving each child's axis-aligned bounding \
box from the given inputs (parent bbox, child specs, relationships). \
This step has no creative latitude — your job is to produce \
coordinates that satisfy every stated constraint simultaneously.

Inputs:
  * Parent bbox (the enclosing zone).
  * A list of child specs — id, prompt, and relationships that target \
    either the parent or another child in this same list.

Relationships carry `kind` (ON, BESIDE, BELOW, ABOVE, ATTACHED) and a \
`reference_point` — a corner of the TARGET's bbox under the canonical \
front view (+X right, +Y up, +Z front).

Each child carries a `proxy_shape` describing its mesh silhouette \
inside its AABB — BOX, SPHERE, CAPSULE, or HEMISPHERE. The PROXY \
SHAPE section below gives the exact surface formula Y_top(x, z) for \
each. When a child with a non-BOX proxy is the TARGET of another \
child's ON relationship, the ON-child rests on the target's proxy \
TOP SURFACE at its XZ centre — NOT on the target's AABB top face. \
There is no automatic correction: YOU must compute the target's \
Y_top(x, z) from the target's AABB and proxy formula, and place the \
ON-child's bbox so its bottom face Y equals that value. Pick \
dimensions so a HEMISPHERE target has vertical headroom above its \
apex for the things that sit on it.

Produce one assignment per child (id + bbox) such that:
  * Every bbox lies fully inside the parent bbox.
  * No two child bboxes overlap volumetrically. Touching at a shared \
    face is fine; eating into another bbox's volume is not.
  * Every relationship is respected: for each one, the child is \
    anchored near the named corner of its target, in the direction \
    implied by `kind` (ABOVE → higher y; BESIDE → adjacent on x or z; \
    ON → the child's bottom face at the target's Y_top(x, z) — the \
    AABB top face y_max for BOX targets, the proxy formula for \
    SPHERE/CAPSULE/HEMISPHERE targets; ATTACHED → touching the \
    target).
  * Dimensions are appropriate to each child's prompt.

Because you are deciding the entire layout at once, RESERVE SPACE for \
every child up front rather than committing each bbox in isolation. A \
later child's requirements must influence earlier siblings' sizing.

Coordinates in meters, centimeter precision (multiples of 0.01). Use \
a signed `dimensions` vector from an `origin` vertex; sign chooses \
expansion direction. Emit exactly one assignment per requested child \
id, no extras, no omissions.

<proxy_shape>
""" + PROXY_SHAPE_DOC + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_zone_bbox_batch(
    *,
    parent_id: str,
    parent_bbox: BoundingBox,
    children: list["ChildNodeSpec"],
) -> str:
    child_lines = "\n\n".join(
        f"  - id={c.id!r}\n"
        f"    prompt: {c.prompt}\n"
        f"    proxy_shape: {_render_proxy_shape(c.proxy_shape)}\n"
        f"    relationships:\n"
        + (
            "\n".join(
                f"      * target={r.target!r} kind={r.kind.value} "
                f"reference_point={r.reference_point.value}"
                for r in c.relationships
            )
            or "      (none)"
        )
        for c in children
    )
    return (
        f"Parent id: {parent_id!r}\n"
        f"Parent bbox: {parent_bbox.model_dump_json()}\n\n"
        f"Children to place ({len(children)}):\n{child_lines}\n\n"
        "Produce a bbox for every child in a single coherent layout."
    )


# ---------- Step 5: object decomposition (Phase 2) --------------------------


class ObjectSpec(ChildNodeSpec):
    """A single object in a zone. Inherits id/prompt/relationships."""

    parent: str
    # Yaw, radians, world-frame rotation about +Y. 0 = front faces world +Z
    # (toward viewer); π/2 = front faces world -X (rotated right-hand). The
    # mesh comes back from Trellis with its intrinsic front along +Z; this
    # field rotates it into the world pose the LLM intends.
    orientation: float = 0.0


class ObjectDecompOutput(BaseModel):
    objects: list[ObjectSpec] = Field(default_factory=list)


SYSTEM_OBJECT_DECOMP = """\
You are enumerating the OBJECTS that populate a 3D scene zone inside \
StarshotBench — a head-to-head competitive benchmark where your scene \
will be rendered and judged against another LLM's rendering of the \
same user prompt.

The object list you produce here is WHAT THE JUDGES ACTUALLY SEE. \
Zones and plans are scaffolding; objects are the scene. Thoughtful \
anchor choices make a zone unmistakably, immediately recognizable as \
its subject — a meeting room's long conference table and screen on \
the end wall; a castle throne room's raised dais, carved chair, and \
flanking banners; an island's lone lightning-split stump, knotted \
roots at the waterline, and the red objective flag planted at its \
crest. Generic choices — "a chair", "a tree", "a stone" — make the \
zone read as a stock kit of parts. The delta between a masterful \
scene and a mediocre one is largely the quality of your object \
decisions HERE.

Push past the obvious first pick. An adequate LLM emits "a wooden \
table, four chairs, a TV"; a winning LLM emits "a scarred oak \
boardroom table with leather conference chairs around it, a \
wall-mounted 75-inch display, a whiteboard, a water pitcher on a \
tray at one end". Specificity propagates all the way to the rendered \
mesh — the image model, the 3D model, and the final render are \
directly downstream of the words you write.

Three modes are available — ANCHOR, ENCAPSULATING, NEGATIVE-SPACE. \
Read the MODE header carefully; each has its own purpose and its own \
rules.

<modes>
You operate in one of three MODES:

* ANCHOR mode — the zone is an atomic leaf (e.g. "meeting room", "toilet \
  area", "hero island"). Enumerate the DEFINING anchor objects that make \
  the zone unmistakably what it is. A meeting room: a large table, chairs \
  around it, a TV on the end wall. A toilet area: a toilet, a toilet \
  paper holder. An island: the flag, the roots at the waterline, a \
  gnarled tree. Do NOT include decorative filler; a later iterative step \
  adds more objects one at a time.

  GROUND-AWARENESS RULE. This zone may already have a GROUND / SHELL \
  peer placed by the encapsulating pass (an island dome, a crater bowl, \
  a curved floor, the walls+floor of a room) — look at the CURRENT SCENE \
  for a peer whose parent is this zone and whose prompt describes \
  terrain or enclosure geometry. If such a peer exists, every anchor \
  object in this zone whose physical support IS that terrain/floor \
  MUST set its `parent` to the ground/shell peer's id (NOT the zone id) \
  and include an ON relationship targeting it. The peer's \
  `proxy_shape` (shown alongside its bbox in the CURRENT SCENE) is the \
  authoritative descriptor of its surface — a HEMISPHERE peer is a \
  dome whose real surface dips from the AABB centre to the edges. \
  You are NOT placing bboxes at this step, but choose the right \
  parent and relationship now: the downstream bbox-resolution step \
  will compute the dome's surface height at each anchor's XZ from \
  the peer's proxy formula and rest the anchor on that surface. Do \
  NOT re-emit the ground itself in anchor mode; the encapsulating \
  pass already placed it.

* ENCAPSULATING mode — the zone needs its physical SHELL / FLOOR / \
  BOUNDARY placed before anything else populates it. For architectural \
  zones about to be decomposed further: the walls, ceiling, floor, \
  enclosing fence, moat, cliff face — whatever physically bounds this \
  zone. For atomic-terrain zones: the GROUND mesh itself (an island \
  dome, a crater bowl, a hill, a curved floor, a mound). Emit one \
  object per shell element. Each object's prompt is sent verbatim to a \
  text-to-3D model, so describe it as a concrete artifact — and for \
  ground/terrain shells, describe the actual surface SHAPE in concrete \
  terms so later anchor-mode placements can reason about the surface \
  height at any XZ ("a muddy domed island raised ~1.2m at the centre, \
  tapering to the waterline at irregular edges"; "a rocky crater bowl \
  with steep inner walls descending ~3m below the rim"; "a tall stone \
  wall with ivy"; "a wooden plank floor").

* NEGATIVE-SPACE mode — you are filling the AMBIENT, CONNECTIVE, \
  INTERSTITIAL space of the scene (or a zone) with drifting, background, \
  or distribution-style content that doesn't belong to any specific \
  zone: lilypads drifting across swamp water between islands, clouds \
  above a cityscape, grass tufts scattered over a meadow, floating \
  debris across open water, mist pooling in low valleys, loose paper \
  blowing across a plaza. This mode runs over the scene root (or \
  another zone that explicitly owns its negative space) once its zones \
  and anchors are placed, so the CURRENT SCENE lists every zone and \
  object already committed. Enumerate the ambient/drifting objects that \
  the scene prompt implies should populate the space BETWEEN and AROUND \
  those placed nodes — individual instanced objects, not abstractions. \
  Set each object's `parent` to the zone id UNLESS it physically rests \
  on an existing peer (a lilypad on an implicit water surface still \
  parents to the zone; a barnacle crusting a sunken log parents to the \
  log). Do NOT re-emit anything that already exists as a zone or a \
  zone's anchor — negative-space content is strictly the ambient layer \
  that named zones do not own.
</modes>

<inputs>
You are given the CURRENT SCENE — every node already placed anywhere in \
the run so far, with id, prompt, bbox, and parent. Reason about it \
before emitting.
</inputs>

<global_rules>
  * DO NOT DUPLICATE GEOMETRY. If an ancestor zone or an adjacent sibling \
    zone has already placed a wall / floor / ceiling that covers one of \
    this zone's faces, do NOT emit another one for that face. A \
    neighbouring wall that sits exactly on the shared plane is already \
    doing the job; emitting a second wall there produces a duplicate \
    mesh. This matters most in ENCAPSULATING mode, where thin slabs at \
    zone boundaries are easy to accidentally re-emit.
</global_rules>

<per_object_fields>
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
  * `proxy_shape` — OPTIONAL. The object's collision-proxy shape if its \
    silhouette is noticeably non-rectilinear (see PROXY SHAPE section \
    below). CRITICAL for TERRAIN SHELLS in encapsulating mode: a domed \
    island MUST set proxy_shape=HEMISPHERE, otherwise every anchor \
    object placed ON it will float above its AABB top instead of \
    resting on the actual dome. Omit for architectural shells (walls, \
    floors, ceilings, fences) and any object whose bbox is already a \
    good silhouette.
  * `orientation` — REQUIRED FLOAT (radians, default 0.0). World-frame \
    yaw about +Y for the generated mesh. The image-to-3D model receives \
    an ORTHOGRAPHIC FRONT VIEW of the object, so its mesh comes back \
    with the visible front face along world +Z. `orientation` is the \
    additional rotation needed to point the object's "front" the right \
    way in the world. Right-handed about +Y: `0.0` keeps the front \
    facing +Z (the viewer); `π/2` (≈1.5708) rotates the front to face \
    -X; `π` (≈3.1416) faces -Z (away); `-π/2` (≈-1.5708) faces +X. \
    Examples: a sofa whose seat opens toward the room centre needs \
    orientation set so its front (the seat side) faces the room's \
    interior, not the wall. A door in a wall on the +X face of a room \
    needs orientation ≈ -π/2 so the door faces +X. The bbox stays an \
    AABB regardless — orientation only rotates the mesh inside it, so \
    a long object's bbox dimensions must match its long axis AFTER \
    rotation. Omit (or set 0.0) for symmetric objects with no preferred \
    facing (boulders, balls, columns, generic terrain).
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
</per_object_fields>

<proxy_shape>
""" + PROXY_SHAPE_DOC + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_object_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, float]],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    mode = {
        "anchor": "ANCHOR",
        "encapsulating": "ENCAPSULATING",
        "negative-space": "NEGATIVE-SPACE",
    }[scenario]
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(proxy)} "
            f"orientation={orient:.3f}rad parent={pid!r}"
            for nid, prompt, bbox, pid, proxy, orient in scene
        )
        if scene
        else "  (none)"
    )
    if prior_attempts:
        attempt_lines = "\n\n".join(
            f"  attempt {i}:\n"
            f"    emitted: [{', '.join(s.model_dump_json() for s in specs)}]\n"
            f"    rejected: {reason}"
            for i, (specs, reason) in enumerate(prior_attempts)
        )
        retry_block = (
            "\n\nPRIOR ATTEMPTS — every decomposition below was ALREADY "
            "rejected. Do NOT re-emit the same set of object specs, and do "
            "not repeat the same structural mistake. Treat every listed "
            "reason as a hard constraint you must satisfy this time:\n"
            f"{attempt_lines}\n\n"
            "Produce a NEW decomposition that fixes every listed reason. In "
            "particular, ensure every object's `relationships` list contains "
            "at least one item whose `target` is EXACTLY EQUAL to that same "
            "object's `parent` field."
        )
    else:
        retry_block = ""
    return (
        f"MODE: {mode}\n"
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "List the objects for this zone in the mode above. Each object has "
        "an id, prompt, parent (zone id or another object in this list), "
        "and at least one relationship whose target is its parent. Respect "
        "the CURRENT SCENE: do not duplicate geometry another zone has "
        "already emitted on a shared face, and keep bboxes inside this "
        "zone so they do not volumetrically overlap any peer."
        f"{retry_block}"
    )


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_BATCH = """\
You are a constraint solver. Place ALL objects for a scene ZONE in one \
shot, deriving each object's axis-aligned bounding box from the given \
inputs (zone bbox, object specs, relationships, peer prompts/bboxes). \
This step has limited creative latitude: your job is to produce \
coordinates that satisfy the stated constraints and that respect the \
actual geometry implied by peer prompts.

Key semantics — an object's SEMANTIC parent does NOT constrain its \
bbox. A lamp's parent is the desk it sits on, but the lamp's bbox is \
NOT inside the desk's bbox — the lamp sits above the desk, anchored by \
an ON relationship. Let the RELATIONSHIPS drive placement, not the \
parent pointer.

AABB vs. actual geometry — an AABB describes each peer's EXTENT, NOT \
the shape of its surface. Each peer also carries a `proxy_shape` \
(BOX, SPHERE, CAPSULE, or HEMISPHERE) that IS the authoritative \
silhouette inside that AABB; the PROXY SHAPE section below gives the \
exact surface formula Y_top(x, z) for each. A dome-shaped island has \
proxy_shape=HEMISPHERE: its real surface dips from the AABB apex \
down to the AABB's bottom face at the footprint edge, NOT a flat top \
face.

There is no automatic correction. When an object anchors ON a peer \
or sibling with a non-BOX proxy, YOU must compute the target's \
Y_top(x, z) from its AABB and proxy formula, and place the \
anchored object's bbox so its bottom face Y equals that value at the \
anchored object's XZ centre. For BOX-proxy targets (walls, floors, \
ceilings, generic slabs) this collapses to the familiar "bottom face \
at y_max".

Inputs:
  * Zone id, prompt, and bbox — the overall region being populated.
  * OBJECTS to place: each with id, prompt, proxy_shape, semantic \
    parent (the zone id, another object in this batch, or a prior \
    peer id), and a list of relationships.
  * PEERS already placed elsewhere in the scene — each with id, \
    PROMPT, bbox, proxy_shape, and parent_id. The proxy_shape is the \
    authoritative surface descriptor for ON placement; the prompt \
    supplies richer visual context. AABB overlap with peers is \
    expected in practice — objects resting on curved terrain share \
    airspace with their ground mesh, and semantically-contained \
    objects live inside their parent's bbox — so don't contort \
    placements to avoid overlap that the underlying geometry will \
    resolve.

Relationships carry `kind` (ON / BESIDE / BELOW / ABOVE / ATTACHED) \
and a `reference_point` — a corner of the TARGET's bbox under the \
canonical front view (+X right, +Y up, +Z front).

Produce one assignment per object (id + bbox) such that:
  * Every bbox lies fully inside the zone bbox.
  * Every relationship is respected.
  * Dimensions are appropriate to each object's prompt (size a chair \
    like a chair, a wall like a wall, a roof like a roof).
  * Avoid placing two clearly unrelated objects in the same XZ \
    footprint when nothing about the scene justifies it (two trees \
    stacked on the same spot). Some AABB overlap is fine and often \
    unavoidable — curved ground meshes, semantic parents, stacking — \
    so treat non-overlap as a soft preference driven by physical \
    plausibility, not a hard rule.

Because you are deciding the full layout at once, RESERVE SPACE for \
every object up front — if the zone needs walls AND a roof, the walls \
must stop short of the ceiling so the roof has somewhere to sit.

Coordinates in meters, centimeter precision (multiples of 0.01). \
Signed `dimensions` from an `origin` vertex. Emit exactly one \
assignment per requested object id — no extras, no omissions.

<proxy_shape>
""" + PROXY_SHAPE_DOC + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_object_bbox_batch(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    objects: list[ObjectSpec],
    peers: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, float]],
) -> str:
    peer_lines = (
        "\n".join(
            f"  - {pid}: prompt={pprompt!r} bbox={pbbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(pproxy)} "
            f"orientation={porient:.3f}rad parent={pparent!r}"
            for pid, pprompt, pbbox, pparent, pproxy, porient in peers
        )
        if peers
        else "  (none)"
    )
    object_lines = "\n\n".join(
        f"  - id={o.id!r}\n"
        f"    prompt: {o.prompt}\n"
        f"    parent: {o.parent!r}\n"
        f"    proxy_shape: {_render_proxy_shape(o.proxy_shape)}\n"
        f"    orientation: {o.orientation:.3f}rad\n"
        f"    relationships:\n"
        + (
            "\n".join(
                f"      * target={r.target!r} kind={r.kind.value} "
                f"reference_point={r.reference_point.value}"
                for r in o.relationships
            )
            or "      (none)"
        )
        for o in objects
    )
    return (
        f"Zone id: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Objects to place ({len(objects)}):\n{object_lines}\n\n"
        f"Peers already placed in the run:\n{peer_lines}\n\n"
        "Produce a bbox for every object in a single coherent layout."
    )


# ---------- Step 7: iterative next-object decision --------------------------


class NextObjectOutput(BaseModel):
    done: bool
    object: ObjectSpec | None = None


SYSTEM_NEXT_OBJECT = """\
You are iteratively refining a 3D scene zone inside StarshotBench — a \
head-to-head competitive benchmark where your scene is rendered and \
judged against another LLM's rendering of the same user prompt. The \
zone's defining anchor objects are already placed. You are being \
asked a single question: is ONE MORE object needed to make this zone \
read as complete, or is it already right?

The threshold between "rich" and "busy" is the judgment call that \
costs runs. Stopping too early leaves the zone sparse and forgettable \
— the judge sees empty dead space and moves on. Adding too much \
turns the zone into incoherent clutter — the judge sees noise and \
can't find the focal point. A masterful build knows when to stop.

Err on the side of `done = true`. Prefer "this zone has what it \
needs" over adding clutter. Only add another object if there is a \
clearly missing element a viewer of the final render would notice \
was absent.

If `done = true`, leave `object` null and stop.

If `done = false`, emit EXACTLY ONE object. Make it COUNT — not \
decorative filler, something that noticeably improves the zone's \
legibility or character. Same rules as the bulk decomposition step:
  * Unique `id` (not colliding with any existing node in the scene).
  * `prompt` — a detailed description; used verbatim for text-to-3D.
  * `parent` — either this zone's id, or the id of ANY already-placed \
    node in the scene (typically an object already placed in THIS \
    zone, like a cup on a previously-placed desk).
  * `orientation` — REQUIRED FLOAT (radians, default 0.0). World-frame \
    yaw about +Y. The mesh comes back from the image-to-3D model with \
    its visible front along world +Z; orientation rotates it into the \
    pose you intend. `0.0` = front faces +Z (toward viewer), `π/2` = \
    front faces -X, `π` = front faces -Z (away), `-π/2` = front faces \
    +X. Pick a non-zero value when the object has a clear "front" that \
    should face a specific direction in the scene; leave 0.0 for \
    symmetric objects.
  * `relationships` — REQUIRED to include at least one relationship \
    whose `target` is EXACTLY EQUAL to this emitted object's `parent` \
    field. This is the primary anchor, it is NOT optional, and any \
    object that lacks it is malformed and will be rejected by the \
    validator. Additional relationships may target already-placed \
    objects.

Parent is semantic ("belongs to"), not a spatial containment \
constraint.

GROUND-AWARENESS RULE. If this zone already has a GROUND / SHELL peer \
placed (a mesh describing terrain or enclosure shape — an island \
dome, a crater bowl, a hill, a curved floor, the room's walls and \
floor), any new object whose physical support is that terrain/floor \
MUST set its `parent` to that peer's id (NOT the zone id) and include \
an ON relationship targeting it. The peer's `proxy_shape` in the \
current scene is authoritative for its surface geometry — a \
HEMISPHERE peer is a dome, and the downstream bbox-resolution step \
will compute the dome's surface height at the new object's XZ from \
the peer's proxy formula and rest the object on that surface. You \
are not placing bboxes at this step; just pick the right parent and \
relationships. Only use the zone id as `parent` for objects \
semantically anchored to the zone rather than to a specific surface.

You MAY emit `proxy_shape` on the new object if its silhouette is \
non-rectilinear (SPHERE for a boulder, CAPSULE for a tree trunk, \
HEMISPHERE for a mound) — omit it otherwise. See the emitter's \
decomposition schema for the full vocabulary; the value set is the \
same.

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


class ImagePromptOutput(BaseModel):
    prompt: str


SYSTEM_IMAGE_PROMPT = """\
You produce ONE short noun phrase naming the object to render. The \
phrase is dropped verbatim into a fixed image-prompt template that \
already handles framing, isolation, backdrop, and hitbox shape — your \
output is the SUBJECT of the sentence and nothing more.

The user message will show you the EXACT wrapper your phrase is \
slotted into, with a `<<<SUBJECT>>>` marker where your output goes. \
Read it before writing. Anything the wrapper already says — \
"orthographic front view", the hitbox shape, the white background, \
"capture the entire model" — must NOT appear in your phrase, or it \
will read twice.

Inputs you receive:
  * The original object prompt.
  * The bounding box dimensions in meters (width +X, height +Y, \
    depth +Z) — use them to pick proportion-sensitive adjectives \
    ("long", "tall", "squat") only when natural.
  * The proxy_shape — already encoded into the wrapper's hitbox \
    language; don't repeat it in your phrase.
  * The full image-prompt template with the `<<<SUBJECT>>>` slot \
    visible.
  * The chronological list of prior subject phrases already submitted \
    in this scene — borrow material vocabulary, palette, and stylistic \
    register so the asset coheres with what came before. Do not copy \
    verbatim.

Rules for the phrase:
  * 5-15 words, lower-case, no trailing period.
  * Names the object directly with its defining attributes (species, \
    material, colour, weathering, character). E.g. "a weathered \
    cypress log with bleached bark and patches of moss", "a sleek \
    matte-black office chair with chromed swivel base", "a low \
    domed mossy island with thin muddy lip".
  * NO scene context: drop "half-submerged in", "surrounded by", \
    "resting in", "on the floor of", "nestled among". Keep only \
    intrinsic features of the object itself.
  * NO camera, framing, backdrop, lighting, or rendering instructions \
    — the wrapper handles all of that.
  * NO mention of orthographic, hitbox, prism, rectangle, or any \
    shape language — the wrapper handles silhouette too.

Respond with ONE JSON object matching the schema. The `prompt` field \
holds the noun phrase only.\
"""


_SUBJECT_SLOT = "<<<SUBJECT>>>"


# (3D hitbox term, 2D silhouette term) — slotted into the image-prompt
# wrapper so the prompt's geometric guidance matches the proxy the rest
# of the pipeline uses.
_HITBOX_TERMS: dict[ProxyShape | None, tuple[str, str]] = {
    None: ("rectangular prism", "rectangle"),
    ProxyShape.SPHERE: ("ellipsoid", "ellipse"),
    ProxyShape.CAPSULE: ("vertical capsule", "pill"),
    ProxyShape.HEMISPHERE: ("dome", "dome"),
}


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def wrap_image_prompt(description: str, proxy_shape: ProxyShape | None) -> str:
    """Slot the LLM's noun phrase into the fixed image-generation
    template. Hitbox + silhouette terms are picked from the proxy."""
    hitbox, silhouette = _HITBOX_TERMS[proxy_shape]
    return (
        f"Generate a realistic orthographic front view of {description} "
        f"that roughly can be captured within {_article(hitbox)} {hitbox} "
        "hitbox without bending or deforming the object's natural "
        f"proportions. The object should not fully be in {_article(silhouette)} "
        f"{silhouette} shape unless it is naturally that shape. Prioritize "
        "realism over confinement to the hitbox shape. Render against a "
        "clean, empty white background with no other objects or graphics. "
        "Capture the entire model in the image."
    )


def render_image_prompt(
    *,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
) -> str:
    w, h, d = bbox.size
    template_preview = wrap_image_prompt(_SUBJECT_SLOT, proxy_shape)
    if prior_prompts:
        prior_lines = "\n".join(
            f"  {i + 1}. {p}" for i, p in enumerate(prior_prompts)
        )
        prior_block = (
            f"Prior subject phrases in this scene ({len(prior_prompts)} "
            f"total):\n{prior_lines}"
        )
    else:
        prior_block = (
            "Prior subject phrases in this scene: (none — this is the "
            "first object; you are setting the aesthetic baseline)."
        )
    return (
        f"Original object prompt: {prompt!r}\n"
        f"Bounding box dimensions: width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m\n"
        f"Proxy shape: {_render_proxy_shape(proxy_shape)}\n\n"
        f"Image-prompt template your phrase will be slotted into "
        f"(`{_SUBJECT_SLOT}` is your output):\n  {template_preview}\n\n"
        f"{prior_block}\n\n"
        "Produce ONE short noun phrase naming the subject."
    )


def render_next_object(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, float]],
    prior_attempts: list[tuple[ObjectSpec, str]] | None = None,
) -> str:
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(proxy)} "
            f"orientation={orient:.3f}rad parent={pid!r}"
            for nid, prompt, bbox, pid, proxy, orient in scene
        )
        if scene
        else "  (none)"
    )
    if prior_attempts:
        attempt_lines = "\n".join(
            f"  attempt {i}: emitted {spec.model_dump_json()}\n"
            f"             rejected: {reason}"
            for i, (spec, reason) in enumerate(prior_attempts)
        )
        retry_block = (
            "\n\nPRIOR ATTEMPTS — every object spec below was ALREADY "
            "rejected. Do NOT re-emit the same spec, and do not repeat the "
            "same structural mistake. Treat every listed reason as a hard "
            "constraint you must satisfy this time:\n"
            f"{attempt_lines}\n\n"
            "Either emit a NEW ObjectSpec that fixes every listed reason, or "
            "set done=true. If you emit an object, its `relationships` list "
            "MUST contain at least one item whose `target` is EXACTLY EQUAL "
            "to that object's `parent` field."
        )
    else:
        retry_block = ""
    return (
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "Decide whether another object is needed in this zone. "
        "If yes, emit exactly one ObjectSpec; otherwise set done=true."
        f"{retry_block}"
    )
