"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Flow per node:
  1. ZONE PLAN (LLM) — high-level character/intent for this zone. Pure
     planning, no structural decisions, no individual objects. Runs for
     EVERY zone, root included; the root's plan IS the scene plan.
  2. (root only) Overall bounding box (LLM) — sizes the canvas to match
     the silhouette implied by the root's plan.
  3. ZONE DECOMPOSE (LLM) — decides atomic vs subdivides; if subdivides,
     emits one (id, prompt) seed per child zone. Children's plans are
     authored later by their own ZONE PLAN call when `_build` recurses.
  4. Children materialization (LLM) — runs only when the decomposer said
     subdivides. Assigns each seed a proxy_shape and the relationships
     anchoring it inside the parent. The sibling-relationship DAG is
     checked for cycles; any cycle is logged and accepted (advisory).
  5. Batch-resolve a bbox for EVERY child in one LLM call.
  6. Hand each placed child to Phase 2 generation for its encapsulating
     geometry (walls, moat, fence, etc.) as objects.
  7. Recurse on each child. Children arrive with `plan=None`; step 1
     authors their plan fresh.

Atomic leaves skip the recursion and are handed to Phase 2 generation for
anchor-object population. Root gets no encapsulating pass; it does get a
final negative-space pass at the end of the run.
"""

from __future__ import annotations

from pathlib import Path

from app.core.prompts import (
    BboxBatchOutput,
    ChildNodeSpec,
    ChildrenDecompOutput,
    OverallBboxOutput,
    SubzoneSeed,
    SYSTEM_CHILDREN_DECOMP,
    SYSTEM_OVERALL_BBOX,
    SYSTEM_ZONE_BBOX_BATCH,
    SYSTEM_ZONE_DECOMPOSE,
    SYSTEM_ZONE_PLAN,
    ZoneDecomposeOutput,
    ZonePlanOutput,
    render_children_decomp,
    render_overall_bbox,
    render_zone_bbox_batch,
    render_zone_decompose,
    render_zone_plan,
)
from app.core.types import BoundingBox, Node
from app.pipeline import generation
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_sibling_relationships_acyclic


async def _pick_overall_bbox(prompt: str, scene_plan: str) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OVERALL_BBOX,
        user=render_overall_bbox(prompt, scene_plan),
        output_schema=OverallBboxOutput,
    )
    return out.bbox


def _prior_zones(all_nodes: list[Node]) -> list[tuple[str, str, str, str]]:
    """Zones already declared AND planned, excluding the root (which is
    surfaced separately as the scene plan). Used by the structural
    materialization step, which still wants the full lateral context."""
    out: list[tuple[str, str, str, str]] = []
    for n in all_nodes:
        if n.mesh_url is not None:
            continue
        if n.parent_id is None or n.plan is None:
            continue
        out.append((n.id, n.prompt, n.plan, n.parent_id))
    return out


def _ancestors(node: Node, all_nodes: list[Node]) -> list[tuple[str, str, str]]:
    """Walk parent_id pointers up to the root, then return root-first →
    parent-of-`node`, excluding `node` itself. Each tuple is (id, prompt,
    plan). Empty for the root."""
    by_id = {n.id: n for n in all_nodes}
    chain: list[Node] = []
    parent_id = node.parent_id
    while parent_id is not None:
        parent = by_id[parent_id]
        chain.append(parent)
        parent_id = parent.parent_id
    chain.reverse()
    out: list[tuple[str, str, str]] = []
    for a in chain:
        assert a.plan is not None, f"ancestor {a.id} must be planned"
        out.append((a.id, a.prompt, a.plan))
    return out


def _generated_objects(all_nodes: list[Node]) -> list[tuple[str, str, str | None]]:
    """Every concrete (mesh-bearing) node placed so far, in declaration
    order. Used to ground planning context in what the scene actually
    looks like, not just what's been promised."""
    return [
        (n.id, n.prompt, n.parent_id) for n in all_nodes if n.mesh_url is not None
    ]


async def _plan_zone(
    *,
    zone_id: str,
    zone_prompt: str,
    ancestors: list[tuple[str, str, str]],
    objects: list[tuple[str, str, str | None]],
) -> str:
    """Author the high-level plan for a zone. Works for any zone (root or
    nested) — the root just passes empty ancestors. Returns the plan string."""
    out = await llm.call_llm(
        system=SYSTEM_ZONE_PLAN,
        user=render_zone_plan(
            zone_id=zone_id,
            zone_prompt=zone_prompt,
            ancestors=ancestors,
            objects=objects,
        ),
        output_schema=ZonePlanOutput,
    )
    return out.plan


async def _decompose_zone(
    *, node: Node, all_nodes: list[Node],
) -> ZoneDecomposeOutput:
    """Decide atomic vs subdivides for an already-planned zone, and (if
    subdividing) emit (id, prompt) seeds for each child."""
    assert node.plan is not None, "zone must be planned before decomposition"
    return await llm.call_llm(
        system=SYSTEM_ZONE_DECOMPOSE,
        user=render_zone_decompose(
            zone_id=node.id,
            zone_prompt=node.prompt,
            zone_bbox=node.bbox,
            zone_plan=node.plan,
            ancestors=_ancestors(node, all_nodes),
            objects=_generated_objects(all_nodes),
        ),
        output_schema=ZoneDecomposeOutput,
    )


async def _materialize_children(
    *,
    node: Node,
    subzones: list[SubzoneSeed],
    all_nodes: list[Node],
) -> ChildrenDecompOutput:
    """Materialize each subzone seed into a ChildNodeSpec by assigning a
    proxy_shape and relationships. Atomicity, ids, and prompts are already
    authored upstream — this step authors only structure."""
    root = all_nodes[0]
    assert root.plan is not None, "root.plan must be set before materialization"
    assert node.plan is not None, "zone plan must be set before materialization"
    return await llm.call_llm(
        system=SYSTEM_CHILDREN_DECOMP,
        user=render_children_decomp(
            prompt=node.prompt,
            bbox=node.bbox,
            plan=node.plan,
            subzones=subzones,
            parent_id=node.id,
            scene_prompt=root.prompt,
            scene_plan=root.plan,
            prior_zones=_prior_zones(all_nodes),
        ),
        output_schema=ChildrenDecompOutput,
    )


async def _resolve_child_bboxes_batch(
    *, parent: Node, children: list[ChildNodeSpec],
) -> dict[str, BoundingBox]:
    out = await llm.call_llm(
        system=SYSTEM_ZONE_BBOX_BATCH,
        user=render_zone_bbox_batch(
            parent_id=parent.id,
            parent_bbox=parent.bbox,
            children=children,
        ),
        output_schema=BboxBatchOutput,
    )
    return {a.id: a.bbox for a in out.assignments}


async def _build(
    *, node: Node, runs_dir: Path, run_id: str, all_nodes: list[Node],
) -> None:
    if node.plan is None:
        logging.emit_step(node.id, "planning")
        plan = await _plan_zone(
            zone_id=node.id,
            zone_prompt=node.prompt,
            ancestors=_ancestors(node, all_nodes),
            objects=_generated_objects(all_nodes),
        )
        # Node is frozen; swap the planned copy into the shared registry so
        # later steps read the plan rather than the stale pre-plan version.
        planned = node.model_copy(update={"plan": plan})
        idx = all_nodes.index(node)
        all_nodes[idx] = planned
        node = planned
        logging.log("divider.zone_plan", node=node.id, plan=plan)

    logging.emit_step(node.id, "decomposing")
    decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
    logging.log(
        "divider.zone_decompose",
        node=node.id,
        is_atomic=decomp.is_atomic,
        subzones=[s.model_dump() for s in decomp.subzones],
    )

    if decomp.is_atomic:
        logging.emit_step(node.id, "generating_anchor")
        await generation.run(
            zone=node, runs_dir=runs_dir, run_id=run_id,
            scenario="anchor", all_nodes=all_nodes,
        )
        logging.emit_step(node.id, "done")
        return

    children_decomp = await _materialize_children(
        node=node, subzones=decomp.subzones, all_nodes=all_nodes,
    )
    logging.log(
        "divider.decompose",
        node=node.id,
        children=[{"id": c.id, "prompt": c.prompt} for c in children_decomp.children],
    )

    try:
        validate_sibling_relationships_acyclic(children_decomp.children)
    except ValueError as e:
        logging.log(
            "divider.validate.relationships.accept_invalid",
            node=node.id, reason=str(e),
        )

    logging.emit_step(node.id, "resolving_bboxes", parent=node.id)
    bboxes = await _resolve_child_bboxes_batch(
        parent=node, children=children_decomp.children,
    )

    placed: list[Node] = []
    for spec in children_decomp.children:
        child_bbox = bboxes[spec.id]
        logging.emit_bbox(
            spec.id, child_bbox,
            parent_id=node.id, prompt=spec.prompt, kind="zone",
            proxy_shape=spec.proxy_shape,
        )
        child = Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=child_bbox,
            proxy_shape=spec.proxy_shape,
            relationships=list(spec.relationships),
            parent_id=node.id,
            plan=None,
        )
        placed.append(child)
        all_nodes.append(child)

    for child in placed:
        logging.emit_step(child.id, "generating_frame")
        await generation.run(
            zone=child, runs_dir=runs_dir, run_id=run_id,
            scenario="encapsulating", all_nodes=all_nodes,
        )

    for child in placed:
        await _build(
            node=child, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes,
        )
    logging.emit_step(node.id, "done")


async def run(
    *, run_id: str, prompt: str, model: str, runs_dir: Path,
) -> Node:
    llm.set_model(model)
    logging.emit_step("root", "planning")
    scene_plan = await _plan_zone(
        zone_id="root", zone_prompt=prompt, ancestors=[], objects=[],
    )
    logging.log("divider.zone_plan", node="root", plan=scene_plan)
    bbox = await _pick_overall_bbox(prompt, scene_plan)
    logging.emit_bbox("root", bbox, parent_id=None, prompt=prompt, kind="zone")
    root = Node(
        id="root", prompt=prompt, bbox=bbox, parent_id=None, plan=scene_plan,
    )
    all_nodes: list[Node] = [root]
    await _build(node=root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run(
        zone=root, runs_dir=runs_dir, run_id=run_id,
        scenario="negative-space", all_nodes=all_nodes,
    )
    logging.emit_step(root.id, "done")
    return root
