"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Flow per node:
  1. Plan this zone (LLM) — produces the zone's character/intent plan,
     decides atomic vs subdivides, and (when subdividing) authors a full
     plan for each subzone in the same call.
  2. Children materialization (LLM) — runs only when the planner said
     subdivides. Takes the planner's subzone seeds and assigns each a
     proxy_shape and the relationships that anchor it inside the parent.
     The sibling-relationship DAG is checked for cycles; any cycle is
     logged and accepted (advisory).
  3. Batch-resolve a bbox for EVERY child in one LLM call, so the set is
     chosen as a mutually consistent layout. The LLM is trusted; nothing
     is validated or retried here.
  4. Hand each placed child to Phase 2 generation for its encapsulating
     geometry (walls, moat, fence, etc.) as objects.
  5. Recurse on each child. Each child Node arrives with its `plan`
     pre-seeded from step 1.

Atomic leaves skip the recursion and are handed to Phase 2 generation for
anchor-object population. Root gets no generation pass: objects belong to
decomposed zones, not to the world-scale canvas.
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
    SYSTEM_ZONE_PLAN,
    ZonePlanOutput,
    render_children_decomp,
    render_overall_bbox,
    render_zone_bbox_batch,
    render_zone_plan,
)
from app.core.types import BoundingBox, Node
from app.pipeline import generation
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_sibling_relationships_acyclic


async def _pick_overall_bbox(prompt: str) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OVERALL_BBOX,
        user=render_overall_bbox(prompt),
        output_schema=OverallBboxOutput,
    )
    return out.bbox


def _prior_zones(all_nodes: list[Node]) -> list[tuple[str, str, str, str]]:
    """Zones already declared AND planned, excluding the root (which is
    surfaced separately as the scene plan)."""
    out: list[tuple[str, str, str, str]] = []
    for n in all_nodes:
        if n.mesh_url is not None:
            continue
        if n.parent_id is None or n.plan is None:
            continue
        out.append((n.id, n.prompt, n.plan, n.parent_id))
    return out


async def _plan_zone(*, node: Node, all_nodes: list[Node]) -> ZonePlanOutput:
    """Plan the given zone — its own plan, the atomic decision, and (when
    subdividing) a fully-authored plan for each subzone, in one LLM call.
    Root is planned with no prior context; non-root zones receive the scene
    plan + every already-planned zone."""
    is_root = node.parent_id is None
    if is_root:
        scene_prompt = None
        scene_plan = None
        prior_zones: list[tuple[str, str, str, str]] = []
    else:
        root = all_nodes[0]
        assert root.plan is not None, "root must be planned before any child"
        scene_prompt = root.prompt
        scene_plan = root.plan
        prior_zones = _prior_zones(all_nodes)
    return await llm.call_llm(
        system=SYSTEM_ZONE_PLAN,
        user=render_zone_plan(
            zone_id=node.id,
            zone_prompt=node.prompt,
            zone_bbox=node.bbox,
            scene_prompt=scene_prompt,
            scene_plan=scene_plan,
            prior_zones=prior_zones,
            inherited_plan=node.plan,
        ),
        output_schema=ZonePlanOutput,
    )


async def _materialize_children(
    *,
    node: Node,
    subzones: list[SubzoneSeed],
    all_nodes: list[Node],
) -> ChildrenDecompOutput:
    """Materialize each pre-planned subzone seed into a ChildNodeSpec by
    assigning a proxy_shape and relationships. Atomicity, ids, prompts, and
    plans are already authored upstream — this step authors only structure."""
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
    logging.emit_step(node.id, "planning")
    plan_out = await _plan_zone(node=node, all_nodes=all_nodes)
    # Node is frozen; swap the planned copy into the shared registry so
    # later steps read the plan rather than the stale pre-plan version.
    # Subzone seeds are stashed on the node as `subzone_plan` (JSON) for
    # traceability; the live seeds also flow directly into materialization
    # below without needing to round-trip through the registry.
    subzone_plan_blob = (
        "\n\n".join(
            f"- {s.id}: {s.prompt}\n  plan: {s.plan}" for s in plan_out.subzones
        )
        if plan_out.subzones
        else None
    )
    planned = node.model_copy(
        update={"plan": plan_out.plan, "subzone_plan": subzone_plan_blob},
    )
    idx = all_nodes.index(node)
    all_nodes[idx] = planned
    node = planned
    logging.log(
        "divider.zone_plan",
        node=node.id,
        plan=plan_out.plan,
        is_atomic=plan_out.is_atomic,
        subzones=[s.model_dump() for s in plan_out.subzones],
    )

    if plan_out.is_atomic:
        logging.emit_step(node.id, "generating_anchor")
        await generation.run(
            zone=node, runs_dir=runs_dir, run_id=run_id,
            scenario="anchor", all_nodes=all_nodes,
        )
        logging.emit_step(node.id, "done")
        return

    seed_by_id = {s.id: s for s in plan_out.subzones}

    logging.emit_step(node.id, "decomposing")
    decomp = await _materialize_children(
        node=node, subzones=plan_out.subzones, all_nodes=all_nodes,
    )
    logging.log(
        "divider.decompose",
        node=node.id,
        children=[{"id": c.id, "prompt": c.prompt} for c in decomp.children],
    )

    try:
        validate_sibling_relationships_acyclic(decomp.children)
    except ValueError as e:
        logging.log(
            "divider.validate.relationships.accept_invalid",
            node=node.id, reason=str(e),
        )

    logging.emit_step(node.id, "resolving_bboxes", parent=node.id)
    bboxes = await _resolve_child_bboxes_batch(parent=node, children=decomp.children)

    placed: list[Node] = []
    for spec in decomp.children:
        child_bbox = bboxes[spec.id]
        logging.emit_bbox(
            spec.id, child_bbox,
            parent_id=node.id, prompt=spec.prompt, kind="zone",
            proxy_shape=spec.proxy_shape,
        )
        seed = seed_by_id.get(spec.id)
        child = Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=child_bbox,
            proxy_shape=spec.proxy_shape,
            relationships=list(spec.relationships),
            parent_id=node.id,
            plan=seed.plan if seed is not None else None,
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
    bbox = await _pick_overall_bbox(prompt)
    logging.emit_bbox("root", bbox, parent_id=None, prompt=prompt, kind="zone")
    root = Node(id="root", prompt=prompt, bbox=bbox, parent_id=None)
    all_nodes: list[Node] = [root]
    await _build(node=root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run(
        zone=root, runs_dir=runs_dir, run_id=run_id,
        scenario="negative-space", all_nodes=all_nodes,
    )
    logging.emit_step(root.id, "done")
    return root
