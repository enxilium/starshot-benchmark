"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Flow per node:
  1. Children decomposition (LLM) — atomic? or list of child zone specs.
  2. Topologically order children by sibling relationships.
  3. For each child zone:
     a. Resolve bbox (LLM).
     b. Validate bbox (contained in parent, no sibling overlap).
     c. Hand to Phase 2 generation to produce its encapsulating geometry
        (walls, moat, fence, etc.) as objects.
  4. Recurse on each child.

Atomic leaves skip the recursion and are handed to Phase 2 generation for
anchor-object population.

Root gets no generation pass: objects belong to decomposed zones, not to
the world-scale canvas.
"""

from __future__ import annotations

from pathlib import Path

from app.core.prompts import (
    BboxResolveOutput,
    ChildrenDecompOutput,
    OverallBboxOutput,
    SYSTEM_BBOX_RESOLVE,
    SYSTEM_CHILDREN_DECOMP,
    SYSTEM_OVERALL_BBOX,
    render_bbox_resolve,
    render_children_decomp,
    render_overall_bbox,
)
from app.core.types import BoundingBox, Node
from app.pipeline import generation
from app.services import llm
from app.utils import logging
from app.utils.topology import toposort_children


async def _pick_overall_bbox(prompt: str) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OVERALL_BBOX,
        user=render_overall_bbox(prompt),
        output_schema=OverallBboxOutput,
    )
    return out.bbox


async def _decompose(
    *,
    prompt: str,
    bbox: BoundingBox,
    parent_id: str,
    all_nodes: list[Node],
) -> ChildrenDecompOutput:
    # all_nodes[0] is always root. Zones have mesh_url is None; objects set it.
    scene_prompt = all_nodes[0].prompt
    prior_zones = [
        (n.id, n.prompt, n.plan, n.parent_id)
        for n in all_nodes
        if n.mesh_url is None
    ]
    return await llm.call_llm(
        system=SYSTEM_CHILDREN_DECOMP,
        user=render_children_decomp(
            prompt=prompt,
            bbox=bbox,
            parent_id=parent_id,
            scene_prompt=scene_prompt,
            prior_zones=prior_zones,
        ),
        output_schema=ChildrenDecompOutput,
    )


async def _resolve_child_bbox(
    *,
    parent: Node,
    siblings: list[Node],
    child_id: str,
    child_prompt: str,
    relationships,
) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_BBOX_RESOLVE,
        user=render_bbox_resolve(
            parent_id=parent.id,
            parent_bbox=parent.bbox,
            child_id=child_id,
            child_prompt=child_prompt,
            siblings=[(s.id, s.bbox) for s in siblings],
            relationships=relationships,
        ),
        output_schema=BboxResolveOutput,
    )
    return out.bbox


def _bboxes_overlap(a: BoundingBox, b: BoundingBox, tol: float = 0.01) -> bool:
    a_min, a_max = a.min_corner, a.max_corner
    b_min, b_max = b.min_corner, b.max_corner
    for i in range(3):
        if a_max[i] - b_min[i] <= tol or b_max[i] - a_min[i] <= tol:
            return False
    return True


def _contained(child: BoundingBox, parent: BoundingBox, tol: float = 0.01) -> bool:
    c_min, c_max = child.min_corner, child.max_corner
    p_min, p_max = parent.min_corner, parent.max_corner
    for i in range(3):
        if c_min[i] < p_min[i] - tol or c_max[i] > p_max[i] + tol:
            return False
    return True


def _validate_zone_bboxes(
    *,
    child_id: str,
    child_bbox: BoundingBox,
    parent_bbox: BoundingBox,
    siblings: list[Node],
) -> None:
    if not _contained(child_bbox, parent_bbox):
        raise ValueError(f"zone {child_id!r} bbox is not contained in parent bbox")
    for s in siblings:
        if _bboxes_overlap(child_bbox, s.bbox):
            raise ValueError(
                f"zone {child_id!r} bbox overlaps sibling {s.id!r}"
            )


async def _build(
    *, node: Node, runs_dir: Path, run_id: str, all_nodes: list[Node],
) -> None:
    decomp = await _decompose(
        prompt=node.prompt, bbox=node.bbox, parent_id=node.id,
        all_nodes=all_nodes,
    )
    logging.log(
        "divider.decompose",
        node=node.id,
        is_atomic=decomp.is_atomic,
        children=[c.id for c in decomp.children],
    )

    if decomp.is_atomic:
        await generation.run(
            zone=node, runs_dir=runs_dir, run_id=run_id,
            scenario="anchor", all_nodes=all_nodes,
        )
        return

    ordered_children = toposort_children(decomp.children)
    if [c.id for c in ordered_children] != [c.id for c in decomp.children]:
        logging.log(
            "divider.toposort",
            node=node.id,
            order=[c.id for c in ordered_children],
        )

    placed: list[Node] = []
    for spec in ordered_children:
        child_bbox: BoundingBox | None = None
        for attempt in range(2):
            candidate = await _resolve_child_bbox(
                parent=node,
                siblings=placed,
                child_id=spec.id,
                child_prompt=spec.prompt,
                relationships=spec.relationships,
            )
            try:
                _validate_zone_bboxes(
                    child_id=spec.id,
                    child_bbox=candidate,
                    parent_bbox=node.bbox,
                    siblings=placed,
                )
                child_bbox = candidate
                break
            except ValueError as e:
                logging.log(
                    "divider.validate.bboxes.fail",
                    reason=str(e), id=spec.id, attempt=attempt,
                )
                if attempt == 1:
                    raise
        assert child_bbox is not None
        logging.emit_bbox(spec.id, child_bbox)
        child = Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=child_bbox,
            relationships=list(spec.relationships),
            parent_id=node.id,
            plan=spec.plan,
        )
        placed.append(child)
        all_nodes.append(child)

        await generation.run(
            zone=child, runs_dir=runs_dir, run_id=run_id,
            scenario="encapsulating", all_nodes=all_nodes,
        )

    for child in placed:
        await _build(
            node=child, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes,
        )


async def run(
    *, run_id: str, prompt: str, model: str, runs_dir: Path,
) -> Node:
    llm.set_model(model)
    bbox = await _pick_overall_bbox(prompt)
    logging.emit_bbox("root", bbox)
    root = Node(id="root", prompt=prompt, bbox=bbox, parent_id=None)
    all_nodes: list[Node] = [root]
    await _build(node=root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
    return root
