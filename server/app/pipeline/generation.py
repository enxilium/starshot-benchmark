"""Phase 2 — populate a zone with objects via Hunyuan.

Two scenarios:
  * "anchor"        — atomic leaf zone; generate defining objects, then
                      iterate (ask "is another object needed?") until the
                      LLM says done.
  * "encapsulating" — non-atomic zone about to be decomposed further;
                      generate the geometry that encapsulates it (walls,
                      moat, fence). One shot, no loop. Replaces the old
                      frame step.

Both share: decompose -> validate relationships -> topo-sort -> per-object
bbox resolution (LLM) -> validate overlap against same-parent peers ->
parallel Hunyuan + rescaling. Events stream via SSE as each mesh lands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import trimesh

from app.core.prompts import (
    BboxResolveOutput,
    NextObjectOutput,
    ObjectDecompOutput,
    ObjectSpec,
    SYSTEM_NEXT_OBJECT,
    SYSTEM_OBJECT_BBOX_RESOLVE,
    SYSTEM_OBJECT_DECOMP,
    render_next_object,
    render_object_bbox_resolve,
    render_object_decomp,
)
from app.core.types import BoundingBox, Node
from app.services import llm, threed
from app.utils import logging
from app.utils.geometry import rescale_mesh_to_bbox
from app.utils.topology import toposort_children, validate_object_relationships


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


def _bboxes_overlap(a: BoundingBox, b: BoundingBox, tol: float = 0.01) -> bool:
    a_min, a_max = a.min_corner, a.max_corner
    b_min, b_max = b.min_corner, b.max_corner
    for i in range(3):
        if a_max[i] - b_min[i] <= tol or b_max[i] - a_min[i] <= tol:
            return False
    return True


async def _decompose_objects(
    *, zone: Node, scenario: Literal["anchor", "encapsulating"],
) -> list[ObjectSpec]:
    out = await llm.call_llm(
        system=SYSTEM_OBJECT_DECOMP,
        user=render_object_decomp(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_bbox=zone.bbox,
            scenario=scenario,
        ),
        output_schema=ObjectDecompOutput,
    )
    return list(out.objects)


async def _next_object(*, zone: Node, all_nodes: list[Node]) -> NextObjectOutput:
    scene = [(n.id, n.prompt, n.bbox, n.parent_id) for n in all_nodes]
    return await llm.call_llm(
        system=SYSTEM_NEXT_OBJECT,
        user=render_next_object(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_bbox=zone.bbox,
            scene=scene,
        ),
        output_schema=NextObjectOutput,
    )


def _lookup_parent(
    spec: ObjectSpec, zone: Node, known: list[Node],
) -> tuple[BoundingBox, Literal["zone", "object"]]:
    if spec.parent == zone.id:
        return zone.bbox, "zone"
    for n in known:
        if n.id == spec.parent:
            return n.bbox, "object"
    raise ValueError(f"parent {spec.parent!r} of object {spec.id!r} not found in known nodes")


async def _resolve_object_bbox(
    *,
    zone: Node,
    spec: ObjectSpec,
    parent_bbox: BoundingBox,
    parent_kind: Literal["zone", "object"],
    peers: list[Node],
) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OBJECT_BBOX_RESOLVE,
        user=render_object_bbox_resolve(
            zone_id=zone.id,
            zone_bbox=zone.bbox,
            object_id=spec.id,
            object_prompt=spec.prompt,
            parent_id=spec.parent,
            parent_kind=parent_kind,
            parent_bbox=parent_bbox,
            peers=[(p.id, p.bbox, p.parent_id) for p in peers],
            relationships=list(spec.relationships),
        ),
        output_schema=BboxResolveOutput,
    )
    return out.bbox


def _check_sibling_overlap(
    *, candidate_id: str, candidate_parent: str, candidate_bbox: BoundingBox,
    peers: list[Node],
) -> None:
    for p in peers:
        if p.id == candidate_id:
            continue
        if p.parent_id != candidate_parent:
            continue
        if _bboxes_overlap(candidate_bbox, p.bbox):
            raise ValueError(
                f"object {candidate_id!r} bbox overlaps sibling {p.id!r} "
                f"(shared parent {candidate_parent!r})"
            )


async def _resolve_and_generate(
    *,
    specs: list[ObjectSpec],
    zone: Node,
    all_nodes: list[Node],
    scenario: Literal["anchor", "encapsulating"],
    runs_dir: Path,
    run_id: str,
) -> list[Node]:
    try:
        validate_object_relationships(
            specs, zone_id=zone.id,
            existing_ids={n.id for n in all_nodes},
        )
    except ValueError as e:
        logging.log("generation.validate.relationships.fail", reason=str(e))
        raise

    ordered = toposort_children(specs)

    resolved: list[Node] = []
    for spec in ordered:
        known = all_nodes + resolved
        bbox: BoundingBox | None = None
        for attempt in range(2):
            parent_bbox, parent_kind = _lookup_parent(spec, zone, known)
            candidate = await _resolve_object_bbox(
                zone=zone, spec=spec,
                parent_bbox=parent_bbox, parent_kind=parent_kind,
                peers=known,
            )
            # Encapsulating geometry (walls, floor, ceiling, moat) is meant
            # to share boundaries with its siblings — skip the overlap check.
            if scenario == "encapsulating":
                bbox = candidate
                break
            try:
                _check_sibling_overlap(
                    candidate_id=spec.id,
                    candidate_parent=spec.parent,
                    candidate_bbox=candidate,
                    peers=known,
                )
                bbox = candidate
                break
            except ValueError as e:
                logging.log("generation.validate.bboxes.fail",
                            reason=str(e), id=spec.id, attempt=attempt)
                if attempt == 1:
                    raise
        assert bbox is not None
        logging.emit_bbox(spec.id, bbox)
        resolved.append(Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=bbox,
            relationships=list(spec.relationships),
            parent_id=spec.parent,
        ))

    return await _generate_parallel(resolved=resolved, runs_dir=runs_dir, run_id=run_id)


async def _generate_parallel(
    *, resolved: list[Node], runs_dir: Path, run_id: str,
) -> list[Node]:
    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)

    async def _one(node: Node) -> Node:
        raw = objs_dir / f"{node.id}.raw.glb"
        path = objs_dir / f"{node.id}.glb"
        logging.log("hunyuan.submit", id=node.id, prompt=node.prompt)
        await threed.generate_mesh(node.prompt, output_path=raw)
        scene = trimesh.load(raw)
        rescaled = rescale_mesh_to_bbox(scene, node.bbox)
        rescaled.export(path, file_type="glb")
        url = _artifact_url(runs_dir, path)
        logging.emit_model(node.id, artifact_kind="object", url=url)
        return node.model_copy(update={"mesh_url": url})

    return await asyncio.gather(*[_one(n) for n in resolved])


async def run(
    *,
    zone: Node,
    runs_dir: Path,
    run_id: str,
    scenario: Literal["anchor", "encapsulating"],
    all_nodes: list[Node],
) -> None:
    specs = await _decompose_objects(zone=zone, scenario=scenario)
    logging.log(
        "generation.decompose",
        zone=zone.id,
        scenario=scenario,
        objects=[s.id for s in specs],
    )

    placed = await _resolve_and_generate(
        specs=specs, zone=zone, all_nodes=all_nodes, scenario=scenario,
        runs_dir=runs_dir, run_id=run_id,
    )
    all_nodes.extend(placed)

    if scenario != "anchor":
        return

    while True:
        decision = await _next_object(zone=zone, all_nodes=all_nodes)
        if decision.done or decision.object is None:
            logging.log("generation.next.done", zone=zone.id)
            return
        logging.log("generation.next", zone=zone.id, id=decision.object.id)
        new_nodes = await _resolve_and_generate(
            specs=[decision.object], zone=zone, all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir, run_id=run_id,
        )
        all_nodes.extend(new_nodes)
