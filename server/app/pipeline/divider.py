"""Phase 1 — recursive top-down decomposition into a tree of Nodes.

Flow per node:
  1. Children decomposition (LLM) — atomic? or list of child specs.
  2. Topologically order children by sibling relationships.
  3. For each child: resolve bbox (LLM), then frame decider (LLM) on the
     child itself — realized frames become concrete child Nodes with
     mesh_url set, attached to the child.
  4. Recurse on each child.

Root gets no frame decider: frames belong to decomposed zones, not to the
world-scale canvas.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from app.core.prompts import (
    BboxResolveOutput,
    ChildrenDecompOutput,
    CurveFrameSpec,
    FrameDeciderOutput,
    FrameSpec,
    GeneratedFrameSpec,
    OverallBboxOutput,
    PlaneFrameSpec,
    SYSTEM_BBOX_RESOLVE,
    SYSTEM_CHILDREN_DECOMP,
    SYSTEM_FRAME_DECIDER,
    SYSTEM_OVERALL_BBOX,
    render_bbox_resolve,
    render_children_decomp,
    render_frame_decider,
    render_overall_bbox,
)
from app.core.types import (
    BoundingBox,
    CurveFrame,
    Frame,
    Node,
    PlaneFrame,
)
from app.services import llm, threed
from app.utils import logging
from app.utils.geometry import mesh_aabb, rescale_mesh_to_bbox, write_frame_glb
from app.utils.topology import toposort_children


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


async def _pick_overall_bbox(prompt: str) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OVERALL_BBOX,
        user=render_overall_bbox(prompt),
        output_schema=OverallBboxOutput,
    )
    return out.bbox


async def _decompose(
    *, prompt: str, bbox: BoundingBox, parent_id: str,
) -> ChildrenDecompOutput:
    return await llm.call_llm(
        system=SYSTEM_CHILDREN_DECOMP,
        user=render_children_decomp(prompt=prompt, bbox=bbox, parent_id=parent_id),
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


async def _decide_frames(*, prompt: str, bbox: BoundingBox) -> list[FrameSpec]:
    out = await llm.call_llm(
        system=SYSTEM_FRAME_DECIDER,
        user=render_frame_decider(prompt=prompt, bbox=bbox),
        output_schema=FrameDeciderOutput,
    )
    return list(out.frames) if out.needs_frame else []


async def _attach_frames(
    *, target: Node, runs_dir: Path, run_id: str,
) -> None:
    specs = await _decide_frames(prompt=target.prompt, bbox=target.bbox)
    for spec in specs:
        frame_node = await _realize_frame_node(
            spec=spec, parent_bbox=target.bbox, runs_dir=runs_dir, run_id=run_id,
        )
        target.children.append(frame_node)


async def _realize_frame_node(
    *,
    spec: FrameSpec,
    parent_bbox: BoundingBox,
    runs_dir: Path,
    run_id: str,
) -> Node:
    frames_dir = runs_dir / run_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(spec, PlaneFrameSpec):
        frame: Frame = PlaneFrame(
            id=spec.id, origin=spec.origin, u_axis=spec.u_axis, v_axis=spec.v_axis,
        )
        path = frames_dir / f"{spec.id}.glb"
        _, aabb = write_frame_glb(frame, path)
        prompt = f"plane frame {spec.id}"
    elif isinstance(spec, CurveFrameSpec):
        frame = CurveFrame(
            id=spec.id, control_points=spec.control_points, height=spec.height,
        )
        path = frames_dir / f"{spec.id}.glb"
        _, aabb = write_frame_glb(frame, path)
        prompt = f"curve frame {spec.id}"
    else:
        assert isinstance(spec, GeneratedFrameSpec)
        raw = frames_dir / f"{spec.id}.raw.glb"
        path = frames_dir / f"{spec.id}.glb"
        logging.log("hunyuan.submit", id=spec.id, prompt=spec.prompt)
        await threed.generate_mesh(spec.prompt, output_path=raw)
        raw_mesh: trimesh.Trimesh = trimesh.load(raw, force="mesh")  # pyright: ignore[reportAssignmentType]
        rescaled = rescale_mesh_to_bbox(raw_mesh, parent_bbox)
        rescaled.export(path, file_type="glb")
        aabb = mesh_aabb(rescaled)
        prompt = spec.prompt

    url = _artifact_url(runs_dir, path)
    logging.emit_bbox(spec.id, aabb)
    logging.emit_model(spec.id, artifact_kind="frame", url=url)
    return Node(id=spec.id, prompt=prompt, bbox=aabb, mesh_url=url)


async def _build(*, node: Node, runs_dir: Path, run_id: str) -> None:
    decomp = await _decompose(
        prompt=node.prompt, bbox=node.bbox, parent_id=node.id,
    )
    logging.log(
        "divider.decompose",
        node=node.id,
        is_atomic=decomp.is_atomic,
        children=[c.id for c in decomp.children],
    )

    if decomp.is_atomic:
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
        child_bbox = await _resolve_child_bbox(
            parent=node,
            siblings=placed,
            child_id=spec.id,
            child_prompt=spec.prompt,
            relationships=spec.relationships,
        )
        logging.emit_bbox(spec.id, child_bbox)
        child = Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=child_bbox,
            relationships=list(spec.relationships),
        )
        await _attach_frames(target=child, runs_dir=runs_dir, run_id=run_id)
        placed.append(child)

    node.children.extend(placed)

    for child in placed:
        await _build(node=child, runs_dir=runs_dir, run_id=run_id)


async def run(
    *, run_id: str, prompt: str, model: str, runs_dir: Path,
) -> Node:
    llm.set_model(model)
    bbox = await _pick_overall_bbox(prompt)
    logging.emit_bbox("root", bbox)
    root = Node(id="root", prompt=prompt, bbox=bbox)
    await _build(node=root, runs_dir=runs_dir, run_id=run_id)
    return root
