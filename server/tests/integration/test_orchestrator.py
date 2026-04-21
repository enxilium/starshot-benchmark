from __future__ import annotations

import contextlib
import json
from pathlib import Path

import trimesh

from app.pipeline.step03_orchestrator import run as run_pipeline
from tests.integration.conftest import write_fixture


def overall_bbox_output() -> dict:
    return {
        "bbox": {"origin": [0, 0, 0], "dimensions": [10, 3, 10]},
        "reasoning": "small two-room floor plan",
    }


def root_breakdown() -> dict:
    return {
        "is_atomic": False,
        "subscenes": [
            {
                "scope_id": "root.bathroom",
                "prompt": "a small bathroom",
                "bbox": {"origin": [0, 0, 0], "dimensions": [3, 3, 4]},
                "high_level_plan": "toilet + sink",
            },
            {
                "scope_id": "root.bedroom",
                "prompt": "a cozy bedroom",
                "bbox": {"origin": [3, 0, 0], "dimensions": [7, 3, 10]},
                "high_level_plan": "bed + nightstand",
            },
        ],
    }


def atomic_breakdown() -> dict:
    return {"is_atomic": True, "subscenes": []}


def frame_floor(obj_id: str, width: float, depth: float) -> dict:
    return {
        "kind": "plane",
        "id": obj_id,
        "origin": [0, 0, 0],
        "u_axis": [width, 0, 0],
        "v_axis": [0, 0, depth],
    }


def no_frame() -> dict:
    return {"needs_frame": False, "frames": [], "reasoning": "none"}


def bathroom_floor() -> dict:
    return {
        "needs_frame": True,
        "reasoning": "bathroom floor",
        "frames": [frame_floor("bathroom_floor", 3, 4)],
    }


def bedroom_floor() -> dict:
    return {
        "needs_frame": True,
        "reasoning": "bedroom floor",
        "frames": [frame_floor("bedroom_floor", 7, 10)],
    }


def phase2_anchor_bathroom() -> dict:
    return {
        "objects": [
            {"id": "toilet", "prompt": "a white porcelain toilet"},
            {"id": "sink", "prompt": "a small pedestal sink"},
        ],
        "relationships": [
            {
                "subject": "toilet",
                "kind": "ATTACHED",
                "target": "bathroom_floor",
                "attachment": [0.8, 0.8],
            },
            {
                "subject": "sink",
                "kind": "ATTACHED",
                "target": "bathroom_floor",
                "attachment": [0.2, 0.2],
            },
        ],
    }


def phase2_bboxes_bathroom() -> dict:
    return {
        "assignments": [
            {
                "object_id": "toilet",
                "bbox": {"origin": [2.2, 0, 2.8], "dimensions": [0.6, 0.6, 0.8]},
            },
            {
                "object_id": "sink",
                "bbox": {"origin": [0.3, 0, 0.3], "dimensions": [0.6, 0.8, 0.6]},
            },
        ]
    }


def phase2_anchor_bedroom() -> dict:
    return {
        "objects": [
            {"id": "bed", "prompt": "a queen-size bed"},
            {"id": "nightstand", "prompt": "a small nightstand"},
        ],
        "relationships": [
            {
                "subject": "bed",
                "kind": "ATTACHED",
                "target": "bedroom_floor",
                "attachment": [0.5, 0.5],
            },
            {"subject": "nightstand", "kind": "BESIDE", "target": "bed"},
        ],
    }


def phase2_bboxes_bedroom() -> dict:
    return {
        "assignments": [
            {
                "object_id": "bed",
                "bbox": {"origin": [4.5, 0, 3], "dimensions": [2, 0.8, 2]},
            },
            {
                "object_id": "nightstand",
                "bbox": {"origin": [6.7, 0, 3.2], "dimensions": [0.6, 0.6, 0.6]},
            },
        ]
    }


def stop() -> dict:
    return {"stop": True}


async def test_full_pipeline_produces_loadable_glb(tmp_path: Path, ctx_factory) -> None:
    fd = tmp_path / "fixtures"

    # Phase 1 fixtures
    write_fixture(fd, "step04", 0, overall_bbox_output())
    write_fixture(fd, "step06", 0, root_breakdown())
    write_fixture(fd, "step06", 1, atomic_breakdown())
    write_fixture(fd, "step06", 2, atomic_breakdown())
    write_fixture(fd, "step05", 0, no_frame())        # root
    write_fixture(fd, "step05", 1, bathroom_floor())  # bathroom
    write_fixture(fd, "step05", 2, bedroom_floor())   # bedroom

    # Phase 2 fixtures — bathroom leaf then bedroom leaf
    write_fixture(fd, "step10", 0, phase2_anchor_bathroom())
    write_fixture(fd, "step11", 0, phase2_bboxes_bathroom())
    write_fixture(fd, "step14", 0, stop())

    write_fixture(fd, "step10", 1, phase2_anchor_bedroom())
    write_fixture(fd, "step11", 1, phase2_bboxes_bedroom())
    write_fixture(fd, "step14", 1, stop())

    ctx = ctx_factory(fixtures_dir=fd)
    url = await run_pipeline("a two-room floorplan", model_id=ctx.model_id)
    assert url == f"/glb/{ctx.run_id}"

    # GLB file exists and is loadable.
    from app.pipeline.step17_publish_glb import glb_path_for

    path = glb_path_for(ctx.run_id)
    assert path.exists()
    scene = trimesh.load(path, force="scene")
    assert isinstance(scene, trimesh.Scene)
    names = set(scene.geometry.keys())
    # All four objects + both floor frames should be present.
    assert {"toilet", "sink", "bed", "nightstand"}.issubset(names)
    assert {"bathroom_floor", "bedroom_floor"}.issubset(names)

    # Event log contains RunStarted + RunCompleted.
    lines = ctx.events.path.read_text().strip().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    # Zero retries across this run.
    completed = json.loads(lines[-1])
    assert completed["retry_summary"] == {}


async def test_failure_emits_run_failed(tmp_path: Path, ctx_factory) -> None:
    fd = tmp_path / "fixtures"
    # Only step04 fixture exists. step06 will fail with FileNotFoundError.
    write_fixture(fd, "step04", 0, overall_bbox_output())
    write_fixture(fd, "step05", 0, no_frame())

    ctx = ctx_factory(fixtures_dir=fd)
    with contextlib.suppress(FileNotFoundError):
        await run_pipeline("will fail", model_id=ctx.model_id)

    lines = ctx.events.path.read_text().strip().splitlines()
    last = json.loads(lines[-1])
    assert last["type"] == "run_failed"
