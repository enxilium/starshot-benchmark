from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.types import BoundingBox, PlaneFrame, SubsceneNode
from app.pipeline.phase2.step14_completion_loop import generate_leaf
from tests.integration.conftest import write_fixture


def make_meeting_room_leaf() -> tuple[SubsceneNode, list[PlaneFrame]]:
    bbox = BoundingBox(origin=(0, 0, 0), dimensions=(6, 3, 4))
    leaf = SubsceneNode(
        scope_id="root.meeting_room",
        prompt="a small corporate meeting room",
        bbox=bbox,
        is_atomic=True,
    )
    floor = PlaneFrame(
        id="floor",
        origin=(0, 0, 0),
        u_axis=(6, 0, 0),
        v_axis=(0, 0, 4),
    )
    north_wall = PlaneFrame(
        id="wall_n",
        origin=(0, 0, 4),
        u_axis=(6, 0, 0),
        v_axis=(0, 3, 0),
    )
    return leaf, [floor, north_wall]


def step2_output(n_objects: int = 3) -> dict:
    objects = [
        {"id": "table", "prompt": "a conference table"},
        {"id": "chair_1", "prompt": "an office chair"},
        {"id": "tv", "prompt": "a wall-mounted television"},
    ]
    relationships = [
        {
            "subject": "table",
            "kind": "ATTACHED",
            "target": "floor",
            "attachment": [0.5, 0.5],
        },
        {"subject": "chair_1", "kind": "BESIDE", "target": "table"},
        {
            "subject": "tv",
            "kind": "ATTACHED",
            "target": "wall_n",
            "attachment": [0.5, 0.7],
        },
    ]
    return {"objects": objects[:n_objects], "relationships": relationships[:n_objects]}


def step4_output_valid() -> dict:
    return {
        "assignments": [
            {
                "object_id": "table",
                "bbox": {"origin": [2, 0, 1.5], "dimensions": [2, 0.75, 1]},
            },
            {
                "object_id": "chair_1",
                "bbox": {"origin": [1, 0, 1.5], "dimensions": [0.8, 0.9, 0.8]},
            },
            {
                "object_id": "tv",
                "bbox": {"origin": [2.5, 1.5, 3.8], "dimensions": [1, 0.7, 0.2]},
            },
        ]
    }


def step4_output_overlapping() -> dict:
    return {
        "assignments": [
            {
                "object_id": "table",
                "bbox": {"origin": [2, 0, 1.5], "dimensions": [2, 0.75, 1]},
            },
            {
                "object_id": "chair_1",
                "bbox": {
                    "origin": [3, 0, 2],  # overlaps table
                    "dimensions": [0.5, 0.9, 0.4],
                },
            },
            {
                "object_id": "tv",
                "bbox": {"origin": [2.5, 1.5, 3.8], "dimensions": [1, 0.7, 0.2]},
            },
        ]
    }


def step8_stop() -> dict:
    return {"stop": True}


def step8_propose_lamp() -> dict:
    return {
        "stop": False,
        "object": {"id": "lamp", "prompt": "a small desk lamp"},
        "new_relationships": [{"subject": "lamp", "kind": "ON", "target": "table"}],
    }


async def test_happy_path_generates_all_meshes_and_writes_realized(
    tmp_path: Path, ctx_factory
) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step10", 0, step2_output())
    write_fixture(fd, "step11", 0, step4_output_valid())
    write_fixture(fd, "step14", 0, step8_stop())

    ctx = ctx_factory(fixtures_dir=fd)
    leaf, frames = make_meeting_room_leaf()

    out = await generate_leaf(leaf=leaf, frames=frames)

    assert out.scope_id == "root.meeting_room"
    assert {o.id for o in out.objects} == {"table", "chair_1", "tv"}
    assert all(o.bbox is not None for o in out.objects)
    assert set(out.meshes) == {"table", "chair_1", "tv"}

    # Every mesh is rescaled to its bbox's max dimension.
    for oid, mesh in out.meshes.items():
        obj = next(o for o in out.objects if o.id == oid)
        assert obj.bbox is not None
        lo, hi = mesh.bounds
        max_dim = float(max(hi - lo))
        assert abs(max_dim - obj.bbox.max_dimension) < 1e-5

    # RealizedEntry written to state repo.
    visible = await ctx.state_repo.read_visible()
    assert len(visible.realized) == 1
    assert visible.realized[0].scope_id == "root.meeting_room"


async def test_completion_loop_adds_one_more_object(tmp_path: Path, ctx_factory) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step10", 0, step2_output())
    write_fixture(fd, "step11", 0, step4_output_valid())
    # Step 8 call 0: propose lamp; call 1: stop.
    write_fixture(fd, "step14", 0, step8_propose_lamp())
    write_fixture(fd, "step14", 1, step8_stop())
    # Step 4 is called again in incremental mode for the lamp.
    write_fixture(
        fd,
        "step11",
        1,
        {
            "assignments": [
                {
                    "object_id": "lamp",
                    "bbox": {
                        "origin": [3.4, 0.75, 1.9],
                        "dimensions": [0.3, 0.3, 0.3],
                    },
                }
            ]
        },
    )

    ctx_factory(fixtures_dir=fd)
    leaf, frames = make_meeting_room_leaf()

    out = await generate_leaf(leaf=leaf, frames=frames)
    assert {o.id for o in out.objects} == {"table", "chair_1", "tv", "lamp"}
    assert "lamp" in out.meshes


async def test_step4_retry_on_bbox_conflict(tmp_path: Path, ctx_factory) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step10", 0, step2_output())
    # First step-4 call: overlapping bboxes → rejected.
    write_fixture(fd, "step11", 0, step4_output_overlapping())
    # Second step-4 call: valid bboxes.
    write_fixture(fd, "step11", 1, step4_output_valid())
    write_fixture(fd, "step14", 0, step8_stop())

    ctx = ctx_factory(fixtures_dir=fd)
    leaf, frames = make_meeting_room_leaf()

    out = await generate_leaf(leaf=leaf, frames=frames)
    assert len(out.objects) == 3

    # Event log should record exactly one StepRetried for step11.
    lines = ctx.events.path.read_text().strip().splitlines()
    retried = [
        json.loads(line)
        for line in lines
        if '"type":"step_retried"' in line
    ]
    assert len(retried) == 1
    assert retried[0]["step_id"] == "step11"


async def test_step4_retry_exhaustion_raises(tmp_path: Path, ctx_factory) -> None:
    from app.core.errors import RetryExhausted

    fd = tmp_path / "fixtures"
    write_fixture(fd, "step10", 0, step2_output())
    # All step-4 calls return overlapping bboxes — always rejected.
    for i in range(5):
        write_fixture(fd, "step11", i, step4_output_overlapping())
    write_fixture(fd, "step14", 0, step8_stop())

    ctx_factory(fixtures_dir=fd, max_retries=2)
    leaf, frames = make_meeting_room_leaf()

    with pytest.raises(RetryExhausted) as excinfo:
        await generate_leaf(leaf=leaf, frames=frames)
    assert excinfo.value.step_id == "step11"
    assert excinfo.value.attempts == 2
