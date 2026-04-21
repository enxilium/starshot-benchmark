from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.phase1.step08_recurse import divide
from app.pipeline.phase1.step09_collect_leaves import (
    collect_frames_for_leaf,
    collect_leaves,
)
from tests.integration.conftest import write_fixture


def overall_bbox_output() -> dict:
    return {
        "bbox": {"origin": [0, 0, 0], "dimensions": [10, 3, 10]},
        "reasoning": "small single-story house footprint",
    }


def root_breakdown() -> dict:
    return {
        "is_atomic": False,
        "subscenes": [
            {
                "scope_id": "root.bathroom",
                "prompt": "a small bathroom",
                "bbox": {"origin": [0, 0, 0], "dimensions": [3, 3, 4]},
                "high_level_plan": "toilet + sink in a compact arrangement",
            },
            {
                "scope_id": "root.bedroom",
                "prompt": "a cozy bedroom",
                "bbox": {"origin": [3, 0, 0], "dimensions": [7, 3, 10]},
                "high_level_plan": "bed + nightstand + dresser",
            },
        ],
    }


def atomic_breakdown() -> dict:
    return {"is_atomic": True, "subscenes": []}


def overlapping_breakdown() -> dict:
    return {
        "is_atomic": False,
        "subscenes": [
            {
                "scope_id": "root.a",
                "prompt": "a",
                "bbox": {"origin": [0, 0, 0], "dimensions": [6, 3, 10]},
                "high_level_plan": "x",
            },
            {
                "scope_id": "root.b",
                "prompt": "b",
                "bbox": {
                    "origin": [5, 0, 0],  # overlaps root.a on x=[5,6]
                    "dimensions": [5, 3, 10],
                },
                "high_level_plan": "x",
            },
        ],
    }


def frame_decider_no_frame() -> dict:
    return {"needs_frame": False, "frames": [], "reasoning": "no frame needed"}


def frame_decider_walls() -> dict:
    return {
        "needs_frame": True,
        "reasoning": "room needs floor + walls",
        "frames": [
            {
                "kind": "plane",
                "id": "floor",
                "origin": [0, 0, 0],
                "u_axis": [3, 0, 0],
                "v_axis": [0, 0, 4],
            },
        ],
    }


async def test_divide_produces_tree_with_atomic_leaves(
    tmp_path: Path, ctx_factory
) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step04", 0, overall_bbox_output())

    # Root breakdown call (0) -> two subscenes.
    write_fixture(fd, "step06", 0, root_breakdown())
    # Bathroom breakdown call (1) -> atomic.
    write_fixture(fd, "step06", 1, atomic_breakdown())
    # Bedroom breakdown call (2) -> atomic.
    write_fixture(fd, "step06", 2, atomic_breakdown())

    # Frame decider: root (no frame), bathroom (walls), bedroom (walls).
    write_fixture(fd, "step05", 0, frame_decider_no_frame())
    write_fixture(fd, "step05", 1, frame_decider_walls())
    write_fixture(fd, "step05", 2, frame_decider_walls())

    ctx = ctx_factory(fixtures_dir=fd)
    root = await divide("a small two-room house")

    assert root.scope_id == "root"
    assert not root.is_atomic
    assert [c.scope_id for c in root.children] == ["root.bathroom", "root.bedroom"]
    assert all(c.is_atomic for c in root.children)
    # Each child has its walls frame.
    assert len(root.children[0].frames) == 1
    assert root.children[0].frames[0].id == "floor"

    # Root got an empty frame list from the frame decider's no_frame response.
    assert root.frames == []

    leaves = collect_leaves(root)
    assert [leaf.scope_id for leaf in leaves] == ["root.bathroom", "root.bedroom"]

    # State repo has 2 plan entries (one per child) — root itself has no plan entry.
    visible = await ctx.state_repo.read_visible()
    assert {p.scope_id for p in visible.plans} == {"root.bathroom", "root.bedroom"}


async def test_step3_retries_on_overlapping_children(
    tmp_path: Path, ctx_factory
) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step04", 0, overall_bbox_output())

    # Root: first breakdown overlaps, second is fine.
    write_fixture(fd, "step06", 0, overlapping_breakdown())
    # Second attempt: use root_breakdown style but with scope_ids matching.
    write_fixture(
        fd,
        "step06",
        1,
        {
            "is_atomic": False,
            "subscenes": [
                {
                    "scope_id": "root.a",
                    "prompt": "a",
                    "bbox": {"origin": [0, 0, 0], "dimensions": [5, 3, 10]},
                    "high_level_plan": "x",
                },
                {
                    "scope_id": "root.b",
                    "prompt": "b",
                    "bbox": {"origin": [5, 0, 0], "dimensions": [5, 3, 10]},
                    "high_level_plan": "x",
                },
            ],
        },
    )
    # Both children atomic.
    write_fixture(fd, "step06", 2, atomic_breakdown())
    write_fixture(fd, "step06", 3, atomic_breakdown())

    write_fixture(fd, "step05", 0, frame_decider_no_frame())
    write_fixture(fd, "step05", 1, frame_decider_no_frame())
    write_fixture(fd, "step05", 2, frame_decider_no_frame())

    ctx = ctx_factory(fixtures_dir=fd)
    root = await divide("test")
    assert [c.scope_id for c in root.children] == ["root.a", "root.b"]

    # One StepRetried event recorded for step06 at the root.
    lines = ctx.events.path.read_text().strip().splitlines()
    retried = [json.loads(line) for line in lines if '"type":"step_retried"' in line]
    assert len(retried) == 1
    assert retried[0]["step_id"] == "step06"


async def test_step3_retry_exhaustion_raises(tmp_path: Path, ctx_factory) -> None:
    from app.core.errors import RetryExhausted

    fd = tmp_path / "fixtures"
    write_fixture(fd, "step04", 0, overall_bbox_output())
    for i in range(5):
        write_fixture(fd, "step06", i, overlapping_breakdown())
    write_fixture(fd, "step05", 0, frame_decider_no_frame())

    ctx_factory(fixtures_dir=fd, max_retries=2)
    with pytest.raises(RetryExhausted) as excinfo:
        await divide("test")
    assert excinfo.value.step_id == "step06"
    assert excinfo.value.attempts == 2


async def test_collect_frames_for_leaf_walks_ancestors(
    tmp_path: Path, ctx_factory
) -> None:
    fd = tmp_path / "fixtures"
    write_fixture(fd, "step04", 0, overall_bbox_output())
    write_fixture(fd, "step06", 0, root_breakdown())
    write_fixture(fd, "step06", 1, atomic_breakdown())
    write_fixture(fd, "step06", 2, atomic_breakdown())
    # Root gets a frame, children get no frames.
    write_fixture(fd, "step05", 0, frame_decider_walls())
    write_fixture(fd, "step05", 1, frame_decider_no_frame())
    write_fixture(fd, "step05", 2, frame_decider_no_frame())

    ctx_factory(fixtures_dir=fd)
    root = await divide("test")

    bathroom = root.children[0]
    inherited = collect_frames_for_leaf(root, bathroom)
    # Bathroom has no frame of its own, but the root's "floor" frame should
    # flow through.
    assert [f.id for f in inherited] == ["floor"]
