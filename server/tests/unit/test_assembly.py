from __future__ import annotations

from pathlib import Path

import trimesh

from app.core.types import PlaneFrame
from app.pipeline.step16_assemble_scene import build_scene
from app.pipeline.step17_publish_glb import write_glb


def test_build_scene_includes_objects_and_frames() -> None:
    cube_a = trimesh.creation.box(extents=(1, 1, 1))
    cube_b = trimesh.creation.box(extents=(2, 2, 2))
    wall = PlaneFrame(
        id="wall_n",
        origin=(0, 0, 0),
        u_axis=(3, 0, 0),
        v_axis=(0, 3, 0),
    )
    scene = build_scene(
        object_meshes=[("cube_a", cube_a), ("cube_b", cube_b)],
        frames=[wall],
    )
    assert "cube_a" in scene.geometry
    assert "cube_b" in scene.geometry
    assert "wall_n" in scene.geometry


def test_glb_roundtrip(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    wall = PlaneFrame(
        id="floor",
        origin=(0, 0, 0),
        u_axis=(4, 0, 0),
        v_axis=(0, 0, 4),
    )
    scene = build_scene(object_meshes=[("box", mesh)], frames=[wall])
    out = tmp_path / "scene.glb"
    write_glb(scene, out)
    assert out.exists()
    assert out.stat().st_size > 0

    # Reload the GLB — must be parseable.
    reloaded = trimesh.load(out, force="scene")
    assert isinstance(reloaded, trimesh.Scene)
    names = set(reloaded.geometry.keys())
    assert "box" in names
    assert "floor" in names
