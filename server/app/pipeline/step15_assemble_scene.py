"""
PIPELINE.md step 15 — assemble every leaf mesh + every frame into a
single `trimesh.Scene`.

Each object mesh and each frame is added as a named geometry so
downstream consumers can inspect the scene graph (names are the `id`
fields from `AnchorObject` / `Frame`). Frames are converted to meshes
deterministically via `geometry.frames.frame_to_mesh`.

Coordinate convention: Y-up, right-handed, meters. trimesh's GLB export
(step 16) writes Y-up natively, matching glTF 2.0.
"""

from __future__ import annotations

from collections.abc import Iterable

import trimesh

from app.core.types import Frame
from app.geometry.frames import frame_to_mesh


def build_scene(
    *,
    object_meshes: Iterable[tuple[str, trimesh.Trimesh]],
    frames: Iterable[Frame],
) -> trimesh.Scene:
    """Assemble a `trimesh.Scene` from `(id, mesh)` pairs plus frame geometry."""
    scene = trimesh.Scene()
    for obj_id, mesh in object_meshes:
        scene.add_geometry(mesh, node_name=obj_id, geom_name=obj_id)
    for frame in frames:
        fmesh = frame_to_mesh(frame)
        scene.add_geometry(fmesh, node_name=frame.id, geom_name=frame.id)
    return scene
