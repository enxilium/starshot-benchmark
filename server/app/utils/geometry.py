"""Mesh post-processing: yaw rotation + rescaling into a target bbox.

Two scaling modes:
  * "cover" — uniform scale by MAX ratio. Mesh covers the bbox on its
              tightest axis and may overflow the others; proportions are
              preserved. For discrete objects (chair, lamp, tree) where
              the bbox is a size guide, not a hard cage.
  * "fill"  — per-axis scale, mesh exactly fills the bbox. Distorts
              proportions. For encapsulating geometry (walls, floors,
              ceilings) where the bbox IS the shape.

Orientation contract:
  Trellis 2 returns a mesh whose intrinsic front face points along +Z
  in mesh frame. The Node's `orientation` is a yaw (radians, right-handed
  about +Y) that rotates the mesh into world pose: 0 leaves the front
  facing world +Z; π/2 turns the front to world -X. The image-prompt
  step requests an orthographic head-on front view so Trellis's output
  frame is predictable, leaving this rotation as the only orientation
  knob.

Order of operations: translate to origin → yaw → scale → translate to
bbox center. Yaw is applied to the unscaled mesh so it composes cleanly
with per-axis "fill" scaling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

from app.core.types import BoundingBox

RescaleMode = Literal["cover", "fill"]


def rescale_mesh_to_bbox(
    mesh: trimesh.Trimesh | trimesh.Scene,
    bbox: BoundingBox,
    *,
    mode: RescaleMode,
    orientation: float = 0.0,
) -> trimesh.Trimesh | trimesh.Scene:
    if mesh.is_empty:
        raise ValueError("cannot rescale an empty mesh")
    out = mesh.copy()
    cur_min, cur_max = out.bounds
    mesh_center = (cur_min + cur_max) / 2.0

    T_origin = trimesh.transformations.translation_matrix(-mesh_center)
    R = trimesh.transformations.rotation_matrix(orientation, [0.0, 1.0, 0.0])
    out.apply_transform(T_origin)
    out.apply_transform(R)

    rotated_min, rotated_max = out.bounds
    rotated_extents = np.asarray(rotated_max - rotated_min, dtype=float)
    if np.any(rotated_extents <= 0):
        raise ValueError("degenerate mesh has zero extent on some axis")
    target_extents = np.asarray(bbox.size, dtype=float)

    if mode == "fill":
        scale_vec = target_extents / rotated_extents
    else:
        s = float(np.max(target_extents / rotated_extents))
        scale_vec = np.array([s, s, s])

    S = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0])
    T_target = trimesh.transformations.translation_matrix(
        np.asarray(bbox.center, dtype=float)
    )
    out.apply_transform(S)
    out.apply_transform(T_target)
    return out
