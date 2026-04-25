"""Mesh post-processing: yaw rotation + rescaling into a target bbox.

Per-axis scaling: each axis is scaled independently so the mesh exactly
fills the bbox on every axis. Proportions are not preserved — the
guarantee is that the mesh is fully contained in the bbox.

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
with per-axis scaling.
"""

from __future__ import annotations

import numpy as np
import trimesh

from app.core.types import BoundingBox


def rescale_mesh_to_bbox(
    mesh: trimesh.Trimesh | trimesh.Scene,
    bbox: BoundingBox,
    *,
    orientation: float = 0.0,
) -> trimesh.Trimesh | trimesh.Scene:
    if mesh.is_empty:
        raise ValueError("cannot rescale an empty mesh")
    out = mesh.copy()

    # Rotate first (around origin), THEN re-derive the AABB and recenter.
    # Rotating an origin-centered AABB does not preserve centering — vertex
    # positions don't have to be symmetric inside their AABB, so a rotation
    # around the AABB centre can land the new AABB off-origin. Centering
    # before rotation and skipping the post-rotation recenter would scale
    # off-centered bounds and translate by `bbox.center`, leaving the mesh
    # overhanging the bbox by `S · rotated_center`. Recenter after R so the
    # subsequent per-axis scale is symmetric.
    R = trimesh.transformations.rotation_matrix(orientation, [0.0, 1.0, 0.0])
    out.apply_transform(R)

    rotated_min, rotated_max = out.bounds
    rotated_center = (rotated_min + rotated_max) / 2.0
    out.apply_transform(
        trimesh.transformations.translation_matrix(-rotated_center)
    )

    rotated_extents = np.asarray(rotated_max - rotated_min, dtype=float)
    if np.any(rotated_extents <= 0):
        raise ValueError("degenerate mesh has zero extent on some axis")
    target_extents = np.asarray(bbox.size, dtype=float)
    scale_vec = target_extents / rotated_extents

    S = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0])
    T_target = trimesh.transformations.translation_matrix(
        np.asarray(bbox.center, dtype=float)
    )
    out.apply_transform(S)
    out.apply_transform(T_target)
    return out
