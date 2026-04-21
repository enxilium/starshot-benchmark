"""
PIPELINE.md step 12 — rescale.

Given a generated mesh and the target `BoundingBox` resolved for its
anchor object, apply a uniform scale so the mesh's maximal dimension
matches the bbox's maximal dimension, then translate it to the bbox
center.

Overshoot on other axes after scaling is acceptable — the mesh's shape
(aspect ratio) is preserved.
"""

from __future__ import annotations

import numpy as np
import trimesh

from app.core.types import BoundingBox


def rescale_mesh_to_bbox(mesh: trimesh.Trimesh, bbox: BoundingBox) -> trimesh.Trimesh:
    """
    Return a new Trimesh translated + uniformly scaled so:
      * the mesh's max-dimension = bbox.max_dimension
      * the mesh's center = bbox.center

    The input mesh is not modified.
    """
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Cannot rescale an empty mesh")

    # Work on a copy so the caller's mesh is untouched.
    out = mesh.copy()

    # Current extents (axis-aligned bbox of the source mesh).
    cur_min, cur_max = out.bounds
    cur_extent = cur_max - cur_min
    cur_max_dim = float(np.max(cur_extent))
    if cur_max_dim == 0.0:
        raise ValueError("Degenerate mesh has zero extent in every axis")

    scale = bbox.max_dimension / cur_max_dim

    # Scale about the mesh's current center, then translate to target center.
    cur_center = (cur_min + cur_max) / 2.0
    T_center_to_origin = trimesh.transformations.translation_matrix(-cur_center)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    T_to_target = trimesh.transformations.translation_matrix(np.asarray(bbox.center, dtype=float))

    out.apply_transform(T_center_to_origin)
    out.apply_transform(S)
    out.apply_transform(T_to_target)
    return out
