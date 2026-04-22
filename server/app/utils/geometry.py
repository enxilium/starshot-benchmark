"""Mesh rescaling."""

from __future__ import annotations

import numpy as np
import trimesh

from app.core.types import BoundingBox


def rescale_mesh_to_bbox(
    mesh: trimesh.Trimesh | trimesh.Scene, bbox: BoundingBox,
) -> trimesh.Trimesh | trimesh.Scene:
    if mesh.is_empty:
        raise ValueError("cannot rescale an empty mesh")
    out = mesh.copy()
    cur_min, cur_max = out.bounds
    cur_max_dim = float(np.max(cur_max - cur_min))
    if cur_max_dim == 0.0:
        raise ValueError("degenerate mesh has zero extent on every axis")

    scale = bbox.max_dimension / cur_max_dim
    cur_center = (cur_min + cur_max) / 2.0

    T_origin = trimesh.transformations.translation_matrix(-cur_center)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    T_target = trimesh.transformations.translation_matrix(
        np.asarray(bbox.center, dtype=float)
    )

    out.apply_transform(T_origin)
    out.apply_transform(S)
    out.apply_transform(T_target)
    return out
