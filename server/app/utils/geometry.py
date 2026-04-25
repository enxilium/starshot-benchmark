"""Mesh rescaling into a target bbox.

Two modes:
  * "cover" — uniform scale by MAX ratio. Mesh covers the bbox on its
              tightest axis and may overflow the others; proportions are
              preserved. For discrete objects (chair, lamp, tree) — the
              bbox is treated as a size *guide* rather than a hard cage,
              so the mesh never ends up tiny floating inside an oversized
              box. Overflow clipping between neighbours is accepted and
              logged as a benchmark signal.
  * "fill"  — per-axis scale, mesh exactly fills the bbox. Distorts
              proportions. For encapsulating geometry (walls, floors,
              ceilings, moats) where the bbox IS the shape.

No rotation. The mesh's orientation is whatever Trellis 2 produced from
the reference image, and the image-prompt step is responsible for shooting
the object from the canonical front view (camera in front of the object,
+X right, +Y up, +Z toward the viewer) so that orientation already matches
the bbox.
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
) -> trimesh.Trimesh | trimesh.Scene:
    if mesh.is_empty:
        raise ValueError("cannot rescale an empty mesh")
    out = mesh.copy()
    cur_min, cur_max = out.bounds
    mesh_extents = np.asarray(cur_max - cur_min, dtype=float)
    if np.any(mesh_extents <= 0):
        raise ValueError("degenerate mesh has zero extent on some axis")
    mesh_center = (cur_min + cur_max) / 2.0
    target_extents = np.asarray(bbox.size, dtype=float)

    if mode == "fill":
        scale_vec = target_extents / mesh_extents
    else:
        s = float(np.max(target_extents / mesh_extents))
        scale_vec = np.array([s, s, s])

    T_origin = trimesh.transformations.translation_matrix(-mesh_center)
    S = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0])
    T_target = trimesh.transformations.translation_matrix(
        np.asarray(bbox.center, dtype=float)
    )

    out.apply_transform(T_origin)
    out.apply_transform(S)
    out.apply_transform(T_target)
    return out
