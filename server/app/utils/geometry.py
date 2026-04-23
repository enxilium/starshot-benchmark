"""Mesh orientation + rescaling into a target bbox.

Two modes:
  * "fit"  — uniform scale by min ratio, mesh sits inside the bbox with its
             proportions preserved. For discrete objects (chair, lamp, tree).
  * "fill" — per-axis scale, mesh exactly fills the bbox. For encapsulating
             geometry (walls, floors, ceilings, moats).

Both modes run an axis-permutation alignment first: pick the permutation of
mesh axes that best matches the target bbox's axis extents (log-ratio loss),
applied as a proper rotation. Then translate mesh AABB center to bbox center.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import trimesh

from app.core.types import BoundingBox

RescaleMode = Literal["fit", "fill"]


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

    best_perm = _best_axis_permutation(mesh_extents, target_extents)
    R = _signed_permutation_matrix(best_perm)

    rotated_extents = mesh_extents[list(best_perm)]
    if mode == "fill":
        scale_vec = target_extents / rotated_extents
    else:
        s = float(np.min(target_extents / rotated_extents))
        scale_vec = np.array([s, s, s])

    T_origin = trimesh.transformations.translation_matrix(-mesh_center)
    S = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0])
    T_target = trimesh.transformations.translation_matrix(
        np.asarray(bbox.center, dtype=float)
    )

    out.apply_transform(T_origin)
    out.apply_transform(R)
    out.apply_transform(S)
    out.apply_transform(T_target)
    return out


def _best_axis_permutation(
    mesh_extents: np.ndarray, target_extents: np.ndarray,
) -> tuple[int, int, int]:
    log_mesh = np.log(mesh_extents)
    log_target = np.log(target_extents)
    best_perm: tuple[int, int, int] = (0, 1, 2)
    best_loss = float("inf")
    for perm in itertools.permutations(range(3)):
        loss = float(np.sum((log_target - log_mesh[list(perm)]) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_perm = perm  # type: ignore[assignment]
    return best_perm


def _signed_permutation_matrix(perm: tuple[int, int, int]) -> np.ndarray:
    """4x4 proper rotation sending mesh axis perm[j] onto target axis j."""
    R3 = np.zeros((3, 3))
    for j, i in enumerate(perm):
        R3[j, i] = 1.0
    if np.linalg.det(R3) < 0:
        R3[0, :] *= -1
    R = np.eye(4)
    R[:3, :3] = R3
    return R
