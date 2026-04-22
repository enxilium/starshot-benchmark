"""Deterministic GLB writers and mesh rescaling.

Frames are defined in world space, so exported meshes need no extra transform.
Curve frames treat `control_points` as a smooth Catmull-Rom spline, not a
literal polyline. Closed loops are capped top and bottom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from app.core.types import BoundingBox, CurveFrame, Frame, PlaneFrame

_SAMPLES_PER_SEGMENT = 16
_CLOSED_EPSILON = 1e-3


def write_frame_glb(frame: Frame, path: Path) -> tuple[Path, BoundingBox]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, PlaneFrame):
        mesh = _plane_mesh(frame)
    elif isinstance(frame, CurveFrame):
        mesh = _curve_mesh(frame)
    else:
        raise TypeError("generated frames are produced via services.threed, not write_frame_glb")
    mesh.export(path, file_type="glb")
    return path, mesh_aabb(mesh)


def rescale_mesh_to_bbox(mesh: trimesh.Trimesh, bbox: BoundingBox) -> trimesh.Trimesh:
    if mesh.vertices.shape[0] == 0:
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


def mesh_aabb(mesh: trimesh.Trimesh) -> BoundingBox:
    cur_min, cur_max = mesh.bounds
    origin = (round(float(cur_min[0]), 2), round(float(cur_min[1]), 2), round(float(cur_min[2]), 2))
    dims = (
        max(round(float(cur_max[0] - cur_min[0]), 2), 0.01),
        max(round(float(cur_max[1] - cur_min[1]), 2), 0.01),
        max(round(float(cur_max[2] - cur_min[2]), 2), 0.01),
    )
    return BoundingBox(origin=origin, dimensions=dims)


def _plane_mesh(frame: PlaneFrame) -> trimesh.Trimesh:
    o = np.asarray(frame.origin, dtype=np.float64)
    u = np.asarray(frame.u_axis, dtype=np.float64)
    v = np.asarray(frame.v_axis, dtype=np.float64)
    vertices = np.stack([o, o + u, o + u + v, o + v])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _curve_mesh(frame: CurveFrame) -> trimesh.Trimesh:
    base = _tessellate(frame.control_points)
    closed = _is_closed_loop(frame.control_points)
    top = base + np.asarray([0.0, frame.height, 0.0])
    vertices = np.concatenate([base, top], axis=0)
    n = len(base)
    faces: list[list[int]] = []

    last = n if closed else n - 1
    for i in range(last):
        b0, b1 = i, (i + 1) % n
        t0, t1 = b0 + n, b1 + n
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    if closed:
        base_center = base.mean(axis=0)
        top_center = top.mean(axis=0)
        vertices = np.concatenate([vertices, [base_center, top_center]], axis=0)
        base_ctr_idx = 2 * n
        top_ctr_idx = 2 * n + 1
        for i in range(n):
            j = (i + 1) % n
            faces.append([base_ctr_idx, j, i])
            faces.append([top_ctr_idx, i + n, j + n])

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _is_closed_loop(control_points: list[tuple[float, float, float]]) -> bool:
    if len(control_points) < 3:
        return False
    a = np.asarray(control_points[0])
    b = np.asarray(control_points[-1])
    return bool(np.linalg.norm(a - b) < _CLOSED_EPSILON)


def _tessellate(control_points: list[tuple[float, float, float]]) -> np.ndarray:
    closed = _is_closed_loop(control_points)
    pts = np.asarray(
        control_points[:-1] if closed else control_points, dtype=np.float64
    )
    n = len(pts)
    if n < 2:
        return pts

    samples: list[np.ndarray] = []
    n_segs = n if closed else n - 1
    for i in range(n_segs):
        if closed:
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
        else:
            p0 = pts[i - 1] if i > 0 else pts[i]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        for k in range(_SAMPLES_PER_SEGMENT):
            t = k / _SAMPLES_PER_SEGMENT
            samples.append(_catmull_rom(p0, p1, p2, p3, t))

    if not closed:
        samples.append(pts[-1])

    return np.stack(samples)


def _catmull_rom(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )
