"""
Frame mesh generation. Takes a `Frame` (PlaneFrame | CurveFrame) from the
phase-1 step-5 frame decider and produces a `trimesh.Trimesh` suitable for
inclusion in the final `trimesh.Scene` during assembly.

Frame geometry is deterministic — no LLM involvement. The Hunyuan backend is
also not used here.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import trimesh

from app.core.types import CurveFrame, Frame, PlaneFrame

_CURVE_SEGMENTS_PER_UNIT = 8  # u-direction tessellation density


def frame_to_mesh(frame: Frame) -> trimesh.Trimesh:
    if isinstance(frame, PlaneFrame):
        return _plane_to_mesh(frame)
    # Frame is a discriminated union of PlaneFrame | CurveFrame; the first
    # branch rules out PlaneFrame so this must be a CurveFrame.
    return _curve_to_mesh(frame)


def _plane_to_mesh(frame: PlaneFrame) -> trimesh.Trimesh:
    o = np.asarray(frame.origin, dtype=float)
    u = np.asarray(frame.u_axis, dtype=float)
    v = np.asarray(frame.v_axis, dtype=float)

    vertices = np.stack([o, o + u, o + u + v, o + v], axis=0)
    # Two triangles forming the quad, counter-clockwise when viewed from
    # the normal side (cross(u, v)).
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _curve_to_mesh(frame: CurveFrame) -> trimesh.Trimesh:
    # Sample the curve into N segments, then build a ribbon from polyline to
    # polyline + (0, height, 0). The ribbon is a strip of quads.
    total_length = 0.0
    pts = [np.asarray(p, dtype=float) for p in frame.control_points]
    for a, b in pairwise(pts):
        total_length += float(np.linalg.norm(b - a))
    # A minimum of 4 segments even for very short curves, scaled by length.
    n_segs = max(4, int(np.ceil(total_length * _CURVE_SEGMENTS_PER_UNIT)))

    # Sample points along arc length.
    sampled: list[np.ndarray] = []
    for i in range(n_segs + 1):
        u = i / n_segs
        p = frame.interp_arc(u)
        sampled.append(np.asarray(p, dtype=float))

    up = np.asarray([0.0, frame.height, 0.0], dtype=float)
    bottom_row = np.stack(sampled, axis=0)
    top_row = bottom_row + up

    vertices = np.concatenate([bottom_row, top_row], axis=0)
    faces: list[list[int]] = []
    top_offset = len(sampled)
    for i in range(n_segs):
        b0 = i
        b1 = i + 1
        t0 = top_offset + i
        t1 = top_offset + i + 1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False
    )
