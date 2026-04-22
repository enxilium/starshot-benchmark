"""
Domain types for the pipeline.

Coordinates: Y-up, right-handed, meters.
Canonical front view: +X = right, +Y = up, +Z = toward the viewer.
All bounding boxes, corners, and relationships are expressed under this
convention.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Vec3Tuple = tuple[float, float, float]

_Cm = Annotated[float, Field(multiple_of=0.01)]
Vec3Cm = tuple[_Cm, _Cm, _Cm]


class Corner(StrEnum):
    """One of the 8 corners of an AABB under the canonical front view."""

    TOP_LEFT_FRONT = "TOP_LEFT_FRONT"
    TOP_LEFT_BACK = "TOP_LEFT_BACK"
    TOP_RIGHT_FRONT = "TOP_RIGHT_FRONT"
    TOP_RIGHT_BACK = "TOP_RIGHT_BACK"
    BOTTOM_LEFT_FRONT = "BOTTOM_LEFT_FRONT"
    BOTTOM_LEFT_BACK = "BOTTOM_LEFT_BACK"
    BOTTOM_RIGHT_FRONT = "BOTTOM_RIGHT_FRONT"
    BOTTOM_RIGHT_BACK = "BOTTOM_RIGHT_BACK"


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box, defined by `origin` vertex and signed `dimensions`.
    The sign of each component of `dimensions` chooses the direction of expansion.
    """

    model_config = ConfigDict(frozen=True)

    origin: Vec3Cm
    dimensions: Vec3Cm

    @classmethod
    def from_center_size(cls, center: Vec3Tuple, size: Vec3Tuple) -> BoundingBox:
        half = (size[0] / 2, size[1] / 2, size[2] / 2)
        return cls(
            origin=(center[0] - half[0], center[1] - half[1], center[2] - half[2]),
            dimensions=size,
        )

    @property
    def min_corner(self) -> Vec3Tuple:
        return (
            min(self.origin[0], self.origin[0] + self.dimensions[0]),
            min(self.origin[1], self.origin[1] + self.dimensions[1]),
            min(self.origin[2], self.origin[2] + self.dimensions[2]),
        )

    @property
    def max_corner(self) -> Vec3Tuple:
        return (
            max(self.origin[0], self.origin[0] + self.dimensions[0]),
            max(self.origin[1], self.origin[1] + self.dimensions[1]),
            max(self.origin[2], self.origin[2] + self.dimensions[2]),
        )

    @property
    def size(self) -> Vec3Tuple:
        return (abs(self.dimensions[0]), abs(self.dimensions[1]), abs(self.dimensions[2]))

    @property
    def center(self) -> Vec3Tuple:
        return (
            self.origin[0] + self.dimensions[0] / 2,
            self.origin[1] + self.dimensions[1] / 2,
            self.origin[2] + self.dimensions[2] / 2,
        )

    @property
    def max_dimension(self) -> float:
        return max(self.size)

    def corner_point(self, corner: Corner) -> Vec3Tuple:
        lo, hi = self.min_corner, self.max_corner
        x = hi[0] if "_RIGHT_" in corner.value else lo[0]
        y = hi[1] if corner.value.startswith("TOP") else lo[1]
        z = hi[2] if corner.value.endswith("FRONT") else lo[2]
        return (x, y, z)


class RelationshipKind(StrEnum):
    ON = "ON"
    BESIDE = "BESIDE"
    BELOW = "BELOW"
    ABOVE = "ABOVE"
    ATTACHED = "ATTACHED"


class Relationship(BaseModel):
    """Anchors a node's bbox to a target corner of another node's bbox."""

    model_config = ConfigDict(frozen=True)

    target: str
    kind: RelationshipKind
    reference_point: Corner


class Node(BaseModel):
    """Tree node for the scene.

    Zones and objects are both Nodes. Zones are abstract (mesh_url is None,
    children populated). Concrete nodes (objects, realized frames) set
    mesh_url and typically have no children.
    """

    id: str
    prompt: str
    bbox: BoundingBox
    relationships: list[Relationship] = Field(default_factory=list)
    mesh_url: str | None = None
    children: list[Node] = Field(default_factory=list)


# --- Frame geometry (deterministic architectural surfaces) -----------------


class _FrameMixin:
    def surface_point(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError

    def surface_normal(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError


class PlaneFrame(BaseModel, _FrameMixin):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["plane"] = "plane"
    origin: Vec3Tuple
    u_axis: Vec3Tuple
    v_axis: Vec3Tuple

    def surface_point(self, u: float, v: float) -> Vec3Tuple:
        return (
            self.origin[0] + u * self.u_axis[0] + v * self.v_axis[0],
            self.origin[1] + u * self.u_axis[1] + v * self.v_axis[1],
            self.origin[2] + u * self.u_axis[2] + v * self.v_axis[2],
        )

    def surface_normal(self, u: float, v: float) -> Vec3Tuple:
        _ = (u, v)
        n = np.cross(np.asarray(self.u_axis), np.asarray(self.v_axis))
        length = float(np.linalg.norm(n))
        if length == 0.0:
            raise ValueError(f"Degenerate plane {self.id!r}")
        n = n / length
        return (float(n[0]), float(n[1]), float(n[2]))


class CurveFrame(BaseModel, _FrameMixin):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["curve"] = "curve"
    control_points: list[Vec3Tuple] = Field(min_length=2)
    height: float = Field(gt=0.0)

    def interp_arc(self, u: float) -> Vec3Tuple:
        u = max(0.0, min(1.0, u))
        pts = self.control_points
        segs = list(pairwise(pts))
        lengths = [float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in segs]
        total = sum(lengths)
        if total == 0.0:
            return pts[0]
        target = u * total
        accum = 0.0
        for (a, b), length in zip(segs, lengths, strict=True):
            if accum + length >= target:
                t = 0.0 if length == 0.0 else (target - accum) / length
                return (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                )
            accum += length
        return pts[-1]

    def surface_point(self, u: float, v: float) -> Vec3Tuple:
        base = self.interp_arc(u)
        return (base[0], base[1] + v * self.height, base[2])

    def surface_normal(self, u: float, v: float) -> Vec3Tuple:
        _ = v
        eps = 1e-4
        p0 = self.interp_arc(max(0.0, u - eps))
        p1 = self.interp_arc(min(1.0, u + eps))
        tangent = np.asarray([p1[i] - p0[i] for i in range(3)])
        t_len = float(np.linalg.norm(tangent))
        if t_len == 0.0:
            raise ValueError(f"Degenerate curve {self.id!r} at u={u}")
        tangent = tangent / t_len
        n = np.cross(tangent, np.asarray([0.0, 1.0, 0.0]))
        n_len = float(np.linalg.norm(n))
        if n_len == 0.0:
            return (1.0, 0.0, 0.0)
        n = n / n_len
        return (float(n[0]), float(n[1]), float(n[2]))


class GeneratedFrame(BaseModel, _FrameMixin):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["generated"] = "generated"
    prompt: str
    bounds: BoundingBox | None = None

    def surface_point(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError

    def surface_normal(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError


Frame = Annotated[
    PlaneFrame | CurveFrame | GeneratedFrame, Field(discriminator="kind")
]
