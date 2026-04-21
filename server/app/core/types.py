"""
Domain types for the pipeline.

Coordinates are **Y-up, right-handed, meters**. All bounding boxes and frames
are expressed in world space of the containing scene.

Organization:
  * Vec3Tuple / MinCorner / MaxCorner : coordinate primitives (NewTypes).
  * BoundingBox                        : axis-aligned box with overlap / contains / union.
  * RelationshipKind, Relationship     : phase-2 anchor-object relationships.
  * AnchorObject                       : phase-2 leaf object with bbox.
  * PlaneFrame / CurveFrame / Frame    : phase-1 frame geometry with (u,v) surface param.
  * SubsceneNode                       : phase-1 recursion tree node.
  * RealizedLeaf                       : phase-2 output written to StateRepository.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, NewType

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---- Coordinate primitives ---------------------------------------------------

Vec3Tuple = tuple[float, float, float]

MinCorner = NewType("MinCorner", Vec3Tuple)
MaxCorner = NewType("MaxCorner", Vec3Tuple)


# ---- Bounding box ------------------------------------------------------------


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box, defined by one vertex (`origin`) and a signed
    `dimensions` vector extending from it. The sign of each component of
    `dimensions` chooses the direction of expansion along that axis:

        origin=(0,0,0), dimensions=(2, 3, 4)      → extends +x, +y, +z
        origin=(2,3,4), dimensions=(-2, -3, -4)   → same region, opposite corner

    Derived `min_corner`, `max_corner`, and `size` are computed from the
    canonicalised extents. Immutable.
    """

    model_config = ConfigDict(frozen=True)

    origin: Vec3Tuple
    dimensions: Vec3Tuple

    @model_validator(mode="after")
    def _check_nonzero(self) -> BoundingBox:
        if any(d == 0 for d in self.dimensions):
            raise ValueError(
                f"dimensions {self.dimensions} must be non-zero on every axis"
            )
        return self

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
    def volume(self) -> float:
        sx, sy, sz = self.size
        return sx * sy * sz

    @property
    def max_dimension(self) -> float:
        return max(self.size)

    def overlaps(self, other: BoundingBox, *, tolerance: float = 1e-6) -> bool:
        """True iff interiors intersect in all three axes. Touching faces don't count."""
        a_min, a_max = self.min_corner, self.max_corner
        b_min, b_max = other.min_corner, other.max_corner
        for i in range(3):
            if a_max[i] <= b_min[i] + tolerance:
                return False
            if b_max[i] <= a_min[i] + tolerance:
                return False
        return True

    def contains(self, other: BoundingBox, *, tolerance: float = 1e-6) -> bool:
        """True iff `other` sits fully inside `self` (boundary-touching allowed)."""
        a_min, a_max = self.min_corner, self.max_corner
        b_min, b_max = other.min_corner, other.max_corner
        for i in range(3):
            if b_min[i] < a_min[i] - tolerance:
                return False
            if b_max[i] > a_max[i] + tolerance:
                return False
        return True

    def union(self, other: BoundingBox) -> BoundingBox:
        a_min, a_max = self.min_corner, self.max_corner
        b_min, b_max = other.min_corner, other.max_corner
        origin = (
            min(a_min[0], b_min[0]),
            min(a_min[1], b_min[1]),
            min(a_min[2], b_min[2]),
        )
        far = (
            max(a_max[0], b_max[0]),
            max(a_max[1], b_max[1]),
            max(a_max[2], b_max[2]),
        )
        return BoundingBox(
            origin=origin,
            dimensions=(far[0] - origin[0], far[1] - origin[1], far[2] - origin[2]),
        )


# ---- Relationships -----------------------------------------------------------


class RelationshipKind(StrEnum):
    ON = "ON"
    BESIDE = "BESIDE"
    BELOW = "BELOW"
    ABOVE = "ABOVE"
    ATTACHED = "ATTACHED"


SYMMETRIC_KINDS: frozenset[RelationshipKind] = frozenset(
    {RelationshipKind.BESIDE, RelationshipKind.ATTACHED}
)

INVERSE_KINDS: dict[RelationshipKind, RelationshipKind] = {
    RelationshipKind.ABOVE: RelationshipKind.BELOW,
    RelationshipKind.BELOW: RelationshipKind.ABOVE,
}


class Relationship(BaseModel):
    """
    Directed relationship between a `subject` (anchor object) and a `target`
    (anchor object or frame). Semantics:

    * ABOVE / BELOW  : `subject` sits above / below `target` (directed).
    * ON             : `subject` sits on top of `target` (directed, contact implied).
    * BESIDE         : `subject` next to `target` (symmetric; one entry suffices).
    * ATTACHED       : `subject` attached to `target` at surface coordinate
                       `attachment=(u, v)`. `target` must be a frame id.
    """

    model_config = ConfigDict(frozen=True)

    subject: str
    kind: RelationshipKind
    target: str
    attachment: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_attachment(self) -> Relationship:
        if self.kind == RelationshipKind.ATTACHED and self.attachment is None:
            raise ValueError("ATTACHED relationships require an (u, v) attachment")
        if self.kind != RelationshipKind.ATTACHED and self.attachment is not None:
            raise ValueError(
                f"{self.kind.value} relationship must not carry an attachment (u, v)"
            )
        return self


# ---- Anchor objects ----------------------------------------------------------


class AnchorObject(BaseModel):
    """A single defining object inside a phase-2 leaf subscene."""

    id: str
    prompt: str
    bbox: BoundingBox | None = None  # filled in phase 2 step 4


# ---- Frames ------------------------------------------------------------------


class _FrameMixin:
    """Every Frame exposes (u, v) ∈ [0, 1]² → world-space surface geometry."""

    def surface_point(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError

    def surface_normal(self, u: float, v: float) -> Vec3Tuple:
        raise NotImplementedError

    def world_point(self, attachment: tuple[float, float]) -> Vec3Tuple:
        return self.surface_point(attachment[0], attachment[1])


class PlaneFrame(BaseModel, _FrameMixin):
    """
    A flat rectangular frame. Spans the parallelogram

        origin + u * u_axis + v * v_axis   for (u, v) ∈ [0, 1]².

    In practice `u_axis` and `v_axis` are orthogonal (walls, floors, ceilings),
    but this is not enforced — non-orthogonal planes are permitted.
    """

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
            raise ValueError(f"Degenerate plane {self.id!r}: u_axis cross v_axis = 0")
        n = n / length
        return (float(n[0]), float(n[1]), float(n[2]))


class CurveFrame(BaseModel, _FrameMixin):
    """
    Vertically-extruded curved frame. `control_points` define a horizontal
    polyline; the frame spans vertically from the polyline to `height` above it.

    * `u` ∈ [0, 1] is the arc-length parameter along the polyline.
    * `v` ∈ [0, 1] is the vertical parameter (0 = polyline, 1 = top).
    """

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


Frame = Annotated[PlaneFrame | CurveFrame, Field(discriminator="kind")]


# ---- Phase-1 tree & phase-2 realized leaves ----------------------------------


class SubsceneNode(BaseModel):
    """
    A node in the phase-1 recursion tree. Non-leaf nodes have `children`;
    leaf nodes (`is_atomic=True`) are passed to phase 2 for anchor generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope_id: str
    prompt: str
    bbox: BoundingBox
    high_level_plan: str = ""
    # Frames anchored to THIS subscene (walls, floors, ceilings). A leaf's
    # PIPELINE.md step 11 references these PLUS all ancestor frames.
    frames: list[Frame] = []
    children: list[SubsceneNode] = []
    is_atomic: bool = False


class RealizedLeaf(BaseModel):
    """Phase-2 output for a single leaf subscene. Written to StateRepository."""

    scope_id: str
    objects: list[AnchorObject]
    relationships: list[Relationship]
