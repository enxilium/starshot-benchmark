from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.types import CurveFrame, PlaneFrame
from app.geometry.frames import frame_to_mesh

# ---- PlaneFrame --------------------------------------------------------------


def test_plane_surface_point_corners() -> None:
    f = PlaneFrame(
        id="wall",
        origin=(0.0, 0.0, 0.0),
        u_axis=(4.0, 0.0, 0.0),
        v_axis=(0.0, 3.0, 0.0),
    )
    assert f.surface_point(0.0, 0.0) == (0.0, 0.0, 0.0)
    assert f.surface_point(1.0, 0.0) == (4.0, 0.0, 0.0)
    assert f.surface_point(0.0, 1.0) == (0.0, 3.0, 0.0)
    assert f.surface_point(1.0, 1.0) == (4.0, 3.0, 0.0)
    assert f.surface_point(0.5, 0.5) == (2.0, 1.5, 0.0)


def test_plane_surface_normal() -> None:
    f = PlaneFrame(
        id="floor",
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    n = f.surface_normal(0.5, 0.5)
    # u cross v = x_hat cross z_hat = -y_hat
    assert n == pytest.approx((0.0, -1.0, 0.0))


def test_plane_degenerate_raises() -> None:
    f = PlaneFrame(
        id="bad",
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(2.0, 0.0, 0.0),  # parallel to u_axis
    )
    with pytest.raises(ValueError, match="Degenerate"):
        f.surface_normal(0.5, 0.5)


def test_plane_to_mesh_has_4_verts_and_2_faces() -> None:
    f = PlaneFrame(
        id="w",
        origin=(0, 0, 0),
        u_axis=(2, 0, 0),
        v_axis=(0, 0, 2),
    )
    m = frame_to_mesh(f)
    assert len(m.vertices) == 4
    assert len(m.faces) == 2


# ---- CurveFrame --------------------------------------------------------------


def test_curve_surface_point_linear_segment() -> None:
    f = CurveFrame(
        id="c",
        control_points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        height=2.0,
    )
    assert f.surface_point(0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0))
    assert f.surface_point(1.0, 0.0) == pytest.approx((10.0, 0.0, 0.0))
    assert f.surface_point(0.5, 0.0) == pytest.approx((5.0, 0.0, 0.0))
    # v=1 raises the point by full height
    assert f.surface_point(0.5, 1.0) == pytest.approx((5.0, 2.0, 0.0))


def test_curve_arc_length_parameterization() -> None:
    # L-shaped polyline: short leg then long leg. u=0.5 should land at the
    # corner IF arc length is parameterized, NOT at a linear segment midpoint.
    f = CurveFrame(
        id="L",
        control_points=[(0, 0, 0), (1, 0, 0), (1, 0, 9)],
        height=1.0,
    )
    # total length = 1 + 9 = 10; arc halfway is at s=5 which is 4 units past
    # the corner along the second leg.
    p = f.surface_point(0.5, 0.0)
    assert p == pytest.approx((1.0, 0.0, 4.0))


def test_curve_normal_is_unit_and_horizontal() -> None:
    f = CurveFrame(
        id="s",
        control_points=[(0, 0, 0), (1, 0, 0)],
        height=1.0,
    )
    n = f.surface_normal(0.5, 0.0)
    mag = math.sqrt(sum(c * c for c in n))
    assert mag == pytest.approx(1.0)
    # tangent is along +x so normal is ±z (horizontal)
    assert abs(n[1]) < 1e-8


def test_curve_to_mesh_is_a_valid_ribbon() -> None:
    f = CurveFrame(
        id="c",
        control_points=[(0, 0, 0), (2, 0, 0), (2, 0, 3)],
        height=2.0,
    )
    m = frame_to_mesh(f)
    # Ribbon: 2 * (n_segs + 1) vertices, 2 * n_segs faces
    assert len(m.vertices) % 2 == 0
    n_verts_per_side = len(m.vertices) // 2
    assert len(m.faces) == 2 * (n_verts_per_side - 1)
    # No zero-area triangles
    areas = m.area_faces
    assert np.all(areas > 0.0)
