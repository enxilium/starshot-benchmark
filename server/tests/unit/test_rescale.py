from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.core.types import BoundingBox
from app.pipeline.phase2.step13_rescale import rescale_mesh_to_bbox


def unit_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def test_rescale_matches_max_dimension() -> None:
    m = unit_cube()
    target = BoundingBox(origin=(0, 0, 0), dimensions=(4, 4, 4))
    out = rescale_mesh_to_bbox(m, target)
    lo, hi = out.bounds
    extent = hi - lo
    assert float(np.max(extent)) == pytest.approx(4.0)


def test_rescale_centers_on_target() -> None:
    m = unit_cube()
    target = BoundingBox(origin=(10, 20, 30), dimensions=(2, 2, 2))
    out = rescale_mesh_to_bbox(m, target)
    lo, hi = out.bounds
    center = (lo + hi) / 2
    assert tuple(center.round(6)) == (11.0, 21.0, 31.0)


def test_rescale_preserves_aspect_ratio_via_max_dim() -> None:
    # Mesh is 2x1x1; target is 4x4x4 cube. After rescale, max dim = 4 → scale=2,
    # result extents = 4x2x2 (overshoots on the "short" axes only in that they
    # don't fill the full bbox). Spec allows this.
    m = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    target = BoundingBox(origin=(0, 0, 0), dimensions=(4, 4, 4))
    out = rescale_mesh_to_bbox(m, target)
    lo, hi = out.bounds
    extent = hi - lo
    assert tuple(extent.round(6)) == (4.0, 2.0, 2.0)


def test_rescale_does_not_mutate_input() -> None:
    m = unit_cube()
    before = m.vertices.copy()
    target = BoundingBox(origin=(5, 5, 5), dimensions=(1, 1, 1))
    _ = rescale_mesh_to_bbox(m, target)
    assert np.array_equal(m.vertices, before)


def test_rescale_raises_on_empty_mesh() -> None:
    empty = trimesh.Trimesh()
    target = BoundingBox(origin=(0, 0, 0), dimensions=(1, 1, 1))
    with pytest.raises(ValueError):
        rescale_mesh_to_bbox(empty, target)
