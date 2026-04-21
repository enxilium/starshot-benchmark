from __future__ import annotations

import pytest

from app.core.types import BoundingBox


def make(
    mn: tuple[float, float, float], mx: tuple[float, float, float]
) -> BoundingBox:
    """Build a box from two opposite corners (min, max) via origin + dimensions."""
    return BoundingBox(
        origin=mn,
        dimensions=(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]),
    )


def test_constructor_rejects_zero_dimension() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        BoundingBox(origin=(0, 0, 0), dimensions=(0, 1, 1))
    with pytest.raises(ValueError, match="non-zero"):
        BoundingBox(origin=(0, 0, 0), dimensions=(1, 1, 0))


def test_signed_dimensions_expand_in_indicated_direction() -> None:
    # Positive dimensions: origin is the min corner.
    b = BoundingBox(origin=(0, 0, 0), dimensions=(2, 3, 4))
    assert b.min_corner == (0, 0, 0)
    assert b.max_corner == (2, 3, 4)

    # Negative dimensions: origin is the max corner. Region is identical
    # to the box above, but the representation differs.
    b2 = BoundingBox(origin=(2, 3, 4), dimensions=(-2, -3, -4))
    assert b2.min_corner == (0, 0, 0)
    assert b2.max_corner == (2, 3, 4)

    # Mixed signs are permitted and expand per-axis from the origin.
    b3 = BoundingBox(origin=(1, 0, 1), dimensions=(2, 3, -1))
    assert b3.min_corner == (1, 0, 0)
    assert b3.max_corner == (3, 3, 1)


def test_from_center_size() -> None:
    b = BoundingBox.from_center_size(center=(1.0, 2.0, 3.0), size=(2.0, 4.0, 6.0))
    assert b.min_corner == (0.0, 0.0, 0.0)
    assert b.max_corner == (2.0, 4.0, 6.0)


def test_size_center_volume_max_dim() -> None:
    b = make((0, 0, 0), (2, 3, 4))
    assert b.size == (2, 3, 4)
    assert b.center == (1, 1.5, 2)
    assert b.volume == 24
    assert b.max_dimension == 4


def test_size_is_absolute_regardless_of_dimension_sign() -> None:
    b = BoundingBox(origin=(5, 5, 5), dimensions=(-2, -3, -4))
    assert b.size == (2, 3, 4)
    assert b.volume == 24
    assert b.max_dimension == 4


def test_overlap_true_for_interior_intersection() -> None:
    a = make((0, 0, 0), (2, 2, 2))
    b = make((1, 1, 1), (3, 3, 3))
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_overlap_false_for_disjoint_boxes() -> None:
    a = make((0, 0, 0), (1, 1, 1))
    b = make((2, 2, 2), (3, 3, 3))
    assert not a.overlaps(b)


def test_overlap_false_for_touching_faces() -> None:
    a = make((0, 0, 0), (1, 1, 1))
    b = make((1, 0, 0), (2, 1, 1))  # touching on x=1 plane
    assert not a.overlaps(b)


def test_overlap_is_sign_agnostic() -> None:
    # Same regions as test_overlap_true_for_interior_intersection but written
    # with reversed origins/signs. Result must be identical.
    a = BoundingBox(origin=(2, 2, 2), dimensions=(-2, -2, -2))
    b = BoundingBox(origin=(3, 3, 3), dimensions=(-2, -2, -2))
    assert a.overlaps(b)


def test_contains_boundary_touching_allowed() -> None:
    outer = make((0, 0, 0), (10, 10, 10))
    inner = make((0, 0, 0), (10, 10, 10))
    assert outer.contains(inner)


def test_contains_strict() -> None:
    outer = make((0, 0, 0), (10, 10, 10))
    inner = make((1, 1, 1), (9, 9, 9))
    assert outer.contains(inner)


def test_does_not_contain_escaping_box() -> None:
    outer = make((0, 0, 0), (5, 5, 5))
    escaping = make((3, 3, 3), (10, 10, 10))
    assert not outer.contains(escaping)


def test_union() -> None:
    a = make((0, 0, 0), (1, 1, 1))
    b = make((-1, 2, 3), (2, 3, 4))
    u = a.union(b)
    assert u.min_corner == (-1, 0, 0)
    assert u.max_corner == (2, 3, 4)
    # Union is canonicalised to positive dimensions with origin at the min corner.
    assert u.origin == (-1, 0, 0)
    assert u.dimensions == (3, 3, 4)


def test_immutable() -> None:
    from pydantic import ValidationError

    b = make((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValidationError):
        b.origin = (0, 0, 0)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        b.dimensions = (1, 1, 1)  # type: ignore[misc]
