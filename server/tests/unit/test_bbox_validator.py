from __future__ import annotations

from app.core.types import BoundingBox
from app.geometry.bbox_validator import validate_boxes, validate_completeness


def bb(
    mn: tuple[float, float, float], mx: tuple[float, float, float]
) -> BoundingBox:
    return BoundingBox(
        origin=mn,
        dimensions=(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]),
    )


def test_validate_boxes_accepts_valid_partition() -> None:
    parent = bb((0, 0, 0), (10, 10, 10))
    children = [
        ("a", bb((0, 0, 0), (5, 5, 5))),
        ("b", bb((5, 0, 0), (10, 5, 5))),  # touching a's x=5 face, not overlapping
        ("c", bb((0, 5, 0), (5, 10, 5))),
    ]
    assert validate_boxes(parent=parent, children=children) is None


def test_validate_boxes_flags_overlap() -> None:
    parent = bb((0, 0, 0), (10, 10, 10))
    children = [
        ("a", bb((0, 0, 0), (6, 6, 6))),
        ("b", bb((5, 0, 0), (10, 5, 5))),  # overlaps a on x=[5,6]
    ]
    conflict = validate_boxes(parent=parent, children=children)
    assert conflict is not None
    assert conflict.validator == "bbox_sibling_overlap"
    assert "a" in conflict.detail and "b" in conflict.detail
    assert conflict.data["a"] == "a"
    assert conflict.data["b"] == "b"


def test_validate_boxes_flags_containment() -> None:
    parent = bb((0, 0, 0), (5, 5, 5))
    children = [("escape", bb((3, 3, 3), (10, 10, 10)))]
    conflict = validate_boxes(parent=parent, children=children)
    assert conflict is not None
    assert conflict.validator == "bbox_containment"
    assert conflict.data["child"] == "escape"


def test_validate_boxes_prefers_containment_over_overlap_when_both_present() -> None:
    parent = bb((0, 0, 0), (5, 5, 5))
    children = [
        ("a", bb((0, 0, 0), (3, 3, 3))),
        ("b", bb((2, 2, 2), (10, 10, 10))),  # b overlaps a AND escapes parent
    ]
    conflict = validate_boxes(parent=parent, children=children)
    assert conflict is not None
    # containment is checked first in the validator's order
    assert conflict.validator == "bbox_containment"


def test_validate_completeness() -> None:
    assert (
        validate_completeness(
            expected_ids=["a", "b"],
            bboxes={"a": bb((0, 0, 0), (1, 1, 1)), "b": bb((2, 2, 2), (3, 3, 3))},
        )
        is None
    )
    conflict = validate_completeness(
        expected_ids=["a", "b", "c"],
        bboxes={"a": bb((0, 0, 0), (1, 1, 1))},
    )
    assert conflict is not None
    assert conflict.validator == "bbox_completeness"
    assert conflict.data["missing"] == ["b", "c"]
