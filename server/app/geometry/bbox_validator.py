"""
Shared bounding-box validator. Used by BOTH:

* Phase 1 step 4 — sibling subscene bboxes must not overlap and each must fit
  inside the parent subscene bbox.
* Phase 2 step 5 — anchor-object bboxes must not overlap and each must fit
  inside the leaf bbox.

Returns a `ValidationConflict` describing the first problem found, or `None`
when the box set is valid. The caller (phase 1 / phase 2 step driver) decides
whether to feed the conflict back to the LLM for a retry.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.errors import ValidationConflict
from app.core.types import BoundingBox


def validate_boxes(
    *,
    parent: BoundingBox,
    children: Iterable[tuple[str, BoundingBox]],
    tolerance: float = 1e-6,
) -> ValidationConflict | None:
    """
    Check a set of named child bboxes against a parent box.

    Produces a conflict for the first violation it finds in this order:
      1. Missing bbox (None) — handled by the caller before it reaches here;
         this function assumes every child has a resolved bbox.
      2. Child escapes parent.
      3. Two children overlap.
    """
    children_list = list(children)

    for name, bbox in children_list:
        if not parent.contains(bbox, tolerance=tolerance):
            return ValidationConflict(
                validator="bbox_containment",
                detail=f"child {name!r} bbox is not contained within parent bbox",
                data={
                    "child": name,
                    "child_bbox": bbox.model_dump(),
                    "parent_bbox": parent.model_dump(),
                },
            )

    for i in range(len(children_list)):
        for j in range(i + 1, len(children_list)):
            a_name, a_bbox = children_list[i]
            b_name, b_bbox = children_list[j]
            if a_bbox.overlaps(b_bbox, tolerance=tolerance):
                return ValidationConflict(
                    validator="bbox_sibling_overlap",
                    detail=f"bboxes {a_name!r} and {b_name!r} overlap",
                    data={
                        "a": a_name,
                        "b": b_name,
                        "a_bbox": a_bbox.model_dump(),
                        "b_bbox": b_bbox.model_dump(),
                    },
                )

    return None


def validate_completeness(
    *,
    expected_ids: Iterable[str],
    bboxes: dict[str, BoundingBox],
) -> ValidationConflict | None:
    """Every expected id must have a bbox. Missing entries are a conflict."""
    missing = [eid for eid in expected_ids if eid not in bboxes]
    if missing:
        return ValidationConflict(
            validator="bbox_completeness",
            detail=f"missing bboxes for: {sorted(missing)}",
            data={"missing": sorted(missing)},
        )
    return None
