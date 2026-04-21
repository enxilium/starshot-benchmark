from __future__ import annotations

from app.core.errors import ValidationConflict
from app.core.types import AnchorObject, Relationship, RelationshipKind
from app.geometry.relationship_graph import GraphResult, validate_and_sort


def obj(oid: str) -> AnchorObject:
    return AnchorObject(id=oid, prompt=f"{oid} prompt")


def rel(
    subject: str,
    kind: RelationshipKind,
    target: str,
    attachment: tuple[float, float] | None = None,
) -> Relationship:
    return Relationship(subject=subject, kind=kind, target=target, attachment=attachment)


def test_valid_graph_with_attachment_and_besides() -> None:
    objects = [obj("table"), obj("chair"), obj("tv")]
    frame_ids = {"floor", "wall_n"}
    relationships = [
        rel("table", RelationshipKind.ATTACHED, "floor", (0.5, 0.5)),
        rel("chair", RelationshipKind.BESIDE, "table"),
        rel("tv", RelationshipKind.ATTACHED, "wall_n", (0.5, 0.7)),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, GraphResult)
    # table must come before chair (chair depends on table)
    assert result.order.index("table") < result.order.index("chair")


def test_implied_inverse_added_for_above_below() -> None:
    objects = [obj("a"), obj("b")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0.1, 0.1)),
        rel("b", RelationshipKind.ABOVE, "a"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, GraphResult)
    # implied inverse (a BELOW b) should be present
    assert any(
        r.subject == "a" and r.kind == RelationshipKind.BELOW and r.target == "b"
        for r in result.normalized
    )


def test_contradiction_above_and_below_same_pair() -> None:
    objects = [obj("a"), obj("b")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0.0, 0.0)),
        rel("b", RelationshipKind.ATTACHED, "floor", (1.0, 1.0)),
        rel("b", RelationshipKind.ABOVE, "a"),
        rel("b", RelationshipKind.BELOW, "a"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_contradiction"


def test_object_uncovered_by_any_relationship() -> None:
    objects = [obj("a"), obj("b")]
    frame_ids = {"floor"}
    relationships = [rel("a", RelationshipKind.ATTACHED, "floor", (0.0, 0.0))]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_coverage"
    assert result.data["uncovered"] == ["b"]


def test_attached_target_must_be_a_frame() -> None:
    objects = [obj("a"), obj("b")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0.0, 0.0)),
        rel("b", RelationshipKind.ATTACHED, "a", (0.0, 0.0)),  # a is an object
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_attached_to_non_frame"


def test_invalid_target_id() -> None:
    objects = [obj("a")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0.0, 0.0)),
        rel("a", RelationshipKind.BESIDE, "ghost"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_invalid_target"


def test_invalid_subject_id() -> None:
    objects = [obj("a")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0, 0)),
        rel("ghost", RelationshipKind.BESIDE, "a"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_invalid_subject"


def test_cycle_detected_and_reported() -> None:
    # Two objects that only reference each other — no frame attachment.
    # The reachability check catches this first (frame_unreachable), but
    # this test validates cycle detection explicitly by attaching BOTH to frames
    # and adding a redundant mutual BESIDE→triggered only when there's a true
    # cycle between objects.
    objects = [obj("a"), obj("b"), obj("c")]
    frame_ids = {"floor"}
    # All three objects attached to floor, and a chain a BESIDE b, b BESIDE c,
    # c BESIDE a. Each BESIDE makes subject depend on target, so edges are
    # b→a, c→b, a→c — that's a cycle.
    relationships = [
        rel("a", RelationshipKind.ATTACHED, "floor", (0, 0)),
        rel("b", RelationshipKind.ATTACHED, "floor", (0.5, 0)),
        rel("c", RelationshipKind.ATTACHED, "floor", (1, 0)),
        rel("a", RelationshipKind.BESIDE, "b"),
        rel("b", RelationshipKind.BESIDE, "c"),
        rel("c", RelationshipKind.BESIDE, "a"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_cycle"


def test_frame_unreachable_flagged_when_no_attached_chain() -> None:
    # Two objects only BESIDE each other — no path to any frame.
    objects = [obj("a"), obj("b")]
    frame_ids = {"floor"}
    relationships = [
        rel("a", RelationshipKind.BESIDE, "b"),
        rel("b", RelationshipKind.BESIDE, "a"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, ValidationConflict)
    assert result.validator == "relationship_frame_unreachable"


def test_topo_order_respects_chains() -> None:
    # A chain: table → lamp → book. Lamp depends on table; book depends on lamp.
    objects = [obj("table"), obj("lamp"), obj("book")]
    frame_ids = {"floor"}
    relationships = [
        rel("table", RelationshipKind.ATTACHED, "floor", (0.5, 0.5)),
        rel("lamp", RelationshipKind.ON, "table"),
        rel("book", RelationshipKind.ON, "lamp"),
    ]
    result = validate_and_sort(
        objects=objects, frame_ids=frame_ids, relationships=relationships
    )
    assert isinstance(result, GraphResult)
    assert result.order == ["table", "lamp", "book"]
