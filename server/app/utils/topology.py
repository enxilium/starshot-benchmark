"""Graph utilities."""

from __future__ import annotations

from typing import TypeVar

from app.core.prompts import ChildNodeSpec, ObjectSpec

T = TypeVar("T", bound=ChildNodeSpec)


def toposort_children(children: list[T]) -> list[T]:
    """Order children so each is placed after all its sibling dependencies.

    Relationships targeting the parent (or any non-sibling id) do not
    constrain order. Self-loops are ignored. Cycles raise ValueError.
    Stable: independent children keep their input order.
    """
    by_id = {c.id: c for c in children}
    deps: dict[str, set[str]] = {c.id: set() for c in children}
    for c in children:
        for r in c.relationships:
            if r.target in by_id and r.target != c.id:
                deps[c.id].add(r.target)

    ordered: list[T] = []
    remaining = {c.id for c in children}
    while remaining:
        ready = [
            c for c in children
            if c.id in remaining and deps[c.id] <= {s.id for s in ordered}
        ]
        if not ready:
            raise ValueError(f"cyclic sibling relationships among: {sorted(remaining)}")
        ordered.extend(ready)
        remaining -= {c.id for c in ready}
    return ordered


def validate_object_relationships(
    specs: list[ObjectSpec],
    *,
    zone_id: str,
    existing_ids: set[str],
) -> None:
    """Raise ValueError if the object graph is malformed.

    Checks:
      1. ids unique within `specs` and disjoint from `existing_ids`.
      2. `spec.parent` resolves to `zone_id`, another spec id, or a prior
         existing id.
      3. At least one relationship per spec targets `spec.parent`.
      4. Every relationship target resolves to `zone_id`, a spec id, or a
         prior existing id.
      5. Object-to-object parent edges among `specs` form a DAG.
    """
    spec_ids = [s.id for s in specs]
    if len(spec_ids) != len(set(spec_ids)):
        raise ValueError(f"duplicate ids among object specs: {spec_ids}")
    collisions = set(spec_ids) & existing_ids
    if collisions:
        raise ValueError(f"object ids collide with existing nodes: {sorted(collisions)}")

    known = {zone_id} | existing_ids | set(spec_ids)
    for s in specs:
        if s.parent not in known:
            raise ValueError(f"object {s.id!r} has unknown parent {s.parent!r}")
        if not any(r.target == s.parent for r in s.relationships):
            raise ValueError(
                f"object {s.id!r} has no relationship targeting its parent {s.parent!r}"
            )
        for r in s.relationships:
            if r.target not in known:
                raise ValueError(
                    f"object {s.id!r} has relationship with unknown target {r.target!r}"
                )

    by_id = {s.id: s for s in specs}
    color: dict[str, int] = {}  # 0=unseen, 1=on-stack, 2=done

    def visit(node_id: str) -> None:
        state = color.get(node_id, 0)
        if state == 1:
            raise ValueError(f"cyclic object-parent chain through {node_id!r}")
        if state == 2:
            return
        color[node_id] = 1
        parent = by_id[node_id].parent
        if parent in by_id:
            visit(parent)
        color[node_id] = 2

    for s in specs:
        visit(s.id)
