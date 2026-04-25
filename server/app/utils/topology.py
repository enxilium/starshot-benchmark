"""Graph utilities."""

from __future__ import annotations

from app.core.prompts import ChildNodeSpec, ObjectSpec


def validate_sibling_relationships_acyclic(children: list[ChildNodeSpec]) -> None:
    """Raise ValueError if any set of siblings forms a cyclic relationship.

    The batch bbox resolver places every sibling in one shot, so ordering
    doesn't matter — but a cyclic relationship graph (A ABOVE B, B ABOVE A)
    is semantically contradictory and should be caught before the LLM is
    asked to satisfy it. Relationships targeting the parent (or any
    non-sibling id) don't contribute edges. Self-loops are ignored.
    """
    by_id = {c.id: c for c in children}
    deps: dict[str, set[str]] = {c.id: set() for c in children}
    for c in children:
        for r in c.relationships:
            if r.target in by_id and r.target != c.id:
                deps[c.id].add(r.target)

    color: dict[str, int] = {}  # 0=unseen, 1=on-stack, 2=done

    def visit(node_id: str) -> None:
        state = color.get(node_id, 0)
        if state == 1:
            raise ValueError(
                f"cyclic sibling relationships through {node_id!r}"
            )
        if state == 2:
            return
        color[node_id] = 1
        for dep in deps[node_id]:
            visit(dep)
        color[node_id] = 2

    for c in children:
        visit(c.id)


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

    validate_sibling_relationships_acyclic(specs)
