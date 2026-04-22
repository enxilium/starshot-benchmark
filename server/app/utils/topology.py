"""Graph utilities."""

from __future__ import annotations

from app.core.prompts import ChildNodeSpec


def toposort_children(children: list[ChildNodeSpec]) -> list[ChildNodeSpec]:
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

    ordered: list[ChildNodeSpec] = []
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
