"""
Relationship-graph validator and topological sort for phase-2 anchor objects.

Responsibilities (shared validator used by PIPELINE.md steps 10 and 14):
  * Inject implied inverses (A ABOVE B ⇒ B BELOW A).
  * Treat BESIDE and ATTACHED as symmetric (no duplicate required).
  * Reject contradictions (A ABOVE B and A BELOW B on the same pair).
  * Reject ATTACHED relationships whose target is not a frame.
  * Require every anchor object to participate in at least one relationship.
  * Require every anchor object to reach a frame via the dependency graph —
    otherwise there is no anchor point and its bbox is undetermined.
  * Detect cycles; a cycle means no valid bbox-resolution order exists.
  * Produce a topological order used by phase 2 step 4: frames first, then
    objects in dependency order.

All checks return a `ValidationConflict` or `None`. The topo sort only runs
once the graph validates; on success the function returns the order, on
failure it returns the conflict.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from app.core.errors import ValidationConflict
from app.core.types import (
    INVERSE_KINDS,
    AnchorObject,
    Relationship,
    RelationshipKind,
)


@dataclass(frozen=True)
class GraphResult:
    """Output of `validate_and_sort` on success."""

    order: list[str]                  # topological order of object ids (frames are prerequisites, not included here)
    normalized: list[Relationship]    # input list with implied inverses added
    dependencies: dict[str, set[str]]  # object id -> set of prerequisite ids (objects or frames)


def _add_implied_inverses(rels: list[Relationship]) -> list[Relationship]:
    out = list(rels)
    present = {(r.subject, r.kind, r.target) for r in rels}
    for r in rels:
        inv = INVERSE_KINDS.get(r.kind)
        if inv is None:
            continue
        key = (r.target, inv, r.subject)
        if key not in present:
            out.append(Relationship(subject=r.target, kind=inv, target=r.subject))
            present.add(key)
    return out


def _check_contradictions(rels: list[Relationship]) -> ValidationConflict | None:
    """A contradicting pair is two directed relationships on the same (subject, target)
    with opposite kinds — e.g. A ABOVE B and A BELOW B."""
    seen: dict[tuple[str, str], set[RelationshipKind]] = defaultdict(set)
    for r in rels:
        seen[(r.subject, r.target)].add(r.kind)

    for (subj, tgt), kinds in seen.items():
        # ABOVE + BELOW on the SAME subject->target ordering is a contradiction.
        if RelationshipKind.ABOVE in kinds and RelationshipKind.BELOW in kinds:
            return ValidationConflict(
                validator="relationship_contradiction",
                detail=(
                    f"{subj!r} is both ABOVE and BELOW {tgt!r}; only one can hold"
                ),
                data={"subject": subj, "target": tgt, "kinds": [k.value for k in kinds]},
            )

    return None


def validate_and_sort(
    *,
    objects: list[AnchorObject],
    frame_ids: set[str],
    relationships: list[Relationship],
) -> GraphResult | ValidationConflict:
    """
    Run all phase-2 step-3 checks and return either a `GraphResult` (topo order +
    normalized relationships) or the first `ValidationConflict` encountered.
    """
    object_ids = {o.id for o in objects}
    all_node_ids = object_ids | frame_ids

    # 1. Duplicate id check (defensive — the LLM should not produce these)
    if len(object_ids) != len(objects):
        return ValidationConflict(
            validator="relationship_duplicate_object",
            detail="duplicate anchor-object ids detected",
            data={"ids": sorted([o.id for o in objects])},
        )

    # 2. Target validity
    for r in relationships:
        if r.subject not in object_ids:
            return ValidationConflict(
                validator="relationship_invalid_subject",
                detail=f"relationship subject {r.subject!r} is not a known anchor object",
                data={"subject": r.subject, "known_objects": sorted(object_ids)},
            )
        if r.target not in all_node_ids:
            return ValidationConflict(
                validator="relationship_invalid_target",
                detail=f"relationship target {r.target!r} is neither an anchor object nor a frame",
                data={
                    "target": r.target,
                    "known_objects": sorted(object_ids),
                    "known_frames": sorted(frame_ids),
                },
            )
        if r.kind == RelationshipKind.ATTACHED and r.target not in frame_ids:
            return ValidationConflict(
                validator="relationship_attached_to_non_frame",
                detail=(
                    f"ATTACHED relationship must target a frame, got {r.target!r}"
                ),
                data={"target": r.target, "known_frames": sorted(frame_ids)},
            )

    # 3. Inject implied inverses
    normalized = _add_implied_inverses(relationships)

    # 4. Contradiction check
    conflict = _check_contradictions(normalized)
    if conflict is not None:
        return conflict

    # 5. Coverage: every object participates in at least one relationship
    participants: set[str] = set()
    for r in normalized:
        participants.add(r.subject)
        participants.add(r.target)
    uncovered = object_ids - participants
    if uncovered:
        return ValidationConflict(
            validator="relationship_coverage",
            detail=f"anchor objects with no relationships: {sorted(uncovered)}",
            data={"uncovered": sorted(uncovered)},
        )

    # 6. Build the dependency graph from the ORIGINAL relationships only.
    # Implied inverses are informational; feeding them back as graph edges
    # would collapse every ABOVE/BELOW pair into a cycle. The LLM chose the
    # direction — that's the one that carries dependency.
    deps: dict[str, set[str]] = {oid: set() for oid in object_ids}
    for r in relationships:
        if r.subject in object_ids:
            deps[r.subject].add(r.target)

    # 7. Every object must transitively reach a frame.
    if frame_ids:
        reachable_to_frame = _find_objects_reaching_frames(deps, object_ids, frame_ids)
        unreachable = object_ids - reachable_to_frame
        if unreachable:
            return ValidationConflict(
                validator="relationship_frame_unreachable",
                detail=(
                    f"anchor objects do not transitively depend on any frame: "
                    f"{sorted(unreachable)}"
                ),
                data={"unreachable": sorted(unreachable)},
            )

    # 8. Topological sort (Kahn) over objects only (frames are pre-resolved).
    # Consider only edges between two objects; frame edges anchor the chain.
    obj_deps: dict[str, set[str]] = {
        oid: {d for d in deps[oid] if d in object_ids} for oid in object_ids
    }
    incoming_count: dict[str, int] = {oid: len(obj_deps[oid]) for oid in object_ids}
    reverse: dict[str, set[str]] = defaultdict(set)
    for oid, ds in obj_deps.items():
        for d in ds:
            reverse[d].add(oid)

    ready: deque[str] = deque(sorted(oid for oid, n in incoming_count.items() if n == 0))
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for dependent in sorted(reverse.get(node, ())):
            incoming_count[dependent] -= 1
            if incoming_count[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(object_ids):
        unresolved = sorted(object_ids - set(order))
        return ValidationConflict(
            validator="relationship_cycle",
            detail=f"cycle detected — could not topologically order: {unresolved}",
            data={"unresolved": unresolved},
        )

    return GraphResult(order=order, normalized=normalized, dependencies=deps)


def _find_objects_reaching_frames(
    deps: dict[str, set[str]],
    object_ids: set[str],
    frame_ids: set[str],
) -> set[str]:
    """
    Return the subset of `object_ids` whose dependency chain reaches at least
    one frame. BFS from each object through its `deps`.
    """
    reachable: set[str] = set()
    for start in object_ids:
        visited: set[str] = set()
        stack: list[str] = [start]
        found = False
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in frame_ids:
                found = True
                break
            for d in deps.get(node, ()):
                if d not in visited:
                    stack.append(d)
        if found:
            reachable.add(start)
    return reachable
