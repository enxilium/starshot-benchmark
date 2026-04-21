"""
PIPELINE.md step 10 — LLM-generated bboxes for anchor objects.

Supports two modes:
  * Initial: `already_resolved={}`, `to_resolve=<all objects in topo order>`.
  * Incremental (step-13 completion loop): `already_resolved=<all prior>`,
    `to_resolve=[new_object_id]`.

Wraps `call_with_validator` with the shared bbox validator. Retries use
the conflict-feedback contract. On success, returns a dict of
`object_id -> BoundingBox` for the resolved subset only (the caller
merges it with `already_resolved` as needed).
"""

from __future__ import annotations

from app.core.context import current
from app.core.errors import ValidationConflict
from app.core.types import AnchorObject, BoundingBox, Frame, Relationship, SubsceneNode
from app.geometry.bbox_validator import validate_boxes
from app.llm.client import PriorAttempt, PromptPayload, call_with_validator
from app.llm.prompts.phase2_step10_object_bboxes import (
    STEP_ID,
    SYSTEM_PROMPT,
    Output,
    render,
)

from .frame_summary import summarize_frames, summarize_relationships


async def run(
    *,
    leaf: SubsceneNode,
    frames: list[Frame],
    objects: list[AnchorObject],
    relationships: list[Relationship],
    topo_order: list[str],
    already_resolved: dict[str, BoundingBox],
    to_resolve: list[str],
) -> dict[str, BoundingBox]:
    ctx = current()
    _ = objects  # kept in signature for symmetry; topo_order carries the ids

    frames_summary = summarize_frames(frames)
    rels_summary = summarize_relationships(relationships)

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(
            system=SYSTEM_PROMPT,
            user=render(
                leaf_prompt=leaf.prompt,
                leaf_bbox=leaf.bbox,
                frames_summary=frames_summary,
                relationships_summary=rels_summary,
                topo_order=topo_order,
                already_resolved=already_resolved,
                to_resolve=to_resolve,
            ),
            prior_attempts=prior,
        )

    def validate(out: Output) -> ValidationConflict | None:
        emitted_ids = [a.object_id for a in out.assignments]
        expected = set(to_resolve)
        got = set(emitted_ids)
        if got != expected:
            missing = sorted(expected - got)
            extras = sorted(got - expected)
            return ValidationConflict(
                validator="step10_coverage",
                detail=(
                    f"assignments mismatch: expected ids {sorted(expected)}, "
                    f"got {sorted(got)} (missing={missing}, extras={extras})"
                ),
                data={"expected": sorted(expected), "got": sorted(got)},
            )
        # Build the full bbox set (already_resolved + new) and validate it
        # against the parent leaf bbox: no overlap, all contained.
        combined: dict[str, BoundingBox] = dict(already_resolved)
        for a in out.assignments:
            combined[a.object_id] = a.bbox
        return validate_boxes(parent=leaf.bbox, children=list(combined.items()))

    out = await call_with_validator(
        step_id=STEP_ID,
        llm=ctx.llm,
        events=ctx.events,
        max_retries=ctx.max_retries,
        output_schema=Output,
        build_prompt=build,
        validate=validate,
    )
    return {a.object_id: a.bbox for a in out.assignments}
