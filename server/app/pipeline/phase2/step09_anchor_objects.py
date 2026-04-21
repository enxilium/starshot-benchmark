"""
PIPELINE.md step 9 — anchor objects + relationships.

Wraps `call_with_validator` with the relationship-graph validator. The
`GraphResult` (topologically sorted order + normalized relationships)
is returned alongside the validated (objects, relationships) pair so
the downstream bbox step (step 10) doesn't need to re-run the DAG
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.context import current
from app.core.errors import ValidationConflict
from app.core.types import AnchorObject, Frame, Relationship, SubsceneNode
from app.geometry.relationship_graph import GraphResult, validate_and_sort
from app.llm.client import PriorAttempt, PromptPayload, call_with_validator
from app.llm.prompts.phase2_step09_anchor_objects import (
    STEP_ID,
    SYSTEM_PROMPT,
    Output,
    render,
)

from .frame_summary import summarize_frames


@dataclass(frozen=True)
class Step9Result:
    objects: list[AnchorObject]
    relationships: list[Relationship]
    graph: GraphResult


async def run(*, leaf: SubsceneNode, frames: list[Frame]) -> Step9Result:
    ctx = current()
    frame_ids = {f.id for f in frames}
    frame_summary = summarize_frames(frames)

    # We need the GraphResult produced during validate(). Stash it in a closure.
    graph_holder: dict[str, GraphResult] = {}

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(
            system=SYSTEM_PROMPT,
            user=render(prompt=leaf.prompt, bbox=leaf.bbox, frame_summary=frame_summary),
            prior_attempts=prior,
        )

    def validate(out: Output) -> ValidationConflict | None:
        objects = [AnchorObject(id=o.id, prompt=o.prompt) for o in out.objects]
        result = validate_and_sort(
            objects=objects, frame_ids=frame_ids, relationships=out.relationships
        )
        if isinstance(result, ValidationConflict):
            return result
        graph_holder["r"] = result
        return None

    out = await call_with_validator(
        step_id=STEP_ID,
        llm=ctx.llm,
        events=ctx.events,
        max_retries=ctx.max_retries,
        output_schema=Output,
        build_prompt=build,
        validate=validate,
    )
    objects = [AnchorObject(id=o.id, prompt=o.prompt) for o in out.objects]
    return Step9Result(
        objects=objects,
        relationships=out.relationships,
        graph=graph_holder["r"],
    )
