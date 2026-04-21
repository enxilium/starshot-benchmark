"""
PIPELINE.md step 6 — break a scene into subscenes.

Wraps `call_with_validator` with the shared bbox validator: siblings
must not overlap and each must be contained in the parent bbox. Retries
on conflict up to `ctx.max_retries`.

When `is_atomic=True`, the bbox validator is skipped (an atomic scene
has no meaningful children to validate).

Reads `StateRepository.read_visible()` once per step invocation and
feeds the summary into the prompt — this is how step 8 surfaces
already-completed subscenes from elsewhere in the recursion for
stylistic consistency.
"""

from __future__ import annotations

from app.core.context import current
from app.core.errors import ValidationConflict
from app.core.types import BoundingBox
from app.geometry.bbox_validator import validate_boxes
from app.llm.client import PriorAttempt, PromptPayload, call_with_validator
from app.llm.prompts.phase1_step06_scene_breakdown import (
    STEP_ID,
    SYSTEM_PROMPT,
    Output,
    render,
)
from app.state_repo import VisibleState


def _summarize_visible(state: VisibleState) -> str:
    parts: list[str] = []
    if state.plans:
        parts.append("Plans in flight:")
        for p in state.plans:
            parts.append(f"  - {p.scope_id!r} -> {p.high_level_plan!r}")
    if state.realized:
        parts.append("Leaves already realized:")
        for r in state.realized:
            object_ids = [o.id for o in r.objects]
            parts.append(f"  - {r.scope_id!r} contains objects {object_ids}")
    if not parts:
        return "  (none)"
    return "\n".join(parts)


async def run(*, scope_id: str, prompt: str, bbox: BoundingBox) -> Output:
    ctx = current()

    async def _visible_summary() -> str:
        visible = await ctx.state_repo.read_visible()
        return _summarize_visible(visible)

    # Pre-fetch once per step invocation (the summary doesn't change across
    # retries within a single step run, only across recursions).
    visible_summary = await _visible_summary()

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(
            system=SYSTEM_PROMPT,
            user=render(
                scope_id=scope_id,
                prompt=prompt,
                bbox=bbox,
                visible_state_summary=visible_summary,
            ),
            prior_attempts=prior,
        )

    def validate(out: Output) -> ValidationConflict | None:
        if out.is_atomic:
            return None  # no children to check for atomic scenes
        if not out.subscenes:
            return ValidationConflict(
                validator="step06_empty_breakdown",
                detail="non-atomic scene must emit at least one subscene",
            )
        children = [(sub.scope_id, sub.bbox) for sub in out.subscenes]
        return validate_boxes(parent=bbox, children=children)

    return await call_with_validator(
        step_id=STEP_ID,
        llm=ctx.llm,
        events=ctx.events,
        max_retries=ctx.max_retries,
        output_schema=Output,
        build_prompt=build,
        validate=validate,
    )
