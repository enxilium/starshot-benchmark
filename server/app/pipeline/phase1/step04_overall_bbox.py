"""
PIPELINE.md step 4 — overall bounding box. Runs ONCE at the root of the
recursion.

No validator retry: the emitted bbox is either syntactically valid
(Pydantic enforces non-zero `dimensions`) or it's a provider-level
failure. There is no deterministic validator on the root bbox beyond
the schema check.
"""

from __future__ import annotations

from app.core.context import current
from app.core.events import StepCompleted, StepStarted
from app.core.types import BoundingBox
from app.llm.client import PromptPayload
from app.llm.prompts.phase1_step04_overall_bbox import (
    STEP_ID,
    SYSTEM_PROMPT,
    Output,
    render,
)


async def run(user_prompt: str) -> BoundingBox:
    ctx = current()
    await ctx.events.emit(StepStarted(step_id=STEP_ID))
    payload = PromptPayload(system=SYSTEM_PROMPT, user=render(user_prompt))
    out = await ctx.llm.call_structured(
        step_id=STEP_ID, prompt=payload, output_schema=Output
    )
    await ctx.events.emit(
        StepCompleted(
            step_id=STEP_ID,
            duration_ms=0.0,
            output_summary={"size": out.bbox.size},
        )
    )
    return out.bbox
