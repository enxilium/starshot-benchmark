"""
PIPELINE.md step 5 — decide whether a subscene needs frames and, if so,
emit concrete PlaneFrame / CurveFrame instances.

Runs for every subscene (leaf AND non-leaf). The resulting frames are
attached to the SubsceneNode and inherited by phase-2 leaves below them.
"""

from __future__ import annotations

from app.core.context import current
from app.core.events import StepCompleted, StepStarted
from app.core.types import BoundingBox, CurveFrame, Frame, PlaneFrame
from app.llm.client import PromptPayload
from app.llm.prompts.phase1_step05_frame_decider import (
    STEP_ID,
    SYSTEM_PROMPT,
    CurveFrameSpec,
    Output,
    PlaneFrameSpec,
    render,
)


def _spec_to_frame(spec: PlaneFrameSpec | CurveFrameSpec) -> Frame:
    if isinstance(spec, PlaneFrameSpec):
        return PlaneFrame(
            id=spec.id, origin=spec.origin, u_axis=spec.u_axis, v_axis=spec.v_axis
        )
    return CurveFrame(id=spec.id, control_points=spec.control_points, height=spec.height)


async def run(*, prompt: str, bbox: BoundingBox) -> list[Frame]:
    ctx = current()
    await ctx.events.emit(StepStarted(step_id=STEP_ID))
    payload = PromptPayload(
        system=SYSTEM_PROMPT, user=render(prompt=prompt, bbox=bbox)
    )
    out = await ctx.llm.call_structured(
        step_id=STEP_ID, prompt=payload, output_schema=Output
    )
    frames: list[Frame] = [] if not out.needs_frame else [_spec_to_frame(s) for s in out.frames]
    await ctx.events.emit(
        StepCompleted(
            step_id=STEP_ID,
            duration_ms=0.0,
            output_summary={"needs_frame": out.needs_frame, "n_frames": len(frames)},
        )
    )
    return frames
