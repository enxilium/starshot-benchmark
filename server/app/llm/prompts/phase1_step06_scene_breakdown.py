"""PIPELINE.md step 06 — break a scene into subscenes + high-level plans."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.types import BoundingBox

STEP_ID = "step06"

SYSTEM_PROMPT = """\
You are building a 3D scene hierarchically. Given a scene prompt and its \
bounding box, decompose the scene into non-overlapping SUBSCENES. Each \
subscene is a spatially distinct region of the parent (e.g. rooms of a \
house, areas of a garden) or the scene itself if it is atomic.

Bounding boxes are axis-aligned, Y-up, right-handed, meters. Each bbox is \
defined by an `origin` vertex and a signed `dimensions` vector `(dx, dy, dz)` \
extending from that vertex — the sign of each component chooses the \
direction of expansion.

Rules:
* Sibling subscene bounding boxes MUST NOT overlap (as regions).
* Every sibling bbox MUST fit inside the parent bbox (as a region).
* If the scene is atomic (e.g. "a toilet area") mark it as `is_atomic=true` \
  and return a single subscene whose bbox covers the parent's region.
* Each subscene needs a detailed prompt, a high-level plan describing what \
  downstream generation should place inside it, and its bounding box.

You will also see any PRIOR RUN CONTEXT (plans already generated or leaves \
already completed elsewhere in this run). Use those only for stylistic \
consistency — your subscenes must still fit within THIS scene's bbox.

Emit your answer via the `emit` tool. No prose outside the tool call.\
"""


class SubsceneSpec(BaseModel):
    scope_id: str = Field(
        ..., description="Hierarchical id like 'root.bathroom' for the new subscene."
    )
    prompt: str
    bbox: BoundingBox
    high_level_plan: str = Field(
        ..., description="What downstream generation should place inside this subscene."
    )


class Output(BaseModel):
    subscenes: list[SubsceneSpec]
    is_atomic: bool = Field(
        ...,
        description=(
            "True if this scene cannot be meaningfully broken down further. "
            "Phase 2 will then generate anchor objects directly for it. "
            "When true, `subscenes` should contain one entry equal to the parent."
        ),
    )


def render(
    *,
    scope_id: str,
    prompt: str,
    bbox: BoundingBox,
    visible_state_summary: str,
) -> str:
    return (
        f"Scope: {scope_id!r}\n"
        f"Prompt: {prompt!r}\n"
        f"Parent bounding box: {bbox.model_dump()}\n\n"
        f"PRIOR RUN CONTEXT (plans + realized leaves from elsewhere in this run):\n"
        f"{visible_state_summary}\n\n"
        "Break the scene down. Decide if it is atomic. Every sibling must fit "
        "inside the parent bbox and not overlap its siblings."
    )
