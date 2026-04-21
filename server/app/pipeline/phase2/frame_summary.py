"""Short, LLM-facing summaries of frame + relationship state for prompt bodies."""

from __future__ import annotations

from app.core.types import Frame, PlaneFrame, Relationship


def summarize_frames(frames: list[Frame]) -> str:
    if not frames:
        return "  (no frames — this is an outdoor / open subscene)"
    lines: list[str] = []
    for f in frames:
        if isinstance(f, PlaneFrame):
            lines.append(
                f"  - id={f.id!r} kind=plane origin={f.origin} u_axis={f.u_axis} v_axis={f.v_axis}"
            )
        else:
            # Frame is a discriminated union of PlaneFrame | CurveFrame.
            lines.append(
                f"  - id={f.id!r} kind=curve height={f.height} control_points={f.control_points}"
            )
    return "\n".join(lines)


def summarize_relationships(rels: list[Relationship]) -> str:
    if not rels:
        return "  (none)"
    lines: list[str] = []
    for r in rels:
        tail = f" attachment={r.attachment}" if r.attachment is not None else ""
        lines.append(f"  - {r.subject} {r.kind.value} {r.target}{tail}")
    return "\n".join(lines)
