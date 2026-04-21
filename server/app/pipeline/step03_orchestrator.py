"""
PIPELINE.md step 3 — top-level pipeline orchestrator.

Phase 1 and phase 2 are **interleaved**: `step08_recurse.divide` dives
DFS and fires the full phase 2 on every leaf before unwinding. The
orchestrator receives back both the fully-populated `SubsceneNode` tree
and the list of `GeneratedLeaf`s in DFS-realization order.

Flow per run:
  * Emit `RunStarted`.
  * Steps 4-14 (`step08_recurse.divide` with `realize_leaves=True`):
      - step 4 at the root (overall bbox)
      - per node: step 5 (frames), step 6 (breakdown)
      - leaves: phase 2 fires inline (steps 9 → 10 → 11 → 12 → 13 loop
        → 14) before the recursion unwinds
      - non-leaves: step 7 (write plans), then recurse (step 8)
  * Step 15 (`step15_assemble_scene.build_scene`) — merge every mesh.
  * Step 16 (`step16_publish_glb.publish`) — write the `.glb`.
  * Emit `RunCompleted` with the URL + retry summary.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from app.core.context import current
from app.core.events import (
    Event,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepRetried,
    parse_event_line,
)
from app.core.types import Frame, SubsceneNode
from app.pipeline import step15_assemble_scene, step16_publish_glb
from app.pipeline.phase1 import step08_recurse


def _collect_all_frames(root: SubsceneNode) -> list[Frame]:
    frames: list[Frame] = []
    _walk_frames(root, frames)
    return frames


def _walk_frames(node: SubsceneNode, out: list[Frame]) -> None:
    out.extend(node.frames)
    for child in node.children:
        _walk_frames(child, out)


def _retry_summary_from_events(events_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not events_path.exists():
        return {}
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event: Event = parse_event_line(line)
            if isinstance(event, StepRetried):
                counts[event.step_id] += 1
    return dict(counts)


async def run(user_prompt: str, model_id: str) -> str:
    """Execute the full pipeline for `user_prompt`. Returns the published GLB URL."""
    ctx = current()
    t0 = time.perf_counter()
    await ctx.events.emit(RunStarted(prompt=user_prompt, model_id=model_id))

    try:
        # Steps 4-14 (interleaved phase 1 + phase 2).
        root, generated_leaves = await step08_recurse.divide(user_prompt)

        # Step 15 — assemble final scene.
        object_meshes: list[tuple[str, object]] = []
        for gl in generated_leaves:
            for oid, mesh in gl.meshes.items():
                object_meshes.append((oid, mesh))
        all_frames = _collect_all_frames(root)
        scene = step15_assemble_scene.build_scene(
            object_meshes=object_meshes,  # type: ignore[arg-type]
            frames=all_frames,
        )

        # Step 16 — publish.
        _, url = step16_publish_glb.publish(ctx.run_id, scene)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        retry_summary = _retry_summary_from_events(ctx.events.path)
        await ctx.events.emit(
            RunCompleted(
                glb_url=url,
                total_duration_ms=duration_ms,
                retry_summary=retry_summary,
            )
        )
        return url

    except Exception as e:
        step_id = getattr(e, "step_id", None)
        await ctx.events.emit(RunFailed(error=repr(e), step_id=step_id))
        raise
