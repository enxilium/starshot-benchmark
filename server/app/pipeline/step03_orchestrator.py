"""
PIPELINE.md step 3 — top-level pipeline orchestrator.

Flow per run:
  * Emit `RunStarted`.
  * Step 8 (`step08_recurse.divide`) — build the `SubsceneNode` tree.
  * Step 9 (`step09_collect_leaves`) — flatten to leaves + frame chains.
  * Per leaf: step 14 (`step14_completion_loop.generate_leaf`) — runs
    phase-2 steps 10 → 11 → 12 → 13 → 14 loop → 15 internally.
  * Step 16 (`step16_assemble_scene.build_scene`) — merge every mesh.
  * Step 17 (`step17_publish_glb.publish`) — write the `.glb`.
  * Emit `RunCompleted` with the URL + retry summary.

Sequential per-leaf iteration keeps the State Repository's realized
entries observable in a deterministic order by any future interleaved
phase-1 / phase-2 variation. Today's implementation runs all of phase 1
first, then phase 2 — the design already accommodates relaxing that.
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
from app.pipeline import step16_assemble_scene, step17_publish_glb
from app.pipeline.phase1 import step08_recurse, step09_collect_leaves
from app.pipeline.phase2 import step14_completion_loop


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
        # Step 8 — recursive divide
        root = await step08_recurse.divide(user_prompt)

        # Step 9 — flatten to leaves + frame chains, then step 14 per leaf
        leaves = step09_collect_leaves.collect_leaves(root)
        generated_leaves = []
        for leaf in leaves:
            ancestor_frames = step09_collect_leaves.collect_frames_for_leaf(root, leaf)
            generated = await step14_completion_loop.generate_leaf(
                leaf=leaf, frames=ancestor_frames
            )
            generated_leaves.append(generated)

        # Step 16 — assemble final scene
        object_meshes: list[tuple[str, object]] = []
        for gl in generated_leaves:
            for oid, mesh in gl.meshes.items():
                object_meshes.append((oid, mesh))
        all_frames = _collect_all_frames(root)
        scene = step16_assemble_scene.build_scene(
            object_meshes=object_meshes,  # type: ignore[arg-type]
            frames=all_frames,
        )

        # Step 17 — publish
        _, url = step17_publish_glb.publish(ctx.run_id, scene)

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
