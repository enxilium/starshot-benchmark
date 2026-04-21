"""
PIPELINE.md step 12 — mesh generation via the injected `MeshGenerator`.

Runs in bounded-concurrency parallel
(`asyncio.Semaphore(mesh_gen_concurrency)`) since mesh generation
(especially against Hunyuan 3.1) is the slowest step. Per-object
`MeshGenerated` events are emitted with timing + backend.

Failures propagate; they are NOT retried under `ctx.max_retries` (that
budget is for validator-driven LLM retries only).
"""

from __future__ import annotations

import asyncio
import time

import trimesh

from app.core.config import get_settings
from app.core.context import current
from app.core.events import MeshGenerated
from app.core.types import AnchorObject


async def run(*, objects: list[AnchorObject]) -> dict[str, trimesh.Trimesh]:
    ctx = current()
    settings = get_settings()
    sem = asyncio.Semaphore(settings.mesh_gen_concurrency)

    async def _one(obj: AnchorObject) -> tuple[str, trimesh.Trimesh]:
        async with sem:
            t0 = time.perf_counter()
            mesh = await ctx.mesh_generator.generate(obj.prompt, obj.id)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            await ctx.events.emit(
                MeshGenerated(
                    object_id=obj.id,
                    duration_ms=duration_ms,
                    backend=ctx.mesh_generator.name,
                )
            )
            return obj.id, mesh

    pairs = await asyncio.gather(*(_one(o) for o in objects))
    return dict(pairs)
