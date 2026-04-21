"""
MeshGenerator protocol. The real Hunyuan 3.1 adapter and the stub (task 4)
both implement this.

The pipeline only depends on this shape; the backend is injected via
`RunContext`. Set `MESH_GEN_BACKEND=stub|hunyuan` in settings to pick one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import trimesh


@runtime_checkable
class MeshGenerator(Protocol):
    """
    Takes a natural-language description of an object and produces a Trimesh
    at arbitrary scale. Phase-2 step 7 rescales the result to fit its
    resolved bounding box.
    """

    @property
    def name(self) -> str:
        """Short identifier for the backend (emitted in MeshGenerated events)."""
        ...

    async def generate(self, prompt: str, object_id: str) -> trimesh.Trimesh:
        """Return a mesh for `prompt`. `object_id` is for logging / caching only."""
        ...
