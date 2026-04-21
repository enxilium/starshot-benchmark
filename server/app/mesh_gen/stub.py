"""
Stub mesh generator. Produces a colored unit cube per prompt so the whole
pipeline (rescale, assembly, GLB export) runs end-to-end without touching
Hunyuan 3.1 or a GPU. Color is derived deterministically from the prompt
hash so two runs with the same prompt yield the same visual.
"""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh


def _prompt_color(prompt: str) -> np.ndarray:
    """Deterministic RGBA from a prompt. Alpha is fixed at 255 (opaque)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return np.asarray([digest[0], digest[1], digest[2], 255], dtype=np.uint8)


class StubMeshGenerator:
    """Implements `MeshGenerator`. Always returns a unit cube, colored per prompt."""

    name = "stub"

    async def generate(self, prompt: str, object_id: str) -> trimesh.Trimesh:
        _ = object_id
        cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        color = _prompt_color(prompt)
        # trimesh's ColorVisuals exposes face_colors at runtime but stubs are
        # incomplete; silence the strict-mode attribute-access warnings here.
        cube.visual.face_colors = np.tile(color, (len(cube.faces), 1))  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        return cube
