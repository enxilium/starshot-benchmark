from __future__ import annotations

import numpy as np

from app.mesh_gen.interface import MeshGenerator
from app.mesh_gen.stub import StubMeshGenerator


async def test_stub_returns_unit_cube() -> None:
    gen = StubMeshGenerator()
    mesh = await gen.generate("a chair", "chair_1")
    lo, hi = mesh.bounds
    extents = hi - lo
    assert np.allclose(extents, (1.0, 1.0, 1.0))


async def test_color_is_deterministic_per_prompt() -> None:
    gen = StubMeshGenerator()
    m1 = await gen.generate("a chair", "chair_1")
    m2 = await gen.generate("a chair", "chair_2")
    # same prompt → same color, even with different object ids
    assert np.array_equal(m1.visual.face_colors[0], m2.visual.face_colors[0])


async def test_different_prompts_yield_different_colors() -> None:
    gen = StubMeshGenerator()
    m1 = await gen.generate("a red chair", "a")
    m2 = await gen.generate("a blue chair", "b")
    assert not np.array_equal(m1.visual.face_colors[0], m2.visual.face_colors[0])


async def test_structurally_matches_protocol() -> None:
    gen: MeshGenerator = StubMeshGenerator()
    assert gen.name == "stub"
    mesh = await gen.generate("x", "y")
    assert len(mesh.vertices) > 0
