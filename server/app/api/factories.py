"""
Factory hooks for building the per-run LLM client + mesh generator.

Centralizing them here lets tests monkeypatch the two functions below to
swap in `RecordedLLMClient` and `StubMeshGenerator` instances without
touching route code.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.llm.client import LLMClient
from app.llm.registry import build_client
from app.mesh_gen.interface import MeshGenerator
from app.mesh_gen.stub import StubMeshGenerator


def make_llm(model_id: str) -> LLMClient:
    return build_client(model_id)


def make_mesh_generator() -> MeshGenerator:
    backend = get_settings().mesh_gen_backend
    if backend == "stub":
        return StubMeshGenerator()
    if backend == "hunyuan":
        raise NotImplementedError("Hunyuan backend ships in a follow-up (task 10).")
    raise ValueError(f"Unknown mesh_gen_backend: {backend}")
