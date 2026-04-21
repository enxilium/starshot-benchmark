"""
Provider factory: maps a `model_id` (from the request payload) to a concrete
`LLMClient` instance. Adding a new provider is a one-line change here plus
one new file under `app/llm/`.
"""

from __future__ import annotations

from app.core.config import MODEL_REGISTRY
from app.llm.anthropic import AnthropicClient
from app.llm.client import LLMClient


def build_client(model_id: str) -> LLMClient:
    provider = MODEL_REGISTRY.get(model_id)
    if provider is None:
        allowed = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {model_id!r}. Allowed: {allowed}")
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"No client factory registered for provider {provider!r}")
