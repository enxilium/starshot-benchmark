from __future__ import annotations

import pytest

from app.core.config import MODEL_REGISTRY
from app.llm.anthropic import AnthropicClient
from app.llm.registry import build_client


def test_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_client("definitely-not-real")


def test_builds_anthropic_client_for_registered_model(monkeypatch) -> None:
    # Ensure API key is present so AnthropicClient can construct.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from app.core.config import reset_settings_for_tests

    reset_settings_for_tests()

    # Pick the first anthropic-provider model in the registry.
    model_id = next(m for m, prov in MODEL_REGISTRY.items() if prov == "anthropic")
    client = build_client(model_id)
    assert isinstance(client, AnthropicClient)
