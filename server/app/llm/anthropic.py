"""
Anthropic provider for LLMClient. Uses tool-use mode to coerce structured
output: the tool's input_schema is the Pydantic model's JSON schema, and we
force `tool_choice` so the model must emit it.

Transient errors (connection, timeout, rate limit) are retried at this layer
with exponential backoff. They are independent from the validator-driven
retries in `call_with_validator`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.context import current
from app.llm.client import PriorAttempt, PromptPayload

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_RETRIES = 4
_BASE_BACKOFF_SECONDS = 1.0


def _render_prior_attempts(prior: list[PriorAttempt]) -> str:
    if not prior:
        return ""
    lines = ["\n\nPrior attempts (all rejected by the validator):"]
    for pa in prior:
        lines.append(
            f"\nAttempt {pa.attempt} output:\n{json.dumps(pa.output, indent=2)}\n"
            f"Rejection ({pa.conflict.validator}): {pa.conflict.detail}\n"
            f"Rejection details: {json.dumps(pa.conflict.data, indent=2)}"
        )
    lines.append(
        "\nRevise your output to fix the listed issues. Emit only the corrected "
        "structured output via the tool."
    )
    return "\n".join(lines)


class AnthropicClient:
    """LLMClient implementation backed by the Anthropic SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError(
                "AnthropicClient requires ANTHROPIC_API_KEY (set it in env or pass api_key=...)"
            )
        self._client = AsyncAnthropic(api_key=key)

    async def call_structured(
        self,
        step_id: str,
        prompt: PromptPayload,
        output_schema: type[T],
    ) -> T:
        ctx = current()
        model_id = ctx.model_id
        tool_name = "emit"
        tool_input_schema = output_schema.model_json_schema()
        tools = [
            {
                "name": tool_name,
                "description": (
                    f"Emit the structured output for pipeline step {step_id}. "
                    "You MUST call this tool exactly once with the final result."
                ),
                "input_schema": tool_input_schema,
            }
        ]

        user_content = prompt.user + _render_prior_attempts(prompt.prior_attempts)

        last_exc: Exception | None = None
        for attempt in range(1, _TRANSIENT_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=model_id,
                    system=prompt.system,
                    messages=[{"role": "user", "content": user_content}],
                    tools=tools,  # pyright: ignore[reportArgumentType]
                    tool_choice={"type": "tool", "name": tool_name},
                    max_tokens=4096,
                )
                break
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_exc = e
                if attempt == _TRANSIENT_RETRIES:
                    raise
                await asyncio.sleep(_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        else:  # pragma: no cover — exhausted above re-raises
            raise RuntimeError("transient-retry loop exited without return") from last_exc

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return output_schema.model_validate(block.input)
        raise RuntimeError(
            f"Anthropic response for step {step_id!r} contained no tool_use block; "
            f"content types: {[b.type for b in response.content]}"
        )
