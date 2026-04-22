"""Single-shot structured call to OpenRouter via forced tool-calling.

The model is run-global. Configure it once via `set_model()` at the start
of a run; every subsequent `call_llm()` uses it.
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel

_URL = "https://openrouter.ai/api/v1/chat/completions"

_current_model: str | None = None


def set_model(model: str) -> None:
    global _current_model
    _current_model = model


async def call_llm[T: BaseModel](
    *,
    system: str,
    user: str,
    output_schema: type[T],
) -> T:
    if _current_model is None:
        raise RuntimeError("llm.set_model() must be called before call_llm()")
    body = {
        "model": _current_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "emit",
                    "description": "Emit the structured output.",
                    "parameters": output_schema.model_json_schema(),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "emit"}},
        "max_tokens": 4096,
    }
    async with httpx.AsyncClient(timeout=180.0) as http:
        resp = await http.post(
            _URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=body,
        )
    args = resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    return output_schema.model_validate_json(args)
