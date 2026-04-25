"""Single-shot structured call to OpenRouter via `response_format: json_schema`.

The model is run-global. Configure it once via `set_model()` at the start
of a run; every subsequent `call_llm()` uses it.

Up to 4 resamples on parse / validation failures.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel, ValidationError

from app.utils import cache, logging

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
    key = cache.hash_llm_call(
        model=_current_model,
        system=system,
        user=user,
        schema_name=output_schema.__name__,
    )
    hit = cache.find_llm_cache_hit(logging.current_events(), key)
    if hit is not None:
        return output_schema.model_validate(hit)

    body: dict[str, object] = {
        "model": _current_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.__name__,
                "strict": True,
                "schema": _strip_array_lengths(output_schema.model_json_schema()),
            },
        },
        "max_tokens": 4096,
    }

    for attempt in range(4):
        async with httpx.AsyncClient(timeout=180.0) as http:
            resp = await http.post(
                _URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json=body,
            )
        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {data}")
        try:
            content = data["choices"][0]["message"]["content"]  # type: ignore[index]
            args = json.loads(content) if isinstance(content, str) else content
            validated = output_schema.model_validate(args)
            logging.log("cache.llm", key=key, output=validated.model_dump(mode="json"))
            return validated
        except (ValidationError, ValueError, KeyError, IndexError, TypeError) as e:
            if attempt == 3:
                raise
            logging.log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
    raise AssertionError("unreachable")


def _strip_array_lengths(schema: object) -> object:
    """Recursively drop `minItems`/`maxItems`. Pydantic emits them for
    fixed-length tuples (e.g. `tuple[float, float, float]` → minItems=3),
    which some providers (Anthropic) reject outright. Pydantic still
    validates the constraint on the parsed response."""
    if isinstance(schema, dict):
        return {
            k: _strip_array_lengths(v)
            for k, v in schema.items()
            if k not in {"minItems", "maxItems"}
        }
    if isinstance(schema, list):
        return [_strip_array_lengths(v) for v in schema]
    return schema
