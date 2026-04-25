"""Single-shot structured call to OpenRouter via `response_format: json_schema`.

The model is run-global. Configure it once via `set_model()` at the start
of a run; every subsequent `call_llm()` uses it.

Up to 4 resamples on parse / validation failures.
"""

from __future__ import annotations

import json
import os

from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

from app.utils import cache, logging

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

    for attempt in range(4):
        async with OpenRouter(
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout_ms=180_000,
        ) as client:
            response = await client.chat.send_async(
                model=_current_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_schema.__name__,
                        "strict": True,
                        "schema_": _normalize_schema(output_schema.model_json_schema()),
                    },
                },
                reasoning={"effort": "xhigh"},
            )
        content: object = None
        try:
            message = response.choices[0].message
            content = message.content
            args = json.loads(content) if isinstance(content, str) else content
            validated = output_schema.model_validate(args)
            reasoning = getattr(message, "reasoning", None) or ""
            logging.log(
                "cache.llm",
                key=key,
                output=validated.model_dump(mode="json"),
                reasoning=reasoning,
            )
            return validated
        except json.JSONDecodeError as e:
            logging.log(
                "llm.json_decode_error",
                reason=f"JSONDecodeError: {e}",
                attempt=attempt,
                final=attempt == 3,
                content=content if isinstance(content, str) else repr(content),
            )
            if attempt == 3:
                raise
        except (ValidationError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            if attempt == 3:
                raise
            logging.log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
    raise AssertionError("unreachable")


def _normalize_schema(schema: object) -> object:
    """Recursively normalize the Pydantic-emitted schema for providers that
    reject draft-2020-12 features. Two transforms:

      * Drop `minItems`/`maxItems` — Anthropic rejects them on `array`.
      * Collapse `prefixItems` (Pydantic emits this for fixed-length
        tuples like `tuple[float, float, float]`) into a single `items`
        schema. Anthropic rejects `prefixItems` outright. We assume
        homogeneous tuples (all our tuples are `Vec3` of floats); the
        first prefix item is reused as `items`.

    Pydantic still enforces the original constraints on the parsed
    response, so loosening the wire schema is safe."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in {"minItems", "maxItems"}:
                continue
            if k == "prefixItems":
                if isinstance(v, list) and v and "items" not in schema:
                    out["items"] = _normalize_schema(v[0])
                continue
            out[k] = _normalize_schema(v)
        return out
    if isinstance(schema, list):
        return [_normalize_schema(v) for v in schema]
    return schema
