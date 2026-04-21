"""
LLMClient protocol + retry helper.

The pipeline's only touch point with the LLM layer is `call_structured`: it
sends a `PromptPayload` and gets back an instance of a Pydantic output schema.
No string parsing happens anywhere in the pipeline.

`call_with_validator` wraps `call_structured` with the validator-driven retry
policy described in CLAUDE.md / plan: the call is repeated until a validator
accepts the output or until `max_retries` attempts have been exhausted, with
each prior attempt's output + conflict being fed back to the LLM in the next
prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.core.errors import RetryExhausted, ValidationConflict
from app.core.events import EventLog, StepRetried

T = TypeVar("T", bound=BaseModel)


class PriorAttempt(BaseModel):
    """A previous attempt at the same step that was rejected by a validator."""

    attempt: int
    output: dict[str, Any]
    conflict: ValidationConflict


class PromptPayload(BaseModel):
    """Provider-agnostic prompt payload."""

    system: str
    user: str
    prior_attempts: list[PriorAttempt] = []


@runtime_checkable
class LLMClient(Protocol):
    async def call_structured(
        self,
        step_id: str,
        prompt: PromptPayload,
        output_schema: type[T],
    ) -> T: ...


async def call_with_validator[T: BaseModel](
    *,
    step_id: str,
    llm: LLMClient,
    events: EventLog,
    max_retries: int,
    output_schema: type[T],
    build_prompt: Callable[[list[PriorAttempt]], PromptPayload],
    validate: Callable[[T], ValidationConflict | None],
) -> T:
    """
    Invoke `llm.call_structured` in a validator-driven retry loop.

    * `build_prompt` receives the running list of prior attempts and returns
      the next PromptPayload. This lets the caller customize how the history
      is rendered for their step.
    * `validate` returns `None` when the output passes, or a
      `ValidationConflict` that will be fed back for the next attempt.

    Retries up to `max_retries` total attempts. Exhaustion raises
    `RetryExhausted` carrying the final conflict.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    prior_attempts: list[PriorAttempt] = []
    last_conflict: ValidationConflict | None = None

    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(prior_attempts)
        output = await llm.call_structured(
            step_id=step_id, prompt=prompt, output_schema=output_schema
        )
        conflict = validate(output)
        if conflict is None:
            return output
        last_conflict = conflict
        await events.emit(StepRetried(step_id=step_id, attempt=attempt, conflict=conflict))
        prior_attempts.append(
            PriorAttempt(attempt=attempt, output=output.model_dump(), conflict=conflict)
        )

    assert last_conflict is not None  # loop ran at least once with max_retries >= 1
    raise RetryExhausted(step_id=step_id, last_conflict=last_conflict, attempts=max_retries)
