from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PipelineError(Exception):
    """Base class for pipeline errors."""


class ValidationConflict(BaseModel):
    """
    Structured conflict returned by a validator. Fed back into LLM retries via
    PromptPayload.prior_attempts so the model can see exactly what went wrong.
    """

    validator: str
    detail: str
    data: dict[str, Any] = {}


class RetryExhausted(PipelineError):
    def __init__(self, step_id: str, last_conflict: ValidationConflict, attempts: int) -> None:
        super().__init__(
            f"Step {step_id!r} exhausted {attempts} retry attempts: {last_conflict.detail}"
        )
        self.step_id = step_id
        self.last_conflict = last_conflict
        self.attempts = attempts
