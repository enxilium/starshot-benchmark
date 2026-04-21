from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.config import MODEL_REGISTRY


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    model: str
    max_retries: int | None = Field(default=None, ge=1, le=20)

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        if v not in MODEL_REGISTRY:
            allowed = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model {v!r}. Allowed: {allowed}")
        return v


class GenerateResponse(BaseModel):
    run_id: str
    events_url: str
    status_url: str
