from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.core.errors import RetryExhausted, ValidationConflict
from app.core.events import EventLog
from app.llm.client import PriorAttempt, PromptPayload, call_with_validator
from app.llm.recorded import RecordedLLMClient


class Answer(BaseModel):
    value: int


def write_fixture(dir_: Path, step_id: str, idx: int, payload: dict) -> None:
    step_dir = dir_ / step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"{idx:03d}.json").write_text(json.dumps(payload))


async def test_recorded_client_returns_fixtures_in_order(tmp_path: Path) -> None:
    write_fixture(tmp_path, "step04", 0, {"value": 10})
    write_fixture(tmp_path, "step04", 1, {"value": 20})

    client = RecordedLLMClient(tmp_path)
    p = PromptPayload(system="s", user="u")
    a = await client.call_structured(step_id="step04", prompt=p, output_schema=Answer)
    b = await client.call_structured(step_id="step04", prompt=p, output_schema=Answer)
    assert a.value == 10
    assert b.value == 20


async def test_recorded_client_missing_fixture_errors(tmp_path: Path) -> None:
    client = RecordedLLMClient(tmp_path)
    with pytest.raises(FileNotFoundError):
        await client.call_structured(
            step_id="nope.step", prompt=PromptPayload(system="s", user="u"), output_schema=Answer
        )


async def test_call_with_validator_accepts_first_try(tmp_path: Path) -> None:
    write_fixture(tmp_path, "x", 0, {"value": 42})
    client = RecordedLLMClient(tmp_path)
    events = EventLog("r", tmp_path / "runs")

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(system="s", user=f"u (prior={len(prior)})")

    def validate(out: Answer) -> ValidationConflict | None:
        return None

    result = await call_with_validator(
        step_id="x",
        llm=client,
        events=events,
        max_retries=3,
        output_schema=Answer,
        build_prompt=build,
        validate=validate,
    )
    assert result.value == 42
    # No StepRetried events should have been written (events file may not even exist).
    assert not events.path.exists() or '"type":"step_retried"' not in events.path.read_text()


async def test_call_with_validator_retries_until_accept(tmp_path: Path) -> None:
    write_fixture(tmp_path, "y", 0, {"value": 1})  # rejected
    write_fixture(tmp_path, "y", 1, {"value": 2})  # rejected
    write_fixture(tmp_path, "y", 2, {"value": 99})  # accepted
    client = RecordedLLMClient(tmp_path)
    events = EventLog("r", tmp_path / "runs")

    accepted_value = 99

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(system="s", user="u", prior_attempts=prior)

    def validate(out: Answer) -> ValidationConflict | None:
        if out.value == accepted_value:
            return None
        return ValidationConflict(
            validator="too_small", detail=f"value {out.value} < {accepted_value}"
        )

    result = await call_with_validator(
        step_id="y",
        llm=client,
        events=events,
        max_retries=5,
        output_schema=Answer,
        build_prompt=build,
        validate=validate,
    )
    assert result.value == accepted_value

    # Event log should contain exactly 2 StepRetried entries for the rejected ones.
    lines = events.path.read_text().strip().splitlines()
    retried = [line for line in lines if '"type":"step_retried"' in line]
    assert len(retried) == 2
    events_data = [json.loads(line) for line in retried]
    assert events_data[0]["attempt"] == 1
    assert events_data[1]["attempt"] == 2


async def test_call_with_validator_exhaustion_raises(tmp_path: Path) -> None:
    write_fixture(tmp_path, "z", 0, {"value": 1})
    write_fixture(tmp_path, "z", 1, {"value": 2})
    client = RecordedLLMClient(tmp_path)
    events = EventLog("r", tmp_path / "runs")

    def build(prior: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(system="s", user="u", prior_attempts=prior)

    def always_reject(out: Answer) -> ValidationConflict | None:
        return ValidationConflict(validator="nope", detail=f"never accept {out.value}")

    with pytest.raises(RetryExhausted) as excinfo:
        await call_with_validator(
            step_id="z",
            llm=client,
            events=events,
            max_retries=2,
            output_schema=Answer,
            build_prompt=build,
            validate=always_reject,
        )
    assert excinfo.value.step_id == "z"
    assert excinfo.value.attempts == 2
    assert excinfo.value.last_conflict.validator == "nope"


async def test_call_with_validator_rejects_zero_max_retries(tmp_path: Path) -> None:
    client = RecordedLLMClient(tmp_path)
    events = EventLog("r", tmp_path / "runs")

    def build(_p: list[PriorAttempt]) -> PromptPayload:
        return PromptPayload(system="s", user="u")

    def validate(_o: Answer) -> ValidationConflict | None:
        return None

    with pytest.raises(ValueError, match="max_retries"):
        await call_with_validator(
            step_id="x",
            llm=client,
            events=events,
            max_retries=0,
            output_schema=Answer,
            build_prompt=build,
            validate=validate,
        )
