from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.core.errors import ValidationConflict
from app.core.events import (
    EventLog,
    PhaseStarted,
    RunCompleted,
    RunStarted,
    StepRetried,
    parse_event_line,
    read_events_jsonl,
)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


async def test_emit_appends_jsonl_and_assigns_seq(runs_dir: Path) -> None:
    log = EventLog("abc123", runs_dir)
    await log.emit(RunStarted(prompt="hello", model_id="claude-opus-4-7"))
    await log.emit(PhaseStarted(phase="phase1", scope_id="root"))
    await log.emit(RunCompleted(glb_url="u", total_duration_ms=1.0))

    lines = (runs_dir / "abc123" / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["seq"] for p in parsed] == [1, 2, 3]
    assert [p["run_id"] for p in parsed] == ["abc123"] * 3
    assert [p["type"] for p in parsed] == ["run_started", "phase_started", "run_completed"]


async def test_terminal_event_closes_log(runs_dir: Path) -> None:
    log = EventLog("abc", runs_dir)
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))
    assert log.closed
    with pytest.raises(RuntimeError):
        await log.emit(RunStarted(prompt="x", model_id="claude-opus-4-7"))


async def test_run_failed_also_closes_log(runs_dir: Path) -> None:
    from app.core.events import RunFailed

    log = EventLog("r", runs_dir)
    await log.emit(RunFailed(error="boom", step_id="step06"))
    assert log.closed


async def test_subscribe_delivers_live_events_in_order(runs_dir: Path) -> None:
    log = EventLog("sub", runs_dir)
    received: list[str] = []

    async def consume() -> None:
        async for event in log.subscribe():
            received.append(event.type)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let subscribe() register the queue

    await log.emit(RunStarted(prompt="p", model_id="claude-opus-4-7"))
    await log.emit(PhaseStarted(phase="phase1", scope_id="root"))
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))

    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["run_started", "phase_started", "run_completed"]


async def test_subscribe_after_close_replays_from_jsonl(runs_dir: Path) -> None:
    log = EventLog("replay", runs_dir)
    await log.emit(RunStarted(prompt="p", model_id="claude-opus-4-7"))
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))
    assert log.closed

    received: list[int] = []
    async for event in log.subscribe(after_seq=0):
        received.append(event.seq)
    assert received == [1, 2]


async def test_subscribe_respects_after_seq(runs_dir: Path) -> None:
    log = EventLog("r", runs_dir)
    await log.emit(RunStarted(prompt="p", model_id="claude-opus-4-7"))
    await log.emit(PhaseStarted(phase="phase1", scope_id="root"))
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))

    received: list[int] = []
    async for event in log.subscribe(after_seq=1):
        received.append(event.seq)
    assert received == [2, 3]


async def test_subscribe_no_duplicates_when_subscribing_between_emits(runs_dir: Path) -> None:
    log = EventLog("r", runs_dir)
    await log.emit(RunStarted(prompt="p", model_id="claude-opus-4-7"))

    received: list[int] = []

    async def consume() -> None:
        async for event in log.subscribe():
            received.append(event.seq)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await log.emit(PhaseStarted(phase="phase1", scope_id="root"))
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))
    await asyncio.wait_for(task, timeout=1.0)

    # seq=1 was emitted before subscribe; still delivered via replay
    assert received == [1, 2, 3]


def test_read_events_jsonl_parses_validation_conflict(runs_dir: Path) -> None:
    # Write a step_retried event via emit, then read back.
    async def _do() -> Path:
        log = EventLog("r", runs_dir)
        await log.emit(
            StepRetried(
                step_id="step06",
                attempt=1,
                conflict=ValidationConflict(
                    validator="bbox_sibling_overlap",
                    detail="A overlaps B",
                    data={"a": "subscene_1", "b": "subscene_2"},
                ),
            )
        )
        await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))
        return log.path

    path = asyncio.run(_do())
    events = list(read_events_jsonl(path))
    assert len(events) == 2
    assert events[0].type == "step_retried"
    assert events[0].conflict.validator == "bbox_sibling_overlap"  # type: ignore[attr-defined]


def test_parse_event_line_round_trip() -> None:
    original = RunStarted(prompt="p", model_id="claude-opus-4-7")
    original.run_id = "r"
    original.seq = 7
    parsed = parse_event_line(original.model_dump_json())
    assert parsed.type == "run_started"
    assert parsed.seq == 7
    assert parsed.run_id == "r"
