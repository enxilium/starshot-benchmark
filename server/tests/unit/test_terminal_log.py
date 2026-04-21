from __future__ import annotations

import asyncio
import io
from pathlib import Path

from rich.console import Console

from app.core.errors import ValidationConflict
from app.core.events import EventLog, RunCompleted, RunStarted, StepRetried
from app.core.terminal_log import TerminalLogger


async def _drive(log: EventLog, logger: TerminalLogger) -> None:
    task = asyncio.create_task(logger.run())
    await asyncio.sleep(0)

    await log.emit(RunStarted(prompt="A modern mansion", model_id="claude-opus-4-7"))
    await log.emit(
        StepRetried(
            step_id="step06",
            attempt=1,
            conflict=ValidationConflict(validator="bbox", detail="overlap"),
        )
    )
    await log.emit(RunCompleted(glb_url="file:///out.glb", total_duration_ms=123.0))

    await asyncio.wait_for(task, timeout=1.0)


async def test_terminal_logger_renders_expected_markers(tmp_path: Path) -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    log = EventLog("abcd1234", tmp_path)
    logger = TerminalLogger(log, console=console)

    await _drive(log, logger)

    output = buffer.getvalue()
    # run id prefix (first 8 chars)
    assert "[abcd1234]" in output
    # key markers for each event type
    assert "run started" in output
    assert "retry #1" in output
    assert "bbox: overlap" in output
    assert "run completed" in output
    # step_id tags should appear in bracket form
    assert "[step06]" in output


async def test_terminal_logger_verbose_includes_summaries(tmp_path: Path) -> None:
    from app.core.events import StepCompleted, StepStarted

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    log = EventLog("r", tmp_path)
    logger = TerminalLogger(log, console=console, verbose=True)

    task = asyncio.create_task(logger.run())
    await asyncio.sleep(0)
    await log.emit(StepStarted(step_id="step10", inputs_summary={"leaf": "bathroom"}))
    await log.emit(
        StepCompleted(
            step_id="step10",
            duration_ms=42.0,
            output_summary={"n_anchors": 3},
        )
    )
    await log.emit(RunCompleted(glb_url="", total_duration_ms=0.0))
    await asyncio.wait_for(task, timeout=1.0)

    output = buffer.getvalue()
    assert "bathroom" in output
    assert "n_anchors" in output
