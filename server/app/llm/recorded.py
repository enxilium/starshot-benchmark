"""
Recorded LLMClient: deterministic fixture-driven replacement for tests and
integration harnesses. Reads JSON fixtures keyed by `(step_id, call_index)`
from a configured directory.

Fixture layout:
    <fixtures_dir>/<step_id>/<call_index>.json   (zero-padded, e.g. 000.json)

Each file is the raw dict the LLM would have emitted as tool input. The
client validates it against the step's output schema on return.

To inject validator-retry conflicts into a test, supply multiple files per
step_id: `000.json` (rejected by validator), `001.json` (accepted), etc.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from app.llm.client import PromptPayload

T = TypeVar("T", bound=BaseModel)


class RecordedLLMClient:
    """LLMClient fixture-reader. Counts per-step calls and returns fixtures in order."""

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir
        self._call_count: dict[str, int] = defaultdict(int)

    async def call_structured(
        self,
        step_id: str,
        prompt: PromptPayload,
        output_schema: type[T],
    ) -> T:
        _ = prompt
        idx = self._call_count[step_id]
        self._call_count[step_id] += 1
        fixture_path = self.fixtures_dir / step_id / f"{idx:03d}.json"
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"No recorded fixture for step {step_id!r} call #{idx} at {fixture_path}"
            )
        data = json.loads(fixture_path.read_text())
        return output_schema.model_validate(data)

    def call_count(self, step_id: str) -> int:
        return self._call_count[step_id]
