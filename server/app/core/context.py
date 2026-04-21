from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.events import EventLog
    from app.llm.client import LLMClient
    from app.mesh_gen.interface import MeshGenerator
    from app.state_repo import StateRepository


@dataclass
class RunContext:
    run_id: str
    model_id: str
    max_retries: int
    state_repo: StateRepository
    events: EventLog
    mesh_generator: MeshGenerator
    llm: LLMClient


_current: ContextVar[RunContext | None] = ContextVar("starshot_run_context", default=None)


def set_current(ctx: RunContext) -> None:
    _current.set(ctx)


def current() -> RunContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("No active RunContext — call set_current() at request entry")
    return ctx


def try_current() -> RunContext | None:
    return _current.get()
