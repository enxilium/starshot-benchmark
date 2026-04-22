"""Event log + SSE subscribers.

Every emitted event is a dict with a `kind` and arbitrary extra fields.
Events fan out to: (a) an in-memory buffer for snapshotting late subscribers,
(b) the rich console as a pretty per-field block, (c) every active SSE queue.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.markup import escape

from app.core.types import BoundingBox

_console = Console()

STATE: dict[str, Any] = {
    "status": "idle",
    "prompt": None,
    "model": None,
    "events": [],
}

_subscribers: list[asyncio.Queue[dict[str, Any]]] = []


def start_run(prompt: str, model: str) -> None:
    STATE["status"] = "running"
    STATE["prompt"] = prompt
    STATE["model"] = model
    STATE["events"] = []
    log("run.start", prompt=prompt, model=model)


def finish_run() -> None:
    STATE["status"] = "done"
    log("run.done")


def log(kind: str, **data: Any) -> None:
    event: dict[str, Any] = {"kind": kind, **data}
    STATE["events"].append(event)
    _print(event)
    for q in _subscribers:
        q.put_nowait(event)


def emit_bbox(node_id: str, bbox: BoundingBox) -> None:
    log(
        "bbox",
        id=node_id,
        origin=list(bbox.origin),
        dimensions=list(bbox.dimensions),
    )


def emit_model(node_id: str, artifact_kind: str, url: str) -> None:
    log("model", id=node_id, artifact_kind=artifact_kind, url=url)


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue[dict[str, Any]]) -> None:
    if q in _subscribers:
        _subscribers.remove(q)


# --- console formatting -----------------------------------------------------

_KIND_COLOR = {
    "run.start": "cyan",
    "run.done": "green",
    "run.error": "red",
    "bbox": "yellow",
    "model": "magenta",
}


def _print(event: dict[str, Any]) -> None:
    kind = str(event.get("kind", "?"))
    color = _KIND_COLOR.get(kind, "blue")
    fields = [(k, v) for k, v in event.items() if k != "kind"]
    _console.print(f"[bold {color}]{kind}[/bold {color}]")
    if not fields:
        return
    width = max(len(k) for k, _ in fields)
    for k, v in fields:
        _console.print(f"  [dim]{k.ljust(width)}[/dim]  {escape(_fmt(v))}")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in value) + "]"
    if isinstance(value, dict):
        parts = [f"{k}={_fmt(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, str):
        return value
    return repr(value)
