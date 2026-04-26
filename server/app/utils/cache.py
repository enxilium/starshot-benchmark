"""Event-log-backed cache.

The event log at runs/current/events.jsonl doubles as the cache. Every
successful LLM call emits a `cache.llm` event carrying the call key and
validated output; every successful mesh generation emits a `cache.artifact`
event carrying the node id and artifact paths. Cache lookup is a backward
scan over the in-memory event buffer; truncating the event log rewinds the
cache in one step.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_SEP = "\x1e"  # ASCII record separator — guards against boundary collisions.


def hash_llm_call(*, model: str, system: str, user: str, schema_name: str) -> str:
    payload = _SEP.join((model, system, user, schema_name)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_llm_cache_hit(
    events: list[dict[str, Any]], key: str,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") == "cache.llm" and event.get("key") == key:
            output = event.get("output")
            return output if isinstance(output, dict) else None
    return None


def find_artifact_cache_hit(
    events: list[dict[str, Any]], node_id: str,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") == "cache.artifact" and event.get("node_id") == node_id:
            return event
    return None


def hash_runware_input(model: str, arguments: dict[str, Any]) -> str:
    """Stable hash over (Runware model id, arguments dict) for
    runware.submit cache lookups. `sort_keys=True` makes dict insertion
    order irrelevant so re-running the same node with the same prompt
    produces the same hash across processes."""
    payload = json.dumps(
        {"model": model, "arguments": arguments}, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_runware_submit(
    events: list[dict[str, Any]],
    node_id: str,
    stage: str,
    input_hash: str,
) -> dict[str, Any] | None:
    """Most recent `runware.submit` matching (node_id, stage, input_hash).
    Used to reattach to an in-flight or recently-completed Runware job
    across process restarts so we don't pay for the same generation
    twice. Mismatched input_hash (prompt change, model swap) misses on
    purpose."""
    for event in reversed(events):
        if (
            event.get("kind") == "runware.submit"
            and event.get("node_id") == node_id
            and event.get("stage") == stage
            and event.get("input_hash") == input_hash
        ):
            return event
    return None


def find_banana_done(
    events: list[dict[str, Any]], node_id: str,
) -> dict[str, Any] | None:
    """Most recent `nano_banana.done` for a node. The Banana-skip gate
    in generate_mesh consults this so a process death between Banana
    and Trellis doesn't re-bill Banana."""
    for event in reversed(events):
        if (
            event.get("kind") == "nano_banana.done"
            and event.get("node_id") == node_id
        ):
            return event
    return None
