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
