"""Fixed benchmark slots. Each is a resumable pipeline run keyed by id."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    id: str
    prompt: str


SLOTS: list[Slot] = [
    Slot("swamp-land", "A swamp with islands"),
    Slot("hotel-room", "A minimal hotel room"),
    Slot("modern-house", "A modern house"),
    Slot("platformer-level", "A platformer level"),
    Slot("battle-arena", "A battle arena"),
    Slot("urban-city", "A beautiful, urban city"),
    Slot("planetary-system", "A planetary system in space"),
]

SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in SLOTS}

DEFAULT_MODEL = "anthropic/claude-opus-4.7"
