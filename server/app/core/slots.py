"""Fixed benchmark slots. Each is a resumable pipeline run keyed by id."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    id: str
    prompt: str


SLOTS: list[Slot] = [
    Slot("swamp-land", "A swamp with islands, designed as a top-down arcade level where a frog can jump from island to island"),
    Slot("hotel-room", "A modern hotel room"),
    Slot("modern-house", "A modern house"),
    Slot("platformer-level", "A super mario bros type platformer level"),
    Slot("battle-arena", "A battle arena for a two player game"),
    Slot("urban-city", "A modern, urban city"),
    Slot("planetary-system", "A planetary system in space with an alien battle happening"),
]

SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in SLOTS}

DEFAULT_MODEL = "anthropic/claude-opus-4.7"
