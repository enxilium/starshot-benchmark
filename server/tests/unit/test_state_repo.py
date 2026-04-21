from __future__ import annotations

import asyncio

from app.core.types import AnchorObject, BoundingBox
from app.state_repo import (
    InMemoryStateRepository,
    PlanEntry,
    RealizedEntry,
    StateRepository,
)


def bb(
    mn: tuple[float, float, float], mx: tuple[float, float, float]
) -> BoundingBox:
    return BoundingBox(
        origin=mn,
        dimensions=(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]),
    )


def make_plan(scope_id: str = "root.bathroom") -> PlanEntry:
    return PlanEntry(
        scope_id=scope_id,
        prompt="opulent bathroom",
        bbox=bb((0, 0, 0), (3, 3, 3)),
        high_level_plan="marble tiles, double vanity, toilet alcove",
    )


def make_realized(scope_id: str = "root.bathroom.toilet_area") -> RealizedEntry:
    return RealizedEntry(
        scope_id=scope_id,
        prompt="toilet area",
        bbox=bb((0, 0, 0), (1, 2, 1)),
        objects=[AnchorObject(id="toilet", prompt="white porcelain toilet")],
    )


async def test_write_plan_and_read_visible() -> None:
    repo = InMemoryStateRepository()
    await repo.write_plan(make_plan())
    visible = await repo.read_visible()
    assert len(visible.plans) == 1
    assert visible.plans[0].scope_id == "root.bathroom"
    assert visible.realized == []


async def test_write_realized_and_read_visible() -> None:
    repo = InMemoryStateRepository()
    await repo.write_realized(make_realized())
    visible = await repo.read_visible()
    assert visible.plans == []
    assert len(visible.realized) == 1
    assert visible.realized[0].scope_id == "root.bathroom.toilet_area"


async def test_plans_and_realized_coexist() -> None:
    repo = InMemoryStateRepository()
    await repo.write_plan(make_plan())
    await repo.write_realized(make_realized())
    visible = await repo.read_visible()
    assert len(visible.plans) == 1
    assert len(visible.realized) == 1


async def test_overwrite_same_scope_id() -> None:
    repo = InMemoryStateRepository()
    a = make_plan(scope_id="root.x")
    b = PlanEntry(
        scope_id="root.x",
        prompt="updated",
        bbox=bb((0, 0, 0), (1, 1, 1)),
        high_level_plan="updated plan",
    )
    await repo.write_plan(a)
    await repo.write_plan(b)
    visible = await repo.read_visible()
    assert len(visible.plans) == 1
    assert visible.plans[0].high_level_plan == "updated plan"


async def test_structurally_matches_protocol() -> None:
    repo: StateRepository = InMemoryStateRepository()
    await repo.write_plan(make_plan())
    visible = await repo.read_visible()
    assert visible.plans[0].prompt == "opulent bathroom"


async def test_concurrent_writes_do_not_race() -> None:
    repo = InMemoryStateRepository()

    async def write_many(prefix: str, n: int) -> None:
        for i in range(n):
            await repo.write_plan(
                PlanEntry(
                    scope_id=f"{prefix}.{i}",
                    prompt="p",
                    bbox=bb((0, 0, 0), (1, 1, 1)),
                    high_level_plan="hp",
                )
            )

    await asyncio.gather(
        write_many("a", 50),
        write_many("b", 50),
        write_many("c", 50),
    )
    visible = await repo.read_visible()
    assert len(visible.plans) == 150
    scope_ids = {p.scope_id for p in visible.plans}
    assert len(scope_ids) == 150
