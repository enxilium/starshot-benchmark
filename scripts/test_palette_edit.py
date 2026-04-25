#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fal-client>=0.5", "httpx>=0.27", "python-dotenv>=1.0"]
# ///
"""Prototype: palette swatch + nano-banana/edit for encapsulation references.

Generates a single material swatch via nano-banana-pro, then reuses that
swatch URL as the reference image in three nano-banana/edit calls — wall,
floor, ceiling — to check whether the edit model respects shape and framing
instructions while honouring the attached texture. This is the load-bearing
assumption behind the per-zone palette proposal.

Usage:
    ./scripts/test_palette_edit.py

Outputs land in /tmp/starshot-palette-test/<timestamp>/ — swatch, each
element's PNG, and the prompts used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import fal_client
import httpx
from dotenv import load_dotenv

# FAL_KEY lives in server/.env in this repo.
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "server" / ".env")


PALETTE_BRIEF = (
    "A seamless material swatch for a tropical hotel's exterior shell: "
    "weathered limestone masonry with soft climbing ivy and subtle moss in "
    "the grout lines. Warm neutral tones, soft overcast daylight. Square "
    "swatch, flat, filling the frame, plain background — a pure material "
    "reference with no objects and no context."
)

ELEMENTS: list[tuple[str, str]] = [
    (
        "wall",
        "A reference photo of a single tall rectangular wall slab, 16m "
        "wide by 3m tall by 0.2m thick, photographed from a natural "
        "three-quarter angle so the front face, top edge, and thin side "
        "edge are all visible in perspective. Solid continuous mass, "
        "unbroken rectangular silhouette, no base trim, no plinth, no "
        "overhangs, no protrusions of any kind. Plain neutral studio "
        "background, soft even lighting. Use the attached image as the "
        "material and style reference — preserve its colour, texture, and "
        "surface treatment exactly.",
    ),
    (
        "floor",
        "A reference photo of a single flat rectangular floor slab, 20m "
        "wide by 0.2m tall by 15m deep, photographed from a natural "
        "three-quarter angle slightly above so the top surface and one "
        "thin side edge are both clearly visible. Solid continuous mass, "
        "unbroken rectangular silhouette, no trim, no separate foundation, "
        "no protrusions. Plain neutral studio background, soft even "
        "lighting. Use the attached image as the material and style "
        "reference — preserve its colour, texture, and surface treatment "
        "exactly.",
    ),
    (
        "ceiling",
        "A reference photo of a single flat rectangular ceiling slab, 20m "
        "wide by 0.2m tall by 15m deep, photographed from a natural "
        "three-quarter angle slightly below so the underside and one thin "
        "side edge are both clearly visible. Solid continuous mass, "
        "unbroken rectangular silhouette, no crown molding, no overhangs, "
        "no protrusions. Plain neutral studio background, soft even "
        "lighting. Use the attached image as the material and style "
        "reference — preserve its colour, texture, and surface treatment "
        "exactly.",
    ),
]

NANO_BANANA_TEXT = "fal-ai/nano-banana-pro"
NANO_BANANA_EDIT = "fal-ai/nano-banana/edit"


async def submit(model: str, args: dict) -> dict:
    handler = await fal_client.submit_async(model, arguments=args)
    return await handler.get()


async def download(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)


async def main() -> int:
    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set (checked env and server/.env)", file=sys.stderr)
        return 1

    out_dir = Path("/tmp/starshot-palette-test") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    (out_dir / "prompts.txt").write_text(
        f"PALETTE_BRIEF:\n{PALETTE_BRIEF}\n\n"
        + "\n\n".join(f"{name.upper()}:\n{prompt}" for name, prompt in ELEMENTS)
    )

    print("\n[1/2] Generating palette swatch (nano-banana-pro, text-to-image)")
    swatch = await submit(NANO_BANANA_TEXT, {"prompt": PALETTE_BRIEF})
    swatch_url = swatch["images"][0]["url"]
    swatch_path = out_dir / "00_swatch.png"
    await download(swatch_url, swatch_path)
    print(f"       {swatch_path}")
    print(f"       (remote: {swatch_url})")

    print(f"\n[2/2] Generating {len(ELEMENTS)} encapsulation elements (nano-banana/edit)")

    async def run_element(i: int, name: str, prompt: str) -> None:
        print(f"   -> {name} submitting...")
        result = await submit(
            NANO_BANANA_EDIT,
            {"prompt": prompt, "image_urls": [swatch_url]},
        )
        url = result["images"][0]["url"]
        path = out_dir / f"{i + 1:02d}_{name}.png"
        await download(url, path)
        print(f"   <- {path}")

    await asyncio.gather(
        *(run_element(i, name, prompt) for i, (name, prompt) in enumerate(ELEMENTS))
    )

    print(f"\nDone. Inspect: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
