"""Two-stage asset generation: text -> image (Nano Banana Pro) -> 3D (Trellis 2).

Text-to-3D alone produces unreliable geometry for thin architectural shells
(walls, ceilings, floors). Going through an image model first gives Trellis
a concrete visual reference with correct proportions, which is much more
robust.

The returned GLB has textures embedded in its binary chunk, but trimesh
cannot decode them unless Pillow is installed at import time. Pillow is a
project dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fal_client
import httpx

from app.utils import cache, logging

NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", "fal-ai/nano-banana-pro")
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "fal-ai/trellis-2")
MAX_ATTEMPTS = 3
# Transient failures we retry on: fal-side HTTP / timeout errors AND the
# httpx network-layer errors (RemoteProtocolError "stream closed",
# ReadError, ConnectError, timeouts, etc.) that surface when the local
# connection drops mid-request.
RETRYABLE: tuple[type[BaseException], ...] = (
    fal_client.FalClientError,
    httpx.HTTPError,
)


async def _submit_with_retry(
    model: str,
    arguments: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    for attempt in range(MAX_ATTEMPTS):
        try:
            handler = await fal_client.submit_async(model, arguments=arguments)
            return await handler.get()
        except RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logging.log(
                f"{stage}.retry",
                attempt=attempt,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
    raise AssertionError("unreachable")


async def _download_with_retry(url: str, *, stage: str) -> bytes:
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=180.0,
                follow_redirects=True,
            ) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                return resp.content
        except httpx.HTTPError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logging.log(
                f"{stage}.download.retry",
                attempt=attempt,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
    raise AssertionError("unreachable")


_CONTENT_TYPE_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


async def generate_mesh(
    prompt: str,
    *,
    output_path: Path,
    image_stem: Path,
) -> dict[str, Path]:
    """Run text -> image -> 3D. Saves the reference image alongside the
    GLB so the client asset browser can display it. Returns both paths."""
    node_id = image_stem.name
    hit = cache.find_artifact_cache_hit(logging.current_events(), node_id)
    if hit is not None:
        cached_raw = Path(hit["raw_glb_path"])
        cached_image = Path(hit["image_path"])
        if cached_raw.exists() and cached_image.exists():
            logging.log(
                "cache.artifact.hit",
                node_id=node_id,
                image_path=str(cached_image),
                raw_glb_path=str(cached_raw),
            )
            return {"glb": cached_raw, "image": cached_image}

    img = await _submit_with_retry(
        NANO_BANANA_MODEL,
        {
            "prompt": prompt,
            "num_images": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "resolution": "1K",
        },
        stage="nano_banana",
    )
    image_info = img["images"][0]
    remote_image_url = image_info["url"]
    ext = _CONTENT_TYPE_EXT.get(image_info.get("content_type", ""), ".png")
    image_path = image_stem.parent / (image_stem.name + ext)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = await _download_with_retry(remote_image_url, stage="nano_banana")
    image_path.write_bytes(image_bytes)
    logging.log("nano_banana.done", remote_url=remote_image_url, saved=str(image_path))

    mesh = await _submit_with_retry(
        TRELLIS_MODEL,
        {
            "image_url": remote_image_url,
            "remesh": False,
            "resolution": 512,
            "texture_size": 1024,
        },
        stage="trellis",
    )
    glb_url = mesh["model_glb"]["url"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = await _download_with_retry(glb_url, stage="trellis")
    output_path.write_bytes(content)
    logging.log(
        "cache.artifact",
        node_id=node_id,
        image_path=str(image_path),
        raw_glb_path=str(output_path),
    )
    return {"glb": output_path, "image": image_path}
