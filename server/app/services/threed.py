"""Single-shot Hunyuan text-to-mesh call via fal_client.

Named `threed.py` because Python module names cannot start with a digit.
"""

from __future__ import annotations

import os
from pathlib import Path

import fal_client
import httpx

HUNYUAN_MODEL = os.environ.get("HUNYUAN_MODEL", "fal-ai/hunyuan-3d/v3.1/pro/text-to-3d")


async def generate_mesh(prompt: str, *, output_path: Path) -> Path:
    handler = await fal_client.submit_async(HUNYUAN_MODEL, arguments={"prompt": prompt})
    result = await handler.get()
    url = result["model_glb"]["url"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as http:
        resp = await http.get(url)
    output_path.write_bytes(resp.content)
    return output_path
