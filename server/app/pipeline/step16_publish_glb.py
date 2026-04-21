"""
PIPELINE.md step 16 — publish the assembled scene as a `.glb` on disk
and return the URL the API will serve it at.

Takes the `trimesh.Scene` produced by step 15, exports it to
`{runs_dir}/{run_id}/scene.glb` via `trimesh.Scene.export('glb')`, and
returns `(path, "/glb/{run_id}")`. The URL propagates up through step 3
into `RunCompleted` and into the `run.json` summary.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from app.core.config import get_settings


def glb_path_for(run_id: str) -> Path:
    """Canonical on-disk path for a run's published `.glb`."""
    return get_settings().runs_dir / run_id / "scene.glb"


def write_glb(scene: trimesh.Scene, path: Path) -> Path:
    """Export `scene` to `path` as GLB. Returns `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scene.export(file_type="glb")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes from GLB export, got {type(data).__name__}")
    path.write_bytes(bytes(data))
    return path


def publish(run_id: str, scene: trimesh.Scene) -> tuple[Path, str]:
    """Write `scene` to disk and return `(path, url)`.

    The URL is a relative path that the `GET /glb/{run_id}` endpoint
    (step 17) resolves.
    """
    path = glb_path_for(run_id)
    write_glb(scene, path)
    return path, f"/glb/{run_id}"
