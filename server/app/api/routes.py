"""
Top-level API router. Aggregates the per-step sub-routers:

  * step 1  — `POST /generate`                    (`step01_http_entry`)
  * step 17 — run status, cancel, events, GLB     (`step17_serve`)

`app.main.create_app` includes this router on the FastAPI instance.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import step01_http_entry, step17_serve

router = APIRouter()
router.include_router(step01_http_entry.router)
router.include_router(step17_serve.router)
