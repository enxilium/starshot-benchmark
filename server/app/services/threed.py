"""Two-stage asset generation: text -> image (Nano Banana Pro) -> 3D (Trellis 2).

Text-to-3D alone produces unreliable geometry for thin architectural shells
(walls, ceilings, floors). Going through an image model first gives Trellis
a concrete visual reference with correct proportions, which is much more
robust.

Both stages run on Runware (https://runware.ai/) over its WebSocket SDK. We
use a caller-supplied `taskUUID` per stage so the resumption record (logged
as `runware.submit`) can survive process restarts: on the next attempt the
same UUID is reused via `getResponse` and the in-flight or recently-finished
job returns its result without re-billing.

The returned GLB has textures embedded in its binary chunk, but trimesh
cannot decode them unless Pillow is installed at import time. Pillow is a
project dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from runware import (
    I3dInference,
    I3dInputs,
    IAsyncTaskResponse,
    IImageInference,
    ISettings,
    Runware,
    RunwareAPIError,
)

from app.utils import cache, logging

NANO_BANANA_PRO = "google:4@2"
NANO_BANANA_2 = "google:4@3"

NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", NANO_BANANA_2)
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "microsoft:trellis-2@4b")


def banana_settings_for(model: str) -> dict[str, Any]:
    """Per-model production settings for the Runware imageInference call.

    Runware rejects `resolution` for text-to-image (the preset is only
    meaningful when `referenceImages` is set, where it matches the input
    aspect ratio). For pure text-to-image, `width`/`height` are required.
    nano-banana-2 (`google:4@3`) runs at 512×512 with thinking=MINIMAL;
    nano-banana-pro (`google:4@2`) runs at 1024×1024.
    """
    if model == NANO_BANANA_2:
        return {
            "width": 512,
            "height": 512,
            "thinking": "MINIMAL",
        }
    if model == NANO_BANANA_PRO:
        return {
            "width": 1024,
            "height": 1024,
        }
    return {}
MAX_ATTEMPTS = 3
# Transient failures we retry on: Runware API errors AND the httpx
# network-layer errors (RemoteProtocolError "stream closed", ReadError,
# ConnectError, timeouts, etc.) raised by the GLB / image downloads.
RETRYABLE: tuple[type[BaseException], ...] = (
    RunwareAPIError,
    httpx.HTTPError,
    ConnectionError,
)


# Module-level singleton Runware client. Lazy-init on first call so the
# server can come up even if Runware is briefly unreachable; reused across
# all calls because the SDK manages a single WebSocket connection that
# multiplexes requests internally.
_client: Runware | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> Runware:
    global _client
    async with _client_lock:
        if _client is None:
            _client = Runware(
                api_key=os.environ["RUNWARE_API_KEY"],
                timeout=180,
                max_retries=0,            # outer loop manages retries
            )
            await _client.connect()
        else:
            await _client.ensureConnection()
        return _client


async def disconnect_runware() -> None:
    """Close the singleton WebSocket. Call once during FastAPI lifespan
    teardown so the server exits cleanly."""
    global _client
    async with _client_lock:
        if _client is not None:
            with contextlib.suppress(Exception):
                await _client.disconnect()
            _client = None


def _build_request(
    stage: str, arguments: dict[str, Any], task_uuid: str,
) -> IImageInference | I3dInference:
    """Translate the stage-agnostic arguments dict into the Runware
    request dataclass for that stage. Caller-supplied `task_uuid` is the
    resumption key — same value goes into the `runware.submit` event log
    and into the request itself."""
    if stage == "banana":
        settings = (
            ISettings(thinking=arguments["thinking"])
            if arguments.get("thinking") is not None
            else None
        )
        return IImageInference(
            taskUUID=task_uuid,
            model=arguments["model"],
            positivePrompt=arguments["positivePrompt"],
            width=arguments.get("width"),
            height=arguments.get("height"),
            resolution=arguments.get("resolution"),
            settings=settings,
            outputFormat=arguments["outputFormat"],
            outputType=arguments["outputType"],
            deliveryMethod="async",
            numberResults=1,
        )
    if stage == "trellis":
        return I3dInference(
            taskUUID=task_uuid,
            model=arguments["model"],
            inputs=I3dInputs(image=arguments["image"]),
            settings=ISettings(
                remesh=arguments["remesh"],
                resolution=arguments["resolution"],
                textureSize=arguments["textureSize"],
            ),
            outputFormat=arguments["outputFormat"],
            outputType=arguments["outputType"],
            deliveryMethod="async",
            numberResults=1,
        )
    raise ValueError(f"unknown stage: {stage!r}")


async def _dispatch(
    client: Runware, stage: str, request: IImageInference | I3dInference,
) -> Any:
    if stage == "banana":
        assert isinstance(request, IImageInference)
        return await client.imageInference(requestImage=request)
    if stage == "trellis":
        assert isinstance(request, I3dInference)
        return await client.inference3d(request3d=request)
    raise ValueError(f"unknown stage: {stage!r}")


def _unwrap(stage: str, item: Any) -> dict[str, Any]:
    """Normalize the SDK's per-stage result dataclass into the plain dict
    `generate_mesh` consumes. Banana yields a single image URL; Trellis
    yields a GLB URL nested under `outputs.files[0]`."""
    if stage == "banana":
        url = getattr(item, "imageURL", None)
        if not url:
            raise RuntimeError(f"Banana result missing imageURL: {item!r}")
        return {"image_url": url}
    if stage == "trellis":
        outputs = getattr(item, "outputs", None)
        files = getattr(outputs, "files", None) if outputs else None
        if not files:
            raise RuntimeError(f"Trellis result missing outputs.files: {item!r}")
        first = files[0]
        url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
        if not url:
            raise RuntimeError(f"Trellis result missing url: {first!r}")
        return {"glb_url": url}
    raise ValueError(f"unknown stage: {stage!r}")


async def _submit_resumable(
    arguments: dict[str, Any],
    *,
    node_id: str,
    stage: str,
) -> dict[str, Any]:
    """Submit a Runware job with restart-resilient resumption.

    On entry, scan the events log for a prior `runware.submit` matching
    (node_id, stage, input_hash). If found, attempt
    `client.getResponse(taskUUID=...)` against the persisted UUID —
    Runware keeps task results around long enough that an in-flight or
    recently-completed job returns immediately with no new billing. On
    any RunwareAPIError from the reattach we treat the prior task as
    expired and fall through to a fresh submit.

    Fresh submits log `runware.submit` *before* awaiting the SDK call.
    Because we generate the taskUUID client-side (UUID v4), the
    resumption record is durable as soon as `SlotLog.log` flushes — no
    dependency on the response landing.
    """
    model = arguments["model"]
    input_hash = cache.hash_runware_input(model, arguments)
    prior = cache.find_runware_submit(
        logging.current_events(), node_id, stage, input_hash,
    )
    client = await _get_client()
    if prior is not None:
        try:
            results = await client.getResponse(
                taskUUID=prior["task_uuid"], numberResults=1,
            )
            if results:
                logging.log(
                    "runware.reattach",
                    node_id=node_id,
                    stage=stage,
                    task_uuid=prior["task_uuid"],
                    outcome="success",
                )
                return _unwrap(stage, results[0])
            # Empty result list — treat as expired and submit fresh.
            logging.log(
                "runware.reattach",
                node_id=node_id,
                stage=stage,
                task_uuid=prior["task_uuid"],
                outcome="expired",
                reason="empty_result",
            )
        except RunwareAPIError as e:
            # v1: any error on reattach -> fall through to fresh submit.
            # The new submit's task_uuid overwrites the lookup, so the
            # next restart reattaches to the new task — no double-bill.
            logging.log(
                "runware.reattach",
                node_id=node_id,
                stage=stage,
                task_uuid=prior["task_uuid"],
                outcome="expired",
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )

    for attempt in range(MAX_ATTEMPTS):
        task_uuid = str(uuid.uuid4())
        try:
            request = _build_request(stage, arguments, task_uuid)
            logging.log(
                "runware.submit",
                node_id=node_id,
                stage=stage,
                model=model,
                task_uuid=task_uuid,
                input_hash=input_hash,
            )
            ack = await _dispatch(client, stage, request)
            if isinstance(ack, IAsyncTaskResponse):
                results = await client.getResponse(
                    taskUUID=task_uuid, numberResults=1,
                )
            else:
                results = ack
            if not results:
                raise RuntimeError(f"empty result list for task {task_uuid}")
            return _unwrap(stage, results[0])
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


def _ext_from_url(url: str) -> str:
    """Pick a saved-image extension from the URL when no Content-Type
    header is available. Runware's image URLs end with `.png` for our
    PNG-output requests; default to `.png` if nothing else matches."""
    lower = url.lower().split("?", 1)[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if lower.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"
    return ".png"


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

    # Banana-skip gate: if Banana already finished for this node and the
    # saved image is still on disk, skip the Banana stage and pass the
    # cached Runware-hosted URL straight to Trellis. Closes the Banana
    # re-bill window for process deaths between Banana and Trellis.
    image_path: Path | None = None
    remote_image_url: str | None = None
    banana_hit = cache.find_banana_done(logging.current_events(), node_id)
    if banana_hit is not None:
        candidate = Path(banana_hit["saved"])
        if candidate.exists():
            image_path = candidate
            remote_image_url = banana_hit["remote_url"]
            logging.log("nano_banana.skip", node_id=node_id)

    if image_path is None or remote_image_url is None:
        banana_args: dict[str, Any] = {
            "model": NANO_BANANA_MODEL,
            "positivePrompt": prompt,
            "outputFormat": "PNG",
            "outputType": "URL",
            **banana_settings_for(NANO_BANANA_MODEL),
        }
        img = await _submit_resumable(
            banana_args, node_id=node_id, stage="banana",
        )
        remote_image_url = str(img["image_url"])
        ext = _ext_from_url(remote_image_url)
        image_path = image_stem.parent / (image_stem.name + ext)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = await _download_with_retry(remote_image_url, stage="nano_banana")
        image_path.write_bytes(image_bytes)
        logging.log(
            "nano_banana.done",
            node_id=node_id,
            remote_url=remote_image_url,
            saved=str(image_path),
        )

    trellis_args: dict[str, Any] = {
        "model": TRELLIS_MODEL,
        "image": remote_image_url,
        "remesh": False,
        "resolution": 512,
        "textureSize": 1024,
        "outputFormat": "GLB",
        "outputType": "URL",
    }
    mesh = await _submit_resumable(
        trellis_args, node_id=node_id, stage="trellis",
    )
    glb_url = mesh["glb_url"]

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
