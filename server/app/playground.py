"""Standalone dev playground: prompt → Nano Banana → Trellis 2 → GLB.

A minimal FastAPI app with:
  * `GET  /`        — single-page UI for prompt experimentation.
  * `POST /banana`  — text → image. Returns the Fal CDN image URL.
  * `POST /trellis` — image → 3D. Returns the Fal CDN GLB URL.

No event log, no caching, no slot machinery — every click hits Fal fresh.
Run via `scripts/run_playground.py`.
"""

from __future__ import annotations

import os
from typing import Literal

import fal_client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", "fal-ai/nano-banana-pro")
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "fal-ai/trellis-2")


class BananaRequest(BaseModel):
    prompt: str


class TrellisRequest(BaseModel):
    image_url: str
    remesh: bool = False
    resolution: Literal[512, 1024, 1536] = 512
    texture_size: Literal[1024, 2048, 4096] = 1024


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:  # pyright: ignore[reportUnusedFunction]
        return _PAGE

    @app.post("/banana")
    async def banana(req: BananaRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        if not req.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        try:
            handler = await fal_client.submit_async(
                NANO_BANANA_MODEL, arguments={"prompt": req.prompt},
            )
            result = await handler.get()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        image = result["images"][0]
        return {"image_url": image["url"], "content_type": image.get("content_type", "")}

    @app.post("/trellis")
    async def trellis(req: TrellisRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        if not req.image_url.strip():
            raise HTTPException(400, "image_url is empty")
        try:
            handler = await fal_client.submit_async(
                TRELLIS_MODEL,
                arguments={
                    "image_url": req.image_url,
                    "remesh": req.remesh,
                    "resolution": req.resolution,
                    "texture_size": req.texture_size,
                },
            )
            result = await handler.get()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        return {"glb_url": result["model_glb"]["url"]}

    return app


app = create_app()


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>nano banana → trellis playground</title>
<style>
  html, body { margin: 0; height: 100%; background: #101114; color: #e6e6e6;
    font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
  #app { display: grid; grid-template-columns: 360px 1fr; height: 100%; }
  #panel { padding: 12px; overflow-y: auto; border-right: 1px solid #2a2d35;
    display: flex; flex-direction: column; gap: 10px; background: #16181d; }
  #stage { display: grid; grid-template-rows: 1fr 1fr; min-width: 0; }
  .stage-pane { position: relative; border-bottom: 1px solid #2a2d35;
    display: flex; align-items: center; justify-content: center; background: #0c0d10; }
  .stage-pane:last-child { border-bottom: none; }
  .pane-label { position: absolute; top: 6px; left: 8px; font-size: 11px;
    color: #8a8f99; text-transform: uppercase; letter-spacing: 0.06em; pointer-events: none; }
  textarea { width: 100%; box-sizing: border-box; min-height: 140px;
    background: #0c0d10; color: #e6e6e6; border: 1px solid #2a2d35;
    border-radius: 4px; padding: 8px; font: inherit; resize: vertical; }
  button { padding: 8px 12px; border-radius: 4px; border: 1px solid #2a2d35;
    background: #1f2229; color: #e6e6e6; font: inherit; cursor: pointer; }
  button:hover:not(:disabled) { background: #2a4a78; border-color: #4a8fd8; }
  button:disabled { opacity: 0.5; cursor: default; }
  label.row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #b8bcc4; }
  label.row select { background: #0c0d10; color: #e6e6e6; border: 1px solid #2a2d35;
    border-radius: 3px; padding: 3px 6px; font: inherit; }
  fieldset { border: 1px solid #2a2d35; border-radius: 4px; padding: 8px;
    display: flex; flex-direction: column; gap: 6px; }
  legend { padding: 0 6px; color: #8a8f99; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; }
  #status { color: #9ad4ff; min-height: 1.4em; white-space: pre-wrap; word-break: break-word; }
  #status.err { color: #ff8080; }
  #status.ok { color: #8bd17c; }
  #image-pane img { max-width: 100%; max-height: 100%; object-fit: contain; }
  #image-pane .empty, #model-pane .empty { color: #5a5e68; font-size: 12px; }
  #model-pane canvas { display: block; }
  #image-url { font-size: 10px; color: #5a5e68; word-break: break-all;
    border-top: 1px solid #2a2d35; padding-top: 6px; margin-top: 6px; }
  .image-url-link { color: #6ac2c2; text-decoration: none; }
  .image-url-link:hover { text-decoration: underline; }
  #history { display: flex; flex-direction: column; gap: 4px; max-height: 30vh; overflow-y: auto;
    border-top: 1px solid #2a2d35; padding-top: 8px; }
  .history-item { padding: 4px 6px; border-radius: 3px; cursor: pointer;
    border: 1px solid #2a2d35; background: #0c0d10; font-size: 11px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .history-item:hover { border-color: #4a8fd8; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.171.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.171.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="app">
  <div id="panel">
    <label style="display:flex;flex-direction:column;gap:4px;">
      <span style="color:#8a8f99;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">prompt</span>
      <textarea id="prompt" placeholder="A weathered cypress log on a plain studio backdrop..."></textarea>
    </label>
    <button id="btn-banana" type="button">1. Generate image (Nano Banana)</button>

    <fieldset>
      <legend>trellis 2 settings</legend>
      <label class="row">
        <input type="checkbox" id="opt-remesh" /> remesh
      </label>
      <label class="row">
        resolution
        <select id="opt-resolution">
          <option value="512" selected>512</option>
          <option value="1024">1024</option>
          <option value="1536">1536</option>
        </select>
      </label>
      <label class="row">
        texture_size
        <select id="opt-texture">
          <option value="1024" selected>1024</option>
          <option value="2048">2048</option>
          <option value="4096">4096</option>
        </select>
      </label>
      <label class="row">
        <input type="checkbox" id="opt-auto" checked /> auto-forward to Trellis when image lands
      </label>
    </fieldset>

    <button id="btn-trellis" type="button" disabled>2. Generate 3D (Trellis 2)</button>

    <div id="status"></div>
    <div id="image-url"></div>

    <div id="history-wrap" style="display:none;">
      <div style="color:#8a8f99;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">history</div>
      <div id="history"></div>
    </div>
  </div>
  <div id="stage">
    <div class="stage-pane" id="image-pane">
      <span class="pane-label">image</span>
      <span class="empty">no image yet — submit a prompt</span>
    </div>
    <div class="stage-pane" id="model-pane">
      <span class="pane-label">3d model</span>
      <span class="empty">no model yet</span>
    </div>
  </div>
</div>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const promptEl = $("prompt");
const statusEl = $("status");
const imageUrlEl = $("image-url");
const imagePane = $("image-pane");
const modelPane = $("model-pane");
const btnBanana = $("btn-banana");
const btnTrellis = $("btn-trellis");
const optRemesh = $("opt-remesh");
const optResolution = $("opt-resolution");
const optTexture = $("opt-texture");
const optAuto = $("opt-auto");
const historyEl = $("history");
const historyWrap = $("history-wrap");

const HISTORY_KEY = "playground.history";
const PROMPT_KEY = "playground.prompt";

let currentImageUrl = null;
const history = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
promptEl.value = localStorage.getItem(PROMPT_KEY) ?? "";
promptEl.addEventListener("input", () => localStorage.setItem(PROMPT_KEY, promptEl.value));
renderHistory();

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

function renderHistory() {
  historyWrap.style.display = history.length ? "" : "none";
  historyEl.innerHTML = "";
  for (let i = history.length - 1; i >= 0; i--) {
    const h = history[i];
    const el = document.createElement("div");
    el.className = "history-item";
    el.title = h.prompt;
    el.textContent = h.prompt.slice(0, 80);
    el.addEventListener("click", () => {
      promptEl.value = h.prompt;
      localStorage.setItem(PROMPT_KEY, h.prompt);
      if (h.imageUrl) showImage(h.imageUrl);
      if (h.glbUrl) loadModel(h.glbUrl);
    });
    historyEl.appendChild(el);
  }
}

function pushHistory(entry) {
  history.push(entry);
  if (history.length > 30) history.shift();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function showImage(url) {
  currentImageUrl = url;
  btnTrellis.disabled = false;
  imagePane.innerHTML = '<span class="pane-label">image</span>';
  const img = document.createElement("img");
  img.src = url;
  imagePane.appendChild(img);
  imageUrlEl.innerHTML = `<a class="image-url-link" href="${url}" target="_blank" rel="noopener">${url}</a>`;
}

btnBanana.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) { setStatus("prompt is empty", "err"); return; }
  btnBanana.disabled = true;
  btnTrellis.disabled = true;
  setStatus("generating image…");
  const t0 = performance.now();
  try {
    const r = await fetch("/banana", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
    const { image_url } = await r.json();
    showImage(image_url);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`image ready (${dt}s)` + (optAuto.checked ? " — forwarding to Trellis…" : ""), "ok");
    pushHistory({ prompt, imageUrl: image_url, glbUrl: null, ts: Date.now() });
    if (optAuto.checked) await runTrellis();
  } catch (e) {
    setStatus(String(e.message ?? e), "err");
  } finally {
    btnBanana.disabled = false;
  }
});

btnTrellis.addEventListener("click", () => runTrellis());

async function runTrellis() {
  if (!currentImageUrl) { setStatus("no image yet", "err"); return; }
  btnTrellis.disabled = true;
  btnBanana.disabled = true;
  setStatus("generating 3D mesh…");
  const t0 = performance.now();
  try {
    const r = await fetch("/trellis", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image_url: currentImageUrl,
        remesh: optRemesh.checked,
        resolution: Number(optResolution.value),
        texture_size: Number(optTexture.value),
      }),
    });
    if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
    const { glb_url } = await r.json();
    loadModel(glb_url);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`3D mesh ready (${dt}s)`, "ok");
    if (history.length) {
      history[history.length - 1].glbUrl = glb_url;
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
  } catch (e) {
    setStatus(String(e.message ?? e), "err");
  } finally {
    btnTrellis.disabled = false;
    btnBanana.disabled = false;
  }
}

// --- 3D viewer --------------------------------------------------------------

let renderer, scene, camera, controls, currentModel = null;

function initViewer() {
  modelPane.innerHTML = '<span class="pane-label">3d model</span>';
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const fitSize = () => {
    const w = modelPane.clientWidth, h = modelPane.clientHeight;
    renderer.setSize(w, h, false);
    if (camera) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  };
  modelPane.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0d10);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(3, 5, 4);
  scene.add(dir);
  scene.add(new THREE.AxesHelper(0.5));
  camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.position.set(2, 1.5, 2.5);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  fitSize();
  window.addEventListener("resize", fitSize);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function loadModel(url) {
  if (!renderer) initViewer();
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const root = gltf.scene;
      // fit camera
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      root.position.sub(center);
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const dist = maxDim * 2.4;
      camera.position.set(dist * 0.7, dist * 0.5, dist * 0.9);
      camera.near = maxDim * 0.01;
      camera.far = maxDim * 50;
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();
      scene.add(root);
      currentModel = root;
    },
    undefined,
    (err) => setStatus(`GLB load failed: ${err.message ?? err}`, "err"),
  );
}
</script>
</body>
</html>
"""
