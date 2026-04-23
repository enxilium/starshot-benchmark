import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const SERVER_URL = document
  .querySelector('meta[name="server-url"]')
  .getAttribute("content");

const MODEL = "anthropic/claude-opus-4.7";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const formEl = document.getElementById("prompt-form");
const inputEl = document.getElementById("prompt-input");
const submitEl = document.getElementById("prompt-submit");
const assetsEl = document.getElementById("assets");
const assetsBodyEl = document.getElementById("assets-body");
const assetsCountEl = document.getElementById("assets-count");
const assetsHeaderEl = document.getElementById("assets-header");
const assetsToggleEl = document.getElementById("assets-toggle");

// --- log panel --------------------------------------------------------------

const KIND_COLOR = {
  "run.start": "#9ad4ff",
  "run.done": "#8bd17c",
  "run.error": "#ff8080",
  "bbox": "#e0c271",
  "image": "#f6a96a",
  "model": "#c586d1",
};

function setStatus(text, cls = "hdr") {
  statusEl.innerHTML = "";
  const p = document.createElement("p");
  p.className = `line ${cls}`;
  p.textContent = text;
  statusEl.appendChild(p);
}

function fmtValue(v) {
  if (Array.isArray(v)) return "[" + v.map(fmtValue).join(", ") + "]";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (v && typeof v === "object")
    return "{" + Object.entries(v).map(([k, x]) => `${k}=${fmtValue(x)}`).join(", ") + "}";
  if (typeof v === "string") return v;
  return String(v);
}

function appendEvent(event) {
  const { kind, ...fields } = event;
  const p = document.createElement("p");
  p.className = "line";

  const tag = document.createElement("span");
  tag.className = "step";
  tag.textContent = `[${kind}]`;
  tag.style.color = KIND_COLOR[kind] ?? "#8bd17c";
  p.appendChild(tag);

  const entries = Object.entries(fields);
  if (entries.length === 0) {
    p.appendChild(document.createTextNode(""));
  } else {
    for (const [k, v] of entries) {
      const kv = document.createElement("span");
      kv.className = "kv";
      const label = document.createElement("span");
      label.className = "k";
      label.textContent = ` ${k}=`;
      kv.appendChild(label);
      kv.appendChild(document.createTextNode(fmtValue(v)));
      p.appendChild(kv);
    }
  }
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.innerHTML = "";
}

// --- asset browser ----------------------------------------------------------

// id -> { imageUrl, prompt, modelUrl, status: "pending" | "loaded" | "error", errorMessage }
const assets = new Map();

function assetStatus(a) {
  return a.status ?? "pending";
}

function upsertAsset(id, patch) {
  const cur = assets.get(id) ?? { imageUrl: null, prompt: null, modelUrl: null, status: "pending" };
  assets.set(id, { ...cur, ...patch });
  renderAsset(id);
  assetsCountEl.textContent = `(${assets.size})`;
}

function renderAsset(id) {
  const a = assets.get(id);
  if (!a) return;
  let card = assetsBodyEl.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!card) {
    card = document.createElement("div");
    card.className = "asset-card";
    card.dataset.id = id;
    card.innerHTML = `
      <a class="asset-thumb-link" target="_blank" rel="noopener">
        <div class="asset-thumb placeholder">no image</div>
      </a>
      <div class="asset-body">
        <div class="asset-id"></div>
        <div class="asset-status pending">pending</div>
        <div class="asset-prompt"></div>
      </div>
    `;
    assetsBodyEl.appendChild(card);
    const promptEl = card.querySelector(".asset-prompt");
    promptEl.addEventListener("click", () => {
      promptEl.classList.toggle("expanded");
    });
  }
  card.querySelector(".asset-id").textContent = id;

  const status = assetStatus(a);
  card.className = `asset-card status-${status}`;
  const statusTag = card.querySelector(".asset-status");
  statusTag.className = `asset-status ${status}`;
  statusTag.textContent = status === "error" && a.errorMessage
    ? `error: ${a.errorMessage}`
    : status;

  const link = card.querySelector(".asset-thumb-link");
  const thumb = card.querySelector(".asset-thumb");
  if (a.imageUrl) {
    const absImg = new URL(a.imageUrl, SERVER_URL).toString();
    link.href = absImg;
    if (thumb.tagName !== "IMG") {
      const img = document.createElement("img");
      img.className = "asset-thumb";
      img.loading = "lazy";
      img.alt = id;
      img.src = absImg;
      thumb.replaceWith(img);
    } else if (thumb.src !== absImg) {
      thumb.src = absImg;
    }
  }

  const promptEl = card.querySelector(".asset-prompt");
  promptEl.textContent = a.prompt ?? "";
}

function clearAssets() {
  assets.clear();
  assetsBodyEl.innerHTML = "";
  assetsCountEl.textContent = "(0)";
}

assetsHeaderEl.addEventListener("click", () => {
  const collapsed = assetsEl.classList.toggle("collapsed");
  assetsToggleEl.textContent = collapsed ? "▸" : "▾";
});

// --- three.js scene ---------------------------------------------------------

const host = document.getElementById("canvas-host");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);

// Bboxes live in a sibling group so they don't participate in fit-to-scene,
// and so clearScene can nuke them independently.
const bboxRoot = new THREE.Group();
scene.add(bboxRoot);
const bboxes = new Map(); // id -> THREE.Box3Helper
let hoveredBboxId = null;

const BBOX_COLOR_DEFAULT = 0xff3b3b;
const BBOX_COLOR_HOVER = 0xffe14a;

const tooltip = document.createElement("div");
tooltip.id = "bbox-tooltip";
tooltip.style.cssText = [
  "position: fixed",
  "padding: 3px 8px",
  "background: rgba(22, 24, 29, 0.92)",
  "color: #e6e6e6",
  "border: 1px solid #2a2d35",
  "border-radius: 4px",
  "font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
  "pointer-events: none",
  "display: none",
  "z-index: 10",
].join("; ");
document.body.appendChild(tooltip);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.05,
  5000,
);
camera.position.set(8, 6, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1, 0);
controls.update();

// Once the user drags the camera, stop auto-fitting so subsequent runs
// preserve their chosen angle. The flag deliberately survives clearScene.
let cameraUserMoved = false;
controls.addEventListener("start", () => {
  cameraUserMoved = true;
});

// --- WASD fly controls (complementary to OrbitControls) --------------------
// WASD strafes on the horizontal plane relative to the camera direction;
// Q/E moves world-down/up; Shift multiplies speed. Translates camera and
// target together so OrbitControls' pivot follows the camera.
const pressedKeys = new Set();
let _lastMoveT = performance.now();
const _MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e"]);

function _isTypingTarget(t) {
  return t instanceof HTMLElement &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

window.addEventListener("keydown", (ev) => {
  if (_isTypingTarget(ev.target)) return;
  const k = ev.key.toLowerCase();
  if (_MOVE_KEYS.has(k)) {
    pressedKeys.add(k);
    ev.preventDefault();
  } else if (k === "shift") {
    pressedKeys.add("shift");
  }
});

window.addEventListener("keyup", (ev) => {
  pressedKeys.delete(ev.key.toLowerCase());
});

// Alt-tab / focus-loss: drop held keys so they don't stick on.
window.addEventListener("blur", () => pressedKeys.clear());

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _move = new THREE.Vector3();

function _applyKeyboardMove(dt) {
  if (pressedKeys.size === 0) return;
  const speed = 2 * (pressedKeys.has("shift") ? 3 : 1) * dt;

  _fwd.subVectors(controls.target, camera.position);
  _fwd.y = 0;
  if (_fwd.lengthSq() === 0) return;
  _fwd.normalize();
  _right.crossVectors(_fwd, _worldUp).normalize();

  _move.set(0, 0, 0);
  if (pressedKeys.has("w")) _move.addScaledVector(_fwd, speed);
  if (pressedKeys.has("s")) _move.addScaledVector(_fwd, -speed);
  if (pressedKeys.has("d")) _move.addScaledVector(_right, speed);
  if (pressedKeys.has("a")) _move.addScaledVector(_right, -speed);
  if (pressedKeys.has("e")) _move.addScaledVector(_worldUp, speed);
  if (pressedKeys.has("q")) _move.addScaledVector(_worldUp, -speed);

  if (_move.lengthSq() === 0) return;
  camera.position.add(_move);
  controls.target.add(_move);
  cameraUserMoved = true;
}

scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(8, 12, 6);
scene.add(dir);
scene.add(new THREE.AxesHelper(1));

// Infinite ground grid: a huge plane with a procedural grid shader. Lines
// antialias via screen-space derivatives and fade with distance so the plane
// never looks like it has an edge. Fade distance is driven from camera
// distance each frame so detail scales naturally as the user zooms.
const gridGeom = new THREE.PlaneGeometry(100000, 100000);
gridGeom.rotateX(-Math.PI / 2);
const gridMat = new THREE.ShaderMaterial({
  uniforms: {
    uCameraPos: { value: new THREE.Vector3() },
    uMinorColor: { value: new THREE.Color(0x202020) },
    uMajorColor: { value: new THREE.Color(0x505050) },
    uMinorSpacing: { value: 1.0 },
    uMajorSpacing: { value: 10.0 },
    uFadeStart: { value: 20.0 },
    uFadeEnd: { value: 200.0 },
  },
  vertexShader: `
    varying vec3 vWorldPos;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: `
    uniform vec3 uCameraPos;
    uniform vec3 uMinorColor;
    uniform vec3 uMajorColor;
    uniform float uMinorSpacing;
    uniform float uMajorSpacing;
    uniform float uFadeStart;
    uniform float uFadeEnd;
    varying vec3 vWorldPos;

    float gridLine(vec2 p, float spacing) {
      vec2 q = p / spacing;
      vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
      return 1.0 - min(min(g.x, g.y), 1.0);
    }

    void main() {
      float minor = gridLine(vWorldPos.xz, uMinorSpacing);
      float major = gridLine(vWorldPos.xz, uMajorSpacing);
      float d = distance(vWorldPos.xz, uCameraPos.xz);
      float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, d);
      float alpha = max(minor * 0.5, major) * fade;
      if (alpha < 0.002) discard;
      vec3 col = mix(uMinorColor, uMajorColor, major);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
});
const groundGrid = new THREE.Mesh(gridGeom, gridMat);
groundGrid.renderOrder = -1;
scene.add(groundGrid);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - _lastMoveT) / 1000);
  _lastMoveT = now;
  _applyKeyboardMove(dt);
  controls.update();

  gridMat.uniforms.uCameraPos.value.copy(camera.position);
  const camDist = Math.max(1, camera.position.distanceTo(controls.target));
  gridMat.uniforms.uFadeStart.value = camDist * 0.5;
  gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

  renderer.render(scene, camera);
}
animate();

function clearScene() {
  while (sceneRoot.children.length > 0) {
    const child = sceneRoot.children[0];
    sceneRoot.remove(child);
    child.traverse?.((n) => {
      if (n.isMesh) {
        n.geometry?.dispose?.();
        const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
        for (const m of mats) m.dispose?.();
      }
    });
  }
  for (const helper of bboxes.values()) {
    bboxRoot.remove(helper);
    helper.geometry?.dispose?.();
    helper.material?.dispose?.();
  }
  bboxes.clear();
  hoveredBboxId = null;
  tooltip.style.display = "none";
}

// Fit controls to the union of all loaded models after each addition.
// Skipped once the user has manually adjusted the camera.
function fitToScene() {
  if (cameraUserMoved) return;
  const box = new THREE.Box3().setFromObject(sceneRoot);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = 0.5 * Math.max(size.x, size.y, size.z);
  if (!isFinite(radius) || radius === 0) return;
  controls.target.copy(center);
  const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
  const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
  camera.position.copy(center).addScaledVector(dirVec, dist * 1.6);
  camera.near = Math.max(0.01, radius / 100);
  camera.far = Math.max(100, radius * 100);
  camera.updateProjectionMatrix();
  controls.update();
}

// --- model loading ----------------------------------------------------------

const loader = new GLTFLoader();

async function loadModel(event) {
  const absUrl = new URL(event.url, SERVER_URL).toString();
  upsertAsset(event.id, { modelUrl: event.url });
  try {
    const gltf = await loader.loadAsync(absUrl);
    gltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of mats) m.side = THREE.DoubleSide;
      }
    });
    gltf.scene.name = `${event.artifact_kind}:${event.id}`;
    sceneRoot.add(gltf.scene);
    fitToScene();
    upsertAsset(event.id, { status: "loaded" });
  } catch (e) {
    appendEvent({ kind: "model.error", id: event.id, message: e.message });
    upsertAsset(event.id, { status: "error", errorMessage: e.message });
  }
}

// --- bbox overlays ----------------------------------------------------------

// `{ id, origin: [x,y,z], dimensions: [dx,dy,dz] }` — matches the Python
// BoundingBox serialization. Signed and zero-valued dimensions are allowed
// (walls/floors are flat).
function loadBbox(event) {
  const { id, origin, dimensions } = event;
  if (bboxes.has(id)) {
    const prev = bboxes.get(id);
    bboxRoot.remove(prev);
    prev.geometry?.dispose?.();
    prev.material?.dispose?.();
    if (hoveredBboxId === id) hoveredBboxId = null;
  }
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const fx = ox + dimensions[0], fy = oy + dimensions[1], fz = oz + dimensions[2];
  const box3 = new THREE.Box3(
    new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
    new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
  );
  const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
  helper.userData.bboxId = id;
  bboxRoot.add(helper);
  bboxes.set(id, helper);
}

const raycaster = new THREE.Raycaster();
// Box3Helper is a LineSegments; a generous threshold makes thin wireframes
// (and zero-thickness walls) comfortably hoverable.
raycaster.params.Line.threshold = 0.1;
const pointer = new THREE.Vector2();

function setHoveredBbox(id) {
  if (id === hoveredBboxId) return;
  const prev = hoveredBboxId !== null ? bboxes.get(hoveredBboxId) : null;
  if (prev) prev.material.color.setHex(BBOX_COLOR_DEFAULT);
  hoveredBboxId = id;
  const next = id !== null ? bboxes.get(id) : null;
  if (next) next.material.color.setHex(BBOX_COLOR_HOVER);
}

function positionTooltip(clientX, clientY, text) {
  tooltip.textContent = text;
  tooltip.style.left = `${clientX + 12}px`;
  tooltip.style.top = `${clientY + 12}px`;
  tooltip.style.display = "block";
}

renderer.domElement.addEventListener("pointermove", (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const targets = Array.from(bboxes.values());
  const hits = raycaster.intersectObjects(targets, false);
  if (hits.length > 0) {
    const id = hits[0].object.userData.bboxId ?? null;
    setHoveredBbox(id);
    if (id !== null) {
      positionTooltip(ev.clientX, ev.clientY, id);
      return;
    }
  }
  setHoveredBbox(null);
  tooltip.style.display = "none";
});

renderer.domElement.addEventListener("pointerleave", () => {
  setHoveredBbox(null);
  tooltip.style.display = "none";
});

// --- event dispatch ---------------------------------------------------------

function dispatch(event) {
  appendEvent(event);
  switch (event.kind) {
    case "run.start":
      setStatus(`run :: ${event.model}`);
      break;
    case "run.done":
      setStatus("run complete");
      finishRun();
      break;
    case "run.error":
      setStatus(`error: ${event.message}`, "err");
      finishRun();
      break;
    case "bbox":
      loadBbox(event);
      break;
    case "image":
      upsertAsset(event.id, { imageUrl: event.url, prompt: event.prompt });
      break;
    case "model":
      loadModel(event);
      break;
    // Everything else is already shown as a log line above.
  }
}

// --- run lifecycle ----------------------------------------------------------

let currentSource = null;

function setRunning(isRunning) {
  submitEl.disabled = isRunning;
  submitEl.textContent = isRunning ? "Running…" : "Run";
}

function finishRun() {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  setRunning(false);
}

async function startRun(prompt) {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  clearScene();
  clearLog();
  clearAssets();
  setRunning(true);
  setStatus("POST /generate …");

  const payload = { model: MODEL, prompt };
  let res;
  try {
    res = await fetch(new URL("/generate", SERVER_URL), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    setStatus(`request failed: ${e.message}`, "err");
    setRunning(false);
    return;
  }
  if (!res.ok) {
    setStatus(`HTTP ${res.status}: ${await res.text()}`, "err");
    setRunning(false);
    return;
  }
  const { run_id, events_url } = await res.json();
  setStatus(`run ${run_id} — streaming events…`);
  subscribe(new URL(events_url, SERVER_URL).toString());
}

function subscribe(url) {
  const es = new EventSource(url);
  currentSource = es;
  es.onmessage = (ev) => {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch {
      return;
    }
    dispatch(data);
  };
  es.onerror = () => {
    // EventSource auto-reconnects on transient failures; only surface a hard close.
    if (es.readyState === EventSource.CLOSED && currentSource === es) {
      setStatus("stream closed", "err");
      setRunning(false);
      currentSource = null;
    }
  };
}

formEl.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const prompt = inputEl.value.trim();
  if (!prompt) {
    setStatus("prompt is empty", "err");
    return;
  }
  startRun(prompt);
});

// Cmd/Ctrl+Enter submits from inside the textarea.
inputEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault();
    formEl.requestSubmit();
  }
});
