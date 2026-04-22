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

// --- log panel --------------------------------------------------------------

const KIND_COLOR = {
  "run.start": "#9ad4ff",
  "run.done": "#8bd17c",
  "run.error": "#ff8080",
  "bbox": "#e0c271",
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

scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(8, 12, 6);
scene.add(dir);
scene.add(new THREE.AxesHelper(1));
scene.add(new THREE.GridHelper(20, 20, 0x404040, 0x202020));

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
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
function fitToScene() {
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
  } catch (e) {
    appendEvent({ kind: "model.error", id: event.id, message: e.message });
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
