# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`starshot-benchmark` is an orchestrator for a **text-to-3D scene pipeline**. A user prompt like "A beautiful modern mansion" or "A swamp with islands" is recursively decomposed by LLMs into subscenes and anchor objects, each generated as a 3D mesh via **Hunyuan 3.1**, then composed into a single `.glb` file.

The broader goal is to **benchmark LLM spatial reasoning** — the dashboard lets you swap the LLM used at every reasoning step in the pipeline and compare outputs.

## Repository layout (planned)

Two top-level parts:

- **`client/`** — Node process running a Three.js sandbox + dashboard. Lets the user pick the LLM used across the pipeline, submits a prompt to the server, then loads and renders the returned `.glb` URL.
- **`server/`** — Orchestrator that runs the full pipeline described below and returns a URL to the final `.glb`.

The repo is currently a clean slate aside from `enx.toml` (project meta for the `enx` task runner). When implementing, choose stacks appropriate for: (a) heavy LLM orchestration and async work on the server, (b) a minimal dashboard + Three.js viewer on the client.

## Pipeline architecture

The pipeline has **two distinct phases**. Phase 1 recursively *divides* the scene down to leaves; phase 2 *generates* geometry for each leaf and composes up.

### Phase 1 — Divider (recursive top-down)

Operates on `(prompt, bounding_box, parent_context)` and recurses.

1. **Input** — user prompt.
2. **Bounding Box Generation (LLM)** — overall bbox sized to scene shape (tall+narrow for a skyscraper, long+flat for a river).
3. **Scene Breakdown (LLM)** — splits the current scene into subscenes. For each subscene, emits:
   - a detailed prompt,
   - a bounding box contained within the parent's bbox,
   - a **high-level plan** of what the subscene will contain downstream. Plans are persisted to a **State Repository** and re-fetched on future recursions so siblings stay consistent.

   The State Repository read at this step also includes **any subscenes that have already completed phase 2** elsewhere in the recursion (see below) — the LLM should use their realized object lists to keep later subscenes stylistically consistent with what's already been generated.
4. **Bounding-box validator (deterministic)** — checks no sibling bboxes overlap. On conflict, record the conflict and loop back to step 3 with the conflict details. **Every retry must be logged** (retry counts are a benchmark signal).
5. **Frame Decider (LLM)** — does this subscene need a frame (walls/floor/ceiling)? If yes, generate frame geometry deterministically (no Hunyuan). Two frame types:
   - `plane` — standard flat surfaces (walls, ceilings, floors).
   - `curve` — parametric equation + endpoints, used for non-planar surroundings.
6. **Recurse** — treat each subscene as the new scene and repeat from step 3. Terminate when step 3 decides a subscene is atomic (e.g., the toilet area of a bathroom).

### Phase 2 — Generation (bottom-up per leaf subscene)

Runs on every leaf subscene emitted by phase 1.

1. **Input** — prompt, bbox, parent ref, parent frame definitions (if any).
2. **Anchor objects (LLM)** — list of defining objects (e.g., a meeting room → table, chairs, TV) plus **relationships** from a fixed vocabulary: `ON`, `BESIDE`, `BELOW`, `ABOVE`, `ATTACHED`. Vocabulary may grow — keep it centrally defined. Example: `CHAIR BESIDE TABLE`.
3. **Relationship validator (deterministic)** — must form a valid DAG. Enforces:
   - implied inverses (`A ABOVE B` ⟹ `B BELOW A`),
   - no contradictions,
   - every object participates in at least one relationship,
   - relationship targets are valid (another object or a parent frame).
4. **Per-object bbox generation** — starting from objects attached to parent frames (whose coords are already known), resolve via **topological sort** down the dependency chain.
5. **Object-bbox validator** — same rules as phase-1 step 4: no overlaps, every object has a bbox, all bboxes fit within the parent bbox. Retries loop back and are logged.
6. **Mesh generation (Hunyuan 3.1)** — one mesh per anchor object using its prompt.
7. **Rescale** — fit each mesh to its bbox via the **maximal dimension**. Overshoot in other dimensions post-scaling is acceptable.
8. **State-driven completion loop (LLM)** — assemble a `state` containing (objects so far, their prompts, their coords) and ask the LLM "does this scene need any more objects?" The LLM returns **exactly one** additional object. Run that single object through steps 3–7, update `state`, and ask again. Loop until the LLM says no more are needed.
9. **Assembly** — merge all meshes + frames into the final `.glb` and return a URL to the client.

On completion, the leaf subscene's **realized contents** (final object list with prompts and coords) are written back to the **State Repository**. Because phase 1 and phase 2 interleave — a leaf like the bathroom's toilet area can finish generating entirely before a sibling like the bedroom has even picked a frame — later phase-1 step 3 calls must be able to read already-completed subscenes and use them as stylistic reference.

## Cross-cutting concerns to preserve in implementation

- **Pluggable LLM selection.** Every LLM call site must read the model from a request-scoped config sourced from the dashboard. Do not hard-code model IDs at call sites.
- **State Repository.** Holds two kinds of entries, both read by phase-1 step 3:
  1. **High-level plans** written when a subscene is first broken down (used so siblings stay consistent with not-yet-generated areas).
  2. **Realized subscene contents** written when phase 2 finishes a leaf (object list, prompts, coords) — used for stylistic consistency with *already-generated* areas, since divider and generator phases interleave across the tree.

  Treat it as a first-class module, not ad-hoc state inside the recursion. Writes come from both phases; reads come primarily from phase-1 step 3.
- **Retry logging.** Both bbox validators (phase 1 step 4, phase 2 step 5) and the relationship validator can force retries. Record every retry with the conflict detail — this is core benchmark output.
- **Deterministic vs. LLM steps.** Validators, frame generation, rescaling, topological sort, and assembly are deterministic. Keep them free of LLM calls so benchmark results isolate LLM quality.
- **Recursion termination** is an LLM decision (phase 1 step 3). There is no hard depth cap in the spec — if one is added, make it configurable.
- **Frames are not Hunyuan.** Plane and curve frames are generated programmatically from their definitions.

## Client ↔ server contract

- Client `POST`s `(prompt, model_selection)` to the server.
- Server runs the full two-phase pipeline and responds with a URL to the final `.glb`.
- Client fetches the `.glb` from that URL and renders it in the Three.js sandbox.

Keep this surface small; the dashboard's primary job is model selection and prompt submission, not pipeline introspection (though surfacing retry counts and phase timings later is valuable for the benchmark use case).

## Commands

No build/test/lint commands are wired up yet — the repo is a fresh scaffold. `enx.toml` exposes `enx up`, `enx down`, `enx start`, and `enx test` hooks; populate them as the client and server stacks are chosen.
