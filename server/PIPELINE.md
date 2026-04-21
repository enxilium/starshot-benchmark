# Server pipeline walkthrough

End-to-end description of a single run — from the raw prompt string
received by `POST /generate` to the final `.glb` served by
`GET /glb/{run_id}`.

The steps below are numbered in a single linear sequence. Recursion and
loops point back to earlier step numbers instead of being renumbered.

File references use `path:line` so you can jump to the source.

---

## 1. HTTP entry — `POST /generate`

**Location:** `server/app/api/step01_http_entry.py:32`

**Input:**
```json
{ "prompt": "A beautiful modern mansion", "model": "<model_id>", "max_retries": 3 }
```

**Work:**
- Mint `run_id` (hex uuid).
- Build an `EventLog` backed by `runs/{run_id}/events.jsonl`.
- Construct a fresh `InMemoryStateRepository` (run-scoped;
  `server/app/state_repo.py:68`).
- Build an `LLMClient` via `factories.make_llm(request.model)`.
- Build a `MeshGenerator` via `factories.make_mesh_generator()` — today
  only `StubMeshGenerator` is wired; `"hunyuan"` raises
  `NotImplementedError` (`server/app/api/factories.py:22`).
- Pack everything into a `RunContext`.
- Write an initial `pending` `run.json`.
- Spawn the pipeline on a detached `asyncio.Task` (step 2).

**Output (synchronous HTTP response):**
```json
{ "run_id": "...", "events_url": "/runs/.../events", "status_url": "/runs/..." }
```

---

## 2. Background run wrapper — `execute_run`

**Location:** `server/app/pipeline/step02_execute_run.py:52`

**Input:** `(RunContext, prompt, TerminalLogger)`

**Work:**
- Activate the `RunContext` as a contextvar so every pipeline module can
  call `current()` without threading it through explicitly.
- Flip `run.json` to `running`.
- Spawn the terminal logger task.
- `await orchestrator.run(prompt, ctx.model_id)` (step 3).
- In a `finally` block, write a terminal `run.json` with status, `glb_url`,
  retry summary (rebuilt from the JSONL event log), and error (if any).

**Output:** the `glb_url` returned from the orchestrator is captured and
written into `run.json`.

---

## 3. Orchestrator top level

**Location:** `server/app/pipeline/step03_orchestrator.py:68`

**Input:** `(user_prompt: str, model_id: str)`

Emits `RunStarted`, then runs the three stages below sequentially:

1. **Divide** (step 4): `root = await divider.divide(user_prompt)` →
   `SubsceneNode` tree.
2. **Generate per leaf** (step 9): iterate `collect_leaves(root)` in
   depth-first order; for each leaf resolve its ancestor frame chain with
   `collect_frames_for_leaf` and call `generator.generate_leaf(...)`.
3. **Assemble + publish** (steps 16–17): flatten meshes + frames, export
   GLB.

Emits `RunCompleted(glb_url, total_duration_ms, retry_summary)` on
success, `RunFailed` on exception.

**Output:** the string `/glb/{run_id}`.

---

## 4. Overall bounding box (root of the divide tree)

**Location:** `server/app/pipeline/phase1/step04_overall_bbox.py:25`

Runs **once**, at the root of the recursion.

**Input:** `user_prompt: str`.

**Work:** one structured LLM call (`ctx.llm.call_structured`) with
`SYSTEM_PROMPT` + `render(user_prompt)` and an `Output` schema containing
a `BoundingBox`.

**Output:** a `BoundingBox` sized to the scene's gross aspect ratio (tall
for a skyscraper, flat for a river).

**Retry policy:** none. Pydantic enforces non-zero `dimensions` on the
emitted `BoundingBox` (`types.py:55`); any other failure is a
provider-level error.

> The `BoundingBox` schema itself is `(origin, dimensions)` with signed
> `dimensions` — either corner can be the origin, the other is implicit.
> `min_corner` / `max_corner` / `size` / `center` are derived properties
> (`types.py:37`).

The bbox is wrapped into the root node:
```python
root = SubsceneNode(scope_id="root", prompt=user_prompt, bbox=overall_bbox, high_level_plan="")
```
and `_recurse(root)` is called (which runs steps 5–8 on each node).

---

## 5. Frame decider (current node)

**Location:** `server/app/pipeline/phase1/step05_frame_decider.py:33`

**Input:** `(prompt, bbox)` of the current node.

**Work:** one structured LLM call returning
`Output(needs_frame: bool, frames: list[PlaneFrameSpec|CurveFrameSpec])`.
`_spec_to_frame` converts each spec into a concrete `PlaneFrame`
(`origin, u_axis, v_axis`) or `CurveFrame` (`control_points, height`).
See `server/app/core/types.py:227` and `:262`.

**Output:** `list[Frame]`, assigned to `node.frames`. Outdoor/open scenes
get `[]`.

---

## 6. Scene breakdown (with bbox validator retry loop)

**Location:** `server/app/pipeline/phase1/step06_scene_breakdown.py:49`

**Input:** `(scope_id, prompt, bbox)` of the current node.

### 6a. Read visible state

Call `ctx.state_repo.read_visible()` (no arguments — the repo returns
everything written so far across the whole run, by design;
`state_repo.py:89`). `_summarize_visible` formats the result into a block
like:
```
Plans in flight:
  - 'living_room' -> 'a cosy reading nook...'
Leaves already realized:
  - 'bathroom.toilet' contains objects ['toilet', 'sink', ...]
```
This is the cross-recursion stylistic-consistency mechanism. It is
fetched **once** per step invocation, not once per retry.

### 6b. LLM loop via `call_with_validator`

Helper: `server/app/llm/client.py:54`.

- `build(prior)` renders the prompt (system + user with the visible-state
  summary + parent bbox + any `prior_attempts`).
- `validate(out)`:
  1. `out.is_atomic` → accept.
  2. Empty `subscenes` on a non-atomic → `step06_empty_breakdown`.
  3. Otherwise run the shared
     `geometry.bbox_validator.validate_boxes(parent=bbox, children=[...])`
     (`server/app/geometry/bbox_validator.py:22`): each child must
     `contains`-fit in the parent, no pairwise `overlaps` between
     siblings. First conflict wins (containment before overlap).
- On conflict: emit `StepRetried(step_id, attempt, conflict)`, append a
  `PriorAttempt(output, conflict)`, re-prompt.
- After `ctx.max_retries` attempts: raise `RetryExhausted` → orchestrator
  converts to `RunFailed`.

**Output:** `Output(is_atomic: bool, subscenes: list[SubsceneSpec])`.
Each spec carries `(scope_id, prompt, bbox, high_level_plan)`.

If `is_atomic`, the node's recursion terminates here (no step 7, no
step 8). Otherwise proceed to step 7.

---

## 7. Write plan entries to the state repository

**Location:** `server/app/pipeline/phase1/step07_write_plans.py:23`

For each `spec` in `breakdown.subscenes`:
- Build `PlanEntry(scope_id, prompt, bbox, high_level_plan)`.
- `await ctx.state_repo.write_plan(entry)`.
- Emit `StateRepoWrite(entry_type="plan", scope_id=spec.scope_id)`.

Writing **before** recursing is what lets a sibling's later step 6 call
(`read_visible` in step 6a) see its earlier siblings' plans.

---

## 8. Recurse into each child → step 5

**Location:** `server/app/pipeline/phase1/step08_recurse.py:44` (the
`_recurse` body; the public `divide()` entry is at `:31`).

For each spec, construct a child
`SubsceneNode(scope_id, prompt, bbox, high_level_plan)` and re-enter the
recursion: run **steps 5 → 6 → (7 → 8) or stop** on the child. Append it
to `node.children`.

Once the root's recursion unwinds, the orchestrator has the fully
populated `SubsceneNode` tree.

---

## 9. Collect leaves and ancestor frame chains

**Location:** `server/app/pipeline/phase1/step09_collect_leaves.py`
(`collect_leaves` at `:19`, `collect_frames_for_leaf` at `:34`). Both
are called from `server/app/pipeline/step03_orchestrator.py:83`.

- `collect_leaves(root)` — DFS-flatten into atomic leaves.
- `collect_frames_for_leaf(root, leaf)` — walk root → leaf and
  concatenate every ancestor's `frames`. This is the frame list the leaf
  inherits.

For each leaf, run steps 10 → 15. After step 15 completes for one leaf,
loop back to step 10 for the next leaf.

---

## 10. Anchor objects + graph validator (current leaf)

**Location:** `server/app/pipeline/phase2/step10_anchor_objects.py:39`

Another `call_with_validator` loop.

- `build()` renders the prompt with leaf prompt + leaf bbox + a frame
  summary (`summarize_frames`: one line per frame with id/kind/axes).
- `validate(out)` delegates to
  `geometry/relationship_graph.validate_and_sort(...)`
  (`server/app/geometry/relationship_graph.py:79`):
  1. Duplicate anchor-id check.
  2. Subject must be a known object; target must be an object or frame;
     `ATTACHED` must target a frame.
  3. Inject implied inverses (`A ABOVE B ⇒ B BELOW A`) via
     `INVERSE_KINDS`. `BESIDE`/`ATTACHED` are symmetric — no inverse
     added (`types.py:160`).
  4. Reject `(A ABOVE B) + (A BELOW B)` on the same ordered pair.
  5. Coverage: every object must participate in at least one
     relationship.
  6. Every object must transitively reach a frame along the **original**
     (non-inverse) edges.
  7. Kahn's topological sort over objects only (frames are prerequisites,
     not nodes). Cycle → `relationship_cycle` conflict.
- On success, the validator closure stashes the resulting `GraphResult`.

**Output:** `Step2Result(objects, relationships, graph)`. `graph.order`
is the topo order used by step 11.

---

## 11. Object bbox generation (with bbox validator retry loop)

**Location:** `server/app/pipeline/phase2/step11_object_bboxes.py:32`

Third `call_with_validator` loop. Used in two modes by the same function:

- **Initial (this call):** `already_resolved={}`,
  `to_resolve=graph.order`.
- **Incremental:** used inside step 14; see below.

- `build()` renders leaf prompt/bbox, frame summary, relationships
  summary (`summarize_relationships`), topo order, already-resolved
  bboxes, and the subset to resolve.
- `validate(out)`:
  1. Coverage: emitted `object_id`s must exactly equal `to_resolve`.
  2. Build `already_resolved ∪ new` and run `validate_boxes(
     parent=leaf.bbox, children=...)`: every bbox contained in the leaf
     bbox, no pairwise overlap.

**Output:** `dict[object_id, BoundingBox]` for the newly-resolved subset
only. Caller merges with `already_resolved`.

Each object is then cloned with its bbox attached (`objects_with_bboxes`).

---

## 12. Mesh generation

**Location:** `server/app/pipeline/phase2/step12_mesh_generation.py:26`

- Runs `ctx.mesh_generator.generate(obj.prompt, obj.id)` for every object
  under an `asyncio.Semaphore(settings.mesh_gen_concurrency)`.
- Emits `MeshGenerated(object_id, duration_ms, backend)` per object.
- Backend is abstracted by the `MeshGenerator` protocol
  (`server/app/mesh_gen/interface.py`). Today only `StubMeshGenerator` is
  registered; Hunyuan 3.1 will be a drop-in replacement.
- Failures propagate. Mesh generation is **not** retried under
  `ctx.max_retries` — that budget is reserved for validator-driven LLM
  retries.

**Input:** `list[AnchorObject]` (with bboxes).
**Output:** `dict[object_id, trimesh.Trimesh]` at the backend's native
scale.

---

## 13. Rescale

**Location:** `server/app/pipeline/phase2/step13_rescale.py:21`, called
from `step14_completion_loop.py:126` (initial) and `:170` (incremental).

Per mesh:
- Compute the mesh's current axis-aligned bounds → current max extent.
- Uniform scale factor = `bbox.max_dimension / current_max_dim` (the
  **maximal** dimension per CLAUDE.md, so overshoot on other axes is
  acceptable).
- Transforms: translate mesh center to origin → uniform scale → translate
  to `bbox.center`.

Produces a new `Trimesh`; the input mesh is left untouched.

After this, `meshes` holds correctly-placed, scaled meshes keyed by
`object_id`.

---

## 14. Completion loop (add objects one at a time until the LLM says stop)

**Location:** `server/app/pipeline/phase2/step14_completion_loop.py`
(`generate_leaf` driver at `:92`, `_propose_one` LLM call at `:70`).

Each iteration:

1. Call the completion-loop LLM with
   `(leaf, frames, placed_objects)`. `_summarize_realized` lists every
   placed object's id/prompt/bbox in the prompt.
2. **Output:** `Output(stop: bool, object: Optional[NewObject],
   new_relationships: list[Relationship])`. If `stop=True` or
   `object is None` → break out of the loop (go to step 15).
3. Build a candidate graph (objects ∪ new object, rels ∪ new rels) and
   re-run `validate_and_sort` from step 10's validator. If the proposal is
   structurally broken (cycle, contradiction, unknown target, ...),
   **break** — do not retry. Keep everything placed so far.
4. Re-enter **step 11 in incremental mode** with
   `already_resolved=<current bboxes>` and `to_resolve=[new_obj.id]`.
   This is a full validator-driven retry loop on just the new bbox — it
   must fit the leaf and not overlap existing bboxes. Retry exhaustion
   fails the whole run.
5. Re-enter **step 12 + step 13** for just the new object; merge the
   rescaled mesh into `meshes`.
6. Append to `placed` and `all_rels`; loop back to 14.1.

---

## 15. Write realized entry to the state repository

**Location:** `server/app/pipeline/phase2/step15_write_realized.py:18`
(called from `step14_completion_loop.py` at the end of `generate_leaf`).

Before returning the leaf result:
- `ctx.state_repo.write_realized(RealizedEntry(scope_id, prompt, bbox,
  objects, relationships))`.
- Emit `StateRepoWrite(entry_type="realized", scope_id=leaf.scope_id)`.

This is what future step 6a (`read_visible`) calls would see as "Leaves
already realized". In today's strictly-sequential orchestrator this
cannot happen within a single run (step 4's recursion fully completes
before step 10 starts), but the write/read contract is already in place
for later interleaving.

**Output (of the leaf loop iteration):**
`GeneratedLeaf(scope_id, leaf, objects, relationships, frames, meshes)`.

After step 15, the orchestrator loops back to step 10 for the next leaf
(or proceeds to step 16 once all leaves are done).

---

## 16. Assemble the scene

**Location:** `server/app/pipeline/step16_assemble_scene.py:24`. Called
from `step03_orchestrator.py:91`.

**Inputs:**
- `object_meshes: list[(object_id, Trimesh)]` — flattened across every
  `GeneratedLeaf.meshes`.
- `all_frames: list[Frame]` — collected by walking the whole
  `SubsceneNode` tree with `_collect_all_frames`.

**Work:**
- Create a fresh `trimesh.Scene`.
- Add each object mesh as `scene.add_geometry(mesh, node_name=object_id,
  geom_name=object_id)`.
- For each frame, build a deterministic mesh with `frame_to_mesh`
  from `server/app/geometry/frames.py:22` (a shared utility, not a step):
  - `PlaneFrame` → two-triangle quad from `origin + u*u_axis + v*v_axis`
    corners.
  - `CurveFrame` → arc-length-sample the control polyline into
    `max(4, ceil(total_length * 8))` segments, extrude vertically by
    `height`, build a triangle-strip ribbon.
  Add it under `node_name=frame.id`.

**Output:** a single `trimesh.Scene` whose scene graph names each mesh
by id.

---

## 17. Publish the GLB

**Location:** `server/app/pipeline/step17_publish_glb.py` (`publish` at
`:35`, `write_glb` at `:25`, `glb_path_for` at `:20`).

- Path: `runs/{run_id}/scene.glb`.
- `scene.export(file_type="glb")` → bytes → write to disk. Y-up,
  right-handed, meters — matches glTF 2.0.

**Output:** `(path, "/glb/{run_id}")`. The URL propagates back up through
`orchestrator.run` → `execute_run`, is written into `run.json` as
`glb_url`, and is emitted on `RunCompleted`.

---

## 18. Serve (independent of the pipeline task)

**Location:** `server/app/api/step18_serve.py` (handlers at `:32`, `:40`,
`:56`, `:84`).

- `GET /glb/{run_id}` resolves `glb_path_for(run_id)` and serves the
  file with `media_type=model/gltf-binary`.
- `GET /runs/{run_id}` returns the `run.json` summary
  (pending/running/completed/failed + timings + retry counts + `glb_url`).
- `DELETE /runs/{run_id}` cancels an in-flight run (idempotent on
  already-terminal runs).
- `GET /runs/{run_id}/events` streams the JSONL event log over SSE: live
  if the task is still running, replay from disk otherwise.

---

# Data flow summary

```
prompt: str
  → [1]  POST /generate              → run_id + detached task
  → [2]  execute_run                 → activates RunContext
  → [3]  orchestrator                → runs stages below
         │
         ├─ [4]  overall bbox (root only)            → root SubsceneNode
         │
         ├─ per node (starting at root):
         │     [5]  frame decider                    → node.frames
         │     [6]  scene breakdown + bbox validator → is_atomic | [subscene specs]
         │       ├─ if atomic: stop recursion
         │       └─ else:
         │            [7]  write plan entries
         │            [8]  recurse into each child → [5]
         │
         ├─ [9]  collect leaves + ancestor frame chains
         │
         ├─ per leaf:
         │     [10] anchor objects + graph validator → objects, rels, topo order
         │     [11] object bboxes (initial) + validator → {id: bbox}
         │     [12] mesh generation                  → {id: raw Trimesh}
         │     [13] rescale                          → {id: placed Trimesh}
         │     [14] completion loop:
         │            propose 1 object + rels
         │            re-validate graph (break on invalid)
         │            → [11] incremental + [12] + [13]
         │            repeat until stop
         │     [15] write realized entry
         │     next leaf → [10]
         │
         ├─ [16] assemble scene (meshes + frames)    → trimesh.Scene
         └─ [17] publish GLB                         → runs/{run_id}/scene.glb + "/glb/{run_id}"
                 emit RunCompleted(glb_url)

  [18] later: GET /glb/{run_id} serves the file
```
