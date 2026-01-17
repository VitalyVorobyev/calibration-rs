# Implementation Plan: Problem Types Refactor + Multi-Camera Rig Pipeline

Date: 2026-01-17

This plan follows from the observations in `docs/PROJECT_AUDIT.md`.

Primary goals:
1. Refactor `crates/calib-pipeline/src/session/problem_types.rs` into a maintainable module layout.
2. Make multi-camera intrinsics + rig extrinsics a first-class pipeline in `calib-pipeline`, surfaced cleanly via the `calib` facade.
3. De-risk coordinate convention confusion with explicit naming/docs + stronger tests.

Non-goals (for this iteration):
- adding new camera model families (e.g. Scheimpflug/linescan) into rig BA,
- expanding CLI beyond a minimal rig command (can be a later step).

---

## 0) Confirmed decisions (2026-01-17)

1. **Rig camera model scope**: all cameras share the same model family; parameters (K/distortion) are per-camera.
2. **Pose naming convention**: use `T_dst_src` in docs and `dst_from_src` in identifiers where feasible (`T_C_T`, `T_R_C`, etc).
3. **Session checkpointing**: session JSON checkpoints must capture init/optim options (avoid “empty-serde” workarounds that drop config).
4. **Rig gauge definition**: rig frame defined by a reference camera (`ref_cam_idx`); fix that camera’s extrinsics and fix (at least) one rig pose.
5. **Rig initialization default**: quality-first; per-camera iterative intrinsics + distortion initialization, and support missing observations.

## 0.1 Status (already implemented)

- `problem_types.rs` split into `crates/calib-pipeline/src/session/problem_types/`.
- Session checkpoints record init/optim options; option types derive `serde` in `calib-linear`/`calib-optim`.
- Rig extrinsics conventions standardized to `rig_from_target` (target → rig).
- `calib-pipeline` exposes `run_rig_extrinsics`; `calib` facade re-exports rig session + pipeline APIs.
- Example added: `crates/calib/examples/rig_extrinsics_session.rs`.

---

## 1) Refactor `problem_types.rs` into a module folder

### 1.1 Target module structure

Create `crates/calib-pipeline/src/session/problem_types/`:
- `mod.rs`
  - re-export public problem types + shared types
  - contains only lightweight glue and shared helper module declarations
- `planar_intrinsics.rs`
  - `PlanarIntrinsicsProblem`, observations/init/results, options
  - planar reprojection error helper (or shared helper use)
  - planar-specific tests
- `handeye_single.rs`
  - `HandEyeSingleProblem`, observations/init/results, options
  - hand-eye defaults helper
  - hand-eye-specific tests
- `rig_extrinsics.rs`
  - `RigExtrinsicsProblem`, observations/init/results, options
  - rig initialization + dataset conversion
  - rig-specific tests (strengthened; see §3)
- `common.rs` (optional)
  - shared small utilities (e.g., `mean_reproj_error_planar`)
  - shared synthetic generators if tests need them

Keep `crates/calib-pipeline/src/session/mod.rs` exposing `pub mod problem_types;` unchanged, so
external module paths remain stable (modulo items moved internally).

### 1.2 Reduce mixed responsibility inside problem modules

Move out of “problem type” modules:
- heavy metric helpers that are used by both session problems and imperative pipelines,
- dataset conversions that should be shared with a non-session pipeline API.

Preferred direction:
- “problem type modules” contain:
  - data types + `ProblemType` impl,
  - thin glue calling into reusable pipeline components.

---

## 2) Make rig extrinsics a first-class pipeline API in `calib-pipeline`

### 2.1 Public pipeline API surface (proposed)

Add `crates/calib-pipeline/src/rig_extrinsics.rs` (or `rig/mod.rs`) with:
- Input data types (serde):
  - reuse session observation structs if they are already appropriate, or
  - define “pipeline-level” types and have the session observation type be a re-export/alias.
- Config (serde):
  - `RigExtrinsicsConfig` (init + BA knobs), using only serde-friendly primitives/enums.
  - include robust loss config as a `RobustLossConfig`-style enum (like planar intrinsics).
- Report/result (serde):
  - `RigExtrinsicsReport` with:
    - `cameras: Vec<CameraParams>`
    - `cam_to_rig: Vec<Iso3>` (camera → rig; rig frame defined by `ref_cam_idx`)
    - `rig_from_target: Vec<Iso3>` per view (target → rig)
    - `final_cost`, optional summary metrics (mean reproj error per camera/view)

Top-level functions:
- `initialize_rig_extrinsics(input, init_cfg) -> RigExtrinsicsInit`
- `optimize_rig_extrinsics(input, init, optim_cfg) -> RigExtrinsicsResult`
- `run_rig_extrinsics(input, cfg) -> RigExtrinsicsReport` (thin convenience wrapper)

This should mirror the planar intrinsics pipeline pattern (`run_planar_intrinsics`) and keep the
session API as an alternative “workflow manager”.

### 2.2 Initialization improvements (use existing low-level components)

Replace/extend the current rig init (Zhang-only, zero distortion) with a per-camera init that can
seed distortion when possible:
- For each camera:
  - extract all views where that camera observed the target,
  - run `planar_init_seed_from_views` (or equivalent extracted helper) to get:
    - `FxFyCxCySkew`, `BrownConrady5`, and `poses` for those views.
- Convert per-camera per-view planar poses into the format required by
  `calib_linear::extrinsics::estimate_extrinsics_from_cam_target_poses`.

Notes:
- handle missing observations (`Option<...>`) carefully: extrinsics init requires overlap graph
  connectivity w.r.t. the reference camera (already checked by calib-linear).
- decide if weights should influence init (likely not in v1).

### 2.3 Optimization configuration surface

Current approach:
- Reuse `calib_optim::problems::rig_extrinsics::RigExtrinsicsSolveOptions` and
  `calib_optim::backend::BackendSolveOptions` directly in serde-facing configs and in session
  checkpoints (these types now derive `Serialize`/`Deserialize` intentionally).

Longer-term (optional):
- If the `calib` facade needs stronger API compatibility guarantees than “whatever calib-optim has
  today”, add stable wrapper config types at the `calib` boundary and map them into `calib-optim`
  options internally.

---

## 3) Fix/lock down pose conventions (docs + tests)

### 3.1 Documentation pass

Audit and correct docstrings/comments for:
- rig extrinsics init (`crates/calib-linear/src/extrinsics.rs`)
- rig extrinsics BA factor docs (`crates/calib-optim/src/factors/reprojection_model.rs`)
- rig extrinsics pipeline/session types (`crates/calib-pipeline/...`)

Goal: a user can take the returned transforms and confidently use them to project a known 3D point.

### 3.2 Strengthen rig synthetic tests to validate semantics (not just cost)

Add assertions that disambiguate transform direction:
- Use the optimized outputs to reproject target points and verify pixel error < ε (per camera).
- Validate both translation **and rotation**:
  - compare `cam_from_rig` (or `rig_from_cam`) against ground truth with a tight tolerance,
  - if gauge is defined by reference camera, ensure ref camera transform is (close to) identity *and fixed as such in solve options*.

Also add tests for:
- missing camera observations in some views (sparse `Option`s),
- overlap connectivity errors (camera with no overlap with reference),
- JSON roundtrip of rig pipeline input/config/report.

---

## 4) Integrate into the `calib` facade (high-level API)

### 4.1 Session API exports

Expose rig extrinsics session problem types via `calib::session` similarly to planar + hand-eye:
- `RigExtrinsicsProblem`
- `RigExtrinsicsObservations`
- `RigExtrinsicsInitOptions`
- `RigExtrinsicsOptimOptions` (or a renamed/stabilized options type if §0.2 changes trait bounds)

### 4.2 Pipeline API exports

Expose the new imperative rig pipeline via `calib::pipeline`:
- `run_rig_extrinsics`, `RigExtrinsicsInput`, `RigExtrinsicsConfig`, `RigExtrinsicsReport`

Update `calib::prelude` to include the rig pipeline types that are expected to be commonly used.

---

## 5) Cleanup tasks that should be done alongside

- Update `crates/calib-pipeline/src/functions.md` to match current code (it is currently stale for rig).
- Add at least one example:
  - `crates/calib/examples/rig_extrinsics_session.rs` (checkpoint-friendly)
  - optionally: `crates/calib/examples/rig_extrinsics.rs` (imperative pipeline)
- Consider extracting shared metric helpers (e.g. `mean_reproj_error_planar`) into a single module.

---

## Suggested execution order (low-risk → higher impact)

1. Split `problem_types.rs` into modules without behavior changes (pure refactor + tests still pass).
2. Remove/adjust `ProblemType` option serde bounds (if decision §0.2 is accepted) and delete “empty-serde” hacks.
3. Add a dedicated rig extrinsics pipeline module and example using the existing session implementation.
4. Improve rig initialization to seed distortion (reuse existing planar init utilities).
5. Tighten rig tests + docs for pose conventions.
6. Update `calib` facade exports + README/functions docs.
