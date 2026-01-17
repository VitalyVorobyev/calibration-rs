# Project Audit (calibration-rs)

Date: 2026-01-17

This document captures **design + implementation observations** from a first-pass read-through of
the workspace, with a focus on the `calib` facade, `calib-pipeline` session/problem integration,
and the upcoming multi-camera intrinsics + rig extrinsics pipeline work.

Scope of code skimmed (non-exhaustive):
- `crates/calib-pipeline/src/session/problem_types.rs`
- `crates/calib-pipeline/src/session/mod.rs`
- `crates/calib-pipeline/src/lib.rs`, `crates/calib-pipeline/src/helpers.rs`, `crates/calib-pipeline/src/handeye_single.rs`
- `crates/calib/src/lib.rs`, `crates/calib-cli/src/main.rs`
- `crates/calib-linear/src/extrinsics.rs`, `crates/calib-linear/src/planar_pose.rs`
- `crates/calib-optim/src/problems/rig_extrinsics.rs`, `crates/calib-optim/src/factors/reprojection_model.rs`

---

## 1) Architecture snapshot (what exists today)

- **Layering matches the workspace rules**: `calib-core` → (`calib-linear`, `calib-optim`) → `calib-pipeline` → (`calib`, `calib-cli`).
- `calib-pipeline` provides:
  - a **session framework** (`CalibrationSession<P>`, `ProblemType`) with JSON checkpointing (`to_json`/`from_json`),
  - an imperative planar intrinsics pipeline (and helpers),
  - a stepwise single-camera hand-eye pipeline,
  - session-friendly problem types for planar intrinsics, hand-eye single camera, and rig extrinsics.
- `calib` facade currently exposes:
  - session API + some re-exported session problem types (planar intrinsics + hand-eye single),
  - full re-exports of lower crates via `calib::core`, `calib::linear`, `calib::optim`, plus `calib::pipeline`.

---

## 2) Main issues / risks observed

### A. `problem_types.rs` is a “mega-module” (high complexity / mixed responsibilities)

`crates/calib-pipeline/src/session/problem_types.rs` is ~1400 LOC and currently mixes:
- problem type trait impls (initialize/optimize),
- data model types for observations/initial/results,
- solver option defaults,
- serialization workarounds,
- helper functions (e.g. reprojection error computation),
- a large `#[cfg(test)]` block for multiple problems.

Practical impact:
- hard to navigate and extend (especially for upcoming multi-camera pipelines),
- encourages copy/paste (already visible for reprojection error),
- makes it harder to reason about invariants (pose conventions, gauge fixes, etc.).

### B. Session trait bounds force awkward serialization hacks

`ProblemType` requires `InitOptions` and `OptimOptions` to be `Serialize + Deserialize`, but
`CalibrationSession<P>` does **not store options**. This has led to manual “empty struct”
serialization implementations that discard fields and round-trip to defaults, e.g.:
- `HandEyeSingleInitOptions`, `HandEyeSingleOptimOptions`
- `RigExtrinsicsOptimOptions`

Practical impact:
- misleading API surface (“options are serializable” but are silently dropped),
- makes checkpoint JSON appear reproducible while actually losing configuration,
- increases maintenance burden and friction when adding new options.

### C. Coordinate convention naming is inconsistent / ambiguous in rig code

Across rig extrinsics initialization and optimization, names and docs like:
- `cam_to_rig`, `rig_to_target`
- “camera-to-rig”, “rig-to-target”
are used, but in several places the *applied math* corresponds to the opposite direction.

One concrete example: the rig reprojection factor uses a pose applied to a target/world point to
produce a rig-space point, i.e. **target → rig**, even though comments say “rig-to-target”.

Practical impact:
- high risk of subtle sign/inversion bugs,
- difficult for users to integrate correctly (and for maintainers to extend),
- tests can pass while semantics are wrong (cost improves, but transforms might be inverted).

Recommendation: adopt and enforce a single convention everywhere (e.g. `T_dst_src` or `dst_from_src`),
and reflect it in identifiers and docstrings.

### D. Duplication of fundamental utilities (example: reprojection error)

`mean_reproj_error_planar` exists in at least two places:
- `crates/calib-pipeline/src/session/problem_types.rs`
- `crates/calib-pipeline/src/handeye_single.rs`

They appear logically identical (same camera model instantiation + loop). Similar duplication exists
around planar pose recovery/homographies in different layers.

Practical impact:
- fixes/improvements (numerics, weighting, filtering invalid projections) must be made multiple times,
- higher chance of inconsistent metrics between pipelines and session problems.

### E. Rig extrinsics pipeline is present but not yet a “first-class” high-level API

`RigExtrinsicsProblem` already exists as a session problem type, and `calib-optim` contains the BA
problem. However:
- there is no high-level pipeline function in `calib-pipeline` analogous to `run_planar_intrinsics`,
- there is no `calib` facade re-export of rig extrinsics session types (unlike planar + hand-eye),
- there is no `crates/calib/examples/*` example demonstrating rig extrinsics usage,
- CLI currently only supports planar intrinsics (`crates/calib-cli/src/main.rs`).

### F. Rig initialization quality likely insufficient for real distorted data

`RigExtrinsicsProblem::initialize` currently:
- uses Zhang intrinsics per camera (no distortion),
- seeds distortion as zero and relies on joint BA to recover it,
- estimates planar poses from homographies.

This is fine for synthetic no-distortion tests, but for real datasets it increases risk of:
- poor initialization, slower/non-convergence,
- drift in extrinsics because intrinsics are biased by distortion.

The workspace already contains a more robust per-camera planar init path (iterative intrinsics +
distortion in `calib-linear`, and `planar_init_seed_from_views` in `calib-pipeline`).

### G. Documentation drift (examples don’t match current APIs)

`crates/calib-pipeline/src/functions.md` contains rig extrinsics snippets that reference option
fields that do not exist in the current `calib-optim::problems::rig_extrinsics::RigExtrinsicsSolveOptions`
(e.g. `fix_reference_camera`, `fix_shared_intrinsics`). This suggests docs have diverged from code.

Practical impact:
- users copy/paste examples that don’t compile,
- extra support burden.

### H. Tests are valuable but in a few places too permissive to validate semantics

Rig extrinsics synthetic tests assert:
- convergence (final cost threshold),
- a loose translation error bound for one camera.

But they do not strongly validate:
- transform direction conventions (rig↔camera, rig↔target),
- rotation correctness,
- consistency of predicted reprojections using the reported outputs.

This can let a “mostly working” pipeline hide convention inversions.

---

## 3) Confirmed decisions (audit review)

1. **Rig camera model scope**: all cameras in a rig share the same model family (initially: pinhole + Brown-Conrady5 + identity sensor),
   but parameters (K/distortion) are per-camera.
2. **Session checkpointing**: session JSON checkpoints must capture the init/optim options used (avoid “empty-serde” workarounds that
   silently drop configuration).
3. **Pose conventions**:
   - `T_C_T`: target → camera
   - `T_R_C`: camera → rig
   - rig frame is defined by `ref_cam_idx` (reference camera)
4. **Rig initialization default**: quality-first; run per-camera iterative intrinsics + distortion initialization before rig extrinsics.
5. **Missing observations**: must be supported (some cameras may be absent in some views).
