# Pre-Release Review — calibration-rs ([Unreleased] → v0.4.0)

*Reviewed: 2026-04-29*
*Scope: full workspace; focused on additions since v0.3.0 (commit `892e020`).*
*Reviewer: Architect (Opus 4.7, 1M context)*

> The prior 2026-04-12 review cycle (R-01..R-13) closed cleanly and shipped
> v0.3.0; its full text remains in git history (file before this commit).
> This document tracks the next cycle.

---

## Context

v0.3.0 shipped on 2026-04-12 and closed all 11 in-scope findings of the prior
cycle (R-01..R-13). Since then, the `[Unreleased]` section of `CHANGELOG.md`
has accumulated a substantial new feature surface that has never been audited:

- **Scheimpflug rig family** (3 new optim functions):
  `optimize_rig_extrinsics_scheimpflug`, `optimize_handeye_scheimpflug`,
  `optimize_rig_laserline`.
- **Three new session-API pipelines:** `rig_scheimpflug_extrinsics` (4 steps),
  `rig_scheimpflug_handeye` (6 steps, EyeInHand), `rig_laserline_device`
  (2 steps, consumes a frozen `RigScheimpflugHandeyeExport`).
- **Facade helper** `pixel_to_gripper_point(cam_idx, pixel, rig_cal,
  laser_planes_rig, base_se3_gripper)` and
  `LaserPlane::transform_by(&Iso3)`.
- **New IR factor kinds** `ReprojPointPinhole4Dist5Scheimpflug2{TwoSE3,
  HandEye, HandEyeRobotDelta}` with TinySolver adapters.
- **New private example crate** `vision-calibration-examples-private`
  (`publish = false`, intentionally outside `[workspace.members]` per commit
  `31e8299`).

**Goal of this review:** confirm the new surface is release-ready as v0.4.0
(additive only, no breaking changes detected in the diff), with particular
attention to Python binding parity, test coverage, and documentation.

---

## Executive Summary

Architecture, error handling, and workspace hygiene are healthy. All R-01..R-13
fixes from the prior cycle remain in place; new code follows the typed-error
contract, marks public types `#[non_exhaustive]`, ships module-level `//!`
docstrings, and re-exports cleanly from the facade. CHANGELOG `[Unreleased]`
accurately matches the actual public-API additions.

**The single critical blocker is Python binding parity.** Four new Rust facade
APIs (the three Scheimpflug rig pipelines and the `pixel_to_gripper_point`
helper) are completely unreachable from Python. `crates/vision-calibration-py/
src/lib.rs` registers exactly seven `#[pyfunction]` items, none of which cover
the new surface, and `__init__.py`/`__init__.pyi` mirror that gap. Releasing
v0.4.0 in this state would silently regress the binding-parity contract that
0.3.0 explicitly established (R-12, `scripts/check_pyi_coverage.py`).

Beyond the bindings, two clusters of P2 findings need attention before tagging:
test coverage (the new pipelines and the gripper helper rely solely on the
private dataset example for validation; CI cannot run that) and minor
documentation gaps (README and one terse struct doc). Everything else is P3
polish.

No P0 issues. No regressions in any prior R-NN finding.

---

## Findings

Severity legend: **P0** correctness/security blocker · **P1** fix before
release · **P2** fix soon · **P3** polish.

### B-01 `run_rig_scheimpflug_extrinsics` is not exposed in Python

- **Severity:** P1
- **Category:** contracts (binding parity)
- **Location:** `crates/vision-calibration-py/src/lib.rs:246-256` (the
  `#[pymodule]` block) — function never declared.
- **Status:** done
- **Resolution:** Added `#[pyfunction] fn run_rig_scheimpflug_extrinsics` in `lib.rs`, registered
  in `#[pymodule]`. Added `RigScheimpflugExtrinsicsDataset` / `RigScheimpflugExtrinsicsCalibrationConfig` /
  `RigScheimpflugExtrinsicsResult` to `models.py`, raw + typed helpers to `_api.py`, imports and
  `__all__` entry in `__init__.py`, typed signature in `__init__.pyi`. pyi coverage check passes.
- **Problem:** The Rust facade exports
  `vision_calibration::rig_scheimpflug_extrinsics::run_calibration`
  (re-exported at `crates/vision-calibration/src/lib.rs:348-356`), but no
  `#[pyfunction]` wraps it. Python users of v0.4.0 cannot run the new
  Scheimpflug rig extrinsics workflow at all. This silently violates the
  binding-parity contract that R-12 was designed to enforce; `scripts/
  check_pyi_coverage.py` checks `__init__.pyi` against `lib.rs`'s
  `#[pymodule]` registrations, so a *missing* `#[pyfunction]` slips past it
  because the `.pyi` is also missing the symbol.
- **Fix:** In `crates/vision-calibration-py/src/lib.rs`, add a thin wrapper
  modelled on `run_rig_extrinsics` (lib.rs:145-157):

  ```rust
  #[pyfunction(signature = (input, config=None))]
  fn run_rig_scheimpflug_extrinsics(
      py: Python<'_>,
      input: &Bound<'_, PyAny>,
      config: Option<&Bound<'_, PyAny>>,
  ) -> PyResult<Py<PyAny>> {
      run_problem::<vision_calibration::rig_scheimpflug_extrinsics::
          RigScheimpflugExtrinsicsProblem, _>(
          py, input, config,
          vision_calibration::rig_scheimpflug_extrinsics::run_calibration,
      )
  }
  ```

  Register it in the `#[pymodule]` block alongside the existing
  `m.add_function(...)` calls. Add the corresponding signature to
  `__init__.pyi` (using the `RigScheimpflugExtrinsicsDataset`/`Config`/
  `Result` typed dataclass models), and add the symbol to both the import
  block and `__all__` in `__init__.py`. Add `_api.py` raw + typed helpers
  parallel to `run_rig_extrinsics`.

### B-02 `run_rig_scheimpflug_handeye` is not exposed in Python

- **Severity:** P1
- **Category:** contracts (binding parity)
- **Location:** `crates/vision-calibration-py/src/lib.rs` (missing).
- **Status:** done
- **Resolution:** Added `#[pyfunction] fn run_rig_scheimpflug_handeye` in `lib.rs`, registered in
  `#[pymodule]`. Added `RigScheimpflugHandeyeDataset` / `RigScheimpflugHandeyeCalibrationConfig` /
  `RigScheimpflugHandeyeResult` (+ sub-configs) to `models.py`, raw + typed helpers to `_api.py`,
  imports and `__all__` entries in `__init__.py`, typed signature in `__init__.pyi`.
- **Problem:** Same shape as B-01.
  `vision_calibration::rig_scheimpflug_handeye::run_calibration` exists at
  `crates/vision-calibration-pipeline/src/rig_scheimpflug_handeye/steps.rs:916`
  and is re-exported at `crates/vision-calibration/src/lib.rs:362-373`, but no
  Python wrapper ships. EyeInHand is the only mode this pipeline supports; the
  Python wrapper does not need to model `EyeToHand` for it.
- **Fix:** Mirror B-01, wrapping `RigScheimpflugHandeyeProblem`. The Python
  typed dataclass models for the input/config/export should follow the
  existing `RigHandeyeDataset`/`RigHandeyeCalibrationConfig`/`RigHandeyeResult`
  pattern in `crates/vision-calibration-py/python/vision_calibration/models.py`.

### B-03 `run_rig_laserline_device` is not exposed in Python

- **Severity:** P1
- **Category:** contracts (binding parity)
- **Location:** `crates/vision-calibration-py/src/lib.rs` (missing).
- **Status:** done
- **Resolution:** Added `#[pyfunction] fn run_rig_laserline_device` in `lib.rs`, registered in
  `#[pymodule]`. Added `RigLaserlineView` / `RigLaserlineDataset` / `RigLaserlineDeviceInput` /
  `RigLaserlineUpstreamCalibration` / `RigLaserlineDeviceCalibrationConfig` /
  `RigLaserlineDeviceResult` to `models.py`, raw + typed helpers to `_api.py`, imports and
  `__all__` entries in `__init__.py`, typed signature in `__init__.pyi`.
- **Problem:** Same shape. Pipeline at `crates/vision-calibration-pipeline/src/
  rig_laserline_device/steps.rs:177`, facade re-export at
  `crates/vision-calibration/src/lib.rs:379-385`. The pipeline consumes a
  frozen `RigUpstreamCalibration` (also re-exported), so the Python typed
  dataclass models must include that input shape.
- **Fix:** Mirror B-01 with `RigLaserlineDeviceProblem`. The fact that the
  pipeline takes an upstream-calibration payload is already implicit in
  `RigLaserlineDeviceInput`, so the Python signature can stay
  `(input, config=None)`.

### B-04 `pixel_to_gripper_point` is not exposed in Python

- **Severity:** P1
- **Category:** contracts (binding parity)
- **Location:** `crates/vision-calibration-py/src/lib.rs` (missing).
  Rust definition at `crates/vision-calibration/src/lib.rs:412-510`.
- **Status:** done
- **Resolution:** Added bespoke `#[pyfunction] fn pixel_to_gripper_point` in `lib.rs` with
  5-argument signature; validates inputs via `reject_non_finite`; maps `InvalidInput` errors to
  `PyValueError` and math failures to `PyRuntimeError`. Added typed `pixel_to_gripper_point` helper
  to `_api.py`, exported via `__init__.py` / `__init__.pyi`. pyi coverage check passes.
- **Problem:** This helper composes the four-step laser-pixel-to-gripper-point
  pipeline (undistort → rig-frame ray → plane intersection → hand-eye
  transform), which is the *primary* downstream user-facing operation of the
  whole Scheimpflug rig laser stack. It is documented and exercised in the
  private example `puzzle_130x130_rig.rs:550`, but Python consumers cannot
  call it. Without it, even with B-01..B-03 fixed the Python user has only
  the calibration data — they still need to reimplement the projection math
  in Python to use it.
- **Fix:** Unlike B-01..B-03, this is not a `run_problem` wrapper. Add a
  bespoke `#[pyfunction]` whose signature accepts:
  - `cam_idx: usize`,
  - `pixel: &Bound<'_, PyAny>` (deserialized via `parse_payload` to
    `vision_calibration_core::Pt2`, i.e. `[f64; 2]`),
  - `rig_cal: &Bound<'_, PyAny>` (deserialized to
    `RigScheimpflugHandeyeExport`),
  - `laser_planes_rig: &Bound<'_, PyAny>` (deserialized to
    `Vec<LaserPlane>`),
  - `base_se3_gripper: Option<&Bound<'_, PyAny>>` (deserialized to
    `Option<Iso3>`).

  Run `reject_non_finite` on each numeric payload, call the facade function,
  and serialize the returned `Pt3` back via `pythonize`. Register, declare in
  `__init__.pyi`, export in `__init__.py`. Optionally also expose
  `LaserPlane::transform_by` as a method on the Python `LaserPlane` model —
  but that's a P3 nice-to-have.

### T-01 No pipeline-level integration tests for the three new sessions

- **Severity:** P2
- **Category:** tests
- **Location:** `crates/vision-calibration-pipeline/tests/` contains only
  `json_contract_traits.rs` and `laserline_device.rs` (single-camera). No
  test exercises `rig_scheimpflug_extrinsics`, `rig_scheimpflug_handeye`, or
  `rig_laserline_device`.
- **Status:** done
- **Resolution:** Added three integration test files:
  `tests/rig_scheimpflug_extrinsics.rs` (3 tests — convergence, rejection, JSON round-trip),
  `tests/rig_scheimpflug_handeye.rs` (2 tests — convergence, rejection),
  `tests/rig_laserline_device.rs` (2 tests — convergence, config JSON round-trip). All pass.
- **Problem:** The three new session pipelines are validated only through the
  private dataset example (`puzzle_130x130_rig.rs`), which CI does not (and
  cannot) run because the dataset is not in the public repo. A regression in
  any of these pipelines will not be caught by `cargo test --workspace`. The
  optim crate has direct tests for two of the three underlying functions
  (`tests/handeye_scheimpflug.rs:143`,
  `tests/rig_extrinsics_scheimpflug.rs:176`), which protects the math but
  not the session-step orchestration (config validation, schema versioning,
  state transitions, export round-trip).
- **Fix:** Add three integration tests that mirror the structure of
  `crates/vision-calibration-pipeline/tests/laserline_device.rs`:
  - `tests/rig_scheimpflug_extrinsics.rs` — synthetic 2-camera rig with
    Scheimpflug tilt, ~3 views, run all 4 steps, assert convergence within
    ~5%, plus a JSON export → import round-trip.
  - `tests/rig_scheimpflug_handeye.rs` — synthetic EyeInHand rig with
    Scheimpflug, run all 6 steps, assert intrinsics + handeye recovery.
  - `tests/rig_laserline_device.rs` — synthetic upstream rig calibration +
    laser planes, run both steps, assert per-camera plane recovery within
    tight tolerance.
  Use existing synthetic generators in `vision_calibration_core::synthetic`;
  if those don't cover Scheimpflug yet, add helpers there rather than
  duplicating in tests.

### T-02 `optimize_rig_laserline` has no synthetic-ground-truth test

- **Severity:** P2
- **Category:** tests
- **Location:** `crates/vision-calibration-optim/tests/` — `handeye_scheimpflug.rs`
  and `rig_extrinsics_scheimpflug.rs` cover their respective functions, but
  there is no `rig_laserline.rs` (existing `laserline_bundle.rs` covers the
  single-camera bundle, not the rig-level joint solve).
- **Status:** done
- **Resolution:** Added `crates/vision-calibration-optim/tests/rig_laserline.rs` with a
  2-camera synthetic GT test. Asserts normal angle <0.5°, distance abs error <0.01,
  reproj RMS <0.5 px. Test passes.
- **Problem:** Of the three new optim entry points, `optimize_rig_laserline`
  is the only one without a direct test. A future refactor of the
  rig-laserline residuals or the upstream-calibration adapter could break it
  silently.
- **Fix:** Add `crates/vision-calibration-optim/tests/rig_laserline.rs`:
  - Synthesize a 2-camera rig with known intrinsics + extrinsics + per-camera
    laser planes (in camera frame).
  - Project laser-line points into each camera and create
    `RigLaserlineDataset` views.
  - Provide a frozen upstream calibration (slightly perturbed extrinsics).
  - Call `optimize_rig_laserline` with initial plane guesses (~10% perturbed).
  - Assert per-camera recovered plane normal angle < 0.5°, distance < 1%,
    per-camera reprojection RMS < 0.5 px.

### T-03 `pixel_to_gripper_point` has no unit test

- **Severity:** P2
- **Category:** tests
- **Location:** `crates/vision-calibration/tests/` — `facade_compile_surface.rs`
  and `scheimpflug_intrinsics.rs` only; no test covering the new helper.
- **Status:** todo
- **Problem:** The helper composes four error-prone geometric steps and has
  multiple documented failure paths (`cam_idx` out of range, ray-plane miss,
  undistortion failure, missing `base_se3_gripper` in `EyeToHand`). Only the
  private example exercises it, and only the happy path. Once it is exposed
  in Python (B-04) it becomes part of the supported public API and needs a
  regression test.
- **Fix:** Add `crates/vision-calibration/tests/pixel_to_gripper_point.rs`
  with at least four cases:
  1. Happy path: synthesize a known rig + plane + gripper pose, project a
     known 3D point onto the camera, then verify
     `pixel_to_gripper_point` recovers the point within 1e-9 m.
  2. `cam_idx >= num_cameras` returns `Err(InvalidInput)` (or whichever
     typed variant — verify against the actual implementation).
  3. Ray parallel to laser plane returns the appropriate intersection error.
  4. `EyeToHand` mode with `base_se3_gripper = None` returns the
     missing-pose error.

### D-01 README does not advertise the new Scheimpflug rig family

- **Severity:** P2
- **Category:** docs
- **Location:** `/README.md` — current text mentions perspective + Scheimpflug
  cameras and rigs, but not the rig-level Scheimpflug pipelines or
  `pixel_to_gripper_point`.
- **Status:** todo
- **Problem:** The single biggest user-visible feature of the upcoming
  release is unstated on the front page. CHANGELOG has the detail, but
  README is the discovery surface for new users and crates.io browsers.
- **Fix:** A 3-5 line addition near the existing high-level summary listing
  the new pipelines (`rig_scheimpflug_extrinsics`, `rig_scheimpflug_handeye`,
  `rig_laserline_device`) and the `pixel_to_gripper_point` helper, with a
  one-line description and a link to the facade rustdoc.

### D-02 `RigLaserlineDataset` doc does not specify per-camera indexing convention

- **Severity:** P2
- **Category:** docs
- **Location:** `crates/vision-calibration-optim/src/problems/laserline_rig_bundle.rs`
  (struct definition near the top of the module).
- **Status:** todo
- **Problem:** The new `RigLaserlineDataset` carries per-view, per-camera
  observations, but the field-level doc does not state whether a camera that
  saw nothing in a given view is encoded as `None`, an empty observation
  list, or omitted from the view entirely. Without this, the integration
  test for T-01/T-02 has to read the source to figure out the contract,
  and Python users (post B-03) will hit confusing errors at the boundary.
- **Fix:** Add an `# Example` section to the struct doc and a one-liner on
  each field explaining the missing-observation convention. Add a short
  panel to the module-level `//!` doc summarising the input shape.

### C-01 No helper to derive `RigUpstreamCalibration` from `RigScheimpflugHandeyeExport`

- **Severity:** P3
- **Category:** api
- **Location:** Implicit; consumers chaining
  `rig_scheimpflug_handeye → rig_laserline_device` must hand-construct
  `RigUpstreamCalibration` field-by-field.
- **Status:** done
- **Resolution:** Added `impl From<&RigScheimpflugHandeyeExport> for RigUpstreamCalibration` in
  `crates/vision-calibration-pipeline/src/rig_laserline_device/problem.rs`. Includes a rustdoc
  example showing the `.into()` conversion. Re-exported through the facade via the existing
  `RigUpstreamCalibration` re-export.
- **Problem:** The example `puzzle_130x130_rig.rs` shows the canonical
  pattern: take the export from the rig handeye stage and feed it to the
  laserline-device pipeline. Without a helper, every user reimplements the
  field-by-field copy, and any field added to `RigScheimpflugHandeyeExport`
  in future will require coordinated updates everywhere.
- **Fix:** Add either
  - `impl From<&RigScheimpflugHandeyeExport> for RigUpstreamCalibration`,
    or
  - a method `RigScheimpflugHandeyeExport::to_upstream_calibration(&self)
    -> RigUpstreamCalibration`.
  Place it in the pipeline crate so it sees both types. Re-export through
  the facade. Add a short rustdoc example showing the chain.

### C-02 `target_poses.first().copied()` semantics are non-obvious in handeye export

- **Severity:** P3
- **Category:** code-quality
- **Location:** `crates/vision-calibration-pipeline/src/rig_scheimpflug_handeye/problem.rs:359-388`
  (specifically the `target_poses.first().copied()` line in the export
  builder).
- **Status:** todo
- **Problem:** `HandEyeScheimpflugEstimate::target_poses` is a `Vec<Iso3>`
  with one entry per view. The export uses `.first().copied()`, which is
  correct (the calibration target is fixed; all view-poses are equivalent
  reconstructions of the same rigid transform), but the bare expression
  invites a future maintainer to "fix" it into a per-camera or per-view
  loop and silently break the export contract.
- **Fix:** Either inline a one-line comment explaining the convention
  (`// All views observe the same fixed target; first() is canonical.`),
  or — cleaner — add an `Estimate::canonical_target_pose()` method that
  returns `Option<Iso3>` and document the rule once.

---

## Strong Points

- **Workspace structure:** the new private examples crate
  (`vision-calibration-examples-private`) is correctly placed outside
  `[workspace.members]` (commit `31e8299`) with `publish = false`. The six
  published crates remain in the layered `core ← {linear, optim} ←
  pipeline ← facade ← py` topology with no cycles.
- **Typed-error contract intact:** all new public functions return
  `Result<T, crate::Error>`. No regression to `anyhow::Result` in any
  public signature. R-01 holds.
- **`#[non_exhaustive]` discipline maintained:** every new pub struct/enum
  in `rig_scheimpflug_extrinsics`, `rig_scheimpflug_handeye`, and
  `rig_laserline_device` carries the attribute (verified by grep).
- **Module-level `//!` docs present** on all three new pipeline modules.
- **`# Errors` rustdoc sections** present on the new optim entry points and
  on `pixel_to_gripper_point` (`crates/vision-calibration/src/lib.rs:407-411`).
- **CHANGELOG accuracy:** every `[Unreleased]` bullet maps cleanly to a real
  public-API addition; no entries are stale and no public additions are
  unmentioned.
- **MSRV pins still in place** (`fixed = 1.30.0`, `kiddo = 5.2.4`); MSRV CI
  job runs with `--locked`; `docs/MSRV.md` warns before `cargo update`.
- **`RUSTSEC-2024-0436` ignore comment** still documented in the audit
  workflow (paste crate, proc-macro only, zero runtime exposure).
- **Facade re-export hygiene:** the three new pipelines and the helper are
  exposed under tidy `pub mod` blocks at
  `crates/vision-calibration/src/lib.rs:348-385`, with one-paragraph
  module-level docs each.
- **No new `_ => panic!` wildcards** in match statements (R-10 holds).
- **No `assert!(true)`, `#[ignore]`, or sleep-based** test smells in the new
  test files reviewed.

---

## Out-of-Scope Pointers

- **Numerical correctness of new Scheimpflug factor residuals**
  (`ReprojPointPinhole4Dist5Scheimpflug2{TwoSE3, HandEye, HandEyeRobotDelta}`)
  → delegate to the **`calibration-review`** skill. Recommend running it
  before tagging v0.4.0 since these residuals are the load-bearing math of
  three new pipelines.
- **Numerical robustness of `pixel_to_gripper_point`** — undistortion
  iteration, ray-plane intersection epsilon, frame conventions →
  **`calibration-review`**.
- **Performance of the joint
  intrinsics+extrinsics+handeye+Scheimpflug bundle** in
  `optimize_handeye_scheimpflug` and `optimize_rig_laserline` (high-DOF
  problems) → **`perf-architect`** if dataset-scale numbers are concerning.
- **R-08 (monolithic `ProblemIR::validate()`)** and **R-11 (property
  tests)** remain deferred to v1.1 per the prior cycle's owner decision.
  This review does not reopen them.

---

## Verification Plan

After Implementer applies the fixes, the Reviewer should run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test --workspace --all-features
cargo doc --workspace --no-deps           # zero warnings
python3 scripts/check_pyi_coverage.py --check    # must pass post-B-01..B-04
maturin develop -m crates/vision-calibration-py/Cargo.toml
python3 -c "import vision_calibration as vc; print(vc.run_rig_scheimpflug_extrinsics, vc.run_rig_scheimpflug_handeye, vc.run_rig_laserline_device, vc.pixel_to_gripper_point)"
python3 -m compileall crates/vision-calibration-py/python/vision_calibration
```

End-to-end smoke test (Python side, after wheel rebuild): run a synthetic
input through each of the three new `run_*` calls, assert it returns a typed
result without raising. Then call `pixel_to_gripper_point` with a known
synthetic geometry and assert the returned 3D point matches the expected
value within `1e-9`. The synthetic generators in
`vision_calibration_core::synthetic` already cover most of what's needed; a
test fixture mirroring the new optim-test setup is fine.

---

## Critical Files

Files the Implementer will need to read/edit, grouped by finding:

**B-01..B-04 (Python bindings):**
- `crates/vision-calibration-py/src/lib.rs` — add 4 `#[pyfunction]` items
  and register them in the `#[pymodule]` block (model after the existing
  `run_rig_extrinsics` etc., lines 145-211).
- `crates/vision-calibration-py/src/validation.rs` — already has
  `reject_non_finite` and `value_err` helpers (R-07 work).
- `crates/vision-calibration-py/python/vision_calibration/__init__.py` —
  imports + `__all__`.
- `crates/vision-calibration-py/python/vision_calibration/__init__.pyi` —
  signatures.
- `crates/vision-calibration-py/python/vision_calibration/_api.py` —
  raw + typed helpers (model after existing `run_rig_extrinsics`).
- `crates/vision-calibration-py/python/vision_calibration/models.py` — typed
  dataclasses for the new datasets / configs / results (mirror existing
  `RigHandeye*` shapes).
- `scripts/check_pyi_coverage.py` — should pass automatically once both
  layers are updated; will fail loudly if either is forgotten.

**T-01 (pipeline integration tests):**
- New: `crates/vision-calibration-pipeline/tests/rig_scheimpflug_extrinsics.rs`,
  `rig_scheimpflug_handeye.rs`, `rig_laserline_device.rs` (model after
  `crates/vision-calibration-pipeline/tests/laserline_device.rs`).

**T-02 (optim test):**
- New: `crates/vision-calibration-optim/tests/rig_laserline.rs` (model after
  `crates/vision-calibration-optim/tests/laserline_bundle.rs`).

**T-03 (facade unit test):**
- New: `crates/vision-calibration/tests/pixel_to_gripper_point.rs`.

**D-01 (README):**
- `/README.md` — short addition.

**D-02 (struct docs):**
- `crates/vision-calibration-optim/src/problems/laserline_rig_bundle.rs` —
  expand `RigLaserlineDataset` and module `//!` doc.

**C-01 (helper conversion):**
- `crates/vision-calibration-pipeline/src/rig_laserline_device/problem.rs` —
  add `From` impl or method.
- `crates/vision-calibration/src/lib.rs:379-385` — re-export.

**C-02 (handeye export comment):**
- `crates/vision-calibration-pipeline/src/rig_scheimpflug_handeye/problem.rs:359`.

**Release housekeeping (after fixes land):**
- Bump `[workspace.package].version` from `0.3.0` to `0.4.0` in root
  `Cargo.toml` (per-crate `version.workspace = true` resolves automatically).
- Promote `[Unreleased]` to `[0.4.0] - <release date>` in `CHANGELOG.md`.
  Add a fresh empty `[Unreleased]` block.

---

## Triage Decisions (Phase 3 — confirmed 2026-04-29)

Confirmed by owner via AskUserQuestion:

- **B-01..B-04 (P1 binding parity):** **include** — release blockers.
- **T-01, T-02, T-03 (P2 tests):** **include all three tiers** — pipeline
  integration tests, `optimize_rig_laserline` optim test, and
  `pixel_to_gripper_point` facade unit test. Closes every CI-runnable
  coverage gap on the new public surface.
- **D-01 (README):** **include**.
- **D-02 (`RigLaserlineDataset` doc):** **include**.
- **C-01 (`From<&RigScheimpflugHandeyeExport> for RigUpstreamCalibration`):**
  **include now** — small additive change; removes the copy-paste trap from
  the canonical chaining pattern.
- **C-02 (handeye export comment):** **include** — one-line addition.

**Release version:** **0.4.0** (minor). Matches the 0.2.0 → 0.3.0 precedent
for new public-surface additions in this repo.

**Active scope: 11 findings (B-01..B-04, T-01..T-03, D-01..D-02, C-01..C-02).**
Nothing deferred.

---

## Implementation Order

Confirmed scope, 11 findings, in priority order. Each bullet is a separate
commit with a `[refs <ID>]` suffix per repo convention.

1. **C-01** `From<&RigScheimpflugHandeyeExport> for RigUpstreamCalibration` +
   facade re-export. Tiny, lands first so subsequent tests can use it.
   `feat(pipeline): add From<RigScheimpflugHandeyeExport> for RigUpstreamCalibration [refs C-01]`
2. **B-01** `run_rig_scheimpflug_extrinsics` Python binding (lib.rs +
   _api.py + __init__.py + __init__.pyi + models.py).
   `feat(py): expose run_rig_scheimpflug_extrinsics [refs B-01]`
3. **B-02** `run_rig_scheimpflug_handeye` Python binding (mirrors B-01).
   `feat(py): expose run_rig_scheimpflug_handeye [refs B-02]`
4. **B-03** `run_rig_laserline_device` Python binding (mirrors B-01,
   accepts `RigUpstreamCalibration` payload via the typed input).
   `feat(py): expose run_rig_laserline_device [refs B-03]`
5. **B-04** `pixel_to_gripper_point` Python binding (bespoke wrapper, not
   a `run_problem` helper).
   `feat(py): expose pixel_to_gripper_point [refs B-04]`
6. **T-01a** `tests/rig_scheimpflug_extrinsics.rs` pipeline integration test.
   `test(pipeline): rig_scheimpflug_extrinsics happy path [refs T-01]`
7. **T-01b** `tests/rig_scheimpflug_handeye.rs` pipeline integration test.
   `test(pipeline): rig_scheimpflug_handeye happy path [refs T-01]`
8. **T-01c** `tests/rig_laserline_device.rs` pipeline integration test.
   `test(pipeline): rig_laserline_device happy path [refs T-01]`
9. **T-02** `tests/rig_laserline.rs` optim synthetic-GT test.
   `test(optim): synthetic GT for optimize_rig_laserline [refs T-02]`
10. **T-03** `tests/pixel_to_gripper_point.rs` facade unit test (4 cases:
    happy path, out-of-range cam, ray-plane miss, missing base_se3_gripper
    in EyeToHand).
    `test(facade): pixel_to_gripper_point happy and error paths [refs T-03]`
11. **D-02** `RigLaserlineDataset` rustdoc expansion.
    `docs(optim): clarify RigLaserlineDataset shape [refs D-02]`
12. **C-02** `target_poses.first().copied()` clarifying comment.
    `docs(pipeline): explain handeye target_poses canonical pose [refs C-02]`
13. **D-01** README mention of new pipelines + helper.
    `docs(readme): advertise Scheimpflug rig family and helper [refs D-01]`
14. **Release housekeeping** (separate from R-NN findings):
    - Bump `[workspace.package].version` from `0.3.0` to `0.4.0` in root
      `Cargo.toml` (per-crate `version.workspace = true` resolves
      automatically).
    - Promote `[Unreleased]` to `[0.4.0] - 2026-04-29` (or release date) in
      `CHANGELOG.md`. Add a fresh empty `[Unreleased]` block.
    - `chore(release): bump workspace to 0.4.0`

If the four `B-NN` bindings end up sharing significant scaffolding (typed
models in `models.py`, `_api.py` raw helpers), the Implementer is free to
merge them into 1-2 commits instead of 4 — bisectability and a clean
`[refs B-NN]` reference are the only hard requirements.

After all commits land, the Reviewer should run the full quality-gate
suite from the **Verification Plan** section, plus the new
`maturin develop` + Python smoke test, and verify the per-finding
Status fields in this document update from `todo` → `done` →
`verified`.
