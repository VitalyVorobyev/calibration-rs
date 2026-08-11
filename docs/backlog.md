# Backlog

Open and parked implementation tasks. One task per commit; on completion,
mark it `[x]` with a dated one-paragraph note, then move the entry to
[`backlog-archive.md`](backlog-archive.md)
so this file stays a picture of what is left rather than what was done
(AGENTS.md §11). `[~]` means parked — a deliberate decision not to pursue it
now, with the reason recorded.

Strategy and phase gates live in the [ROADMAP](ROADMAP.md); design decisions
in [ADRs](adrs/).

## B-QUAL / B-UX / B-DIST — app to production grade (Phase III)

- [ ] B-QUAL-HOOKS7 - `eslint-plugin-react-hooks` 7 (adopted 2026-08-11
  with the eslint 10 bump) ships two new rules, both set to `warn` in
  `app/eslint.config.js` rather than blocking CI:
  - `react-hooks/set-state-in-effect` — five real sites
    (`useImageData.ts` ×2, `DiagnoseWorkspace/index.tsx` ×3) reset derived
    state from an effect when their input changes. Correct, but one render
    pass more than keying the component would cost. Rewrite when touching
    those components anyway.
  - `react-hooks/purity` — one report, a **false positive**: the
    `Date.now()` stamping a run's start time in RunWorkspace's async
    submit handler. The rule cannot distinguish an event handler from
    render. Re-check on plugin updates; drop the override if it learns to.
- [ ] B-QUAL-TS7 - TypeScript 7 is **blocked upstream**: `typescript-eslint`
  hard-errors on it (`typescript-eslint does not support TS 7.0`, tracking
  issue typescript-eslint#10940). Adopting it today means dropping the
  type-aware lint config B-QUAL1-LINT-CI deliberately built. Pinned at
  TypeScript 6; revisit when typescript-eslint ships TS 7 support.
- [ ] B-UX2-ELEVATION - Workspace-by-workspace elevation: empty states, error
  surfaces (ADR 0019 fail-fast shown well), progress for long calibrations,
  manifest-sniff UX polish. Absorbs **B-LASER** (laser-pixel overlay in
  Diagnose compare mode, point-to-plane mm panel, single-cam laser plane in
  3D) and **B-EXPLORE** (pre-calibration image grid, detection-cache overlay,
  coverage map) as scoped sub-items. Gate: frontend-review pass with no
  high-severity findings.
  - **Progress 2026-07-10 (core features).** Shipped: (1) **stage-progress
    streaming** for long solves via a request-scoped
    `tauri::ipc::Channel<RunProgress>` — the runner announces `detect →
    solve → export` (the only boundaries it honestly owns; per-camera /
    per-LM granularity has no pipeline hook, deliberately not faked); Run
    workspace shows a stage checklist + live elapsed clock. (2)
    **Cancellation** — `AtomicBool` per `runId` in a managed `RunRegistry`,
    `cancel_run_cmd` flips it, runner stops at the next stage boundary and
    returns `RunResponse::Cancelled` (distinct "Run cancelled" UI, not an
    error). (3) **Multi-pose residual stats** panel in Diagnose (sortable
    per-pose mean/median/max px table, click-to-jump). (4) **Cross-camera
    residual matrix** for multi-camera exports (cameras × poses grid on the
    FrameCanvas severity scale, click-to-jump). (5) **Single-cam laser plane
    in 3D** (B-LASER close-out) — `laserline_device` exports are lifted into
    a one-camera rig at the origin so the camera-frame plane + poses render.
    New pure/tested modules: `runStages.ts`, `lib/residualStats.ts`,
    `lib/sceneExport.ts`. `RunProgress`/`RunStage` schema-generated. Still
    open under B-UX2: **B-EXPLORE** (pre-calibration image grid,
    detection-cache overlay, coverage map), empty-state / error-surface
    polish, manifest-sniff UX polish.

## V — rtv3d validation

- [~] V7-RTV3D-INTRINSICS-FLOOR - **PARKED** (user call 2026-06-14;
  disposition confirmed 2026-07-02). Drive the rtv3d from-scratch
  reprojection floor below 0.4 px, or prove the blocking term — isolated to
  detector/target/model, not rig-chain. Seeded init (ADR 0022) is the
  official acceptance path. **Do not reopen unprompted.**

## P — Performance & profiling

Opened 2026-06-16 after the from-scratch Scheimpflug rig calibration
(`rtv3d_ref_rig`) took 30+ min on the dense `puzzle_board` dataset (~200
corners/view). Root-caused to dense linear-algebra hot paths, not the
algorithms. Full profiling:
`docs/internal/archive/report/2026-06-16-perf-from-scratch-rig-profiling.md`.

Systemic causes: (1) `nalgebra::svd(true, true)` pervasive (~20 sites),
pathologically slow on tall/dense matrices (P1, closed); (2) `tiny-solver`
recomputes an autodiff Jacobian every LM iteration, re-evaluates residuals on
up to 32 damping retries, single-threaded (P3, parked); (3) no data-density
control for the joint rig/hand-eye BA — cost scales linearly with corner
count though extrinsics/hand-eye don't need full density (P2, conditional).

- [ ] P2-BA-DENSITY - **Conditional (2026-07-02):** schedule only if the S4
  acceptance-runner wall time for the 6-camera rtv3d rig is painful (>~5 min);
  otherwise stays parked. Principled corner budget for the joint rig +
  hand-eye BA (spatially-distributed subsample preserving coverage, or
  per-stage decimation knobs). Extrinsics/hand-eye converge on a fraction of
  the corners; the per-camera intrinsics stage already uses a cheap subsampled
  tilt sweep + a full-data refine (`optimize_scheimpflug_intrinsics_staged`).
- [~] P3-BACKEND-COST - **Parked post-1.0 (2026-07-02)** — Track O's
  second-backend premise is dead. Would profile the tiny-solver
  autodiff/assembly/solve split and evaluate analytic Jacobians / caching /
  rayon parallelism for the hot `ReprojPoint` factor.

## M — camera models (gated on M0)

- [~] M-WIRE - Pipeline-selection plumbing for the new distortion models.
  **PlanarIntrinsics vertical slice done 2026-06-21**:
  `PlanarIntrinsicsConfig.distortion_model: DistortionKind` selects
  BC5/Rational8/ThinPrism9/Division1; model-agnostic export contract; 7 new
  E2E tests. Remaining scope split 2026-07-02: intrinsics-bearing rig/
  Scheimpflug problem types → Q4-MWIRE-SCHEIMPFLUG (done); config placement →
  R2/R3 (done); Python binding → R5-PY-PARITY (done 2026-07-09); app
  selector → B-QUAL2-TSRS (open).
- [~] M4-FISHEYE - **Parked post-1.0 (2026-07-02)** — no fisheye dataset in
  the acceptance set. Would add Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`.

## C — MVG (multiple-view geometry)

- [~] C1-FOLLOWUP - De-duplicate geometric solvers shared between `linear`
  and `geometry`. **Resolved 2026-07-04** — low-level `math` primitives
  deduped into `vision_calibration_core::linalg`; higher-level solvers
  (`homography`, `epipolar`, `camera_matrix`, `triangulation`) were already
  deduped in PR #72 (net −1811 LoC), confirmed via Q7-SOLVER-DEDUP.
  - [x] Promote `vision-geometry`/`vision-mvg` to the crates.io publish set.
    **Done 2026-06-17** (user call); the actual first `cargo publish` is a
    manual step.
  - [ ] PyO3 bindings for the MVG surface — **deferred** (A5 Python parity was
    dropped: no Python consumer, and the py crate binds the calibration facade
    only). Revisit if a consumer appears.
- [~] C5-DENSE - Dense stereo matcher. **Direction reset + implementation
  done 2026-06-21** (user-supervised): amended ADR 0015 — the matcher ships
  pure-Rust in `vision-mvg::dense` (block matching + SGM aggregation, ZNCC
  via summed-area tables, no new deps), scored by a bench harness; synthetic
  slanted-plane recovery hits 94% density at 0.18 px RMS. OpenCV SGBM
  baseline **closed as env-blocked** 2026-07-02 (no OpenCV env available).

## D — Earn v1.0

- [ ] D5-CHARUCO-LABELS - **Blocked on upstream
  ([calib-targets-rs#86](https://github.com/VitalyVorobyev/calib-targets-rs/issues/86)).**
  `calib-targets` 0.12's ChArUco detector can emit corner labels that are not
  projectively consistent. Two shapes, one cause: a *collapsed* assignment
  (distinct lattice nodes `(u=13..16, v=8)` sharing one `position`, with
  differing `score`s) and a *scrambled* one (all positions distinct, but no
  homography fits any subset — 3 of 18 inliers after refitting on the best
  12). Both make the affected view's pose meaningless; on `rtv3d` camera 4
  they took the per-camera residual from 1.10 px to 20.12 px.

  Diagnostic that separates them cleanly, needing no ground truth: fit a
  homography to the detection's own `(grid, position)` pairs. Over 20 views of
  that camera, 18 healthy views fit at 0.56–0.99 px median, the two broken
  ones at 7.7–7.8 px.

  **Ours:** `reject_ambiguous_detection` (detect crate) rejects the collapsed
  case — the invariant "one pixel is one board point" holds under any lens, so
  it is always safe. The scrambled case is *not* guarded: the projective test
  assumes distortion is small over the observed corners, so applying it at the
  detect boundary would reject good views on wide-angle cameras. Fix belongs
  in the grid builder upstream.

  **Measured impact, `rtv3d` (6-camera ChArUco rig hand-eye, 720×540 tiles),
  baseline `0.7.0` → `0.8.0` defaults:**

  | camera | 0.7.0 mean px | 0.8.0 mean px | features 0.7.0 → 0.8.0 |
  |---|---|---|---|
  | 0 | 1.048 | 0.964 | 234 → 197 (−16 %) |
  | 1 | 1.050 | 1.038 | 229 → 197 (−14 %) |
  | 2 | 1.099 | 1.545 | 238 → 186 (−22 %) |
  | 3 | 1.092 | 1.777 | 229 → 163 (−29 %) |
  | 4 | 1.261 | **10.926** | 254 → 158 (−38 %) |
  | 5 | 1.538 | 1.452 | 289 → 245 (−15 %) |
  | overall | 1.196 | 2.664 | 1473 → 1146 (−22 %) |

  Degradation tracks corner loss monotonically, and the loss is the new
  `min_corner_strength = 33.0` floor cutting into small, soft tiles. Setting
  `min_corner_strength: 0.0` for this dataset recovers camera 4 but leaves
  camera 3 at 3.38 px — so the floor is the *trigger*, not the whole cause:
  `calib-targets` 0.12's ChArUco grid builder is worse on this data at either
  setting. **`rtv3d` therefore fails its 2.5 px gate under 0.8.0 and its
  baseline is deliberately left frozen at the 0.7.0 numbers** — re-freezing
  would launder an upstream regression into an accepted result. The other 12
  private and all 6 public datasets pass. Re-run `calib-bench accept --only
  rtv3d` when the upstream fix lands; that is the acceptance criterion for
  closing this item.
- [ ] D4-NALGEBRA-035 - **Blocked on `tiny-solver`.** `nalgebra` 0.35,
  `faer` 0.24 and `faer-ext` 0.8 are all unadoptable while `tiny-solver`
  0.18 (latest) is built against 0.34 / 0.23 / 0.7:
  `vision-calibration-optim` passes both nalgebra and faer types straight
  across that boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`), so a bump
  produces ~33 trait-mismatch errors. `calib-targets` 0.12 already uses
  nalgebra 0.35 internally — harmless, because `vision-calibration-detect`
  is nalgebra-free and converts to plain arrays at its boundary. **Keep
  that property**: it is what lets the two nalgebra versions coexist.
  Re-check on each `tiny-solver` release; the alternative is replacing the
  solver backend (see O-track).
- [~] D3-PY-PARITY - Audit the PyO3 binding surface against the Rust facade.
  **Audit done 2026-06-21** (`docs/python-parity-audit.md`): 7/8 workflows
  bound; gaps G0 (`rig_handeye_laserline` unbound, highest priority), G1
  (MVG surface unbound), G2 (`distortion_model` field), G3 (low-level
  modules, by design). Fill work **absorbed into R5-PY-PARITY** 2026-07-02.
- [ ] D4-RELEASE - v1.0 gate — **checklist replaced 2026-07-02** by the
  production-grade program exit criteria (ROADMAP): S4 acceptance command
  green on all on-disk datasets, Q proof packs complete, R-track API/config
  freeze done, app CI green (B-QUAL), docs current, plus the standing
  requirement that the API has been stable across two minor releases.

## B — app (extend; sequencing serves V-track)

- [~] B-LASER - **Re-scoped into B-UX2-ELEVATION** 2026-07-02: laser-pixel
  overlay in Diagnose compare mode, point-to-plane (mm) panel, single-cam
  laser plane in the 3D viewer. (Core laser views shipped 2026-06-12.)
- [~] B-EXPLORE - **Re-scoped into B-UX2-ELEVATION** 2026-07-02: per-camera/
  pose image grid, detection-cache overlay, board coverage map.
- [~] B-INFRA - **Absorbed 2026-07-02** into B-QUAL1-LINT-CI (CI entry),
  B-QUAL2-TSRS (ts-rs codegen), B-QUAL4-SMOKE (`resource_dir` presets +
  Playwright). Vitest unit-slice sub-item shipped 2026-06-15 (18 tests over
  `inferExportKind`/`exportKindLabel`/`mergeConfig`).

## Closed tracks

Fully delivered; notes in the
[archive](backlog-archive.md).

- **S — Device-spec → seed initialization (Phase I)** — 4 tasks.
- **Q — Algorithmic soundness: proofs + regression (Phase I)** — 8 tasks.
- **R — API/config/design revision (Phase II)** — 7 tasks.
- **O — apex-solver backend (O1/O2 WON'T-DO 2026-07-04; O3 DONE)** — 1 task.
- **Benchmark** — 6 tasks.
