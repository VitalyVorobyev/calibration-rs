# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-07-02)

- **Version line:** 0.x (latest release 0.6.0, nine publishable crates). v1.0
  (= stable public API) is deferred until the API has been stable across two minor
  releases without breaking changes. Pre-1.0 means breaking changes are acceptable.
- **Active branch:** `main`.
- **Production-grade program (approved 2026-07-02):** a phased plan to v1.0 —
  three new tracks **S** (device-spec → seed initialization), **Q** (algorithmic
  soundness: proof packs + regression gates), **R** (API/config revision), plus
  app-quality extensions of Track B (**B-QUAL → B-UX → B-DIST**). See
  [Production-grade program](#production-grade-program--path-to-v10) below;
  open tasks live in the [backlog](backlog.md).
- **Track A — Calibration core: COMPLETE.** A1 (manual init), A2 (per-feature
  residuals), A4 (Scheimpflug EyeToHand) shipped. A3 closed (false premise — the
  reported Zhang failure was a fixed puzzleboard-detector bug). A5 dropped (no real
  Python consumer; revisit after the Rust API stabilises). A6 (`rig_family` sensor-
  axis refactor) shipped via PRs #36 + #37 + #38; see
  [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md).
- **Track B — Tauri viewer:** B0 (PR #40), B0.5/B0.6 (PR #42), B1 (PR #43), B2
  (PR #44), **B3a + B3b (PR #45) SHIPPED**. The `app/` shell hosts five workspaces
  (Diagnose, 3D, Epipolar, Depth, Run); the Run workspace covers PlanarIntrinsics +
  chessboard end-to-end. Bench crate + multi-level reprojection report shipped
  (PR #49). A 2026-06-11 workspace review (internal) confirmed the
  extend-don't-rebuild verdict for the app. **B3c coverage completed
  2026-06-14:** B3c-1/2/3 wired the 8 topologies + charuco/laser; B3c-4 added
  the puzzleboard + ring-grid detectors, so **all four target detectors now
  calibrate end-to-end** (charuco dedup deferred, see backlog). **B3d
  (manifest UX) completed 2026-06-14:** the `sniff_folder` heuristic + CLI
  (B3d-1) and the "Sniff folder" / `_unresolved` / AskUser-modal front end
  (B3d-2) let the user point at a foreign folder and edit an auto-generated
  manifest. The remaining app work (formerly B-explore / B-infra / B3e) was
  re-scoped 2026-07-02 into the production-grade program's
  **B-QUAL → B-UX → B-DIST** series (see below).
- **New tracks (2026-06-11):** V (real-data validation on the private rtv3d dataset),
  O (apex-solver optimization backend), M (camera-model expansion — supersedes the
  former "new camera models out of scope" line).
- **In-flight PRs:** none blocking. PR #85 (`c5-ui-pointcloud`:
  depth-from-disparity + 3D point cloud in the Depth workspace) was reviewed
  and **merged 2026-07-02** (with the codex reference-frame fix), closing the
  Phase 0 disposition.

## Production-grade program — path to v1.0

Approved 2026-07-02. Goal: every registered dataset processes successfully
(the hardest being the private rtv3d sensor — six Scheimpflug camera-laser
pairs — via **spec-based seeded initialization**, the ADR 0022 supported
route), algorithmic soundness is backed by proofs and regression benchmarks,
the API/config surface is revised and frozen, and the Tauri app reaches
production quality. Task IDs and acceptance criteria live in the
[backlog](backlog.md).

```
Phase 0    D5-DOCS-TRUTH (this update) + PR #85 disposition
Phase I    Track S (spec → seed) + Track Q (proofs + regression)
Phase II   Track R (API/config revision) + Python parity fill
Phase III  Track B extensions (B-QUAL → B-UX → B-DIST)
Exit       D4-RELEASE v1.0 gate
```

### Track S — Device-spec → seed initialization

Grow the hand-coded per-example seeds into a structured `DeviceSpec` layer:
lens focal (mm) + pixel pitch → fx/fy, mount angle → Scheimpflug tilt,
mechanical layout → extrinsics/hand-eye seeds, feeding the existing manual-init
surface (ADR 0011). S1 (ADR 0023 schema) → S2 (intrinsics seeding, replaces
the `RTV3D_RINGGRID_FOCAL` env sweep) → S3 (extrinsics + hand-eye from layout)
→ S4 (one-command acceptance harness over all registered datasets; absent
datasets print `UNAVAILABLE`, never a silent pass; the failing 0.4 px
from-scratch gate demotes to informational per the parked V7).
**Status 2026-07-02: Track S complete (S1–S4).** ADR 0023 + `device_seed`
facade module; both intrinsics examples and `rtv3d_rig` spec-seeded (bootstrap
behind `RTV3D_SEED=generic|oracle`); `calib-bench accept` runs all registered
datasets through the seeded route with per-entry hard gates (19 passed / 2
UNAVAILABLE / exit 0), the cheap stereo subset gates CI, and the 0.4 px
from-scratch diagnose check is demoted to informational. **Update 2026-07-08
(Q3):** ringgrid gates tightened to ≤ 0.7 px (per-cam means 0.42–0.50 px);
the projective ellipse-center bias is already corrected inside the
`ringgrid` 0.7 detector, and the remaining floor is small-marker
localization noise (`docs/notes/ringgrid-bias.md`).

### Track Q — Algorithmic soundness: proofs + regression

A **proof pack** per algorithm family: a math note (`docs/notes/<family>.md` —
model equations, cost, identifiability/degeneracy, gauge), a synthetic-GT
matrix test (parameter grid × noise levels), property tests where a real
invariant exists, and a committed regression Fit record + gate via the bench
machinery. Init routines additionally get a **convergence-basin study**
(perturb the spec seed, measure gate-pass rate) — the quantitative evidence
behind ADR 0022. Q2 (regression wiring) lands deliberately early so every
later algorithm change is guarded. Absorbs: V6 scale (→ Q5), ringgrid
ellipse-center bias (→ Q3), M-WIRE's Scheimpflug distortion-model wiring
(→ Q4), C1-FOLLOWUP solver dedup (→ Q7). **Q7 resolved-as-already-done
2026-07-04:** the dedup it was meant to do had already landed in PR #72
(2026-06-21); see `docs/backlog.md` Q7-SOLVER-DEDUP.

### Track R — API/config/design revision

**R1–R4 done 2026-07-08.** R1: facade gained `dataset`/`dataset_runner`/
`detect` modules; the app is facade-only; one `ScheimpflugFixMask`;
diagnostics live under `vision_calibration::analysis`; dead shims deleted.
R2: ADR 0024 (one config vocabulary, grouped shapes, shared sub-structs).
R3: executed across all 8 configs + every consumer; `fix_first_rig_pose` and
`fix_first_camera_extrinsic` removed (reference-camera gauge only — the
joint BA now honors `reference_camera_idx`). R4: `vision-calibration-linear`
is anyhow-free. R5 done 2026-07-09: `run_rig_handeye_laserline` bound,
`distortion_model` + `SensorMode` mirrored (typed results made
model-polymorphic), `check_binding_parity.py` CI guard; MVG bindings stay
deferred. R6 done 2026-07-09: distortion-model-selection, single-cam-handeye
(+ `spec.json`), and app-walkthrough tutorials; existing six verified on the
R3 shapes. **Track R is complete.**

### Track B extensions — app to production grade

B-QUAL1–4 done 2026-07-10 (ESLint 9 + Prettier + two app CI jobs;
schemars-driven TS wire types replacing `inferExportKind` — mechanism
changed from ts-rs, see ADR 0018 amendment; 35 component tests; Playwright
smoke + repo-root preset resolution). B-UX1 done 2026-07-10 (ui/
component set + token fixes + dark-mode audit). B-UX2 core done
2026-07-10 (stage progress + cancel, pose stats, camera residual
matrix, single-cam laser 3D) — B-EXPLORE sub-items remain open.
B-DIST done 2026-07-10 as unsigned bundles + `app-bundle.yml`
artifacts; signing/notarization deferred (needs certs).

### Exit criteria (the new D4-RELEASE checklist)

1. One acceptance command runs every on-disk registered dataset through the
   seeded official route with hard gates (rtv3d_ref ≤ 0.5 px all six cameras;
   rtv3d full rig beats the oracle; ringgrid ≤ 0.7 px per camera — met, Q3).
2. Proof pack per shipped algorithm family; basin study for seeded init.
3. API frozen: facade-only consumers, normalized configs, typed errors
   everywhere, no duplicate public type names, no deprecated shims.
4. App CI green (lint, typecheck, unit, component, smoke); generated IPC types.
5. Docs current (this file, backlog, ADR statuses, tutorials, README/AGENTS).

## Tracks

### Track A — Calibration core (DONE)

Eight problem types across four workflow modules: `planar_intrinsics`,
`scheimpflug_intrinsics`, `single_cam_handeye`, `laserline_device`,
`rig_extrinsics` (pinhole + Scheimpflug via `SensorMode`), `rig_handeye`
(pinhole + Scheimpflug via `SensorMode`), and `rig_laserline_device`. All carry
manual init (ADR 0011) and per-feature residuals on export (ADR 0012).

| Item | Status | Notes |
|------|--------|-------|
| **A1** Manual init | **SHIPPED** (PR #32) | ADR 0011, all problem types, `manual_init_proof` example, tutorial. |
| **A2** Per-feature residuals | **SHIPPED** (PR #33 + #35) | ADR 0012, every `*Export` carries `per_feature_residuals`, tutorial. |
| **A3** Zhang lambda-sign fallback | **CLOSED — false premise** | The reported failure was a puzzleboard-detector bug, since fixed. No real-data Zhang failure to defend against. |
| **A4** Scheimpflug EyeToHand | **SHIPPED** | `RigHandeyeProblem` (Scheimpflug variant) supports both `EyeInHand` and `EyeToHand` via `RigHandeyeInitConfig::handeye_mode`. |
| **A5** Python parity | **DROPPED** | No real Python consumer; revisit *after* the Rust API stabilises, as one coherent build, not parity patches. |
| **A6** `rig_family` refactor | **SHIPPED** (PRs #36 + #37 + this PR) | Sensor-axis-only collapse. Five rig sibling modules → three. Net ~−2,300 LoC. See [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md). |

**Track A exit criterion (met):** the `vision-calibration` facade is stable enough
that B0 (Tauri scaffold) can compile against it with no breaking churn back into core.

### Track B — Tauri 2 + React + TypeScript desktop app

A production-grade internal tool wrapping the calibration library.
[ADR 0014](adrs/0014-tauri-desktop-app.md) records the framework choice
(Tauri 2 + React + TS over `rerun.io` and `egui`) and the diagnose-first
v0 scope. After B0–B2 shipped, a 2026-05-02 grill session committed the
track to a much larger goal: make the app the **primary calibration
tool**, not a passive viewer.

**End-state vision (settled 2026-05-02):** point the app at any
foreign dataset → AI inspects the layout → emits a canonical
[`DatasetSpec`](adrs/0016-dataset-manifest.md) manifest with fields
the AI couldn't determine listed under `_unresolved` (no silent
guessing, [ADR 0019](adrs/0019-fail-fast-on-ambiguity.md)) → schema-
driven forms ([ADR 0018](adrs/0018-schema-driven-ui.md)) let the user
edit any of the manifest's or the per-problem-type config's fields →
Run dispatches detection (cached, [ADR 0017](adrs/0017-detection-cache.md))
+ calibration in-process and routes the export into `/diagnose`. All
8 problem types and 4 target detectors (chessboard / charuco /
puzzleboard / ringgrid) are supported.

**Phase 1 — passive viewer (DONE, 2026-05-02).**

- **B0 — diagnose viewer v0** (PR #40). Passive viewer of one
  `PlanarIntrinsicsExport`. `ImageManifest` Export-side contract.
- **B0.5/B0.6 — real-data acceptance + viewer UX** (PR #42). Manifest
  extended to `RigHandeyeExport`; puzzle 130×130 Scheimpflug rig
  rendered correctly; design tokens, navigation, theme toggle.
- **B1 — 3D rig viewer** (PR #43). React-Three-Fiber scene with rig
  origin, per-camera frustums, target boards. Multi-workspace shell.
- **B2 — epipolar workspace** (PR #44). Server-side
  `compute_epipolar_overlay` Tauri command via canonical camera
  models. Two-pane viewer with click-to-pick.

**Phase 2 — self-contained calibration app (in flight).**

- **B3a — foundation** (PR #45). `schemars` derives across every
  `*Config` + shared option types; `cargo xtask emit-schemas` →
  `app/src/schemas/`. New crates `vision-calibration-dataset`
  (`DatasetSpec` + validator) and `vision-calibration-detect`
  (`Detector` trait, `ChessboardDetector`, `DetectionCache` trait
  + filesystem impl). `pipeline::dataset_runner::build_planar_input`
  wires manifest → cache → detect-on-miss → `PlanarDataset` IR.
  ADRs 0016–0019.
- **B3b — Tauri runner + Run workspace.** Tauri `run_calibration`
  command (Planar+Chessboard end-to-end), schema-driven
  `<ConfigForm/>` React component, Run workspace replaces the stub.
  Vertical slice ships first; coverage to all 8 topologies + 4
  detectors follows in B3c.
- **B3c — coverage.** Wire the remaining 7 problem types through
  dispatch; wire charuco / puzzleboard / ringgrid detectors;
  per-problem-type `DatasetSpec → *Input` converters; manifest
  sweep finish on `SingleCamHandeyeExport`,
  `ScheimpflugIntrinsicsExport`, and `LaserlineDeviceExport`.
  **Sequencing (2026-06-11): RigHandeye + RigLaserlineDevice + charuco
  detector first** — they serve the V-track (rtv3d) directly.
  - **B3c-1 (2026-06-12, PRs #51–#54):** charuco detector
    (`vision-calibration-detect`), robot-pose loading + view pairing +
    rig converters in `dataset_runner`, topology dispatch in the Tauri
    runner (+`default_config_cmd`), TS topology selector + presets.
    Covers PlanarIntrinsics, ScheimpflugIntrinsics (incl. its export's
    `image_manifest`), RigExtrinsics, RigHandeye.
  - **B3c-2 (in flight):** SingleCamHandeye **shipped 2026-06-12** —
    `rowmajor4x4` headerless pose-file format (`DatasetSpec`), shared
    pose-to-view matching, `build_single_cam_handeye_input`,
    `image_manifest` on `SingleCamHandeyeExport`, Tauri dispatch arm,
    KUKA preset enabled over the committed `data/kuka_1` manifest
    (no pose-file conversion needed after all).
  - **B3c-3 (2026-06-12): laser topologies SHIPPED** —
    [ADR 0021](adrs/0021-laser-frame-manifest.md): `laser_images` per
    camera + `[laser]` extraction spec + `upstream_calibration` +
    `matrix_field` pose shape in `DatasetSpec`; injected
    `LaserPixelExtractor` (vision-metrology is not on crates.io — the
    app implements it, published crates only define the trait);
    `build_laserline_device_input` / `build_rig_laserline_device_input`
    (frozen `RigHandeyeExport` → per-view `rig_se3_target` via the
    hand-eye chain); both Tauri dispatch arms; `image_manifest` on
    `LaserlineDeviceExport`; rtv3d presets with per-preset
    `configOverrides`. Two-stage rtv3d acceptance
    (`rtv3d_laser_end_to_end`): hand-eye 1.56 px, all six planes at
    0.85–1.15 mm point-to-plane against the frozen upstream (sub-0.1 mm
    needs the V5 joint-BA runner).
  - **B3c-4 (2026-06-14): detector coverage complete** — puzzleboard
    (`calib-targets`) and coded ring-grid (`ringgrid` 0.6) detectors added
    to `vision-calibration-detect` behind the sealed `Detector` trait;
    `dataset_runner` dispatches `TargetSpec::Puzzleboard` (named-layout
    resolver) and `TargetSpec::Ringgrid` (realigned to the real hex-lattice
    `BoardLayout` model). **All four target detectors (chessboard / charuco /
    puzzleboard / ringgrid) now calibrate end-to-end**, the Run workspace
    surfaces them schema-driven. Charuco dedup is deferred (its numeric gate
    needs the private golden datasets, absent from CI). In-app "save export
    to file" remains in B3e.
- **B-laser — laserline visualization (SHIPPED 2026-06-12).**
  `FrameRef.kind` discriminator closes ADR 0021 §5: both laser
  topologies now splice laser-kind frames into their export manifests.
  Diagnose gains a Laser view — observed pixels colored by
  point-to-plane distance (thresholds at the 0.2 mm device norm),
  projected laser line overlay, mm-domain stats legend. The 3D rig
  viewer renders `laser_planes_rig` as bounded translucent quads
  anchored at each owning camera. Follow-ups: laser plane for the
  single-cam `LaserlineDeviceExport` in 3D (viewer is rig-only),
  laser-pixel overlay in compare mode.
- **B-explore — dataset exploration (→ B-UX2, 2026-07-02).** Browse a dataset
  *before* calibrating: image grid per camera/pose, detection overlay from the
  cache, board coverage map. Today the app only visualizes exports.
- **B3d — manifest UX (in flight).**
  - **B3d-1 (2026-06-14): heuristic sniffer SHIPPED** —
    `vision_calibration_dataset::sniff_folder` walks a dataset folder and
    emits a `DatasetSpec` skeleton, inferring only structurally-unambiguous
    fields (camera dirs/globs, robot-pose file format, `by_index` pairing)
    and leaving board geometry / target kind / frame convention / ambiguous
    topology at placeholders with their dotted paths in `_unresolved`
    (ADR 0019, no silent guessing). The `generate-manifest` CLI (`cli`
    feature → TOML) and the app's Tauri `sniff_folder` command share the one
    inference. Acceptance round-trips `data/kuka_1` + `data/stereo`.
    Heuristic-only v0 (no LLM / README scraping yet).
  - **B3d-2 (2026-06-14): manifest UX SHIPPED** — "Sniff folder" button
    (calls `sniff_folder`), `UnresolvedNotice` with vendor-aware field hints
    + per-field "mark resolved", red `_unresolved` badge + blocked Run, and
    an `AskUserModal` (click-to-apply suggestion buttons + free-text)
    replacing the inline AskUser banner. Vendor guidance lives front-end-side
    so runner suggestions stay raw click-to-apply values. B3d complete; the
    remaining app work now lives in the B-QUAL/B-UX/B-DIST series.
- **B3e — iteration polish (→ B-UX2, 2026-07-02).** Cancellability for long
  solves; progress event streaming; multi-pose residual stats panel;
  cross-camera residual matrix; experiments directory storing
  `(dataset.toml, config.json, export.json)` tuples for
  reproducibility.

**Phase 3 — re-scoped 2026-07-02 into the production-grade program.**

- Signed installers per OS → **B-DIST**.
- Infra ratchet (ts-rs codegen + export discriminator tag,
  `resource_dir` presets, component/Playwright tests) →
  **B-QUAL1–B-QUAL4**.
- Still deferred (post-1.0): LLM-backed manifest inference (separate ADR;
  opt-in, behind API key configuration); init-failure diagnosis sweeps
  (perturbed re-runs).

### Track V — Real-data validation: rtv3d (V1–V5, V8 DONE; V6 → Q5 DONE; V7 parked)

Prove the library functional on the rtv3d sensor — a private dataset from a
6-device laser-plane-triangulation head (Scheimpflug camera + laser projector
per device), with a legacy-system oracle calibration to beat.

- **V1 (DONE)** `rtv3d_rig` example in `examples-private`: ChArUco detection,
  `RigHandeye(Scheimpflug)`, oracle comparison tables. The empirical
  convention checks settled hand-eye mode (EyeToHand) and cell size.
- **V2 (DONE)** Full pipeline: laser detection, `RigLaserlineDevice`, joint
  BA; laser-plane comparison vs the oracle.
- **V3 (DONE)** Beat-the-oracle validation: all pass criteria met
  (per-camera reprojection below the oracle, sane recovery of the camera the
  oracle solved degenerately, plane-fit σ below the oracle on all planes).
- **V4 (DONE)** Bench registry entry (`registry/private.json`, local-only) →
  regression tracking.
- **V5 (DONE 2026-06-13)** Full `RigLaserlineDevice` + joint-BA runner in the
  bench. Local floor: 1.19 px mean reprojection, laser point-to-plane RMS
  0.018–0.035 mm over 10,797 points. Laser criterion passes; reprojection
  stays above the (now-parked) 0.4 px from-scratch target.
- **V6 → Q5 (DONE 2026-07-08).** Settled: no metric ambiguity. The apparent
  90.1 mm (ours) vs ~98.5 mm (oracle) hexagon gap was `rtv3d_rig.rs` comparing
  against the pre-laser hand-eye-stage extrinsics; the joint (laser-informed)
  BA extrinsics measure 98.21 ± 0.41 mm, matching the oracle's own
  healthy-camera hexagon (98.13 ± 1.10 mm) to 0.08 %. 5.2 mm cells confirmed
  correct. See `docs/notes/rtv3d-scale.md`.
- **V7 (PARKED, user call 2026-06-14).** Drive the rtv3d from-scratch
  reprojection floor below 0.4 px. Isolated to a detector/target/model floor,
  not rig-chain error (best centered means 0.747–1.199 px). Superseded as an
  acceptance criterion by the seeded route (ADR 0022); S4 demotes the bench's
  0.4 px from-scratch gate to informational. Q4 (richer distortion models on
  the Scheimpflug path) may move this floor as a side effect.
- **V8 (DONE 2026-06-30, PR #86)** Full ringgrid calibration on the private
  `rtv3d_ringgrid` dataset (different physical rig, no laser/oracle).
  Root-caused and fixed a real correctness bug in the shared 4×4 pose loader
  (`dataset_runner/poses.rs`): nalgebra's iterative `from_matrix` silently
  mis-converges on exact 180° rotations → replaced with exact/SVD-polar
  conversion + regression tests. Ring-grid reprojection floor **median
  ≈ 4.7 px** (ellipse-center perspective bias under Scheimpflug tilt; cam5 a
  ~38 px geometric outlier) vs puzzleboard 0.25–0.30 px — driving it down is
  Q3.

### Track O — Optimization backends (O1/O2 WON'T-DO 2026-07-04; O3 DONE)

The premise was that `OptimBackend` (ADR 0008) gets a second real
implementation:
[apex-solver](https://crates.io/crates/apex-solver) 1.3 (LM/GN/DogLeg, Lie-group
support), behind an `apex-solver` cargo feature in `vision-calibration-optim`.

The **O1 pre-verify gate failed** (2026-06-14, report:
`docs/internal/archive/report/2026-06-14-O1-apex-solver-preverify.md`); the track was **closed
won't-do on 2026-07-04** after a fresh assessment (backlog Track O carries the
full note and the revive triggers). The sharpened rationale: the IR is
backend-neutral and the `fn residual<T: RealField>()` kernels are
autodiff-*capable* rather than autodiff-*dependent* — a dual-number adapter
could in principle feed apex-solver's hand-Jacobian `Factor::linearize`, so
the mismatch is bridgeable, not fundamental. It is not worth bridging:
apex-solver 1.3 (still the latest as of 2026-07-04) lacks an S2 manifold
(laser-plane normals), documented robust losses, and documented SE3 /
Jacobian-parameterization conventions, and its LM/GN/DogLeg + sparse
Cholesky/QR core duplicates our homegrown LM + faer stack — leaving A/B
validation as the only payoff, which Q-track baselines + OpenCV cross-checks
already largely cover. A revived second backend should target an
autodiff-native stack (`factrs`/`num-dual`), not apex-solver 1.x.

- **O1 (WON'T-DO 2026-07-04)** `ApexSolverBackend` — closed; see above.
- **O2 (WON'T-DO 2026-07-04)** Backend A/B validation — closed with O1.
- **O3 (DONE 2026-06-15)** Dropped the dead `BackendKind::Ceres` stub from
  `optim/src/backend/mod.rs` (and the now-orphaned `Error::numerical` helper).
  `BackendKind` is now a single-variant enum; `solve_with_backend` no longer has
  an unreachable "backend not available" arm.

### Track P — Performance & profiling (opened 2026-06-16)

From-scratch Scheimpflug **rig** calibration on the dense `puzzle_board` dataset
(~200 corners/view) exposed that the pipeline's cost is dominated by dense
linear-algebra hot paths, not the algorithms. Two `svd(true, true)` sites that
accumulate the U factor across thousands of rows were hanging the linear init
(homography DLT >15 min, distortion fit >11 min) before being fixed in place;
the joint rig + hand-eye bundle adjustments remain heavy on full corner density.
Full profiling + the tiny-solver cost model:
`docs/internal/archive/report/2026-06-16-perf-from-scratch-rig-profiling.md`. Work items P1–P7 in
the [backlog](backlog.md#p--performance--profiling): SVD-sweep (P1, DONE),
joint-BA data density (P2 — open, scheduled only if the S4 acceptance runtime
hurts), tiny-solver backend cost (P3 — parked post-1.0), criterion guards
(P4, DONE), per-stage timing (P5, DONE), from-scratch per-camera convergence
(P6, DONE), seeded-default promotion (P7, DONE → ADR 0022). This track is
the natural home for reviving an autodiff-capable second backend (see Track O).

**Scheimpflug intrinsics — seeded init is the supported path (ADR 0022,
2026-06-17).** From-scratch Scheimpflug *intrinsics* is unstable under the
tilt↔focal↔distortion degeneracy and is now demoted to **experimental** (it logs a
warning). The supported default seeds a coarse focal + the nominal Scheimpflug
mount tilt and lets BA refine. The private `rtv3d_ref_intrinsics` harness
calibrates all 6 cameras from one coarse seed and **every camera clears the hard
≤ 0.5 px gate** (`[0.373, 0.267, 0.282, 0.473, 0.342, 0.321]` px). A reprojection
error > 0.5 px is never accepted as success. P6's from-scratch *rig* path remains
experimental.

### Track M — Camera-model expansion (M0 DONE 2026-06-12)

Supersedes the former "new camera models out of scope" rule — all four models
below are user-requested. The gate was M0: the `FactorKind` IR used to
enumerate projection × distortion × sensor × chain combinations, so new
models would have multiplied variants.

- **M0 (DONE)** Factor generification
  ([ADR 0020](adrs/0020-camera-model-as-data-factor-ir.md)): one factor
  family per residual type, camera model and chain as data, layout-derived
  validation, kernel monomorphization in the backend. Also folded the
  export-path residual helper into the generic `CameraProject` path and
  unblocked pinhole rig laserline.
- **M1/M2/M3 — additive layer DONE (2026-06-14).** Rational k4–k6 (OpenCV),
  thin-prism s1–s4, and the Fitzgibbon division model are implemented at the
  core runtime model + optim IR/backend layers (new `DistortionParams` /
  `AnyDistortion` / `DistortionKind` variants, `CameraModelDesc` constants, ZST
  kernels, dispatch rows, synthetic-GT tests). Strictly additive — the
  Brown-Conrady production paths are unchanged. The **PlanarIntrinsics
  M-WIRE slice landed 2026-06-21** (PR #79). The remaining M-WIRE work is
  split across the production-grade program: Scheimpflug + rig config wiring
  → **Q4** (it is the suspected V7 reprojection-floor factor), config
  normalization → **R2/R3**, Python field → **R5**, app selector → **B-QUAL2**.
- **M4 (PARKED post-1.0)** Kannala-Brandt fisheye (new projection slot +
  linear-init changes — the biggest lift). No fisheye dataset exists in the
  acceptance set, so it is out of scope for the v1.0 program.

### Track C — MVG (C1 landed 2026-06-14)

Two-view geometry is split into `vision-geometry` (deterministic solvers) and
`vision-mvg` (pipelines, robust estimation). The track extends to multi-view
geometry over already-calibrated rigs.
[ADR 0015](adrs/0015-mvg-ceiling.md) caps the ceiling explicitly: no in-house
dense matcher, no full SfM.

- **C1 — DONE (2026-06-14).** Rather than merge the stale `mvg` branch (~136
  commits behind, predating four current crates + a conflicting
  `vision-calibration-linear` refactor), the two crates were **ported fresh and
  additively** onto `main`: `vision-geometry` (20 tests) + `vision-mvg` (31
  tests), both `publish = false`, no change to `vision-calibration-linear`. The
  `linear`→`vision-geometry` de-duplication was a deliberate follow-up (ADR
  0015) — **done, PR #72 (2026-06-21)**: `linear` now depends on
  `vision-geometry` and its duplicate `homography`/`epipolar`/
  `camera_matrix`/`triangulation` modules are gone (net −1811 LoC). See
  `docs/backlog.md` Q7-SOLVER-DEDUP (resolved-as-already-done, 2026-07-04).
- **C2** N-view triangulation + nonlinear refinement.
- **C3** Bundle adjustment with frozen intrinsics, free poses, free structure.
- **C4** Stereo rectification — including **Scheimpflug-aware rectification** (genuinely
  novel for this project).
- **C5** Dense matcher. ADR 0015's ceiling was **amended** (2026-06-21): the matcher
  ships **pure-Rust in `vision-mvg`**, with `opencv-rust` SGBM demoted to a
  benchmark-only baseline in the unpublished bench crate. The **block-matching MVP
  plus opt-in semi-global (SGM) aggregation** (`vision_mvg::dense::match_block`,
  ZNCC + sub-pixel + LR-consistency + 8-path P1/P2 aggregation) landed 2026-06-21 —
  94% density at 0.18 px RMS on synthetic GT; on the real `data/stereo` rig SGM
  doubles board coverage (21%→49%) at 0.44–0.74 px board-planarity RMS (visual-
  evidence demos under `target/fixtures/`). The OpenCV SGBM baseline was
  **closed as env-blocked 2026-07-02** (no OpenCV-equipped environment;
  re-openable if one materializes).
- **C-UI** MVG visualizations layered into the Tauri app. **Landed 2026-06-21:** a
  **Depth (dense stereo) workspace** — server-side `compute_disparity` Tauri command
  (rectify → match → colormap, block or SGM) over a rig export's stereo pair, with
  **rectified-pair / disparity / overlay / depth / 3D-point-cloud** views + a metrics
  strip. The command reprojects the disparity to metric depth + a coloured 3D point
  cloud (`Z = f·B/d`, expressed in the reference-camera frame); the cloud renders in
  a lazy-loaded React-Three-Fiber view. Merged as PR #85 (2026-07-02, closing the
  Phase 0 disposition). Remaining (future): multi-pose cloud fusion, textured-mesh
  export.

### Track D — Earn v1.0 (continuous ratchet)

- **D1 (DONE 2026-06-21)** Typed errors only — no `String`-typed escape hatches
  in public APIs; `anyhow` dropped from every published crate's `[dependencies]`
  (PRs #72, #77, #78). One holdout remains *inside* `vision-calibration-linear`
  (private helpers still on anyhow) — tracked as **R4**.
- **D2** Doc-warning-free, MSRV pinned at v1.0 cut (currently 1.93). **Ratchet
  landed 2026-06-17:** `[workspace.lints.rust] missing_docs = "warn"` enforced
  across every member crate (CI clippy `-D warnings` makes it a hard gate); all
  public items documented. The `RUSTDOCFLAGS="-D warnings"` rustdoc gate was
  already clean. Test/bench targets carry a local `#![allow(missing_docs)]`
  (not public API).
- **D3** Python binding parity. **Audit DONE 2026-06-21** (PR #81,
  `docs/python-parity-audit.md`): seven of the eight facade workflows are bound;
  gaps G0 (`run_rig_handeye_laserline` unbound), G1 (MVG surface), G2
  (`distortion_model` field). The fill is **R5** in the production-grade program.
- **D4** v1.0 release gate — checklist replaced 2026-07-02 by the
  [production-grade program exit criteria](#exit-criteria-the-new-d4-release-checklist):
  dataset acceptance command green, proof packs complete, API frozen
  post-R-track, app CI green, docs current, plus the standing requirement that
  the API has been stable across two minor releases.
- **D5 (DONE 2026-07-02)** Documentation truth pass — this update: roadmap
  refreshed to HEAD, S/Q/R/B-QUAL tracks added, backlog dispositions applied,
  ADR status notes (0008/0015/0020/0022), README/AGENTS crate lists fixed,
  stale internal handoffs archived.

## Load-bearing path

As of 2026-07-02 the load-bearing path **is the production-grade program**:

**D5 (docs truth) → S1–S4 (spec → seed + acceptance harness) → Q2 (regression
wiring, early) → Q3–Q7 (soundness) → R1–R6 (API/config revision + Python
parity) → B-QUAL1–4 → B-UX1–2 → B-DIST → D4-RELEASE.**

The earlier V→B path completed: V1–V5 + V8 proved the library on rtv3d, all
8 topologies + 4 detectors are wired in the app, and manifest UX shipped.
O1/O2 (apex-solver) are closed **won't-do** 2026-07-04 (bridgeable but not
worth it — see Track O); M4 fisheye and P3 backend-cost are parked post-1.0.

## Out of scope (explicit)

- Camera models beyond the M-track set (omnidirectional / MEI, double-sphere,
  telecentric, spline). Defer until a concrete project demands one. M4 fisheye
  itself is parked post-1.0 (no dataset in the acceptance set).
- Full structure-from-motion (incremental SfM, pose graph, loop closure).
  (The former "no in-house dense matcher" line was **amended** — ADR 0015,
  2026-06-21: the dense matcher ships pure-Rust in `vision-mvg`; only full SfM
  remains out of scope.)
- `rerun.io` / `egui` as the UI.
- OSS-grade community surface and multi-platform install docs (internal-first; defer until
  the tool earns it).
- What-if interactive re-optimize (B6 stretch at earliest).

## See also

- [ADR index](adrs/README.md) — design records.
- [Tutorials](tutorials/README.md) — hands-on walkthroughs for new users.
- [MSRV notes](MSRV.md) — MSRV 1.93 history and bump policy (lockfile pins
  dropped 2026-05-23; `cargo update` is safe).
- Per-track ADRs:
  [`0011-manual-initialization-workflow.md`](adrs/0011-manual-initialization-workflow.md) (A1, landed in PR #32);
  [`0012-per-feature-reprojection-residuals.md`](adrs/0012-per-feature-reprojection-residuals.md) (A2, landed in PR #33 + #35);
  [`0013-rig-family-sensor-axis-refactor.md`](adrs/0013-rig-family-sensor-axis-refactor.md) (A6, landed in PRs #36 + #37 + #38);
  [`0014-tauri-desktop-app.md`](adrs/0014-tauri-desktop-app.md) (B0, landed in PR #40);
  [`0016-dataset-manifest.md`](adrs/0016-dataset-manifest.md) (B3a, PR #45);
  [`0017-detection-cache.md`](adrs/0017-detection-cache.md) (B3a, PR #45);
  [`0018-schema-driven-ui.md`](adrs/0018-schema-driven-ui.md) (B3a, PR #45);
  [`0019-fail-fast-on-ambiguity.md`](adrs/0019-fail-fast-on-ambiguity.md) (B3a, PR #45);
  [`0015-mvg-ceiling.md`](adrs/0015-mvg-ceiling.md) (C1, landed 2026-06-14; ceiling amended 2026-06-21 for C5);
  [`0020-camera-model-as-data-factor-ir.md`](adrs/0020-camera-model-as-data-factor-ir.md) (M0);
  [`0021-laser-frame-manifest.md`](adrs/0021-laser-frame-manifest.md) (B3c-3);
  [`0022-scheimpflug-intrinsics-seeded-default.md`](adrs/0022-scheimpflug-intrinsics-seeded-default.md) (P7 — the Track S foundation).
