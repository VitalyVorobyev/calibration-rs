# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-06-14)

- **Version line:** 0.x (latest release 0.5.1). v1.0 (= stable public API) is deferred
  until the API has been stable across two minor releases without breaking changes.
  Pre-1.0 means breaking changes are acceptable.
- **Active branch:** `main`.
- **Track A — Calibration core: COMPLETE.** A1 (manual init), A2 (per-feature
  residuals), A4 (Scheimpflug EyeToHand) shipped. A3 closed (false premise — the
  reported Zhang failure was a fixed puzzleboard-detector bug). A5 dropped (no real
  Python consumer; revisit after the Rust API stabilises). A6 (`rig_family` sensor-
  axis refactor) shipped via PRs #36 + #37 + #38; see
  [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md).
- **Track B — Tauri viewer:** B0 (PR #40), B0.5/B0.6 (PR #42), B1 (PR #43), B2
  (PR #44), **B3a + B3b (PR #45) SHIPPED**. The `app/` shell hosts four workspaces
  (Diagnose, 3D, Epipolar, Run); the Run workspace covers PlanarIntrinsics +
  chessboard end-to-end. Bench crate + multi-level reprojection report shipped
  (PR #49). A 2026-06-11 workspace review (internal) confirmed the
  extend-don't-rebuild verdict for the app. **B3c coverage completed
  2026-06-14:** B3c-1/2/3 wired the 8 topologies + charuco/laser; B3c-4 added
  the puzzleboard + ring-grid detectors, so **all four target detectors now
  calibrate end-to-end** (charuco dedup deferred, see backlog). **B3d
  (manifest UX) completed 2026-06-14:** the `sniff_folder` heuristic + CLI
  (B3d-1) and the "Sniff folder" / `_unresolved` / AskUser-modal front end
  (B3d-2) let the user point at a foreign folder and edit an auto-generated
  manifest. B-explore (pre-calibration dataset browse) + B-infra (TS codegen,
  tests) are the remaining app slices.
- **New tracks (2026-06-11):** V (real-data validation on the private rtv3d dataset),
  O (apex-solver optimization backend), M (camera-model expansion — supersedes the
  former "new camera models out of scope" line).
- **In-flight PRs:** none blocking. The MVG crate split (formerly PR #28 /
  `mvg` branch) was **superseded** by a fresh additive port — `vision-geometry`
  + `vision-mvg` landed on `main` 2026-06-14 (C1), see
  [ADR 0015](adrs/0015-mvg-ceiling.md). The stale `mvg` branch / PR #28 can be
  closed.

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
- **B-explore — dataset exploration.** Browse a dataset *before*
  calibrating: image grid per camera/pose, detection overlay from the
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
    so runner suggestions stay raw click-to-apply values. B3d complete; B-explore
    / B-infra are the remaining app slices.
- **B3e — iteration polish.** Cancellability for long solves;
  progress event streaming; multi-pose residual stats panel;
  cross-camera residual matrix; experiments directory storing
  `(dataset.toml, config.json, export.json)` tuples for
  reproducibility.

**Phase 3 — deferred until B3 stabilises.**

- LLM-backed manifest inference (separate ADR; opt-in, behind API
  key configuration).
- Init-failure diagnosis sweeps (perturbed re-runs).
- Signed installers per OS.
- Infra ratchet: ts-rs (or specta) TS codegen replacing hand-written
  wire types + an export discriminator tag; `resource_dir`-based
  presets; Vitest unit + Playwright smoke tests.

### Track V — Real-data validation: rtv3d (V1–V4 DONE 2026-06-11)

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
- **V5** Full `RigLaserlineDevice` + joint-BA runner in the bench (today the
  bench only profiles laser extraction).
- **V6** Settle the dataset's absolute scale against the head's mechanical
  camera spacing.

### Track O — Optimization backends (NEW 2026-06-11)

`OptimBackend` (ADR 0008) gets a second real implementation:
[apex-solver](https://crates.io/crates/apex-solver) 1.3 (LM/GN/DogLeg, Lie-group
support), behind an `apex-solver` cargo feature in `vision-calibration-optim`.

- **O1** `ApexSolverBackend` implementing `OptimBackend`, mirroring
  `compile_factor` from the tiny-solver backend. Pre-verify: whether apex-solver
  accepts generic factors (its API is graph-flavored), SE3 quaternion order vs
  our `[qx,qy,qz,qw,tx,ty,tz]` (round-trip unit test), S2 manifold availability
  (fallback: R3 + renormalize), robust-loss coverage.
- **O2** Backend A/B in bench: param parity < 1e-4 on synthetic IR, final cost
  within 0.1 % on bench datasets, timing comparison on rtv3d.
- **O3** Drop the `BackendKind::Ceres` stub.

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
  Brown-Conrady production paths are unchanged. Remaining: **M-WIRE**, the
  user-facing pipeline-selection plumbing (config → builder, fix-mask
  generalization, per-model init/pack/export), deferred to a supervised slice.
- **M4** Kannala-Brandt fisheye (new projection slot + linear-init changes —
  the biggest lift; last).

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
  `linear`→`vision-geometry` de-duplication is a deliberate follow-up; ADR 0015.
- **C2** N-view triangulation + nonlinear refinement.
- **C3** Bundle adjustment with frozen intrinsics, free poses, free structure.
- **C4** Stereo rectification — including **Scheimpflug-aware rectification** (genuinely
  novel for this project).
- **C5** Dense matcher integration behind a `dense-opencv` feature flag, wrapping
  `opencv-rust` SGBM.
- **C-UI** MVG visualizations layered into the Tauri app (point clouds, depth maps,
  rectified pairs).

### Track D — Earn v1.0 (continuous ratchet)

- **D1** Typed errors only — no `String`-typed escape hatches in public APIs.
- **D2** Doc-warning-free, MSRV pinned at v1.0 cut (currently 1.93).
- **D3** Python binding parity audited at every minor version bump.
- **D4** v1.0 release once the puzzle rig runs green end-to-end via the Tauri app,
  PR #28 + the diagnose viewer + C4 have all landed, and the API has been stable across two minor
  releases.

## Load-bearing path

**V1 → V2 → V3 (prove the library on rtv3d) → V4 + B3c coverage (rig +
laserline + charuco first) → B-laser visualization → B3c-4 (all four target
detectors) → B3d manifest UX → B3e polish.** B3c coverage **and** B3d
manifest UX are complete as of 2026-06-14 (all 8 topologies + all 4 detectors
wired; sniff-folder → editable auto-manifest shipped). **B-explore /
B-infra / B3e are the remaining app slices**; the load-bearing app path is no
longer blocking. O1/O2 (apex-solver) are parallelizable with the V-track; M0
(factor generification) is done and M1–M4 can proceed. C resumes now that
B3c has landed; D is a continuous ratchet.

## Out of scope (explicit)

- Camera models beyond the M-track set (omnidirectional / MEI, double-sphere,
  telecentric, spline). Defer until a concrete project demands one.
- In-house dense stereo matching. External (`opencv-rust` SGBM) only.
- Full structure-from-motion (incremental SfM, pose graph, loop closure).
- `rerun.io` / `egui` as the UI.
- OSS-grade community surface and multi-platform install docs (internal-first; defer until
  the tool earns it).
- What-if interactive re-optimize (B6 stretch at earliest).

## See also

- [ADR index](adrs/README.md) — design records.
- [Tutorials](tutorials/README.md) — hands-on walkthroughs for new users.
- [MSRV notes](MSRV.md) — why the lockfile is frozen below latest releases.
- Per-track ADRs:
  [`0011-manual-initialization-workflow.md`](adrs/0011-manual-initialization-workflow.md) (A1, landed in PR #32);
  [`0012-per-feature-reprojection-residuals.md`](adrs/0012-per-feature-reprojection-residuals.md) (A2, landed in PR #33 + #35);
  [`0013-rig-family-sensor-axis-refactor.md`](adrs/0013-rig-family-sensor-axis-refactor.md) (A6, landed in PRs #36 + #37 + #38);
  [`0014-tauri-desktop-app.md`](adrs/0014-tauri-desktop-app.md) (B0, landed in PR #40);
  [`0016-dataset-manifest.md`](adrs/0016-dataset-manifest.md) (B3a, PR #45);
  [`0017-detection-cache.md`](adrs/0017-detection-cache.md) (B3a, PR #45);
  [`0018-schema-driven-ui.md`](adrs/0018-schema-driven-ui.md) (B3a, PR #45);
  [`0019-fail-fast-on-ambiguity.md`](adrs/0019-fail-fast-on-ambiguity.md) (B3a, PR #45);
  `0015-mvg-ceiling.md` (C1, pending).
