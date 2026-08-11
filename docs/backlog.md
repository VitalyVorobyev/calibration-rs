# Backlog

Execution status for agent-driven implementation tasks. Each completed task gets
a dated, concise completion note here (the durable record — pointers to
ADRs/notes/PRs where they exist; `docs/report/` is retired, historical entries
archived under `docs/internal/archive/report/`) and a task-scoped commit
(AGENTS.md §11).

Open tasks derive from the
[2026-06-11 workspace review](internal/archive/report/2026-06-11-workspace-review.md)
(findings F1–F6), the V/O/P/M/C/D/B tracks, and — since 2026-07-02 — the
**production-grade program** (Tracks S/Q/R + B-QUAL/B-UX/B-DIST below; phase
plan and exit criteria in the [ROADMAP](ROADMAP.md)). Sequencing:
S1→S2→S3→S4→Q2→{Q3,Q4,Q5,Q7}; Q4→R2→R3→{R5, B-QUAL2};
B-QUAL1→…→B-QUAL4→B-UX1→B-UX2→B-DIST.

## S — Device-spec → seed initialization (Phase I)

Structured "device specification → initialization seed" layer replacing the
hand-coded per-example constants. Spec-seeded init is the **official
acceptance route** (ADR 0022); from-scratch stays experimental.

- [x] S1-SPEC-ADR - **Done 2026-07-02.** ADR 0023 accepted: `DeviceSpec`
  schema + `device_seed` derivation fns, re-exported via the facade. See
  ADR 0023, `docs/DESIGN-device-spec.md`.
- [x] S2-SPEC-INTRINSICS - **Done 2026-07-02.** Both intrinsics examples load
  `privatedata/<ds>/spec.json` and derive seeds via
  `device_seed::scheimpflug_seed`; env-knob focal overrides deleted. Gate:
  rtv3d_ref all 6 cams ≤0.5 px; rtv3d_ringgrid 5/6 ≤0.5 px (cam1 at 0.5008 px,
  a pre-existing seed-independent knife-edge, tracked under Q3).
- [x] S3-SPEC-EXTRINSICS - **Done 2026-07-02.** New
  `device_seed::rig_layout_seed` builds the coupled `RigHandeyeRigManualInit`
  from the spec's nominal mounts + a measured target pose (ADR 0011 coupling
  honored). `rtv3d_rig.rs` defaults to `RTV3D_SEED=spec`. Gate: all
  beat-the-oracle verdicts PASS in both seed modes, converging to the same
  optimum (seeds change the start point, not the answer).
- [x] S4-ACCEPT-HARNESS - **Done 2026-07-02.** `calib-bench accept` iterates
  every registered dataset, runs the seeded official route, and hard-gates
  per entry; absent datasets print `UNAVAILABLE`, ungated entries print
  `NO-GATE`. Full suite: **19 passed, 0 failed, 2 UNAVAILABLE**; CI runs the
  cheap stereo subset.

## Q — Algorithmic soundness: proofs + regression (Phase I)

A **proof pack** per algorithm family: math note (`docs/notes/<family>.md` —
model equations, cost, identifiability/degeneracy, gauge, noise sensitivity),
synthetic-GT matrix test (parameter grid × noise levels), property tests where
a real invariant exists, committed regression Fit record + gate. Init routines
additionally get a convergence-basin study. Families: planar intrinsics,
Scheimpflug intrinsics (seeded), rig extrinsics, hand-eye, laserline bundle,
ringgrid detection/bias, rectification (short note only — C4 gate exists),
two-view/triangulation.

- [x] Q1-PROOF-STANDARD - **Done 2026-07-02.** Standard codified in
  `docs/notes/README.md` (math note + synthetic-GT matrix test + property
  tests + committed Fit record + basin study for init). Template pack:
  `docs/notes/planar-intrinsics.md` +
  `crates/vision-calibration/tests/planar_intrinsics_matrix.rs` (18 cells).
- [x] Q2-REGRESSION-WIRING - **Done 2026-07-02.** New
  `vision_calibration_bench::baseline` module: committed `baselines/<id>.json`
  snapshots + `compare_to_baseline` (5% default tol); `calib-bench accept`
  compares every run against its baseline, `--freeze-baselines` to accept a
  changed fit. All 19 baselines frozen; trip-tests confirm both the hard gate
  and the drift comparison catch deliberate kernel bias.
- [x] Q3-RINGGRID-BIAS - **Done 2026-07-08** — resolved as
  already-corrected-at-detector-level (no pipeline correction; would
  double-correct). `ringgrid` 0.7 defaults to
  `CircleRefinementMethod::ProjectiveCenter` (Wang 2019), removing the
  projective bias before calibration sees the centers; remaining ~0.47 px
  floor is small-marker localization noise, detector-side. See
  `docs/notes/ringgrid-bias.md`; ringgrid accept gates tightened 1.0→0.7 px.
- [x] Q4-MWIRE-SCHEIMPFLUG - **Done 2026-07-04.** `distortion_model:
  DistortionKind` added to `ScheimpflugIntrinsicsConfig` and rig
  `SensorMode::Scheimpflug` (BC5-typed); `fix_distortion` masks translate by
  name onto each model's packed layout. **Measurement:** under the
  production `radial_only` mask, model choice alone does not move the
  seeded floor. Committed bench defaults stay BC5, zero baseline drift.
- [x] Q5-RTV3D-SCALE - **Done 2026-07-08** (absorbs V6-SCALE). Settled: no
  real scale ambiguity — 5.2 mm cell size confirmed correct. The
  "90.1 mm vs ~98.5 mm oracle" gap was a bug comparing against
  hand-eye-stage (pre-laser) extrinsics, which under-determine scale by
  ~10%; fixed to compare against joint-BA extrinsics (98.21±0.41 mm vs
  oracle's 98.13±1.10 mm, 0.08% agreement). See `docs/notes/rtv3d-scale.md`.
- [x] Q6-BASIN-STUDY - **Done 2026-07-04.** New `calib-bench basin` command:
  three independent per-axis sweeps (focal, tilt, principal point) over the
  ADR 0022 seeded route on both private rtv3d families. **Decision rule
  outcome: PASS** — both families clear the all-camera-pass envelope at
  focal ×[0.85, 1.30] and tilt [−4°, +4°], comfortably exceeding the ±5%/±2°
  requirement; no reopening of the Phase A sweep design.
- [x] Q7-SOLVER-DEDUP - **Resolved-as-already-done 2026-07-04** (absorbs the
  C1-FOLLOWUP remainder). The dedup of `homography`/`epipolar`/
  `camera_matrix`/`triangulation` between `linear` and `geometry` already
  landed in PR #72 (net −1811 LoC, commit `6818cde`); `linear` now depends on
  `vision-geometry` directly. Existing GT-relative tests + Q2 baselines
  already gate numeric drift (no new golden-pin suite needed).
- [x] Q8-PROOF-PACKS - **Done 2026-07-08.** Five proof-pack notes (hand-eye,
  rig-extrinsics, laserline-bundle, two-view-triangulation, rectification) +
  four matrix/property test files (GT-grid × noise sweep + gauge/degeneracy
  properties). Findings: `fix_first_rig_pose` redundant with the
  reference-camera gauge fix (R2/ADR 0024 candidate); linear rig init needs
  a direct reference↔camera co-visibility edge.

## R — API/config/design revision (Phase II)

- [x] R1-API-AUDIT - **Done 2026-07-08.** API-surface audit over the facade:
  deleted dead shims, merged duplicate `ScheimpflugFixMask`, consolidated
  residual/histogram diagnostics under `vision_calibration::analysis`;
  **app/src-tauri is now facade-only**.
- [x] R2-CONFIG-ADR - **Done 2026-07-08**
  (`docs/adrs/0024-config-vocabulary.md`). One config vocabulary: grouped
  shape everywhere, `init_iterations` the one name, per-parameter masks the
  one fix idiom; D2 (delete `fix_first_rig_pose`/`fix_first_camera_extrinsic`)
  grounded in Q8 gauge evidence.
- [x] R3-CONFIG-IMPL - **Done 2026-07-08** in three waves: executed ADR 0024
  across all 8 configs (pipeline, bench, app, Python, examples-private,
  facade tests); net code shrink (wave 3 alone −293 LoC). Rename gate: zero
  old-name hits in code (tutorials/book updated in R6).
- [x] R4-LINEAR-ERRORS - **Done 2026-07-08.** `anyhow` dropped from
  `vision-calibration-linear`; `handeye.rs`/`extrinsics.rs` return
  `linear::Error` (new `NoValidMotionPairs` variant). Public signatures
  unchanged.
- [x] R5-PY-PARITY - **Done 2026-07-09** (G1 MVG bindings deferred per user
  decision 2026-07-08, pending a consumer). G0: `run_rig_handeye_laserline`
  bound (typed wrapper + `RigHandeyeLaserline{Dataset,View,CalibrationConfig,
  BaConfig,Result,PerCamStats}` models + stubs) — was the one unbound facade
  workflow. G2: `distortion_model` on planar/Scheimpflug configs and
  `sensor: SensorMode` (`PinholeSensorMode`/`ScheimpflugSensorMode`) on both
  rig configs; typed results made model-polymorphic (`PinholeCamera` /
  `PinholeScheimpflugCamera` with a `Distortion` union of BC5/Division1/
  Rational8/ThinPrism9) so every `distortion_model` value round-trips typed;
  rig/hand-eye/laserline results stay strictly BC5-typed. Guard:
  `scripts/check_binding_parity.py --check` (facade `run_calibration` module ↔
  `run_<module>` pyfunction) wired into CI next to the pyi coverage check.
  Facade now re-exports `DistortionKind` (was a public field type on three
  facade configs with no facade path). 40 runtime/schema tests.
- [x] R6-TUTORIALS - **Done 2026-07-09.** New `docs/tutorials/`:
  `distortion-model-selection.md` (Q4-grounded, verified against the
  `planar/scheimpflug_distortion_models` test numbers),
  `single-cam-handeye.md` (kuka_1-backed walkthrough incl. `spec.json` /
  device-spec section and the honest dataset.toml loading path), and
  `app-walkthrough.md` (five-workspace desktop app tour). Existing six
  verified already on ADR-0024 shapes (zero stale fields); one genuinely
  stale `pixel_to_gripper_point` facade path fixed in the puzzle-130x130
  walkthrough (R1-audit casualty). Index updated.
- [x] R7-EXPORT-DISCRIMINATOR - **Done 2026-07-10.** Added a shared
  `ExportKind` unit enum (`vision-calibration-pipeline` `common/export_kind.rs`,
  `#[serde(rename_all = "snake_case")]`, one variant per problem type; re-exported
  as `vision_calibration::common::ExportKind`) and a **required** `kind` field —
  the first field — on all eight `*Export` structs, set on construction.
  Deserialize is strict (no `serde(default)`): a missing tag errors, making the
  discriminator a hard contract. The enum derives `JsonSchema`, so it flows
  through `emit_schemas` → `diagnose_wire.json` → `diagnose-wire.ts` as a
  `"planar_intrinsics" | …` union. App `detectExportKind` collapsed from ~50
  lines of field probes to a validated read of `data.kind`; a label
  `Record<ExportKind, string>` is now the single source of both labels and the
  recognised-kind set (pinned to the generated union). Regenerated committed
  exports (`data/stereo{,_charuco}/viewer_export.json`) and every hand-built
  export JSON in tests; Python export parsers ignore `kind` harmlessly (40 py
  tests green). ADR 0018 amended. Gates: fmt/clippy/test/doc, src-tauri
  fmt/clippy/test + `emit_schemas --check`, app lint/format/typecheck/vitest
  (27)/e2e (7), TS-gen idempotent.

## B-QUAL / B-UX / B-DIST — app to production grade (Phase III)

- [x] B-QUAL1-LINT-CI - **Done 2026-07-10.** ESLint 9 flat config
  (typescript-eslint `recommendedTypeChecked`, react-hooks v5) + Prettier
  (printWidth 90, churn-minimized empirically); lint/format/typecheck
  scripts; 27 files formatted, 10 with genuine fixes — lint caught a real
  `useMemo` defeat in `Scene.tsx` (fresh `?? []` arrays), an `any`-typed
  parse leak in RunWorkspace, and a dead initializer in AppShell. New CI
  jobs `app-frontend` (bun: lint/format:check/typecheck/vitest) and
  `app-src-tauri` (fmt/clippy/test via `--manifest-path`).
- [x] B-QUAL2-TSRS - **Done 2026-07-10** (mechanism changed from ts-rs to
  schemars per ADR 0018 amendment — single source of truth, no new dep in
  published crates). `JsonSchema` extended to all 8 `*Export` types +
  closure; `emit_schemas` bin (feature `schema-export`) → draft-07 schema →
  `json-schema-to-typescript` → `app/src/types/generated/diagnose-wire.ts`
  (33 interfaces, idempotent). `inferExportKind` deleted; `detectExportKind`
  grounded in generated types (fixed latent misclassification of planar/
  scheimpflug/laserline exports). CI drift checks in both app jobs.
- [x] B-QUAL3-COMPONENT-TESTS - **Done 2026-07-10.** 35 vitest tests
  (jsdom via `environmentMatchGlobs`): ConfigForm against the real planar
  schema fixture (number/bool/enum/oneOf edits propagate), Diagnose mounts
  for all 8 export kinds, Run happy path over a mocked `invoke` seam
  (`@tauri-apps/api/mocks`).
- [x] B-QUAL4-SMOKE - **Done 2026-07-10.** Playwright smoke (7 tests, CI
  chromium in `app-frontend`): boot, five workspaces mount with zero console
  errors, Diagnose fixture renders end-to-end through a
  `__TAURI_INTERNALS__` init-script mock (mocked-IPC boundary documented in
  `app/README.md`). Absorbed the `resource_dir` preset item: hard-coded
  `REPO_ROOT` replaced by `repo_root_cmd` (`CARGO_MANIFEST_DIR` walk-up) +
  repo-root-relative preset paths + a guard test rejecting absolute paths.
  A tiny bundled-dataset Run stays out of CI (private datasets can't ship);
  the mocked Run happy path covers the UI flow.
- [x] B-UX1-DESIGN-SYSTEM - **Done 2026-07-10.** `app/src/components/ui/`
  set (Button with ARIA-pressed toggles, Panel, SectionHeader, Select,
  Table, Banner, Badge, EmptyState + shared ZoomControls); all five
  workspaces, AppShell, and configForm migrated (+347/−593). The audit
  found three real bugs, not just drift: raw `var(--brand)` (an H S% L%
  triplet) used as a CSS color — the Run button fill was silently invalid;
  the never-declared `bg-bg` class left every schema-form field
  transparent; light-mode `--brand` failed WCAG AA (fixed 55%→34% L).
  `--success`/`--warning` tokens added for both themes; design note +
  "use ui/, don't hand-roll" rule in `app/README.md`.
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
- [x] B-DIST-INSTALLERS - **Done 2026-07-10** (scoped to unsigned bundles +
  CI artifacts only, per standing decision — no code-signing certs
  available). `bun run tauri build` now produces a launchable macOS
  `.app`/`.dmg` under `app/src-tauri/target/release/bundle/`: fixed
  `tauri.conf.json` (`bundle.active: true`, `mainBinaryName`, standard
  `icon` array) and generated the missing `.icns`/`.ico`/PNG icon set via
  `bunx tauri icon` (desktop targets only — iOS/Android/Windows-Store
  variants discarded, no mobile target configured). Fixed a real conflict
  between two `[[bin]]` targets: Tauri's bundler copies every declared
  `[[bin]]` target unconditionally regardless of `required-features`, so
  the schema-export gate on `emit_schemas` (B-QUAL2) made every release
  build fail with "does not exist". Fix: `default-run` in `Cargo.toml`
  disambiguates the *main* binary for tools that don't honor
  `required-features`, and `emit_schemas.rs` itself now always compiles
  (`required-features` dropped from its `[[bin]]`) but is a two-line stub
  without `--features schema-export` — the `schemars` dependency stays
  feature-gated, so `tauri dev`/default `clippy`/`test` are unaffected.
  New `.github/workflows/app-bundle.yml` (`workflow_dispatch` + `v*` tag
  trigger, not in `ci.yml`'s per-PR path): macOS (`.app`/`.dmg`) and Linux
  (`.AppImage`/`.deb`, reusing `ci.yml`'s webkit2gtk/appindicator/rsvg/
  patchelf deps) jobs, `actions/upload-artifact@v4`. Deferred, with
  prerequisites: macOS signing needs an Apple Developer ID Application
  certificate + notarization (`xcrun notarytool`, Apple Developer Program
  membership); Windows signing needs an EV/OV code-signing certificate
  from a CA. Updater/update-channel also deferred (needs a signed
  manifest + key pair on top of the above). Local DMG creation could not
  be end-to-end verified in the sandboxed dev shell — `create-dmg`'s
  Finder-styling AppleScript step hit "AppleEvent timed out" (no
  interactive Aqua session available there); the `.app` itself built,
  launched, and quit cleanly. `macos-latest` GitHub-hosted runners have a
  full desktop session and are the standard environment for this exact
  Tauri flow, so CI is expected to succeed — flagged as a known risk
  category in the workflow's comments regardless.

## V — rtv3d validation

- [x] V1-EXAMPLE - **Done 2026-06-11.** `rtv3d_rig` example: ChArUco rig
  hand-eye. Findings: hand-eye is **EyeToHand** (dataset.json is wrong — 3×
  residual evidence), cell size **5.2 mm**.
- [x] V2-LASER - **Done 2026-06-11.** rtv3d full pipeline joint BA: 1.16 px
  mean reproj, laser point-to-plane σ 0.017–0.031 mm over 1672–1868
  points/camera.
- [x] V3-REPORT - **Done 2026-06-11.** Beat-the-oracle: all criteria PASS
  (reproj < oracle all cams, σ < oracle all planes, extrinsic scale within
  10%). See
  `docs/internal/archive/report/2026-06-11-rtv3d-validation.md`.
- [x] V4-BENCH - **Done 2026-06-11.** `rtv3d` registry entry (+laser
  extraction profile), smoke-tested (1.86 px unseeded). 2026-06-12: redundant
  `rtv3d_2` deleted, `rtv3d_1` renamed to `rtv3d`.
- [x] V5-BENCH-LASER - **Done 2026-06-13.** Full `RigLaserlineDevice` + joint
  BA runner in `bench/src/run.rs`: 1.19 px mean reprojection, laser
  point-to-plane RMS 0.018–0.035 mm over 10,797 points. See
  `docs/internal/archive/report/2026-06-13-V5-BENCH-LASER-rtv3d-calibration-quality.md`.
- [x] RTV3D-FROZEN-LASER-POSES - **Done 2026-06-13.** `RigLaserlineDevice`
  now prefers upstream `rig_se3_target` by view token, fixing a coherent
  54 px reprojection drift in the rtv3d laser preset. See
  `docs/internal/archive/report/2026-06-13-RTV3D-FROZEN-LASER-POSES-frozen-rig-laserline-poses.md`.
- [x] RTV3D-JOINT-LASERLINE-APP - **Done 2026-06-13.** rtv3d laser preset
  runs `RigHandeye → RigLaserlineDevice → optimize_rig_handeye_laserline`
  directly from `dataset_laser.toml`, fixing `cx/cy` in joint BA. See
  `docs/internal/archive/report/2026-06-13-RTV3D-JOINT-LASERLINE-APP-joint-app-topology.md`.
- [x] RTV3D-LASER-CUTS - **Done 2026-06-13.** 3D viewer draws clipped
  active-pose target-plane intersections for the six laser planes. See
  `docs/internal/archive/report/2026-06-13-RTV3D-LASER-CUTS-viewer-laser-cuts.md`.
- [x] RTV3D-INTRINSICS-FOCUS - **Done 2026-06-14.** `calib-bench diagnose
  intrinsics` isolates per-camera Scheimpflug intrinsics; floor
  0.747–1.199 px (all 6 cams fail the raw <0.4 px gate), pointing to a
  detector/target/model floor, not rig-chain error. See
  `docs/internal/archive/report/2026-06-14-RTV3D-INTRINSICS-FOCUS-scheimpflug-intrinsics.md`.
- [x] V6-SCALE - Absorbed into Q5-RTV3D-SCALE 2026-07-02, **settled (no
  metric anchor needed) 2026-07-08** — see Q5's note and
  `docs/notes/rtv3d-scale.md`.
- [~] V7-RTV3D-INTRINSICS-FLOOR - **PARKED** (user call 2026-06-14;
  disposition confirmed 2026-07-02). Drive the rtv3d from-scratch
  reprojection floor below 0.4 px, or prove the blocking term — isolated to
  detector/target/model, not rig-chain. Seeded init (ADR 0022) is the
  official acceptance path. **Do not reopen unprompted.**
- [x] V8-RINGGRID - **Done 2026-06-30.** Full calibration on the private
  `rtv3d_ringgrid` dataset (different physical rig, no laser/oracle).
  Root-cause: nalgebra's iterative `from_matrix` mis-converges on exact 180°
  robot rotations, corrupting hand-eye — fixed with exact
  `Rotation3::from_matrix_unchecked`/SVD polar conversion. Bumped `ringgrid`
  0.6→0.7. Ring-grid reprojection median ≈4.7 px vs puzzleboard 0.25–0.30 px
  (ellipse-center bias + sparse markers, detector-side follow-up).

## O — apex-solver backend (O1/O2 WON'T-DO 2026-07-04; O3 DONE)

Pre-verify failed 2026-06-14; closed won't-do 2026-07-04. apex-solver 1.3
lacks an S2 manifold, documented robust losses, and documented SE3/Jacobian
conventions, and its solver core duplicates the homegrown LM + faer stack —
not worth bridging. Pre-verify findings:
`docs/internal/archive/report/2026-06-14-O1-apex-solver-preverify.md`.

**Revive triggers:** apex-solver ships autodiff/S2-manifold/robust-loss
support, or a solver-trust/perf need the Q-track baselines can't address;
preferred target then is an autodiff-native stack (`factrs`/`num-dual`), not
apex-solver 1.x.

- [-] O1-BACKEND - **WON'T-DO 2026-07-04.** `ApexSolverBackend` closed —
  bridgeable via a dual-number adapter but not worth it; see track note above
  for rationale/revive triggers.
- [-] O2-AB - **WON'T-DO 2026-07-04** (backend A/B validation; depended on O1).
- [x] O3-CERES - **Done 2026-06-15.** Dropped the dead `BackendKind::Ceres`
  stub + orphaned `Error::numerical` helper; `BackendKind` is now
  single-variant. See
  `docs/internal/archive/report/2026-06-15-O3-CERES-drop-ceres-stub.md`.

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

- [x] P1-SVD-SWEEP - **Done 2026-06-17.** Replaced `svd(true,true)` with
  hang-proof `AᵀA`+eigen (null-space) / ridge-regularized normal-eq (least
  squares) across all ~20 sites in `linear`/`geometry`/`mvg`, centralized
  behind `math::null_space`/`ridge_lstsq`/`project_to_so3`. See
  `docs/internal/archive/report/2026-06-16-P1-SVD-SWEEP-finish-centralize.md`.
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
- [x] P4-CRITERION - **Done 2026-06-16.** `criterion` dev-dep + `[[bench]]`
  targets for `linear/benches/linear_init.rs` (homography DLT 225pts ~8.6µs,
  was a >15min hang) and `optim/benches/ba_iter.rs` (~16.9ms per-camera BA
  solve). See
  `docs/internal/archive/report/2026-06-16-P4-CRITERION-hot-path-benches.md`.
- [x] P5-STAGE-TIMING - **Done 2026-06-16.** Additive `StageTiming` (6
  optional per-stage `*_ms` fields, serde-backward-compatible) on the bench
  `Timing` struct; `run_rig_extrinsics`/`run_rig_handeye` now time each
  optimize sub-stage. See
  `docs/internal/archive/report/2026-06-16-P5-STAGE-TIMING-bench-per-stage.md`.
- [x] P6-PERCAM-CONVERGENCE - **Done 2026-06-16.** Tilt-aware linear
  Scheimpflug initializer + rig-handeye auto-recovery pass; `rtv3d_ref_rig`
  from-scratch reaches 0.4057 px mean reprojection (per-cam ≤0.4725 px), all
  `tau_x` within ~1.5° of oracle. See
  `docs/internal/archive/report/2026-06-16-P6-PERCAM-CONVERGENCE-tilt-aware-init.md`
  (supersedes the `-diagnosis.md` report).
- [x] P7-SCHEIMPFLUG-SEEDED-DEFAULT - **Done 2026-06-17.** Made user-seeded
  Scheimpflug intrinsics the default (ADR 0022): trusts the seed instead of a
  cold multi-start sweep, frees the pose gauge, escapes the spurious `k1≈0`
  local minimum via `k1` multi-start. All 6 `rtv3d_ref` cameras pass the hard
  ≤0.5 px gate from one coarse shared seed.

## M — camera models (gated on M0)

- [x] M0-GENERIFY - **Done 2026-06-12** (ADR 0020). `FactorKind` = 4 families
  (`ReprojPoint`, `LaserPointToPlane`, `LaserLineDistance`, `Se3TangentPrior`)
  with `CameraModelDesc` as data, ZST-kernel monomorphization; net ~-1.9k LoC
  in optim, numerics bit-identical on all production paths.
- [x] M1-RATIONAL / M2-THINPRISM / M3-DIVISION - **Additive layer done
  2026-06-14.** `RationalPolynomial`, `ThinPrism`, `Division` distortion
  models added at core/optim IR layers, strictly additive (BC5 production
  paths byte-identical). See
  `docs/internal/archive/report/2026-06-14-M-distortion-models.md`.
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

- [x] C1-CRATES - **Done 2026-06-14.** Landed `vision-geometry` (20 tests) +
  `vision-mvg` (31 tests, optional `refine` feature), ported fresh from the
  stale `mvg` branch. ADR 0015 caps the MVG ceiling (no dense matcher, no
  full SfM). See
  `docs/internal/archive/report/2026-06-14-C1-mvg-crates.md`.
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
- [x] C2-TRIANGULATION - **Done 2026-06-17.** N-view triangulation +
  nonlinear refinement (`triangulate_point`, `refine_point`,
  `vision-mvg::triangulate_nview`); migrated the last `svd(true,true)`
  leftover onto `core::linalg::null_space`, closing P1.
- [x] C3-BA - **Done 2026-06-21** (PR #73). `vision-mvg::bundle_adjust`
  (behind `refine`): frozen-intrinsics tiny-solver LM, SE3 pose blocks + 3D
  point blocks, `fix_first_camera` gauge (default). 9 synthetic-GT tests.
- [x] C4-RECTIFY - **Done 2026-06-21** (PR #74, the D4 gate). New
  `vision-mvg::rectification::rectify_stereo_pair` — pre-multiplying by
  `H_tilt⁻¹` collapses Scheimpflug to frontal pinhole, then standard
  Fusiello/Bouguet applies. `rtv3d_ref_rectify` gate: worst row disagreement
  3.4e-13 px across all oracle camera pairs.
- [x] C-FACADE-MVG - **Done 2026-06-21.** New `vision_calibration::mvg`
  module re-exporting the full MVG surface; `bundle_adjust` behind facade
  `refine` feature. Surface locked by `tests/facade_compile_surface.rs`.
- [x] C-MVG-TUTORIAL - **Done 2026-06-21.**
  `docs/tutorials/multiple-view-geometry.md` + runnable
  `examples/mvg_two_view.rs` (pose recovery → BA → rectification),
  synthetic-GT end-to-end demo.
- [~] C5-DENSE - Dense stereo matcher. **Direction reset + implementation
  done 2026-06-21** (user-supervised): amended ADR 0015 — the matcher ships
  pure-Rust in `vision-mvg::dense` (block matching + SGM aggregation, ZNCC
  via summed-area tables, no new deps), scored by a bench harness; synthetic
  slanted-plane recovery hits 94% density at 0.18 px RMS. OpenCV SGBM
  baseline **closed as env-blocked** 2026-07-02 (no OpenCV env available).
- [x] C-UI-DEPTH - **Done 2026-06-21.** New `compute_disparity` Tauri command
  + React `DepthWorkspace` (pose/camera steppers, SGM toggle, view-mode
  switch); rectifies + dense-matches a synchronized pair, returns PNGs +
  metrics. End-to-end Rust test on the committed `data/stereo` rig.
- [x] C-UI-POINTCLOUD - **Done 2026-06-21.** `compute_disparity` now
  reprojects disparity to a metric depth colormap + grid-subsampled 3D point
  cloud; new `depth`/`3D` view modes with a lazy-loaded R3F `PointCloudView`
  (Three.js code-split, main chunk 1203→469 kB).

## D — Earn v1.0

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
- [x] D2-DOCS - **Done** (PR #69). `missing_docs = warn` enforced
  workspace-wide; all public items documented.
- [x] D1-TYPED-ERRORS - **Done** across PR #72 (geometry/mvg) + PR-1 (optim)
  + PR-2 (detect/pipeline). Dropped `anyhow` from every published crate's
  public surface onto `thiserror` enums; caught and fixed a NaN-rejection
  regression (`ensure!(x>0.0)` → hand-written `if` silently accepted NaN)
  across 10 sites.
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
- [x] D5-DOCS-TRUTH - **Done 2026-07-02.** Documentation truth pass: ROADMAP
  refreshed to HEAD, this backlog gained the S/Q/R/B-QUAL sections, ADR
  status notes added, README/AGENTS.md crate tables brought to the 10-crate
  reality, stale internal handoffs archived to `docs/internal/archive/`.

## B — app (extend; sequencing serves V-track)

- [x] B3C-PUZZLEBOARD - **Done 2026-06-14.** `PuzzleboardDetector` wraps
  `calib-targets::detect_puzzleboard` behind the sealed `Detector` trait;
  `dataset_runner` resolves `"puzzle_<R>x<C>"` layouts. See
  `docs/internal/archive/report/2026-06-14-B3C-PUZZLEBOARD-puzzleboard-detector.md`.
- [x] B3C-RINGGRID - **Done 2026-06-14.** `RinggridDetector` wraps `ringgrid`
  0.6 behind the sealed `Detector` trait; `TargetSpec::Ringgrid` realigned
  (breaking) to the real hex-lattice `BoardLayout` model. All four target
  detectors now calibrate end-to-end. See
  `docs/internal/archive/report/2026-06-14-B3C-RINGGRID-ringgrid-detector.md`.
- [x] B3C-CHARUCO-DEDUP - **Closed as blocked 2026-07-08** (R1 audit
  triage). Clean design known (bench/examples delegate to
  `vision-calibration-detect`'s canonical `CharucoDetector`), but the
  byte-identical gate needs private golden datasets absent from CI. Revive
  trigger: commit a synthetic ChArUco fixture.
- [x] B3C-RIG - **Already shipped** in B3c-1/B3c-3 (2026-06-12); checkbox was
  stale, verified 2026-06-14. All 8 topologies dispatch via an exhaustive
  match in `app/src-tauri/src/run.rs`; no code change needed.
- [x] B3D-SNIFF - **Done 2026-06-14.**
  `vision_calibration_dataset::sniff_folder` infers structurally-unambiguous
  manifest fields from a dataset directory, leaving ambiguous ones in
  `_unresolved` (ADR 0019); new `generate-manifest` CLI + Tauri command share
  the inference. See
  `docs/internal/archive/report/2026-06-14-B3D-SNIFF-heuristic-manifest-sniffer.md`.
- [x] B3D-UX - **Done 2026-06-14.** "Sniff folder" button, `UnresolvedNotice`
  strip with vendor-aware hints, red unresolved-count badge, Run blocked
  while unresolved, `AskUserModal`. See
  `docs/internal/archive/report/2026-06-14-B3D-UX-manifest-sniff-unresolved-askuser.md`.
- [~] B-LASER - **Re-scoped into B-UX2-ELEVATION** 2026-07-02: laser-pixel
  overlay in Diagnose compare mode, point-to-plane (mm) panel, single-cam
  laser plane in the 3D viewer. (Core laser views shipped 2026-06-12.)
- [~] B-EXPLORE - **Re-scoped into B-UX2-ELEVATION** 2026-07-02: per-camera/
  pose image grid, detection-cache overlay, board coverage map.
- [~] B-INFRA - **Absorbed 2026-07-02** into B-QUAL1-LINT-CI (CI entry),
  B-QUAL2-TSRS (ts-rs codegen), B-QUAL4-SMOKE (`resource_dir` presets +
  Playwright). Vitest unit-slice sub-item shipped 2026-06-15 (18 tests over
  `inferExportKind`/`exportKindLabel`/`mergeConfig`).

## Benchmark

- [x] BENCH-W2C - **Done 2026-05-31.** Compact benchmark reports (schema v3),
  private 130x130 puzzle rig wiring, optional laser extraction,
  deterministic hand-eye diagnostic sweeps.
- [x] BENCH-W2D - **Done 2026-05-31.** Compact `BenchRecord` dashboard mode
  in the viewer, robot-pose correction magnitudes (mm/degrees), `diagnose
  stages` target/laser timing.
- [x] BENCH-W2E - **Done 2026-05-31** (superseded by BENCH-W2F on mode
  interpretation). Confirmed DS8's 10x14/52mm checkerboard, rejected partial
  detections, added alternate-mode hand-eye comparison.
- [x] BENCH-W2F - **Done 2026-05-31.** `scripts/bench-viewer.sh`, stderr run
  progress, dashboard artifact output, topological chessboard dispatch,
  corrected DS8 to physical EyeInHand / `gripper_se3_base`.
- [x] BENCH-W2G - **Done 2026-05-31.** Viewer temp output routed through
  `/tmp` for Vite's `/@fs`; verified `kuka_1` via the plain chessboard
  topological detector path.
- [x] BENCH-W2H - **Done 2026-05-31.** Typed ChESS threshold overrides,
  private ChArUco rig wired to EyeToHand Scheimpflug staged BA,
  Intrinsic/RigExtrinsic/HandEye level reporting, robot-pose-correction
  flagging.
