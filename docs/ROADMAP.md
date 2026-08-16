# calibration-rs Roadmap

Where the project is and where it is going. Design reasoning lives in
[ADRs](adrs/); the task list lives in the [backlog](backlog.md); completed work
is recorded in [`backlog-archive.md`](backlog-archive.md).

## Status (2026-08-16)

**Version line 0.x**, latest published release **0.7.0**, with **0.8.0**
prepared and awaiting its tag — nine crates.io crates plus the PyPI
extension. Pre-1.0: breaking changes are acceptable and are listed in the
[CHANGELOG](../CHANGELOG.md) with migration notes.

The **production-grade program** approved 2026-07-02 is through its planned
phases. Tracks **S** (device-spec → seed initialization), **Q** (proof packs +
regression gates) and **R** (API/config revision) are complete, as are Tracks
**A** (calibration core), **O** (closed won't-do) and the benchmark harness.
What remains before v1.0 is the app-quality tail, the v1.0 gate itself, and one
externally-blocked item.

| Track | State | What's left |
|---|---|---|
| **A** — calibration core | done | — |
| **S** — spec → seed init | done | — |
| **Q** — soundness proofs + regression | done | — |
| **R** — API/config revision | done | — |
| **B** — desktop app | in progress | B-UX2 elevation; two lint/toolchain items |
| **C** — MVG | done, tail parked | PyO3 bindings for MVG (deferred, no consumer) |
| **M** — camera models | additive layer done | M4 fisheye parked post-1.0 |
| **P** — performance | done bar two | P2 conditional, P3 parked post-1.0 |
| **V** — rtv3d validation | done | V7 intrinsics floor parked (user call) |
| **D** — earn v1.0 | ratcheting | D4 release gate; one upstream block |

## Path to v1.0

The exit criteria, in the order they gate a release:

1. **Acceptance.** One command runs every on-disk registered dataset through
   the seeded official route with hard gates (`calib-bench accept`).
   *Met as of 2026-08-16:* 21 of 21 registered datasets pass, 0 fail. The
   `rtv3d` ChArUco regression that had held this red is fixed upstream in
   `calib-targets` 0.12.1 and its baseline is re-frozen on the improved
   numbers.
2. **Soundness.** A proof pack per shipped algorithm family (math note,
   synthetic-GT matrix, property tests, committed regression record + gate),
   plus a convergence-basin study for seeded init. *Met.*
3. **API frozen.** Facade-only consumers, normalized configs, typed errors
   throughout, no duplicate public type names, no deprecated shims. *Met as of
   0.7.0/0.8.0; the standing requirement is that it then holds across two minor
   releases without breaking changes.*
4. **App CI green** — lint, typecheck, unit, component, and smoke tests, with
   generated IPC types. *Met; two lint rules run as warnings pending upstream
   fixes.*
5. **Docs current** — this file, the backlog, ADR statuses, tutorials, README
   and AGENTS.

## Live tracks

### B — Tauri 2 + React desktop app

The app hosts five workspaces (Diagnose, 3D, Epipolar, Depth, Run), covers all
eight topologies and all four detectors end-to-end, and can sniff a foreign
folder into an editable manifest. Progress streaming, cancellation, per-pose
and cross-camera residual panels, and the dense-stereo point cloud have
shipped.

Remaining: **B-UX2-ELEVATION** — empty states, error surfaces, long-run
progress polish, manifest-sniff UX, plus the absorbed pre-calibration image
grid / detection-cache overlay / coverage map. Gate: a frontend review with no
high-severity findings. Two toolchain items (`B-QUAL-HOOKS7`, `B-QUAL-TS7`)
wait on upstream `eslint-plugin-react-hooks` and `typescript-eslint` releases.

### C — Multiple-view geometry

`vision-geometry` (deterministic two-view solvers) and `vision-mvg` (N-view
pipelines: robust pose recovery, triangulation, bundle adjustment,
Scheimpflug-aware rectification, dense stereo) both ship, are published, and
are re-exported through the facade. [ADR 0015](adrs/0015-mvg-ceiling.md) caps
the ceiling: full SfM stays out of scope. The dense matcher is pure-Rust —
block matching plus optional SGM aggregation — with the OpenCV SGBM comparison
closed as environment-blocked.

Deferred: PyO3 bindings for the MVG surface, until a Python consumer exists.

### M — Camera models

The additive layer is done: rational k4–k6, thin-prism s1–s4, and the
Fitzgibbon division model exist at the core-model and optim-IR layers alongside
Brown-Conrady, selected by `distortion_model` on the two single-camera
intrinsics workflows. Kannala-Brandt fisheye (**M4**) is parked post-1.0 — it
needs a new projection slot and linear-init changes, and no fisheye dataset
exists in the acceptance set.

### P — Performance

Opened after a from-scratch Scheimpflug rig calibration took 30+ minutes on a
dense board. Root cause was dense linear algebra, not the algorithms:
`nalgebra::svd(true, true)` was pathologically slow across ~20 sites and has
been replaced (init 30 min → 9 ms). Benchmarks, per-stage timing records and
the remaining hot-path work are all closed. **P2** (a corner budget for the
joint rig + hand-eye BA) is conditional on acceptance-run wall time becoming
painful; **P3** (tiny-solver autodiff/assembly cost, analytic Jacobians,
parallelism) is parked post-1.0.

### D — Earn v1.0

Typed errors (**D1**), doc-warning-free with `missing_docs` enforced
workspace-wide (**D2**), and the Python parity audit and fill (**D3**/**R5**)
are done. Open:

- **D4-RELEASE** — the v1.0 gate above.
- **D4-NALGEBRA-035** — `nalgebra` 0.35 / `faer` 0.24 / `faer-ext` 0.8 are
  unadoptable while `tiny-solver` 0.18 is built against 0.34 / 0.23 / 0.7 and
  `vision-calibration-optim` exchanges both libraries' types across that
  boundary. Re-check on each tiny-solver release.

## Closed tracks

Full completion notes in
[`backlog-archive.md`](backlog-archive.md).

- **A — Calibration core.** (Closed before the archive was split by track,
  so it has no archive section of its own.) Manual init ([ADR 0011](adrs/0011-manual-initialization-workflow.md)),
  per-feature residuals ([ADR 0012](adrs/0012-per-feature-reprojection-residuals.md)),
  Scheimpflug EyeToHand, and the `rig_family` sensor-axis collapse
  ([ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md)) that unified the
  pinhole and Scheimpflug rig modules behind `SensorMode`.
- **S — Device-spec → seed initialization.** A `DeviceSpec` schema and
  `device_seed` derivation replace hand-coded per-example constants; seeded
  init is the official acceptance route
  ([ADR 0022](adrs/0022-scheimpflug-intrinsics-seeded-default.md),
  [ADR 0023](adrs/0023-device-spec-seed-derivation.md)).
- **Q — Algorithmic soundness.** Proof pack per algorithm family plus the
  committed regression baselines and the convergence-basin study.
- **R — API/config revision.** The shared grouped-config vocabulary
  ([ADR 0024](adrs/0024-config-vocabulary.md)), the `ExportKind` discriminator,
  the two-view solver relocation into `vision-geometry`, and the typed-error
  sweep.
- **O — Alternative solver backend.** Closed **won't-do** 2026-07-04:
  apex-solver 1.3 lacks an S2 manifold, documented robust losses, and
  documented SE3/Jacobian conventions, and duplicates the existing LM + faer
  stack. Revive only if it gains autodiff/S2/robust-loss support, or if a
  solver-trust need appears that the Q-track baselines cannot address — the
  preferred target then is an autodiff-native stack, not apex-solver 1.x.
- **V — rtv3d validation.** The private six-camera Scheimpflug laser rig
  calibrates and beats its oracle on every criterion. **V7** (driving the
  from-scratch reprojection floor below 0.4 px) is parked by user decision —
  the blocking term is isolated to the detector/target/model, not the rig
  chain. Do not reopen unprompted.
- **Benchmark.** The registry-driven `calib-bench` harness, its acceptance
  command, and the committed regression baselines.

## Out of scope

- Camera models beyond the M-track set — omnidirectional/MEI, double-sphere,
  telecentric, spline. Defer until a concrete project demands one.
- Full structure-from-motion: incremental SfM, pose graph, loop closure.
  (The dense matcher is *in* scope and shipped — [ADR 0015](adrs/0015-mvg-ceiling.md)
  was amended for it.)
- `rerun.io` / `egui` as the UI.
- OSS-grade community surface and multi-platform install docs — internal-first
  until the tool earns them.

## See also

- [ADR index](adrs/README.md) — every design record, with status.
- [Backlog](backlog.md) — open and parked tasks.
- [Tutorials](tutorials/README.md) — hands-on walkthroughs.
- [MSRV notes](MSRV.md) — 1.93, history and bump policy.
- [Release runbook](RELEASE-RUNBOOK.md) — the version-lockstep publish process.
