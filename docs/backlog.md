# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §9); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the v1.0 criteria these serve.

## App

- [ ] B-UX2-ELEVATION - Workspace-by-workspace polish: empty states, error
  surfaces (fail-fast errors from ADR 0019 shown well), manifest-sniff UX,
  and three exploration views — a pre-calibration per-camera/pose image
  grid, a detection-cache overlay, and a board coverage map. Gate: a
  frontend review with no high-severity findings.
- [ ] B-QUAL-TS7 - Move to TypeScript 7. **Blocked**: `typescript-eslint`
  rejects TS 7 (typescript-eslint#10940), and adopting it would drop the
  type-aware lint config. Revisit when typescript-eslint supports TS 7.

## Solver backends

Run in this order; each lands as one PR.

- [ ] O-FACTRS-BACKEND - factrs as a first-class second backend:
  `SolverBackend { TinySolver, Factrs }` selectable through
  `BackendSolveOptions` and `SolverConfig.backend` (Rust, JSON, Python, app);
  one shared LM over a linearize/cost/retract trait for both engines; full
  factor coverage (block fusion for the 6-variable arity limit, a custom
  S²×ℝ plane variable, masked fixed components, Arctan loss, post-step
  bounds); cross-backend parity tests; ADR 0025; book "Solver backends".
- [ ] O-BACKEND-COMPARE - Backend axis in `calib-bench solver` and a backend
  override in `calib-bench run`; publish the comparison in the book and
  recommend a default (the default changes only on the user's decision).

## Calibration quality

Found by `calib-bench solver` (well-conditioned synthetic scenes; the named
scenes are the gate).

- [ ] Q-LASERLINE-OUTLIERS - With 5 % of target corners displaced 10–30 px,
  `laserline_device` under Huber or Cauchy (scale 1 px) settles off the
  minimum: inlier RMS 0.16–0.30 px at σ 0.1 px (floor 0.14), focal error
  0.4–1.8 %, laser plane up to 0.29° / 7.6 mm. The other problems (the
  Scheimpflug rig aside, below) reach the floor on the same contamination,
  and the robust-cost LM did not change it, so the init is the suspect. Gate:
  `laserline_device/pinhole/*/n0.1/{huber,cauchy}` at the noise floor.
- [ ] Q-SCHEIMPFLUG-RIG-PERCAM - The Scheimpflug rig's per-camera stage
  (staged init, radial-only BA) lands off the minimum on clean data, even
  seeded with each camera's nominal tilt: inlier RMS 0.21–0.26 px at
  σ 0.1 px, principal point 26–44 px and tilt about 2.5° off. The rig BA
  keeps intrinsics fixed, so it cannot recover; the single-camera Scheimpflug
  problem reaches the floor on comparable data (principal point 1.4 px).
  Gate: `rig_extrinsics/scheimpflug/*/clean` at the noise floor.

## v1.0

- [ ] D4-RELEASE - Cut v1.0 once the ROADMAP exit criteria hold.
- [ ] D4-NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 /
  `faer-ext` 0.8. **Blocked on `tiny-solver`**, still built against
  0.34 / 0.23 / 0.7: `vision-calibration-optim` passes nalgebra and faer
  types across that boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`), so a bump fails
  to compile. `vision-calibration-detect` must stay nalgebra-free (it
  converts to plain arrays at its boundary); that is what lets
  `calib-targets`' nalgebra 0.35 coexist. Re-check on each tiny-solver
  release (0.18.3 is still on 0.34 / 0.23 / 0.7). `factrs` 0.3 pins the
  same versions, so once `O-FACTRS-BACKEND` lands both backends must
  move.

## Deferred and parked

- [ ] BENCH-DETECT-DEDUP - `vision-calibration-bench`'s Tier-B adapters
  (`src/detect.rs`: chessboard, ChArUco, puzzleboard) duplicate the
  detectors in `vision-calibration-detect`; route them through the detect
  crate so detector changes land once. Unpublished crate, so not urgent.

- [ ] C-PYO3-MVG - Python bindings for the MVG surface. Deferred: the
  Python package binds the calibration facade only, and nothing in Python
  needs two-view/N-view geometry yet.
- [ ] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Schedule only
  if the acceptance run for the six-camera rtv3d rig exceeds ~5 min.
- [~] P3-BACKEND-COST - Profile tiny-solver's autodiff/assembly/solve split
  and evaluate analytic Jacobians, caching or rayon for the `ReprojPoint`
  factor. Parked until after 1.0.
- [~] M4-FISHEYE - Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`. Parked until after 1.0; no fisheye dataset exists in
  the acceptance set.
- [~] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by user
  decision — **do not reopen unprompted.**
