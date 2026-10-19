# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §9); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the v1.0 criteria these serve.

## Parameter uncertainty

Every calibration result carries the covariance of its estimated parameters
and the projection uncertainty it implies, validated against Monte-Carlo
scatter. Design: ADR 0026; evidence: `notes/uncertainty.md`. What is left
changes results, so each needs its measurement first.

- [ ] UQ-LASER-LOSS - The laser loss defaults (`LaserlineSolveOptions`,
  `LaserlineDeviceOptimizeConfig`, Python) are `Huber { scale: 0.01 }`, a
  scale chosen for the point-to-plane residual in metres, while the default
  residual is the line distance in pixels: at 0.01 px the fit is almost L1.
  On the laserline Monte-Carlo scene that costs efficiency (σ of `plane.d`
  1.58e-4 against 1.45e-4 at 1 px) and calibration (predicted σ 11 % high,
  from the noisy density estimate an L1 covariance needs). Make the default
  scale follow the residual's unit (1 px for the line distance), measure on
  the rtv3d laser data, migration note. The frozen synthetic laser entries,
  solved with the default, sit further from their truth than their σ
  allows: `synth_laserline` Σz² = 29.6 over 12 parameters (rank 11),
  `synth_rig_handeye_laserline` `cam1.cam_se3_rig.rx` at +4.5σ.
- [ ] UQ-PRIOR-WEIGHTS - Robot-pose refinement weighs pixel residuals at
  unit σ against the priors' `sqrt_info`, so with sub-pixel corners the
  priors are far too weak: on `kuka_1` (σ̂ 0.11 px) the robot-pose deltas
  absorb signal and σ(fx) is 0.95 px against 0.30 px with exact poses.
  Rescale the reprojection family by its σ̂ (one re-solve) or the priors by
  σ̂², and measure hand-eye accuracy against ground truth on synthetic
  scenes; results change, so it needs a migration note. With exact
  synthetic stations (`synth_rig_handeye_laserline`), every hand-eye and
  target coordinate lands within 0.7σ of the truth: σ reflects the priors,
  not the data.
- [ ] UQ-RIG-INTRINSICS - Rig and rig hand-eye results hold the per-camera
  intrinsics fixed in their final solve by default, so their covariance has
  no intrinsics block and the extrinsics' σ is conditional on them. Decide:
  refine intrinsics in the final solve by default (results change), or
  report the per-camera intrinsics stage covariance alongside (loses the
  intrinsics–extrinsics correlation). Measure both on `stereo_rig` and `ds8`.
  How optimistic the conditional σ is: on `stereo_rig` the two backends'
  intrinsics differ by 0.006σ(fx), and that moves `cam1.cam_se3_rig.tz` by
  0.4 of its conditional σ (3.6e-5 m); `suite check --backend factrs
  --sigma 1` passes only by that margin, and should tighten to ~0.05σ once
  the covariance includes the intrinsics.
  `synth_rig`, with `refine_intrinsics_in_rig_ba`, agrees across the
  backends within 3e-5σ.
  The per-camera Scheimpflug stage of both workflows still holds view 0
  fixed (`fix_poses: vec![0]` in `rig_extrinsics/steps.rs` and
  `rig_handeye/steps.rs`), which the single-camera Scheimpflug default no
  longer does; drop it with this change.

## Regression suite

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR;
images live in a versioned Zenodo record. Publishable: `kuka_1`, `stereo`
(ours), DS8 (Tabb & Ahmad Yousef, USDA Ag Data Commons). `stereo_charuco`
stays local-only, as do the private sets.

- [ ] DATA-IMAGES - Zenodo record per suite version with a SHA-256 manifest;
  `calib-bench suite fetch` downloads into a cache and verifies; a scheduled
  CI job runs `suite freeze` from the images and checks the observations
  still match `SHA256SUMS`. Then retire `bench/registry/public.json` and the
  acceptance job's image run, and move `kuka_1` out of LFS (the
  `planar_real`/`handeye_session` examples then name the fetch command). Our
  captures are CC-BY-4.0. Once `data/stereo` is fetchable, its app presets
  move from the local preset file back into `BUILTIN_PRESETS`, and a
  real-image dense-stereo example can return.
- [ ] DATA-OPENCV - OpenCV harness (`tools/compare/opencv/`, `uv`, pinned
  `opencv-python`) reading `ObservationSet`: `calibrateCameraExtended` (with
  its standard deviations), `stereoCalibrate`, `calibrateHandEye`,
  `calibrateRobotWorldHandEye`. Emit a JSON Schema of the observation format
  for harness authors (the problem inputs need `JsonSchema` derives).
  `calib-bench suite compare` evaluates every result with one reprojection
  evaluator on the same observations and reports
  parameter differences in units of our σ and our σ against OpenCV's.
- [~] DATA-MRCAL - mrcal comparison (parameters and projection uncertainty)
  in an Ubuntu CI job (`apt install mrcal`). Parked until DATA-OPENCV lands.
- [~] DATA-MORE - More public entries, each after a licence check: OpenCV's
  sample stereo chessboards; reference-only third-party sets (URL + checksum,
  never redistributed) such as RWHE-Calib; purpose-designed captures of our
  own under CC-BY-4.0. Parked until DATA-IMAGES.

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

- [ ] TS-UPSTREAM - Propose the remaining tiny-solver fork branches upstream:
  variables no residual uses left out of the solve; a sphere manifold; the
  nalgebra 0.35 / faer 0.24 bump; the internal perf changes (Mutex-free
  assembly, sort-free symbolic structure). Keep the backend's layout for a
  fully fixed manifold block (no manifold, so no columns): upstream keeps its
  tangent columns with a zero Jacobian, which would make `JᵀJ` singular for
  the covariance. On release: replace
  `backend/s2.rs` with the upstream sphere manifold, and see
  O-TINYSOLVER-PERF and D4-NALGEBRA-035.
- [ ] O-TINYSOLVER-PERF - Adopt the tiny-solver speed-ups once released.
  Measured with all three on 12 threads: solves 3.9× faster than with
  0.18.x (1.8–5.8× per scene), 1.5× faster than parallel factrs,
  bit-identical results, no code change here.
  - Mutex-free residual and Jacobian assembly (1.18×; fits 0.18.x).
  - Jacobian pattern by counting sort, values scattered in place (1.20× on
    top; fits 0.18.x).
  - Stack duals in passes of 4 directions, as Ceres'
    `DynamicAutoDiffCostFunction` (adds supertraits to `FactorImpl` and
    `Manifold`, so a 0.19 change).

  **Trigger**: the release that carries them. Bump the floor, rerun
  `calib-bench solver`.
- [~] O-FACTRS-PARALLEL - Parallel linearization for factrs: rpl-cmu/fact-rs
  #56 is merged, unreleased (the pending release is breaking, so 0.4.0 rather
  than 0.3.1). Measured 3.9× over serial factrs on 12 threads (2.6–5.1× per
  workflow), bit-identical, about +5 % on the smallest scenes.
  **Trigger**: that release. Then, in one commit: require it with
  `features = ["rayon"]` (off, nothing runs in parallel); take its Jacobian
  as a faer-0.23 view (D4-NALGEBRA-035); rerun `calib-bench solver` for both
  backends.

## Calibration quality

Found by `calib-bench solver` (well-conditioned synthetic scenes; the named
scenes are the gate) and the regression suite.

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
- [ ] Q-CHESS-SHARED-FRAME - A plain chessboard detection labels corners
  from the detection's own top-left corner, and accepts a board seen turned
  by 90° with its labels rotated. Where views share one target frame (rig
  and hand-eye topologies) a partial or turned board therefore moves the
  target frame between views or cameras: DS8 with partial boards kept
  reprojects at 4.5 px instead of 0.26 px. Manifests can keep complete boards
  only (`detector.min_features_per_view` = corner count); make the shared-
  frame topologies reject inconsistent plain-chessboard labelings by default
  (complete and in the declared orientation) and say so in the manifest docs.

## v1.0

- [ ] D4-RELEASE - Cut v1.0 once the ROADMAP exit criteria hold.
- [ ] D4-NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 /
  `faer-ext` 0.8. **Blocked on both backends**, built against 0.34 / 0.23 /
  0.7: `vision-calibration-optim` passes nalgebra and faer types across the
  tiny-solver boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`) and the factrs one
  (its variables and residual traits), so a bump fails to compile.
  `vision-calibration-detect` stays nalgebra-free at its boundary, which is
  what lets `calib-targets`' nalgebra 0.35 coexist.
  - tiny-solver: the bump is a fork branch (TS-UPSTREAM).
  - factrs `main` is on faer 0.24 / faer-ext 0.8 (nalgebra 0.34). The shared
    LM owns its sparse solve and reads a Jacobian only as pattern and values
    (`backend/normal_equations.rs`), so its next release needs a zero-copy
    faer-0.23 view of those slices, not a blocker.

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
- [~] UQ-LONG-TERM - Beyond the first uncertainty release, each its own task
  when picked up: projection uncertainty across the reference-frame shift
  (the mrcal method); robot-pose noise in hand-eye; target manufacturing and
  flatness error; capture advice (which views would shrink σ most) in the
  app; propagation to triangulated and laser-triangulated 3D points. The
  cross-frame projection needs the intrinsics–pose cross-covariance, which
  results do not carry. Parked until the result-changing items above land.
- [~] P3-BACKEND-COST - Close the remaining gap to Ceres. Planar
  intrinsics against Ceres 2.2 at matched tolerances: 7–9× slower on 12
  threads and about 21× on one; with the O-TINYSOLVER-PERF changes, 2.6× and
  4.5×. What remains is linearization: per-block library overhead, num-dual
  against Ceres' `Jet`, and differentiation through the SE(3) retraction. An
  analytic `ReprojPoint` Jacobian is the next lever. Parked until after 1.0.
- [~] M4-FISHEYE - Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`. Parked until after 1.0; no fisheye dataset exists in
  the acceptance set.
- [~] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by the
  maintainer; reopen only on their request.
