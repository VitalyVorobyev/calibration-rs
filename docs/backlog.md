# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §9); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the v1.0 criteria these serve.

## Parameter uncertainty

Every calibration result carries the covariance of its estimated parameters
and the projection uncertainty it implies, validated against Monte-Carlo
scatter. Order: UQ-ADR → UQ-CORE → UQ-EXPORT → UQ-PROJECTION → UQ-APP.

- [ ] UQ-ADR - ADR 0026 *Parameter uncertainty* and the proof-pack skeleton
  `notes/uncertainty.md`. Decide and record:
  - **Estimator**: first-order `Σ = σ̂²·(JᵀWJ)⁻¹` at the converged optimum over
    the free tangent coordinates. `J` is the raw Jacobian from one loss-free
    linearization at the final state; `W` holds the IRLS weights `ρ′(‖rᵢ‖²)`
    from `RobustLoss`. One routine for both backends (today their robust
    corrections differ: Triggs for tiny-solver, √ρ′ for factrs). The sandwich
    estimator is evaluated in the proof pack and shipped only if Monte-Carlo
    shows the simple form is miscalibrated under outliers.
  - **Noise**: a posteriori `σ̂²` per residual family (reprojection, laser);
    mixed problems take one Förstner variance-component iteration. Priors
    (`Se3TangentPrior`) are known information; observation weights are
    relative.
  - **Conventions** (into ADR 0009): SE(3) covariance in the right
    perturbation `T·exp([δθ, δt])`, rotation first; S² planes as a joint
    `[n(2), d(1)]` block; covariance is conditional on fixed blocks, and the
    result lists them.
  - **Availability as a type**: available, or unavailable with a reason
    (rank-deficient with the unobservable labels, not converged, parameter at
    a bound). A singular final Hessian is reported, not an error.
  - Always computed; no config knob.
- [ ] UQ-CORE - Covariance in the shared LM (`backend/covariance.rs`). Reuse
  `NormalEquations`: assemble at λ = 0, factor with the existing symbolic
  LLT, solve identity columns, unscale by the Jacobi scaling; on LLT failure
  name the null directions from a dense eigendecomposition.
  `LinearizationEngine` gains a raw linearization and a column layout (IR
  block → columns); `SolveReport` gains `converged`, residual count and
  degrees of freedom; `BackendSolution` gains the block-labelled covariance.
  Gates: backend parity (with and without robust loss); Monte-Carlo on the
  planar-intrinsics scene (≥ 200 seeded realizations, predicted σ within
  ±10 % of empirical, NEES consistent with χ²); a free-gauge problem reports
  the right unobservable labels.
- [ ] UQ-EXPORT - `Uncertainty` on every `*Export`. Core type with labelled
  covariance over the global parameters (intrinsics, distortion, sensor,
  extrinsics, hand-eye, planes), per-family `σ̂`, degrees of freedom and
  per-view 6×6 pose marginals. Labels from one source
  (`DistortionKind::param_names()` next to `dim()`). Intrinsics family first,
  then hand-eye, rig and laser problems; only the final solve feeds the
  export. Python dataclass with `std_dev(label)` / `correlation(a, b)`,
  `types.py`, `__init__.pyi`, contract tests; a book chapter, README line and
  the five-minute tutorial show `fx ± σ`. Gate: Monte-Carlo per problem
  family.
- [ ] UQ-PROJECTION - Projection uncertainty per camera: major-axis σ (px)
  on an image grid at a stated range (default: median target distance),
  `J_q Σ_θθ J_qᵀ` with exact derivatives from the generic camera kernels,
  plus a centre/corner/max summary. Camera-frame (intrinsics-only) variant,
  documented as such. Gate: Monte-Carlo.
- [ ] UQ-APP - Diagnose shows value ± σ per parameter (fixed ones marked), a
  correlation heatmap and the projection-σ map.

## Regression suite

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR;
images live in a versioned Zenodo record. Publishable: `kuka_1`, `stereo`
(ours), DS8 (Tabb & Ahmad Yousef, USDA Ag Data Commons). `stereo_charuco`
stays local-only, as do the private sets.

- [ ] DATA-OBS - Observation tier and suite manifest.
  - Promote `FrozenFixture` (`bench/src/fixtures.rs`) to a public, versioned,
    schema-emitted `ObservationSet` in `vision-calibration-dataset`:
    per-view, per-camera 2D–3D target correspondences with ids, robot poses
    with explicit convention, laser pixels, image sizes, units.
  - `regression/suite.toml` with `suite_version`; per entry: id, problem,
    licence, citation, source URL, observations path + SHA-256, optional
    image archive (DOI, file, SHA-256), config and gates. Replaces
    `bench/registry/public.json`; the private registry keeps the machinery.
  - Observations for `kuka_1`, `stereo_left`/`stereo_right`/`stereo_rig`,
    DS8 (confirm its licence) and frozen synthetic sets with ground truth
    (planar, Scheimpflug, rig, hand-eye, laser), so generator changes cannot
    silently change the data. No LFS; a few MB in total.
  - Reference results with full parameters, not only reprojection
    statistics.
  - `calib-bench suite run|check|freeze`: verify checksums first (fail, never
    skip), run from observations, gate parameter and reprojection drift. CI
    runs `suite check` on every PR.
  - A book chapter: the format, its conventions against OpenCV (distortion
    order, pose direction, units, the reprojection metric), how to run it.
- [ ] DATA-IMAGES - Zenodo record per suite version with a SHA-256 manifest;
  `calib-bench suite fetch` downloads into a cache and verifies; a scheduled
  CI job runs detection + pipeline from images. Move `kuka_1` out of LFS.
- [ ] DATA-OPENCV - OpenCV harness (`tools/compare/opencv/`, `uv`, pinned
  `opencv-python`) reading `ObservationSet`: `calibrateCameraExtended` (with
  its standard deviations), `stereoCalibrate`, `calibrateHandEye`,
  `calibrateRobotWorldHandEye`. `calib-bench suite compare` evaluates every
  result with one reprojection evaluator on the same observations and reports
  parameter differences in units of our σ and our σ against OpenCV's. After
  UQ-EXPORT, parameter drift gates switch to k·σ.
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

- [ ] TS-0187 - Raise the tiny-solver floor to 0.18.7: exact `SO3::exp` at
  small angles (our SE(3) and quaternion retractions) and manifold blocks held
  constant when fully fixed. Rerun `calib-bench solver` for both backends.
- [ ] TS-UPSTREAM - Propose the remaining tiny-solver fork branches upstream:
  variables no residual uses left out of the solve; a sphere manifold; the
  nalgebra 0.35 / faer 0.24 bump; the internal perf changes (Mutex-free
  assembly, sort-free symbolic structure). On release: replace
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
  app; propagation to triangulated and laser-triangulated 3D points. Parked
  until UQ-PROJECTION.
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
