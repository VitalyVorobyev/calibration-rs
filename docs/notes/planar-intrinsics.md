# Planar intrinsics — proof pack

Family: single-camera intrinsics from planar-target views
(`PlanarIntrinsicsProblem`; init in `vision-calibration-linear`, refinement
in `vision-calibration-optim::problems::planar_intrinsics`).

## Model

Composable camera (ADR 0005), planar specialization:

```
pixel = K( distort( project(X_c) ) ),   X_c = T_C_T · X_t
```

- `project`: pinhole, `(x, y, z) ↦ (x/z, y/z)`.
- `distort`: Brown–Conrady 5 on normalized coordinates,
  `x_d = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 x y + p2 (r² + 2x²)` (and
  symmetrically for `y`), `r² = x² + y²`.
- `K`: `u = fx·x_d + cx`, `v = fy·y_d + cy` (no skew term; the optimizer packs
  `[fx, fy, cx, cy]`).
- `T_C_T` (`cam_se3_target`, ADR 0009): one SE(3) pose per view; the
  target frame is the board plane `z = 0`.

## Initialization (Zhang, iterative)

`estimate_intrinsics_iterative` (`vision-calibration-linear`). Per view, a
Hartley-normalized DLT homography `H_i` maps board points to pixels.
Writing `H = K [r1 r2 t]`, the orthonormality of `r1, r2` gives two linear
constraints per view on the image of the absolute conic `ω = K⁻ᵀK⁻¹`;
stacking views gives `V b = 0`, whose null vector comes from the `VᵀV`
symmetric eigendecomposition (`math::null_space`), and Zhang's closed forms
recover `K` from `b` (`zhang_intrinsics`). The skew of the
general solve is discarded. Poses follow from `K⁻¹H` (`planar_pose`). Distortion
alternates with `K`: fit Brown–Conrady coefficients by linear least squares
against the homography residuals (`distortion_fit`; `k1, k2, p1, p2` by
default, `k3` fixed), undistort the pixels, re-estimate `K`.

**Conditioning.** `V` is built from pixel-unit homographies with no
normalizing camera `K₀`, and `null_space` squares its condition number.
Measured for ten views tilted 10°–40° with unit-norm `H`: `σ₁/σ₅(V)` is
2.3e3, 1.1e4, 2.1e4, 3.7e4 at fx = 800, 4000, 8000, 16 000 px (linear in
f, not quadratic), so `κ²` ≤ 1.4e9 and noise-free homographies return `K`
to ~1e-13 relative at every focal length. With σ ≈ 0.1 px corner noise
(50 draws), the median |Δfx|/fx is 0.019 % at 800 px (1280 px sensor) and
0.006 % at 16 000 px (8192 px sensor), but 0.082 % (max 0.35 %) at
16 000 px on a 2448 px sensor: the narrow field of view loses perspective,
not precision. Normalizing by `K₀` from the image extent before forming
`V` is no better (0.089 %, max 0.36 %), so it is not done.
(`long_focal_lengths_stay_exact`, `long_focal_lengths_initialize_accurately`.)

## Cost

Non-linear refinement minimizes total squared reprojection error

```
E(K, d, {T_i}) = Σ_i Σ_j ρ( ‖ π(K, d, T_i · X_j) − u_ij ‖² )
```

over `fx, fy, cx, cy`, the unfixed distortion coefficients, and all
per-view poses (SE(3) manifold, ADR 0009); `ρ` is the configured robust
loss (default: none; Huber for real detector tails). Factors are
autodiff-generic (ADR 0008).

## Identifiability and degeneracies

- **View count**: each homography gives 2 constraints on `ω`. The linear
  solve estimates all 5 DOF of the general `K` (skew included, then
  discarded), so it needs ≥ 3 views. The pipeline requires ≥ 3 views.
- **Plane-orientation diversity**: each view's two constraints depend only
  on its plane's circular points, i.e. its normal (in-plane rotation and
  translation add nothing). One normal for all views (fronto-parallel, or
  all tilted the same way) leaves `V` rank 2; two distinct normals rank 4
  (one focal length free: tilts about one axis both ways); three distinct
  normals, even about a single axis, determine `ω`. This is why the matrix
  test's pose set mixes pitch and yaw.
- **Parallel-board detection** (`DegenerateViews`). Two checks, because
  neither alone covers the case. (1) Before the solve,
  `estimate_intrinsics_iterative` measures the board-orientation spread —
  the largest angle between a normal `K₀ᵀ(h1 × h2)` and the mean plane —
  through a reference `K₀` (square pixels, centred on the observed corners'
  bounding box, focal = its larger side) and rejects < 1°. Parallel planes
  read zero through any `K₀`; the extent focal reads the *apparent*
  perspective, which is what Zhang's constraints feed on (a telephoto needs
  more tilt). Without it, noisy parallel boards return an arbitrary `K`
  (fx ≈ 54 000 px fronto-parallel, (2914, 150) all tilted 20°, pinhole,
  ±0.35 px). (2) Radial distortion mimics tilt at second order across a
  board (an off-centre fronto-parallel board at `k1 = −0.15` reads ≈ 10°),
  so on distorted data the spread cannot separate parallel boards from a
  3° capture (11°); there, Zhang's conic is not definite (`λ/B11 ≤ 0`) and
  the solve itself returns `DegenerateViews`. The homographies alone carry
  no image extent, so `estimate_intrinsics_from_homographies` has only
  check (2). Not caught: two orientations (rank 4) pass init with a wrong
  focal length (±20° about x: fx 1374 px pinhole, 258 px at `k1 = −0.15`,
  truth 1000) and refine to a σ(fx) of ≈ 96 px (`capture_planning`); a
  zero-skew Zhang (b12 = 0, 5 unknowns) would not fix it, since fx is
  unobservable from those two orientations to first order.
- **Fronto-parallel bias**: with little perspective foreshortening, focal
  length trades off against target distance (`fx·z` nearly constant per
  view); expect inflated focal variance, not a wrong minimum.
- **Distortion collinearity**: over a limited radius range the monomials
  `r², r⁴, r⁶` are nearly collinear, so `k3` is weakly identifiable unless
  the corners reach deep into the image periphery — hence `fix_k3: true`
  by default; enabling it on narrow-FOV data degrades conditioning without
  improving fit.
- **Principal point**: weakly observable under weak distortion + narrow
  FOV (it trades against `t_x, t_y` per view); observability improves with
  strong radial distortion, which pins the distortion center.

## Gauge

None: the board fixes each view's target frame, `K` and `d` are shared,
and every `T_i` is an independent free block (`fix_poses` is empty by
default). Contrast rig extrinsics, where a reference camera pins the rig
frame.

## Noise sensitivity

For iid pixel noise with per-axis std `σ` and `N ≫ #params` residuals, the
post-fit mean reprojection approaches the noise floor (for uniform
`[-a, a]` per-axis noise the expected error-norm mean is `≈ 0.765·a`) and
parameter errors scale linearly in `σ` through `(JᵀJ)⁻¹`. Empirically
(matrix test below): focal recovery within 2 % and `k1` within 0.05 at
`a = 0.3 px` on an 8-view / 99-point board; exact data reproduces GT with
reprojection < 1e-4 px.

## Evidence

- **Matrix test**: `crates/vision-calibration/tests/planar_intrinsics_matrix.rs`
  (`planar_intrinsics_recovers_gt_across_grid_and_noise`) — GT grid
  (2 focal regimes × 3 distortion strengths) × noise `{0, 0.15, 0.3} px`,
  standard `step_init → step_optimize` pipeline on both backends; asserts
  focal / principal point / `k1` recovery and the residual noise floor per
  cell.
- **Acceptance gates**: `calib-bench accept` — `stereo_left` /
  `stereo_right` gated at ≤ 0.4 px per camera, with committed Fit
  baselines.
- **Regression suite**: `stereo_left`, `stereo_right`, `synth_planar`.
- **Degenerate captures**: `iterative_intrinsics` unit tests (parallel
  boards fronto-parallel and tilted 20°, pinhole and `k1 = −0.15`; tilts of
  0.5° (spread 0.79°, rejected) and 3° (4.28°, fx 1047 px); ±x and three
  tilts about x), `zhang_intrinsics` unit tests (spread = tilt through the
  true `K`, zero for parallel boards through any `K₀`, threshold either
  side), pipeline `tests/degenerate_capture.rs` (plain and Scheimpflug).
  Spread margin on the regression suite's planar inits: smallest 8.7°
  (`stereo_left`/`stereo_right`), the rest 15–34°.
- **Related**: ADR 0011 (the Scheimpflug sibling's seeded route — the
  tilt↔focal↔distortion trade-off that makes from-scratch init fragile
  there does not arise in the frontal planar family).
