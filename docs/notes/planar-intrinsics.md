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
- `K`: `u = fx·x_d + skew·y_d + cx`, `v = fy·y_d + cy`.
- `T_C_T` (`camera_se3_target`, ADR 0009): one SE(3) pose per view; the
  target frame is the board plane `z = 0`.

## Initialization (Zhang)

Per view, a Hartley-normalized DLT homography `H_i` maps board points to
pixels. Writing `H = K [r1 r2 t]`, the orthonormality of `r1, r2` yields
two linear constraints per view on the image of the absolute conic
`ω = K⁻ᵀK⁻¹`; stacking views gives `V b = 0`, solved by SVD, and `K` is
recovered from `ω` by Cholesky-style closed forms. Poses follow from
`K⁻¹H` with unit-norm scaling; distortion is estimated iteratively by
alternating a linear least-squares fit of `(k1, k2)` against the current
undistorted projections (`linear::distortion_fit`).

## Cost

Non-linear refinement minimizes total squared reprojection error

```
E(K, d, {T_i}) = Σ_i Σ_j ρ( ‖ π(K, d, T_i · X_j) − u_ij ‖² )
```

over `fx, fy, cx, cy [, skew]`, the unfixed distortion coefficients, and
all per-view poses (SE(3) manifold, ADR 0009); `ρ` is the configured
robust loss (default: none; Huber for real detector tails). Factors are
autodiff-generic (ADR 0008).

## Identifiability and degeneracies

- **View count**: each homography gives 2 constraints on `ω`. With skew
  free, `K` has 5 DOF → ≥ 3 views; with `skew = 0` (our default), ≥ 2.
  The pipeline requires ≥ 3 usable views.
- **Plane-orientation diversity**: constraints from views whose plane
  normals are (near-)parallel are (near-)dependent — a pose set that only
  translates, or only rotates about the optical axis, leaves `ω` rank
  deficient. This is why the matrix test's pose set mixes pitch and yaw.
- **Fronto-parallel bias**: with little perspective foreshortening, focal
  length trades off against target distance (`fx·z` nearly constant per
  view); expect inflated focal variance, not a wrong minimum.
- **Distortion collinearity**: over a limited radius range the monomials
  `r², r⁴, r⁶` are nearly collinear, so `k3` is weakly identifiable unless
  the corners reach deep into the image periphery — the rationale for
  `fix_k3: true` by default (CLAUDE.md); enabling it on narrow-FOV data
  degrades conditioning without improving fit.
- **Principal point**: weakly observable under weak distortion + narrow
  FOV (it trades against `t_x, t_y` per view); observability improves with
  strong radial distortion, which pins the distortion center.

## Gauge

None global: the board fixes each view's target frame, `K` and `d` are
shared, and every `T_i` is an independent free block. (Contrast with rig
extrinsics, where a reference camera pins the rig frame.)

## Noise sensitivity

For iid pixel noise with per-axis std `σ` and `N ≫ #params` residuals, the
post-fit mean reprojection approaches the noise floor (for uniform
`[-a, a]` per-axis noise the expected error-norm mean is `≈ 0.765·a`) and
parameter errors scale linearly in `σ` through `(JᵀJ)⁻¹`. Empirically
(matrix test below): focal recovery within 2 % and `k1` within 0.05 at
`a = 0.3 px` on an 8-view / 99-point board; exact data reproduces GT to
solver tolerance (< 1e-4 px).

## Evidence

- **Matrix test**: `crates/vision-calibration/tests/planar_intrinsics_matrix.rs`
  — GT grid (2 focal regimes × 3 distortion strengths) × noise
  `{0, 0.15, 0.3} px`, standard `step_init → step_optimize` pipeline;
  asserts focal/pp/k1 recovery and the residual noise floor per cell.
- **Acceptance gates**: `calib-bench accept` — `stereo_left`/`stereo_right`
  (planar entries) gated at ≤ 0.4 px per camera from measured baselines
  (S4); Q2 adds committed Fit records with drift gates.
- **Related**: ADR 0022 (the Scheimpflug sibling's seeded route and why
  from-scratch is fragile there — the tilt↔focal↔distortion trade-off does
  not arise in the frontal planar family).
