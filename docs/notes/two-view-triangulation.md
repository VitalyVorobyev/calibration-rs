# Two-view geometry & triangulation — proof pack

Family: relative pose and structure from two (or N) calibrated views. Solvers
in `vision-geometry` (`epipolar::{fundamental, essential, decomposition}`,
`homography`, `triangulation`, `camera_matrix`); orchestration and diagnostics
in `vision-mvg` (`pose_recovery`, `cheirality`, `triangulation`, `degeneracy`,
`robust`, `homography`). Crate split per ADR 0006.

## Model

Two pinhole views of a rigid scene. Pixel points are made calibrated by
`x = K⁻¹ u` (`pixel_to_normalized`, ADR 0009 conventions), so `x₁, x₂` are
normalized (`z = 1`) rays. Relative pose `(R, t)` maps camera 1 into camera 2:
`X_c2 = R·X_c1 + t` (`R` = cam1→cam2, `t` up to scale).

- **Epipolar constraint** (calibrated): `x₂ᵀ E x₁ = 0`, `E = [t]× R`, rank 2,
  singular values `(σ, σ, 0)`.
- **Fundamental** (uncalibrated, pixels): `u₂ᵀ F u₁ = 0`, `F = K₂⁻ᵀ E K₁⁻¹`,
  rank 2.
- **Homography** (planar scene *or* pure rotation): `x₂ ~ H x₁`,
  `H = R + t nᵀ/d` (`homography_from_pose_and_plane`).
- **Triangulation**: given projection matrices `P_i` and observations `x_i`,
  find `X` with `x_i ~ P_i X̃`.

## Solvers (initialization)

- **Fundamental, normalized 8-point** (`fundamental_8point`): Hartley-normalize
  each point set (`normalize_points_2d` — translate centroid to origin, scale to
  mean distance `√2`), stack the 9-column epipolar design `A`, take the 1-D null
  space, reshape to `F_raw`, force rank 2 by zeroing the smallest singular value
  of the `3×3` SVD, then de-normalize `F = T₂ᵀ F_raw T₁`.
- **Fundamental, 7-point** (`fundamental_7point`): minimal solver; a **2-D** null
  space spans `F₁, F₂`, and `det(F₂ + λF₁) = 0` (a cubic via `solve_cubic_real`)
  gives up to three rank-2 candidates.
- **Essential, 5-point** (`essential_5point`, Nistér): a **4-D** null space
  spans `E₁..E₄`; the ten cubic constraints (`build_polynomial_system`) are
  reduced by Gauss-Jordan (LU) elimination to a `10×10` action matrix whose real
  Schur eigenvalues yield up to ten candidate `E`. Calibrated coordinates are
  already `O(1)`, so **no Hartley normalization** is applied (documented in
  `essential.rs`).
- **Essential, linear ≥8-point** (`essential_linear`): 1-D null space, then
  projection onto the essential manifold (singular values → `(σ, σ, 0)`). Used
  for the RANSAC refit on large inlier sets.
- **Camera matrix DLT** (`dlt_camera_matrix`): normalized 6+-point DLT for
  `P = K[R|t]`, with a coplanarity guard (scatter `σ_min/σ_max`) and a rank
  guard; `rq_decompose` / `decompose_camera_matrix` recover `K, R, t`.

**Hartley normalization rationale.** With raw pixels the `u·x` columns of the
epipolar design are `O(10⁵)` while the last column is `O(1)`; the design is
badly scaled and the recovered `F` fails `|u₂ᵀ F u₁|/‖F‖` by orders of
magnitude. Centering and isotropic scaling equalize the columns. Regression:
`fundamental_8point_epipolar_constraint_pixel_coords` asserts the normalized
residual `< 1e-6` at pixel scale, which does not hold unnormalized.

**Null-space extraction (P1-SVD-SWEEP).** The **1-D** homogeneous solves
(8-point `F`, `essential_linear`, homography, camera-matrix, and triangulation
DLTs) take their null vector from `vision_calibration_core::linalg::null_space`
— the smallest-eigenvalue eigenvector of `AᵀA` via a *symmetric*
eigendecomposition — **not** `A.svd(...)`. nalgebra's Golub-Kahan QR sweep can
fail to converge (minutes-long hangs) on real dense designs, and the
non-convergence persists even with `compute_u = false`; `AᵀA` is always `k×k`
(`k ≤ 12` here) and converges in a few sweeps. The **multi-vector** null spaces
(7-point's 2-D, 5-point's 4-D) still use `svd(true, true)` because `null_space`
returns only the single smallest vector. (`ridge_lstsq`, the regularized
normal-equation sibling, lives in `vision-calibration-linear` and is used by the
calibration solvers, not this family.)

## Pose recovery, the 4-fold ambiguity, and cheirality

`decompose_essential` projects `E` to the manifold and returns **four**
candidates `{R₁, R₂} × {+t, −t}` (`R₁ = U W Vᵀ`, `R₂ = U Wᵀ Vᵀ`,
`t = ±u₃`), with a raw-`σ` degeneracy guard on the input (`σ₀ < 1e-10` or
`σ₁/σ₀ < 0.1` ⇒ `Err` — catches rank-1 / zero `E`). Only one candidate places
triangulated points in front of **both** cameras; `cheirality::select_pose`
counts positive-depth points (`cheirality_count`) and picks the winner.

`recover_relative_pose` samples up to 20 evenly-spaced 5-point subsets
(deterministic, input-ordering independent), scores each candidate `E` by mean
Sampson distance over **all** correspondences, decomposes, cheirality-selects,
and tie-breaks by residual — then triangulates with the chosen pose. The robust
variant (`recover_relative_pose_robust` → `estimate_essential`) runs 5-point +
Sampson inside `ransac_fit`, refits inliers with `essential_linear`, and
disambiguates cheirality on inliers only.

## Cost

The solvers minimize algebraic / Sampson error; the one geometric optimizer in
the family is triangulation refinement. `refine_point` (a self-contained 3-DOF
Gauss-Newton, 10 iterations) minimizes total squared reprojection error
`Σ_i ‖π(P_i, X) − x_i‖²` — the maximum-likelihood point under isotropic Gaussian
image noise — starting from the DLT estimate. **No midpoint or optimal
(Hartley–Sturm polynomial) triangulator is implemented**; DLT seed + reprojection
GN is the chosen path. Sampson distance (`residuals::sampson_distance`, the
first-order geometric epipolar residual) is the RANSAC residual for `E`/`F`;
symmetric transfer error serves `H`.

## Triangulation and its failure modes

`triangulate_point_linear` builds the `2N×4` DLT, solves it via `null_space`,
guards rank with `dlt_rank_ok`, and rejects a point at infinity (`w ≈ 0`).
`triangulate_point` appends `refine_point`; `triangulate_nview` adds diagnostics
(RMS reprojection, **widest** pairwise parallax, all-view cheirality). Failure
modes:

- **Zero / low parallax** (narrow baseline or far points): rays are nearly
  parallel, the DLT boundary singular value collapses, `dlt_rank_ok` rejects the
  exactly-degenerate case, and near-degenerate points have depth variance `∝
  1/parallax` (quantified below).
- **Point on the baseline / near the epipole**: the two viewing rays are
  collinear ⇒ rank-deficient ⇒ rejected per point. `triangulate_two_view_partial`
  skips such points instead of failing the whole set (used by `analyze_scene`).
- **Identical cameras** (zero baseline): rejected (`triangulation_rejects_identical_cameras`).

## Identifiability and degeneracies

- **Pure rotation** (`t = 0`): `E = [0]× R = 0` (rank 0) — `decompose_essential`
  rejects it. There is no baseline, so triangulation is meaningless; the correct
  model is `H = R`. `detect_pure_rotation` flags this from the **median parallax
  angle** (scale-invariant) rather than `‖E‖` (which is scale-dependent and
  would misfire on a down-scaled valid `E`).
- **Planar scene**: coplanar structure makes the epipolar design rank-deficient
  — `F`/`E` are under-determined, but `H` is well-posed. `detect_planar_scene`
  compares homography vs essential inlier support. `dlt_camera_matrix` likewise
  rejects coplanar 3D points (estimate a homography instead). The matrix and
  property tests deliberately use **non-coplanar** clouds.
- **Narrow baseline / low parallax**: `E` rank is fine, but the translation
  *direction* and the reconstructed depth are ill-conditioned, degrading `∝
  1/parallax`. This is the dominant real-data sensitivity (see below).
- **Critical surfaces** (structure + both centers on a ruled quadric): a known
  theoretical `F` ambiguity; not separately guarded, as the planar case
  dominates in practice.

## Gauge

Two-view reconstruction is fixed only up to a **similarity** (7 DOF). The
conventions pin most of it: fixing camera 1 at `[I | 0]` removes the 6-DOF rigid
gauge, and `E` is defined only up to scale — `decompose_essential` returns a
**unit** `‖t‖`, so metric depth is unobservable from two views. The remaining
1-DOF scale must come from downstream context (a known baseline, frozen
extrinsics, or bundle adjustment against metric structure). This is why the
matrix test reports rotation and translation *direction* for pose recovery, and
resolves scale for the triangulation cells by handing the solver the true metric
cameras.

## Noise sensitivity

Grounded in the matrix test (42-point non-coplanar cloud, `f ≈ 800`, uniform
per-axis pixel noise `a`). At `a = 0.5 px`:

- **Rotation** error `0.06–0.12°`, nearly flat across baseline.
- **Translation direction** climbs sharply as the baseline narrows —
  `0.19°` (wide, `‖t‖ = 0.8`) → `0.32°` → `0.58°` → `1.91°` (narrow,
  `‖t‖ = 0.10`): the parallax degeneracy made quantitative.
- **Metric 3D triangulation** median error `9.6e-3` (wide) → `7.1e-2` (narrow),
  a `~7×` spread at fixed noise, same `1/parallax` law.
- **Post-fit reprojection** tracks the noise floor: median `≈ 0.15 px` at
  `a = 0.5 px` (uniform `[−a, a]` per axis has error-norm mean `≈ 0.765·a`).
- **Exact data**: rotation / t-direction at solver tolerance (`< 1e-3°`),
  reprojection `~1e-13 px`, 3D error `~1e-15`.

Parameter errors scale ~linearly in `a`; translation direction and depth
additionally scale ~`1/parallax`.

## Evidence

- **Matrix test**: `crates/vision-mvg/tests/two_view_matrix.rs` — 4
  baseline/parallax regimes (`wide-sideways`, `wide-rotated`, `medium`,
  `narrow`) × noise `{0, 0.2, 0.5} px`; per cell asserts rotation deg,
  translation-direction deg, reprojection px, and 3D error against the ground
  truth, scaled to the noise floor.
- **Property tests**: `crates/vision-mvg/tests/two_view_properties.rs` — over a
  pose grid: `E = [t]× R` annihilates GT correspondences and decomposes to the
  GT pose; estimated-`F` epipolar residual `< 1e-6`; triangulate → reproject
  round trip (`< 1e-6 px`, `< 1e-6` in 3D).
- **Unit tests** (per module): Hartley regression and the dense-DLT anti-hang
  guard (`epipolar/fundamental.rs`, `homography.rs`), essential-manifold and
  degeneracy guards (`epipolar/essential.rs`, `epipolar/decomposition.rs`),
  cheirality (`cheirality.rs`), triangulation recovery/refinement
  (`triangulation.rs`), and scene degeneracy detectors (`degeneracy.rs`).
- **Related**: ADR 0006 (the `vision-geometry` / `vision-mvg` split); the
  P1-SVD-SWEEP (`docs/internal/archive/report/2026-06-16-P1-SVD-SWEEP-finish-centralize.md`) —
  `null_space` / `AᵀA` replacing dense SVD in the 1-D null-space solvers;
  `docs/notes/planar-intrinsics.md` (Zhang init shares the Hartley-normalized
  DLT homography).
