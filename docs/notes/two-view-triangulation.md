# Two-view geometry & triangulation — proof pack

Family: relative pose and structure from two (or N) calibrated views. Solvers
in `vision-geometry` (`epipolar::{fundamental, essential, decomposition}`,
`homography`, `triangulation`, `camera_matrix`); orchestration and diagnostics
in `vision-mvg` (`pose_recovery`, `cheirality`, `triangulation`, `degeneracy`,
`robust`, `homography`). Crate split per ADR 0006.

## Model

Two views of a rigid scene through calibrated cameras (ADR 0028). The calibrated
functions take a `CalibratedRig` and **pixels**; each pixel becomes a unit
**bearing** `b` through its camera's own model (`backproject_pixel`,
normalized), so every camera model works, including a fisheye beyond 90° from
its axis, where `z = 1` normalized coordinates do not exist. For a pinhole,
`(x, y, 1)` of `K⁻¹ u` is a bearing. Relative pose `second_se3_first = (R, t)`
maps camera 1 into camera 2 (ADR 0009/0024): `X_c2 = R·X_c1 + t` (`t` up to
scale, unit in the result).

- **Epipolar constraint** (calibrated, on bearings): `b₂ᵀ E b₁ = 0`,
  `E = [t]× R`, rank 2, singular values `(σ, σ, 0)`.
- **Fundamental** (pixels): `u₂ᵀ F u₁ = 0`, `F = K₂⁻ᵀ E K₁⁻¹`, rank 2.
- **Homography** (planar scene *or* pure rotation): `x₂ ~ H x₁`,
  `H = R + t nᵀ/d` (`homography_from_pose_and_plane`).
- **Triangulation**: given projection matrices `P_i` and observations `x_i`,
  find `X` with `x_i ~ P_i X̃`.

## Solvers (initialization)

- **Fundamental, normalized 8-point** (`fundamental_8point`): Hartley-normalize
  each point set (`normalize_points_2d` — centroid to origin, mean distance
  `√2`), stack the 9-column epipolar design `A`, take the 1-D null space,
  force rank 2 by zeroing the smallest singular value of the `3×3` SVD, then
  de-normalize `F = T₂ᵀ F_raw T₁`.
- **Fundamental, 7-point** (`fundamental_7point`): minimal solver; a **2-D**
  null space spans `F₁, F₂`, and `det(F₂ + λF₁) = 0` (a cubic via
  `solve_cubic_real`) gives up to three rank-2 candidates.
- **Essential, 5-point** (`essential_5point`, Nistér): a **4-D** null space
  spans `E₁..E₄`; the ten cubic constraints (`build_polynomial_system`) are
  reduced by Gauss-Jordan elimination to a `10×10` action matrix whose real
  eigenvalues (Schur decomposition) yield up to ten candidate `E`. Unit bearings
  are already `O(1)`, so **no Hartley normalization**. The design row of a pair
  is the Kronecker product `b₂ ⊗ b₁` (each normalized first); the earlier
  normalized-coordinate row is its `z = 1` case.
- **Essential, linear ≥8-point** (`essential_linear`): 1-D null space, then
  projection onto the essential manifold (singular values → `(σ, σ, 0)`).
  Used for the RANSAC refit on inlier sets.
- **Camera matrix DLT** (`dlt_camera_matrix`): normalized 6+-point DLT for
  `P = K[R|t]`, with a coplanarity guard (scatter `σ_min/σ_max`) and a rank
  guard; `rq_decompose` / `decompose_camera_matrix` recover `K, R, t`.

**Hartley normalization.** With raw pixels the `u·x` columns of the epipolar
design are `O(10⁵)` while the last column is `O(1)`; the recovered `F` fails
`|u₂ᵀ F u₁|/‖F‖` by orders of magnitude. Centering and isotropic scaling
equalize the columns. Regression:
`fundamental_8point_epipolar_constraint_pixel_coords` (normalized residual
`< 1e-6` at pixel scale).

**Null-space extraction.** The **1-D** homogeneous solves (8-point `F`,
`essential_linear`, homography, camera-matrix and triangulation DLTs) take
their null vector from `vision_calibration_core::linalg::null_space` — the
smallest-eigenvalue eigenvector of `AᵀA` by *symmetric* eigendecomposition —
not `A.svd(...)`: nalgebra's Golub-Kahan sweep can fail to converge
(minutes-long hangs) on real dense designs, even with `compute_u = false`,
whereas `AᵀA` is `k×k` (`k ≤ 12`) and converges in a few sweeps. The
**multi-vector** null spaces (7-point 2-D, 5-point 4-D) use
`svd(true, true)`, since `null_space` returns a single vector.

## Pose recovery, the 4-fold ambiguity, and cheirality

`decompose_essential` projects `E` to the manifold and returns **four**
candidates `{R₁, R₂} × {+t, −t}` (`R₁ = U W Vᵀ`, `R₂ = U Wᵀ Vᵀ`,
`t = ±u₃`), with a raw-`σ` guard on the input (`σ₀ < 1e-10` or
`σ₁/σ₀ < 0.1` ⇒ `Err` — rank-1 / zero `E`). Only one candidate places
the points ahead along **both** bearings; `cheirality::select_pose` counts
them (`cheirality_count`) and picks the winner. A candidate `(R, t)` with
`p₂ = R p₁ + t` and `p₁ = λ₁ b₁`, `p₂ = λ₂ b₂` solves the `3×2` least squares
`[R b₁, −b₂] [λ₁, λ₂]ᵀ = −t`; the point is in front iff `λ₁ > 0` and
`λ₂ > 0`. This is a ray test, not a sign test on `z`, so it holds beyond 90°.

`recover_relative_pose` (`pose_recovery`) samples up to 20 evenly-spaced
5-point subsets (deterministic, input-order independent), decomposes and
cheirality-selects each candidate `E`, keeps the one with the most
cheirality inliers (ties broken by the mean weighted Sampson error over
**all** matches), then triangulates with the chosen pose. The robust variant
(`recover_relative_pose_robust` → `estimate_essential`) runs 5-point + the
weighted Sampson error inside `ransac_fit` (scoring minimal-sample models),
fits the consensus set with `essential_linear`, starts from its
decomposition with the most inliers ahead of both cameras, and refines it
through the problem builder (below); cheirality is voted on inliers only.

## Cost

The solvers minimize algebraic / first-order error. The RANSAC residual for
`E` is the covariance-weighted Sampson error, in **σ** of the stated pixel
noise,

`e² = g² / (g₁ᵀ m₁ g₁ + g₂ᵀ m₂ g₂)`, `g = b₂ᵀ E b₁`,
`g₁ = (I − b₁b₁ᵀ) Eᵀ b₂`, `g₂ = (I − b₂b₂ᵀ) E b₁`,

with `mₖ = Jₖ Σₖ Jₖᵀ` the bearing covariance a pixel's stated covariance
maps to through the camera model. It is invariant to the scale of `E`, χ²₁
to first order under the stated noise, and `∞` at an epipole (vanishing
denominator).

**Refinement (`robust::refine`).** The consensus set is refined to
`min Σ e²` as a relative pose, `E = [t]× R` over an `So3` block (`R`,
right perturbation) and a `UnitVector` block (`t`, the gauge `|t| = 1`),
with `e` evaluated generically so autodiff differentiates the denominator
too, `Noise::Known(1)` (already whitened). The inliers are re-selected at
the refined `E` and refined again from the previous fit (≤ 5 refinements;
a shrinking set stops the rounds and keeps the fit of the set before it), so
`inliers` are the matches the estimate is fitted to. The covariance is the
builder's `(JᵀJ)⁻¹` over `[rx ry rz]` and the ambient `[x y z]` of `t`
(rank 2); `E`'s 9×9 (rank 5) is `J_E Σ J_Eᵀ` with `∂E/∂δωₖ = [t]× R [eₖ]×`,
`∂E/∂tₖ = [eₖ]× R`. The cameras enter as exact: the calibration's share is
not in it (open, MVG-TWO-VIEW).

Holding each `σ_g` fixed within a step (the deleted local Gauss–Newton
polish, an iteratively reweighted scheme) converges linearly to a fixed
point that is **not** the minimizer: the stationarity it solves drops
`Σ eᵢ² ∂ log σᵢ`, a systematic term that grows as `N` as the information
does, so the offset stays put while σ shrinks as `1/√N`: its bias in σ
grows as `√N`. On
2000 fisheye + pinhole matches it stopped at `Σ e² = 2120.71` against the
minimum `2108.02`. Over 300 noise draws (fisheye `Σ = [0.25 0.08; 0.08
0.09]` px², pinhole 0.4 px) its bias was `−0.66, −0.69, −0.02, −0.86,
+0.34` σ in `(rx, ry, rz, t_a, t_b)` with 120 matches (SE 0.058) and
`−2.18, −2.07, +0.12, −2.38, +1.24` σ with 1000, its RMS error over the
stated σ `1.29–1.47` (120) and up to `2.63` (1000); the exact minimizer's
bias is `≤ 0.10` σ and its RMS over σ `0.97–1.07` at both sizes. The
pixel residuals (`residuals::sampson_distance` on `F`, symmetric transfer on
`H`) are unchanged and serve the uncalibrated estimators. The linear DLT is
the seed of `vision-mvg`'s `triangulate`, which refines the points against
pixels through the public problem builder (ADR 0028), where the camera model
and the calibration's covariance are available; no midpoint or optimal
(Hartley–Sturm) triangulator exists.

## Triangulation and its failure modes

`vision_geometry::triangulation::triangulate_point_linear(cam_se3_world,
bearings)` stacks the three rows of `[b]× P` (rank 2, `P = [R | t]`) of every
view into a `3N×4` DLT, solves it via `null_space`, guards rank with
`dlt_rank_ok`, and rejects a point at infinity (`w ≈ 0`).
`vision_mvg::triangulation::triangulate` builds the bearings from pixels
through a `CalibratedRig` (camera `i` at capture `k` sees the world through
`cam_se3_rig[i] · rig_se3_world[k]`) and adds diagnostics: RMS pixel
reprojection through each camera's model (`None` if the point does not
project), the **widest** pairwise parallax between rays from the camera centres,
and `in_front` (`b · p_cam > 0` for every bearing). It is all-or-nothing and
names the failing track. Failure modes:

- **Zero / low parallax** (narrow baseline or far points): rays are nearly
  parallel and the DLT's boundary singular value collapses; `dlt_rank_ok`
  rejects the exactly-degenerate case, and near-degenerate points have depth
  variance `∝ 1/parallax` (quantified below).
- **Point on the baseline / near the epipole**: the viewing rays are
  collinear ⇒ rank-deficient ⇒ rejected per point; `triangulate` fails the
  call naming the track, and `RelativePose` fails the call naming the match.
  `degeneracy::analyze_scene` (now `pub(crate)`, computed into
  `RelativePose::scene`) takes the points that triangulated, and measures depth
  along the first camera's bearing (`‖p‖` in its frame), not `z`.
- **Identical cameras** (zero baseline): rejected
  (`triangulation_rejects_identical_cameras`).

## Laser points

A laser-line pixel `u` seen by rig camera `k` lies on the viewing ray and on
the camera's laser plane `nᵀp + d = 0`, which fixes one point. The point
solves `F(p; u, θ, n, d) = [π(p; θ) − u; nᵀp + d] = 0` (`vision_mvg::laser`):
the ray's intersection `p = t b`, `t = −d / (nᵀb)`, is the start, polished by
Newton steps on `F` to 1e-9 px so the forward model holds exactly (Brown–
Conrady undistortion is iterative; the forward projection is closed form).
`t ≤ 0` (the plane behind the camera) and `|nᵀb| ≤ 1e-9` (a grazing ray) are
typed errors; the incidence `asin|nᵀb|` is reported. Rays beyond 90° from the
axis (a fisheye) need nothing special: `p` is a 3-vector throughout.

First order, with `A = [∂π/∂p; nᵀ]` (3×3, camera frame):

```text
∂p/∂u = A⁻¹ [I₂; 0]      ∂p/∂θ = −A⁻¹ [∂π/∂θ; 0]
∂p/∂n = −A⁻¹ [0; pᵀ]     ∂p/∂d = −A⁻¹ [0; 1]
```

`q = R p + t` (`rig_se3_cam = (R, t)`) is the rig-frame point. The camera's
extrinsic is perturbed as `cam_se3_rig · Exp(ω, v)` (the tangent of the
export's covariance), which moves `q` by `[q]ₓ ω − v`. With the calibration's
covariance `Σ_c` over the coordinates it lists (camera `θ`, plane, extrinsic,
all with their correlations) and the pixels' stated `Σ_u`,

```text
Σ_q = J_c Σ_c J_cᵀ + blockdiag(J_u Σ_u J_uᵀ)
```

over all points of a call (the calibration is shared, so they are
correlated). The plane's normal enters through its three components; its
covariance is rank 2 (⟂ `n`), so the unconstrained derivative is exact to
first order. The σ is conditional on what the calibration held fixed, and the
pixel noise is the laser-peak detector's, stated. Cost `O(N² |c|)`, `3N×3N`
memory.

**Evidence.** The derivatives against central differences (camera fields,
renormalised plane, camera 1's extrinsic tangent, pixel) agree to 3e-8
relative. Monte Carlo in parameter space (4000 draws of the camera, plane and
extrinsic from a synthetic `Σ_c` with camera-plane correlations, a rank-2
plane block, and fresh pixels): per-point 3×3 sample against stated, relative
Frobenius error 0.015–0.036 (gate 0.08), a cross block 0.021 (gate 0.1); the
pixel-only variant 0.013–0.034; dropping the camera×plane cross terms moves
the stated σ by 0.23. End to end (`tests/laser_uncertainty.rs`, 300
calibrations of a laserline device from 0.3 px pixels, eight plane points at
0.1 px): depth scatter over stated σ 1.02 (gate 0.85–1.15), the length between
two points 0.99 (gate 0.8–1.25); the σ from the pixels alone is 0.017 mm
against 4.7 mm.

## Identifiability and degeneracies

- **Pure rotation** (`t = 0`): `E = 0` — `decompose_essential` rejects it;
  the correct model is `H = R`. `detect_pure_rotation` flags it from the
  **median parallax angle** (scale-invariant) rather than `‖E‖`, which is
  scale-dependent and would misfire on a down-scaled valid `E`.
- **Planar scene**: coplanar structure makes the epipolar design
  rank-deficient — `F`/`E` are under-determined, `H` is well-posed.
  `detect_planar_scene` compares homography vs essential inlier support;
  `dlt_camera_matrix` rejects coplanar 3D points. The matrix and property
  tests use **non-coplanar** clouds.
- **Narrow baseline / low parallax**: `E` rank is fine, but translation
  *direction* and depth are ill-conditioned, degrading `∝ 1/parallax` — the
  dominant real-data sensitivity.
- **Critical surfaces** (structure + both centres on a ruled quadric): a
  known theoretical `F` ambiguity; not guarded, as the planar case dominates
  in practice.

## Gauge

Two-view reconstruction is fixed only up to a **similarity** (7 DOF). Fixing
camera 1 at `[I | 0]` removes the 6-DOF rigid gauge, and
`decompose_essential` returns a **unit** `‖t‖`, so metric depth is
unobservable from two views; the remaining scale must come from context (a
known baseline, frozen extrinsics, or bundle adjustment against metric
structure). The matrix test therefore reports rotation and translation
*direction* for pose recovery, and gives the triangulation cells the true
metric rig.

## Noise sensitivity

Matrix test (42-point non-coplanar cloud, `f ≈ 800`, uniform per-axis noise
`a`). At `a = 0.5 px`:

- **Rotation** error `0.06–0.12°`, nearly flat across baseline.
- **Translation direction** climbs as the baseline narrows — `0.19°` (wide,
  `‖t‖ = 0.8`) → `0.32°` → `0.58°` → `1.91°` (narrow, `‖t‖ = 0.10`).
- **Metric 3D triangulation** median error `9.6e-3` (wide) → `7.1e-2`
  (narrow), a `~7×` spread at fixed noise — the same `1/parallax` law.
- **Post-fit reprojection** median `≈ 0.15 px`, tracking the noise floor.
- **Exact data**: rotation / t-direction `< 1e-3°`, reprojection
  `~1e-13 px`, 3D error `~1e-15`.

With the linear triangulation alone (no Gauss-Newton polish) the median 3D
errors at 0.5 px are `9.3e-3` (wide) → `7.0e-2` (narrow), within 3 % of the
polished values the table above was measured with; pose errors are unchanged
(the pose pipeline is the same, on bearings).

Parameter errors scale ~linearly in `a`; translation direction and depth
additionally scale ~`1/parallax`.

## Evidence

- **Relative pose covariance**: `crates/vision-mvg/tests/relative_pose_cov.rs`
  — fisheye (anisotropic, correlated noise) + distorted pinhole, 120 points,
  600 draws: scatter over stated σ `1.001, 1.037, 1.055` (rotation) and
  `1.031, 0.978` (direction tangent), NEES `5.29` (5 expected, SE 0.13),
  `E` pooled `1.029`; gate `[0.9, 1.1]` per coordinate (≥ 3 SE wide). The
  labels, rank, observer determinism and cancellation are checked there too;
  the cheirality re-solve from a non-physical decomposition in `robust.rs`.
- **Pixel-noise scale**: `crates/vision-mvg/tests/pixel_noise.rs` — inlier
  RMS `1.0147` σ at the stated noise (`1.0181` before the exact refinement),
  `0.5073` with doubled σ; at `f = 400 / 1600` a 3 σ threshold keeps
  `100.0 % / 99.5 %` of the inliers and `0.50 %` of the outliers.

- **Matrix test**: `crates/vision-mvg/tests/two_view_matrix.rs` — 4
  baseline/parallax regimes (`wide-sideways`, `wide-rotated`, `medium`,
  `narrow`) × noise `{0, 0.2, 0.5} px`; per cell asserts rotation,
  translation direction, reprojection and 3D error, scaled to the noise.
- **Camera-model tests**: `crates/vision-mvg/tests/two_view_camera_models.rs`
  — a Brown–Conrady camera (`k₁ = −0.2`) from raw distorted pixels (noiseless
  `1e-9` relative in 3D, `1e-7°` in pose; 0.3 px noise within the matrix
  bounds) and an equidistant + Kannala–Brandt fisheye rig with a third of the
  points beyond 90° off the first camera's axis.
- **Property tests**: `crates/vision-mvg/tests/two_view_properties.rs` —
  over a pose grid: `E = [t]× R` annihilates GT correspondences and
  decomposes to the GT pose; estimated-`F` epipolar residual `< 1e-6`;
  triangulate → reproject round trip (`< 1e-6 px`, `< 1e-6` in 3D).
- **Unit tests**: the Hartley regression (`epipolar/fundamental.rs`), the
  dense-DLT anti-hang guard `dense_homography_is_exact_and_fast`
  (`homography.rs`), essential-manifold and degeneracy guards
  (`epipolar/essential.rs`, `epipolar/decomposition.rs`), cheirality
  (`cheirality.rs`), triangulation (`triangulation.rs`), and the scene
  degeneracy detectors (`degeneracy.rs`).
- **Related**: ADR 0006 (the `vision-geometry` / `vision-mvg` split);
  [planar-intrinsics.md](planar-intrinsics.md) (Zhang init shares the
  Hartley-normalized DLT homography).
