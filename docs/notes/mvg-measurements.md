# MVG measurements — proof pack

Family: measurements from a calibrated system with their σ (ADR 0028), in
`vision-mvg`, re-exported as `vision_calibration::mvg`:

- the pose of a known object (`pose::estimate_pose`);
- laser points, a pixel on a calibrated laser plane (`laser::triangulate_laser`);
- metrology over points with a joint covariance: distance, angle, plane fit,
  point to plane (`metrology`);
- the dense cloud of a disparity map (`StereoRectification::dense_cloud`,
  `rectification/cloud.rs`).

Triangulation (`triangulation::triangulate`, `StereoRectification::triangulate`)
shares the machinery of the object pose and is summarized here for the
evidence; its linear start, diagnostics and failure modes are in
[two-view-triangulation.md](two-view-triangulation.md), the rectification
geometry in [rectification.md](rectification.md), the calibration covariance
every measurement consumes in [uncertainty.md](uncertainty.md).

## Model

A measurement takes a `CalibratedRig`: the cameras, each camera's
`cam_se3_rig` (the reference camera's the identity), optionally a laser plane
per camera and a hand-eye, and the calibration's `Uncertainty`, one joint
covariance `Σ_c` over the coordinates `c` it lists, by the export's labels
(`fx`, `cam1.k1`, `cam1.cam_se3_rig.rx`, `cam1.plane.nx`,
`gripper_se3_rig.tx`). Poses are perturbed on the right, `T · Exp(ω, v)`,
`[ω, v]` = `[rx ry rz tx ty tz]`.

Observations are pixels `u` with a **stated** covariance `Σ_u`
(`PointCovariance`: general, isotropic, or the calibration's own
reprojection level, `CalibratedRig::reprojection_noise`). No measurement
estimates its noise: a laser or dense point has no redundancy, and a pose
or a triangulated point too little to estimate a level the detector
already states. Each pixel enters through its camera's own model
(bearings for closed-form starts, the forward projection `π(p; θ)` for
refinement and derivatives), so every camera model works without a
model-specific path.

A measurement `x̂(u, c)` depends on its pixels and on the calibration. To
first order

```text
Σ_x = J_u Σ_u J_uᵀ + J_c Σ_c J_cᵀ,     J_c = ∂x̂/∂c
```

the cross term being zero (the measurement's images are not the
calibration's). The calibration's share is common to every quantity
estimated with it, so the points of one call are correlated through
`J_c Σ_c J_cᵀ`, and a measurement combining them (a length) must read their
joint covariance. Every API therefore returns the dense joint covariance of
the points of one call (a dense cloud, of the points asked for), and a
measurement takes all its points from one call.

`J_c` comes from the implicit-function theorem on the condition that defines
`x̂`, by one of two routes:

- **Fitted (the builder).** When `x̂` minimizes a cost
  `Σᵢ ρ(‖Lᵢ⁻¹ rᵢ(x; c)‖²)` over redundant pixels (object pose,
  triangulation), it satisfies `g(x̂; c) = R_xᵀ W r = 0`, with `R_x = ∂r/∂x`,
  `R_c = ∂r/∂c` and `W` the whitening and robust weights. Differentiating,
  `S = ∂x̂/∂c = −H⁻¹ H_xc`, `H = R_xᵀ W R_x`, `H_xc = R_xᵀ W R_c`. The
  builder (`optim::problem`, `add_calibrated_rig(rig, Use::Consider)`) holds
  `c` fixed during the solve, linearizes once more with `c` free, and adds
  `S Σ_c Sᵀ` to the sandwich covariance of the observations
  (`backend/covariance.rs::consider_share`).
- **Determined (closed form).** When `x̂` solves as many equations as it has
  unknowns, `F(x; u, c) = 0` (laser point: 3 equations; dense point: 3), then
  `∂x̂/∂c = −A⁻¹ ∂F/∂c`, `A = ∂F/∂x`, and `∂x̂/∂u` likewise. This is the
  fitted route with a square `R_x`: `H⁻¹ H_xc = R_x⁻¹ R_c`. It needs no
  solver; the derivatives of `π` are exact, by dual numbers
  (`optim::CameraJacobian`, the solver's own kernels).

Derived quantities (metrology) propagate the joint covariance of their points
linearly, `σ² = g Σ gᵀ` with `g = ∂f/∂x` by dual numbers.

**Held exact, in every measurement.**

- Calibration parameters the covariance does not list: the calibration held
  them fixed (`fix_k3`, a reference camera's pose, a staged problem's
  upstream), so σ is conditional on them. A pose is listed whole or not at
  all; a partly listed `cam_se3_rig` is `InvalidInput`.
- The known geometry: object points of a pose, the board's corners.
- The estimate itself: asking for σ never moves it. The consider route
  solves with the calibration's values; the pose, laser, triangulation and
  stereo gates assert the estimate with and without the calibration's share
  agrees to 1e-9 m.

**Unavailable, never partial.** A rig without a covariance gives the
measurement and `UncertaintyUnavailable::CalibrationWithoutCovariance`
(likewise a laser rig whose covariance lists none of an observing camera's
intrinsics): a σ from the pixels alone would omit the larger share (below).
A user who wants that σ adds the cameras fixed through the builder. A
refinement that stops on its iteration budget gives `NotConverged`.

**Frames (gauge).** A measurement is in the rig frame, which the reference
camera fixes, or in the frame `Captures` names: `Rig`, `World` (exact
`rig_se3_world` per capture), `Handeye(frame)` or `Robot { frame, robot }`
(gripper or base, through the calibration's hand-eye; see "In the robot's
frames"). The calibration fixes the metric scale (board squares or a stereo
baseline), so no measurement here has a free gauge; bundle adjustment, which
does, is in [uncertainty.md](uncertainty.md).

## Object pose

`estimate_pose(rig, observations, opts)`: known object points `Pᵢ` (object
frame) seen at pixels `uᵢ` by rig cameras `kᵢ`; the result is
`rig_se3_object`.

1. **Start.** The camera with the most observations (lowest index on a tie)
   maps its pixels to bearings, keeps those with `z > 1e-3`, and solves
   `cam_se3_object` on their normalized coordinates with EPnP (a planar
   object through its homography inside `epnp`), or, with
   `PoseOptions::ransac`, P3P on three points with the fourth choosing among
   its roots, inside a RANSAC whose residual is the reprojection distance in
   σ of the stated noise (`whitened_norm`, so a threshold of 3 means the same
   for every focal length and model). `rig_se3_object =
   cam_se3_rig[k]⁻¹ · cam_se3_object`.
2. **Refine** on every inlier of every camera, a generalized PnP:

   ```text
   min over T = rig_se3_object of  Σᵢ ρ(‖Lᵢ⁻¹ (π_kᵢ(cam_se3_rig[kᵢ] · T · Pᵢ; θ_kᵢ) − uᵢ)‖²),   Σ_uᵢ = Lᵢ Lᵢᵀ
   ```

   through the builder: the rig added with `Use::Consider`, one SE3 block
   `object`, one `PixelNoise::Stated` reprojection per observation
   (`TargetPoint::Known`). With RANSAC the inliers are re-selected over every
   camera at the refined pose and refined again until the set settles (at
   most three rounds).
3. **σ.** `Σ_T = H⁻¹ + S Σ_c Sᵀ` over `object.rx … object.tz`, the
   right-perturbation tangent (`ObjectPose::pose_covariance`, 6×6). Stated
   noise is known information (no variance factor); with a robust loss the
   covariance takes the residuals at their stated variance.

Held exact: the object's geometry, the robust weights. Observable: four
non-collinear points in the initializing camera fix the pose; a planar
object is fine. The translation along the viewing axis and the lateral
translation carry most of the calibration's share: the principal point and
the focal length move the whole object (pixels-only σ of `tx` is 26 times
too small in the gate below, of `tz` 3.4 times).

## Triangulated points

`triangulate(rig, captures, tracks, control)`: each track starts from the
bearing DLT of `vision-geometry` and all tracks are refined in one builder
problem on the stated reprojection error, with the rig as consider
parameters (and on a robot each capture's pose as a fixed SE3 block whose
tangent joins the consider coordinates, `Problem::add_rig_pose`). The result
is the joint `3N × 3N` covariance `H⁻¹ + S Σ_c Sᵀ` (labels `p{i}.x/y/z`). A
track behind a camera keeps its linear point and has no covariance.
`StereoRectification::triangulate` maps rectified matches back to raw
pixels and calls it as one stereo capture.

## Laser points

A laser-line pixel `u` seen by rig camera `k` lies on the viewing ray and on
the camera's laser plane `nᵀp + d = 0`, which fixes one point. The point
solves `F(p; u, θ, n, d) = [π(p; θ) − u; nᵀp + d] = 0` (`vision_mvg::laser`):
the ray's intersection `p = t b`, `t = −d / (nᵀb)`, is the start, polished by
Newton steps on `F` to 1e-9 px (at most five; a residual above 1e-6 px is
`Numerical`) so the forward model holds exactly (Brown–Conrady
undistortion is iterative; the forward projection is closed form).
`t ≤ 0` (the plane behind the camera) and `|nᵀb| ≤ 1e-9` (a grazing ray) are
`Degenerate`; the incidence `asin|nᵀb|` is reported. Rays beyond 90° from
the axis (a fisheye) need nothing special: `p` is a 3-vector throughout.

First order, with `A = [∂π/∂p; nᵀ]` (3×3, camera frame):

```text
∂p/∂u = A⁻¹ [I₂; 0]      ∂p/∂θ = −A⁻¹ [∂π/∂θ; 0]
∂p/∂n = −A⁻¹ [0; pᵀ]     ∂p/∂d = −A⁻¹ [0; 1]
```

`q = R p + t` (`rig_se3_cam = (R, t)`) is the rig-frame point. The camera's
extrinsic is perturbed as `cam_se3_rig · Exp(ω, v)` (the tangent of the
export's covariance), which moves `q` by `[q]ₓ ω − v`. With the calibration's
covariance `Σ_c` over the coordinates it lists (camera `θ`, plane, extrinsic,
all with their correlations: the planes and cameras were estimated in one
solve) and the pixels' stated `Σ_u`,

```text
Σ_q = J_c Σ_c J_cᵀ + blockdiag(J_u Σ_u J_uᵀ)
```

over all points of a call. The plane's normal enters through its three
ambient components; its covariance is rank 2 (⟂ `n`), so the unconstrained
derivative is exact to first order, and the normal's sign (an S² discrete
gauge) does not reach the point. The covariance's `dof` is 0 and
`num_residuals` the pixel count: the points are determined, not fitted.
Cost `O(N² |c|)`, `3N×3N` memory.

Held exact: what the calibration held (a plane fitted against a frozen rig
by `rig_laserline_device` carries that upstream's share in its own
covariance, ADR 0028; the hand-eye of a staged export is exact, parked as
MVG-HANDEYE-STAGED). Rejected: the builder route; a single laser point has
no redundancy, and the closed form is the same linear algebra without a
solver.

### In the robot's frames

A rig with a hand-eye `H` (`gripper_se3_rig` eye-in-hand, `rig_se3_base`
eye-to-hand) measures in the gripper or base frame. Capture `k`'s pose in the
measurement's frame, `rig_se3_world`, is the chain of `H` and the robot's
`P = base_se3_gripper`, and its right perturbation is `ξ = A ε_h + B ε_r`
(`Ad(T) = [[R, 0], [[t]ₓR, R]]` over `[ω, v]`):

| mode, frame | `rig_se3_world` | `A` | `B` |
|---|---|---|---|
| eye-in-hand, gripper | `H⁻¹` | `−Ad(H)` | — |
| eye-in-hand, base | `(P H)⁻¹` | `−Ad(P H)` | `−Ad(P)` |
| eye-to-hand, base | `H` | `I` | — |
| eye-to-hand, gripper | `H P` | `Ad(P⁻¹)` | `I` |

A laser point `x = rig_se3_world⁻¹ q` rotates every rig-frame derivative by
the capture's rotation and moves with `ξ` by `[x]ₓ ω − v`, so `J_c` gains the
hand-eye's six columns `[[x]ₓ, −I] A` under its covariance labels (correlated
with the cameras, planes and extrinsics) and each robot pose six columns
`[[x]ₓ, −I] B` with its stated `Σ_r`, block-diagonal against `Σ_c`. A
triangulated point goes through the builder: the capture's pose is a fixed
SE3 block (`Problem::add_rig_pose`) whose tangent joins the consider
coordinates with `Cov(ξ, c) = A Cov(h, c)`, `Cov(ξ_k, ξ_l) = A_k Σ_hh A_lᵀ +
δ_kl B_k Σ_rk B_kᵀ`, and `S Σ_c Sᵀ` does the rest. A robot's systematic
(kinematic) error across poses is not modelled.

## Metrology

`PointSet` holds points and their joint `3N × 3N` covariance, from one
`triangulate`, `triangulate_laser`, `bundle_adjust` or `DenseCloud`
call (or given; it must be finite, symmetric within 1e-9 of its largest
entry, with a non-negative diagonal).

- **Distance** `‖a − b‖`; coincident points are `Degenerate`.
- **Angle** at a vertex, as `2 atan(|u × w| / (|u||w| + u·w))`, which keeps
  its slope near π/2 where `acos` and the dual `atan2` lose it; 0 and π are
  `Degenerate` (not differentiable).
- **Plane** `n·x = d` by total least squares: `n` the eigenvector of the
  smallest eigenvalue of the scatter about the centroid, oriented `d ≥ 0`.
  Collinear points (middle eigenvalue below 1e-12 of the largest) are
  `Degenerate`. `rms` is the orthogonal misfit.
- **Point to plane**, the signed distance `n·p − d`; `p` may be one of the
  plane's points.

σ: `g Σ gᵀ` with `g` by dual numbers over the coordinates of the points
involved and `Σ` their block of the joint covariance, cross terms included.
The plane's parameters are `θ = (α, β, d)`, `n(α, β) = normalize(n₀ + α t₁ +
β t₂)` in a deterministic tangent basis of the fitted `n₀` (`tangent_basis`).
The fit minimizes `F(θ, p) = Σ (n(θ)·pᵢ − d)²`, so at the fit `g(θ, p) =
∂F/∂θ = 0` and

```text
∂θ/∂p = −(∂g/∂θ)⁻¹ ∂g/∂p,     Σ_θ = (∂θ/∂p) Σ_p (∂θ/∂p)ᵀ
```

both blocks from one dual pass over `(θ, p)`. The points are the plane's
consider parameters: the plane moves with them, and their misfit does not
widen it. Point to plane chains `∂s/∂p + ∂s/∂θ · ∂θ/∂p`.

Held exact: nothing beyond the input covariance. Gauge: `(α, β)` are
coordinates of the normal's rotation in `tangent_basis` (rad), a choice of
chart; the normal's sign is fixed by `d ≥ 0` (ambiguous only for a plane
through the origin).

## Dense cloud

A dense match is a pixel `(x, y)` of the rectified left image, a grid point
and so exact, and a disparity `d` measured along the row of the rectified
right image. The rectified images were resampled with the calibration's
values, so both maps are fixed: the left pixel is the raw pixel
`u_L = raw_L(x, y)`, and the right measurement says only that the right
camera saw the point on the raw-image curve of rectified column `x − d`, at
`u_R = raw_R(x − d, y)` to first order along it. The row there (the vertical
disparity a calibration error causes) is not measured.

`dense_cloud` builds each point from the fixed rectified geometry
(`X = rig_se3_left · R_leftᵀ · P_rect(x, y, d)`). The point is the implicit
function

```text
F(X; θ, ξ, d) = [ π_L(X) − u_L ;  r (π_R(X) − u_R(d)) ] = 0
```

with `π_k(X) = π(cam_se3_rig_k · Exp(ξ_k) · X; θ_k)` and `r = e₁ᵀ M⁻¹`, `M =
∂raw_R/∂(rectified pixel)`, the row of the fixed right map's inverse that
gives the rectified column. Three equations, three unknowns, so with
`A = ∂F/∂X = [D_L R_L; r D_R R_R]`

```text
G = ∂X/∂c = −A⁻¹ ∂F/∂c        g = ∂X/∂d = −A⁻¹ e₃ = −(X − c_L)/d
Σ_i = G_i Σ_c G_iᵀ + σ_d² g_i g_iᵀ,     Cov(X_i, X_j) = G_i Σ_c G_jᵀ
```

(`c_L` the left camera's centre). The cloud stores the points only and
computes `G_i` when asked (`point_covariance(i)`, `covariance(&indices)`,
`PointSet::from_dense_cloud`): `G_i` is `3 × |c|` doubles a point, about 30
columns for a stereo pair, too much for a full map.

Held exact: the rectification (how the measurement was taken, not a
parameter of the point, so it is never differentiated), the left pixel, the
right row. `σ_d` is one stated value for the map, the disparities'
errors independent. Rejected: `triangulate` with isotropic noise on both raw
pixels, a cost cubic in the point count and another measurement model (it
splits a vertical disparity between the cameras: the depth σ agrees within
0.5 %, the across-row σ differs up to 40 %).

## Evidence

**Monte-Carlo through the calibration** (the facade's
`crates/vision-calibration/tests/*_uncertainty.rs`). Each trial calibrates
from freshly noisy board views with the public workflow, then measures from
freshly noisy pixels stated at the true σ; the gate compares the scatter of
the measurement about the truth with the σ it states, and the σ without the
calibration's share (`without_calibration_share`, `Σ_c → 0`) must
understate. The stereo scene: two 820–850 px cameras 12 cm apart, eight
calibration views of an 8×6-corner board (30 mm) about 0.6 m away that fill
the field (tilts up to 0.5 rad, moved 12 cm sideways and 9 cm up or down),
0.3 px Gaussian noise. The laser scene: one 900 px camera, a plane at 16.7°
crossing the axis at 0.5 m, a laserline-device calibration from six views
of a 9×7 board with 0.3 px corners and stripe pixels. Measured on
2026-11-12, release build:

| Gate | Measurement | Trials | Assertion | Result |
|---|---|---|---|---|
| `pose_uncertainty` | 8×6 board (30 mm) at 0.7 m, camera 0 alone | 300 | scatter/stated per component within [0.85, 1.15]; pixels-only `tz` > 1.15 | `rx…tz` 1.068, 0.988, 0.985, 0.963, 1.038, 0.933; pixels only 1.82, 2.33, 2.34, 26.4, 18.1, 3.44 |
| `laser_uncertainty` | 8 plane points 0.4–0.6 m at 0.1 px, and the length between two | 300 | depth within [0.85, 1.15], pixels-only > 1.5; length within [0.8, 1.25] | depth σ 4.70 mm, 1.021 (pixels only 0.017 mm, 288); length (RMS error 0.65 mm) 0.990 (pixels only 7.6) |
| `metrology_uncertainty` | held-out 8×6 board (25 mm) at 0.7 m, both cameras: 3- and 7-square lengths, right angle, plane `d` | 300 | ratio within [0.85, 1.15]; mean error under 0.3 σ | 0.998, 1.019, 0.970, 0.891; bias +0.159, +0.066, −0.029, −0.067 σ; observations only 1.002, 1.030, 0.971, 2.195 |
| `triangulation_uncertainty` | 10 points 0.6–0.8 m, both cameras | 200 | depth within [0.85, 1.15]; observations-only > 1.2 | 3.04 mm, 0.97; observations only 2.05 mm, 1.46 |
| `stereo_uncertainty` | the same points as rectified matches (`StereoRectification::triangulate`) | 200 | as above | 3.04 mm, 0.97; observations only 1.46; `disparity_covariance` alone 2.06 mm, 1.46 |
| `dense_cloud_uncertainty` (`#[ignore]`, CI uncertainty job) | 10 grid pixels of a 0.6–0.8 m scene, disparity simulated on each calibration's rectified images, σ_d = 0.2 px | 200 | depth within [0.85, 1.15]; disparity-only > 1.5 | 2.55 mm, 0.907 (0.93–0.95 on three other seed ranges); disparity only 0.97 mm, 2.46; largest bias 0.07 σ |

Reading it: the calibration's share is most of the σ wherever a measurement
depends on an absolute position (the pose's lateral translation, a laser
point's depth, the board plane's distance, a stereo depth), and almost none
of it for a short length on this well-spread calibration (3.45e-4 against
3.43e-4 m at 75 mm), where the shared error cancels. The dense gate's 0.91
is the calibration covariance's conservatism, not the dense model's: at
`σ_d = 0` `dense_cloud` and `triangulate` state the same 2.36 mm against an
error 0.88 times that.

**Derivatives and the closed forms** (unit tests, `vision-mvg`):

- laser (`laser/tests.rs`): ground truth through every camera model; the
  derivatives against central differences (camera fields, renormalized
  plane, extrinsic tangent, pixel) to 3e-8 relative, in the robot frames
  7e-10; Monte-Carlo in parameter space (4000 draws of camera, plane,
  extrinsic and hand-eye from a `Σ_c` with cross terms, fresh pixels):
  per-point 3×3 sample against stated, relative Frobenius error 0.015–0.056
  (gate 0.08), cross blocks 0.021–0.047 (gate 0.1); dropping the camera ×
  plane cross terms moves the stated σ by 0.23.
- dense cloud (`rectification/cloud/tests.rs`, a three-camera rig measured
  with the pair `[1, 2]`, a Scheimpflug right camera): `G` against central
  differences of an independent re-solve 7.9e-7; the joint covariance
  against `triangulate` with the dense noise model (left σ 1e-3 px, right
  σ_d along `r`, 100 px across) 9.6e-8; against isotropic `triangulate`,
  depth σ ratio 1.000–1.005, across-row σ up to 1.40.
- metrology (`metrology.rs`, `tests/metrology.rs`): dual gradients against
  central differences 1.3e-10 (distance) to 5.4e-9 (point to plane); the
  plane sensitivity against refitting 7.4e-9; 20 000 draws of eight points
  with a correlated covariance (a common translation, scale, depth stretch
  and shear plus independent noise): scatter/stated 0.990–1.002 for every
  quantity, 0.44–0.91 with the cross terms dropped; the plane's `(α, β, d)`
  Mahalanobis² mean 2.99 (3 expected).
- pose (`tests/pose.rs`): noise-free poses exact (below 1e-12 rad, 1e-13 m)
  for pinhole, Brown–Conrady and KB4 fisheye cameras, planar and spatial
  objects, each camera alone and both; pixel noise alone (600 draws,
  correlated `Σ_u`): scatter/stated 0.98–1.02 with one camera, 0.99–1.06
  with two (gate [0.9, 1.1]); RANSAC in σ rejects 24 of 24 outliers among
  70 observations; two cameras shrink the translation σ 2.0× against the
  better one alone.
- in the robot's frames (`tests/robot_frames.rs`, core `handeye.rs`): `A`
  and `B` against central differences under 1e-7; triangulated points of
  two stations, 3000 draws: worst point block 0.024 / 0.059, cross block
  0.020 / 0.060; end to end (`tests/handeye_frame_uncertainty.rs`, 300
  joint rig hand-eye laserline calibrations, base frame): scatter over
  stated 1.008 for laser points (0.100 mm) and 1.011 for triangulated ones
  (0.45 mm); without the hand-eye's cross terms the laser σ is 0.41 mm
  (scatter/stated 0.29).

**Real data, gated.** The `opencv_stereo` entry of the regression suite
(`regression/suite.toml`, `measure_board = { square_size_m = 1.0 }`, code in
`vision-calibration-bench/src/suite/measure.rs`, which compiles the facade
example's `examples/support/board_measure.rs` unchanged) runs the
`measure_board` route on the committed observations: the 7 even views
calibrate (5 seen by both cameras; mean reprojection 0.199 px, σ̂ 0.146 px
per camera, view scatter 4.17), the 6 odd views' 54 corners are triangulated
per view with that σ̂, and the 2070 corner pairs on a row or column, 1–7
squares apart, are measured against the board. Lengths in squares:

| squares | pairs | mean error | RMS error | mean σ | z RMS |
|---|---|---|---|---|---|
| 1 | 558 | −0.0009 | 0.0096 | 0.0103 | 0.85 |
| 2 | 468 | −0.0021 | 0.0111 | 0.0107 | 1.00 |
| 3 | 378 | −0.0043 | 0.0125 | 0.0113 | 1.15 |
| 4 | 288 | −0.0067 | 0.0153 | 0.0122 | 1.30 |
| 5 | 198 | −0.0101 | 0.0177 | 0.0133 | 1.41 |
| 6 | 108 | −0.0138 | 0.0203 | 0.0147 | 1.51 |
| 7 | 72 | −0.0123 | 0.0202 | 0.0160 | 1.40 |

Per view the z RMS stays at or below 1.2 at every length in five views; one
(view 7) comes out 0.5 % small, z 2.9 at six squares, and carries the long
lengths' excess. The reference stores these rows per view and in total;
`suite check` re-measures and fails on any number beyond the entry's
tolerance (`rel 1e-6`), so a 1e-4 relative change of the distance σ fails it
(checked by mutation); `--backend factrs --sigma 1` compares each row's mean
error in units of the row's mean σ (a bound on the σ of a mean of correlated
lengths) and moves by 1.2e-7 σ at most. The fisheye walk-through of the book
(14 measured held-out views, z RMS 0.64–0.98) is not gated: its suite
entries are frozen from OpenCV's published corners, which the walk-through
does not use (it detects the images).

## Known limits

- **First order.** Every σ is linearized at the estimate. A length's
  second-order bias is 0.16 σ at 7.5 cm in the metrology gate; a grazing
  laser ray (σ ∝ 1/sin incidence), a near-straight angle and a far stereo
  point leave the linear regime first.
- **The calibration's σ is only as good as its covariance.** The share is
  `Σ_c` propagated; a calibration whose views barely separate parameters
  (centred, mildly tilted) has joint errors 2.5–3× its σ along a few
  combinations (see [uncertainty.md](uncertainty.md), "Second-order error"),
  which every measurement inherits; the facade gates use views that fill
  the field. On real images the covariance is scaled by the view scatter;
  on `opencv_stereo` the long lengths are still understated (z 1.4–1.5), a
  per-view scale error that a σ on independent pixels cannot describe.
- **Independent pixel errors.** Neighbouring corners of a real detection
  share errors, so short real-data lengths are conservative (z 0.85 at one
  square above; 0.64 on the fisheye set); a matcher's disparities are
  correlated over its window and biased toward integers (pixel locking),
  neither modelled.
- **Exact geometry.** Object and board points are exact: manufacturing and
  flatness errors are not in a pose's or a length's σ; a plane's covariance
  ignores its misfit (`rms` reports it).
- **Conditional on what the calibration held.** Fixed parameters, a staged
  laser export's hand-eye, and a robot's systematic error count as exact.
- **Dense covariance.** Joint covariances are dense: `O(N² |c|)` for laser
  points, `O(k² |c|)` for `k` dense points; a measurement asks for the
  points it combines, not a whole map.
- **Pose start.** RANSAC samples only the initializing camera; the other
  cameras' inliers are selected at the refined pose.
