# ADR 0028: Measurements from a Calibration

## Context

`vision-mvg` holds the multiple-view geometry: robust two-view estimation,
pose recovery, triangulation, bundle adjustment, rectification and dense
matching. It does not yet measure with error bars:

- it takes a bare `K` and normalized coordinates, so distortion, fisheye
  and Scheimpflug cameras reach it only through a caller's own
  undistortion, and rectification wants undistorted pixels;
- no result carries a covariance, against the results contract (ADR 0026);
- bundle adjustment and point refinement run their own tiny-solver
  problems outside the optimization IR, and `vision-geometry` and
  `vision-mvg` each have a triangulator and a `refine_point`;
- bundle adjustment leaves the scale gauge to LM damping.

A measurement from a calibrated system has two sources of error: the noise
of its own observations, and the calibration's. The second is shared by
every measurement made with that calibration. In the book's "Compose Your
Own Pipeline" setup (a part's depth at 0.5 m), a camera held fixed at its
calibrated values states σ(depth) 0.91 mm against an actual 2.3 mm. The
public problem builder (ADR 0027) can get it right only by solving jointly
with the calibration's observations, which a user holding an export does
not have.

## Decision

### `vision-mvg` builds on the public problem builder

`vision-mvg` depends on `vision-calibration-optim` and does all its
refinement and covariance through `optim::problem`, the API a user has.
The crate boundary enforces this: MVG cannot reach the private IR, so
what it needs from the builder, a user gets too. `vision-mvg` drops its
direct tiny-solver dependency and its `refine` feature; its deterministic
parts (RANSAC, pose recovery, rectification maps, dense matching) are
unchanged. The facade keeps every `mvg` path.

Rejected: moving bundle adjustment into `optim` (the first plan). It
splits the multiple-view geometry across two crates and lets MVG code use
optim's internals, so the dogfooding would hold only by convention.
`vision-mvg` gains the solver's compile weight; a user who only wants
two-view solvers takes `vision-geometry`.

The edge is new in the DAG (ADR 0006): `vision-mvg` may depend on core,
geometry and optim. Nothing depends on `vision-mvg` but the facade, so no
cycle is possible.

### A calibration is a value: `CalibratedRig`

`vision-calibration-core` gains `CalibratedRig`, what a measurement needs
from a calibration:

- the cameras (`CameraParams`), in order;
- each camera's `cam_se3_rig`, the reference camera's the identity; a
  single camera is a rig of one;
- the calibration's `Uncertainty`, joint over every camera's parameters
  and the extrinsics, with the labels the exports use (`fx`, `cam1.k1`,
  `cam1.cam_se3_rig.rx`). The reprojection noise level of the calibration
  (its `noise` entry) is read from that covariance, not carried as a
  separate field.

Every export with cameras produces one (`CalibratedCameras::calibrated_rig`
in the pipeline). MVG functions take a `&CalibratedRig`, never a bare `K`.

### Observations are pixels through the camera model

Closed-form steps take bearings, unit rays from each camera's own
unprojection, so they work for every camera model, including fisheye
beyond 90°. Refinement compares pixels with the camera's projection. An
observation is a pixel with its noise:

- a stated 2×2 covariance per observation (Mahalanobis reprojection);
- or one σ for all;
- or the calibration's reprojection noise level, named explicitly.

There is no implicit default. A triangulation from two views has one
degree of freedom left to estimate a noise level from, so measurement
noise is stated, not estimated. Bundle adjustment, with many residuals,
may also estimate it.

### The calibration's uncertainty enters as "consider" parameters

`optim::problem` gains `add_calibrated_rig(&CalibratedRig, Use)`, which
adds its cameras and extrinsics as blocks:

- **`Use::Consider`** (the default for measurements): the calibration's
  parameters stay fixed at their values, and their covariance `Σ_c` is
  propagated into the solution. With `H` the Gauss–Newton matrix over the
  free parameters `x`, weighted as the solve weighted it (whitening,
  robust weights), and `H_xc` its block against the calibration's
  parameters `c`, the estimate moves with the calibration as
  `S = ∂x̂/∂c = −H⁻¹ H_xc`, and

  ```text
  Σ_x = Σ_x,observations + S Σ_c Sᵀ
  ```

  `Σ_x,observations` is the sandwich covariance of ADR 0026. The cross
  term is zero: the measurement's images are not the calibration's.

  The estimate is the one a fixed camera gives, so asking for σ never
  changes a result; the σ carries the calibration's share. Every
  quantity solved in one problem shares the same `Σ_c`, so their
  cross-covariance comes with them. Two separate solves lose it, so a
  measurement API takes all its points in one call.
- **`Use::Prior`**: the calibration's parameters are free, held by a
  Gaussian prior `L⁻¹(c − c₀)` with `Σ_c = L Lᵀ` (whitened, unit
  variance, like `Noise::Stated`). This is for bundle adjustment that
  refines the cameras with the new observations. The prior weighs in
  units of its covariance, so every other residual's noise must be stated
  too: the solve weighs a residual of estimated noise as if its σ were 1
  in its unit, which at 0.3 px makes pixels count 11 times too little
  against the prior. `solve` refuses the mix.

A problem holds one calibrated rig, so its parameters keep the labels of
the calibration's export (`fx`, `cam1.k1`, `cam1.cam_se3_rig.rx`).

The consider share is computed after the solve: the problem is linearized
once more with the consider coordinates free, `H` and `H_xc` are weighted
as the sandwich weights them (each family's `κ`, priors at 1), and
`S Σ_c Sᵀ` is added to the observations' covariance. With priors in a
problem, a family's noise level is scaled by `m_obs / (m − n)`: a prior
holds the coordinates it is over, so its rows add no degrees of freedom to
the observations' (before this, `m / (m − n)` overstated a Prior-held
measurement's σ̂² 1.6-fold).

Parameters the calibration held fixed have no covariance and stay fixed
in either use. A calibration without a covariance can still be added
(fixed), and the solution's uncertainty then names the missing source as
a typed `Unavailable` reason, rather than reporting σ from the
observations alone as if it were complete. A user who wants that σ
anyway says so, by adding the cameras fixed with `add_camera`.

The same builder change serves a user's own problem: a camera from an
export enters with its covariance, so the joint solve in "Compose Your
Own Pipeline" is no longer the only exact route.

### Constructors the measurements need

`optim::problem` also gains, as typed constructors (ADR 0027), one
`add_reprojection(camera, PoseChain, TargetPoint, pixel, PixelNoise,
loss)`:

- reprojection of a **point block** (a free 3D point,
  `TargetPoint::Block`), next to the constant point
  (`TargetPoint::Known`); in the IR a `ReprojFreePoint` factor;
- reprojection through a **rig chain** (`PoseChain::Rig`):
  `cam_se3_rig · rig_se3_world`, a `ReprojChain::Rig` storing
  `cam_se3_rig` itself, so its tangent is the one the exports report;
- a **whitened reprojection** for an observation's 2×2 covariance
  (`PixelNoise::Stated`): known information, like `Noise::Stated`, not a
  family whose noise level is estimated.

### Gauge and scale in bundle adjustment

Bundle adjustment takes an explicit gauge; there is no damping-held
gauge:

- **the frame:** the first rig pose is fixed, so the world is the rig
  frame at the first capture;
- **the scale:** a rig of two or more cameras fixes the scale through its
  extrinsics. A single camera needs a stated distance (between two points
  or two poses, with σ). Without one, the call is an `InvalidInput`
  before any work.

The covariance is then in that gauge, and its documentation says so.
Inner-constraint (free-network) covariance is not offered until a use
needs it.

### One of each

- **Triangulation:** `vision-geometry` keeps the linear DLT on bearings.
  `vision-mvg` refines through the problem builder and returns the point
  with its covariance and diagnostics: reprojection error, parallax, and
  in front of every camera.
- **Point refinement:** both `refine_point`s go.
- **Homography refinement** moves onto the builder.
- Names follow ADR 0024.

### Derived quantities

Distances, angles and plane fits over triangulated points propagate the
joint covariance of their points linearly: `J Σ Jᵀ`, with `J` from dual
numbers. This lives in `vision-mvg`, next to the measurements. A plane fit
is a small problem of its own; its covariance holds the points'
uncertainty as consider parameters.

### Laser points: the plane is part of the calibration

A laser pixel on a known plane is one 3D point: the viewing ray's
intersection with the plane. `CalibratedRig` carries each camera's laser
plane (`with_laser_planes`, in that camera's frame, `Option` per camera),
so a laser point's covariance has the same three sources as a triangulated
one: the pixel, the camera, and the plane, and the plane and camera
parameters are correlated in the calibration's covariance (they were
estimated in one solve). `LaserPlane` moves to core with the rig, its
serialized form unchanged; the facade path is `core::LaserPlane` alone.

The point is the implicit function `F(p; u, θ, n, d) = [π(p; θ) − u;
nᵀp + d] = 0`, so its first-order covariance is closed form from the
exact derivatives of the forward projection (`CameraJacobian` in optim,
the same dual-number kernels as the solver): `∂p/∂x = −A⁻¹ ∂F/∂x` with
`A = [∂π/∂p; nᵀ]`. The calibration's covariance enters as `J_c Σ_c J_cᵀ`
over the coordinates the covariance lists (camera, plane, and the camera's
extrinsic, whole or held), the pixels' as `J_u Σ_u J_uᵀ`. Points of one
call share the calibration, so the joint covariance is dense and the
cross-covariance feeds `metrology`.

Rejected: the builder route (`Use::Consider` with a laser-plane residual).
A single point has one residual per axis and zero degrees of freedom left,
and the builder's consider share is computed from a Gauss-Newton system
that is exactly determined here; the closed form is the same linear
algebra without a solver, and exact. Revisit when a measurement mixes
laser pixels with redundant observations of the same point (a point seen
by a camera and by the laser), where a joint solve has something to
adjust.

### Dense clouds: the matcher's measurement, closed form, on demand

A dense match is a rectified left grid pixel and a disparity along the row.
The rectification maps are fixed (built with the calibration's values), so
the left pixel is a fixed raw pixel, exact, and the right measurement says
only which rectified column the right camera saw the point on; the row there
(the vertical disparity a calibration error causes) is not measured. The
point is the implicit function `F(X; c, d) = [π_L(X) − u_L; r(π_R(X) −
u_R(d))] = 0` with `r` the row of the fixed right map's inverse that gives
the column: three equations, three unknowns, so its sensitivity to the
calibration is closed form, `G = −A⁻¹ ∂F/∂c`, from the same `CameraJacobian`
as the laser points, and the disparity's is `−(X − c_L)/d`.
`StereoRectification::dense_cloud` returns a `DenseCloud` in the rig frame
(as `triangulate`'s): it stores the points only, and computes `G_i` when a
covariance is asked for, per point or jointly over chosen indices
(`PointSet::from_dense_cloud` for metrology), since a full map's `G_i` would
take hundreds of megabytes. The rectification is never differentiated: it is
how the measurement was taken, not a parameter of the point.

Rejected: `triangulate` with isotropic noise on both raw pixels per point. It
is the builder route at a dense cost that grows as the cube of the point
count, and it is another measurement model: it shares a vertical disparity
between the two cameras (across-row σ up to 40 % off; the depth σ agrees
within 0.5 %). With the left pixel exact and the right one weighted only
along `r` it states the dense covariance to 1e-7, which the tests check.

### A staged laser calibration carries its upstream as consider parameters

`rig_laserline_device` fits each camera's plane against a frozen rig
calibration, so a plane's own covariance is conditional on it. Its
`RigUpstreamCalibration` carries the upstream's `Uncertainty` over the
parameters the plane fit consumes: the cameras' parameters, `cam_se3_rig`,
and each view's `rig_se3_target` (`view<v>.rig_se3_target.*` among the
shared parameters, so the poses stay correlated with each other and with
the cameras). The fit is unchanged; after it, each camera's laser IR is
linearized once more with the listed camera coordinates and its views'
`camera_se3_target` free, `S = −H⁻¹ H_xc` comes from the same
`consider_sensitivity` the builder's `Use::Consider` uses, and a right
perturbation `ε` of `cam_se3_rig` maps onto `camera_se3_target =
cam_se3_rig · rig_se3_target` as `Ad(rig_se3_target⁻¹) ε`. With `A` the
chained `∂plane/∂u`, the export's covariance is the cameras' and
extrinsics' upstream block, their cross terms `A Σ_u` with the planes, and
the planes' `Σ_obs + A Σ_u Aᵀ`, so a laser point from this export has the
three sources a joint solve gives it. The target poses enter the share but
are not listed: the export is a device calibration.

The hand-eye export's covariance is over the hand-eye and the target's
pose, not over per-view poses, so `RigHandeyeExport::to_upstream_calibration`
maps it: `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T` moves by `−Ad(T_R_T⁻¹) ε`
with the hand-eye and by `ε` with the target (`EyeToHand`, `T_R_T = T_R_B ·
T_B_G · T_G_T`: `Ad(T_R_T⁻¹ · T_R_B) ε` and `ε`). The robot poses, with any
per-view corrections, count as exact: the export states no correlation
between a correction and the shared parameters, and the per-view blocks
alone would not be a valid approximation. An upstream without a covariance
holds it exact, bit for bit the conditional covariance.

Rejected: carrying the hand-eye chain (mode, poses, robot poses) in the
upstream so the plane fit linearizes through it. The upstream's
`rig_se3_target` would then have two sources, and a rig-extrinsics
upstream (per-view target poses of its own) would need a second path.

### On a robot: the hand-eye is part of the calibration

A rig calibrated on a robot measures in the robot's frames. `CalibratedRig`
carries the hand-eye (`with_handeye`), a core `Handeye` enum with the one
pose of its mode, `EyeInHand { gripper_se3_rig }` or `EyeToHand {
rig_se3_base }`, the pose the export estimates; its covariance labels are
the export's (`gripper_se3_rig.rx`, `rig_se3_base.rx`, `gripper_se3_camera`
and `camera_se3_base` for one camera), whole or held as for the planes.
`CalibratedCameras::calibrated_rig` attaches it for the three hand-eye
exports. The robot's pose at a capture is a core `RobotPose`
(`base_se3_gripper`, exact or with a 6×6 covariance over the right
perturbation `base_se3_gripper·Exp(δ)`, the frame the robot-pose
corrections of the calibration use), and `RobotFrame` names the frame.

A measurement states where the rig was at each capture with `mvg::types::Captures`:
`Rig` (one capture, the rig frame), `World(&rig_se3_world)` (exact poses),
`Handeye(frame)` (the frame the rig is fixed in: one capture, no robot) or
`Robot { frame, robot }` (the frame the rig moves in, a robot pose per
capture). The variant and the frame are checked against the rig's mode
before any work. Capture `k`'s `rig_se3_world` is the hand-eye chain, and
its right perturbation `ξ = A ε_h + B ε_r` (core's `Handeye::rig_in_frame`,
with `Ad`):

| mode, frame | `rig_se3_<frame>` | `A` | `B` |
|---|---|---|---|
| eye-in-hand, gripper | `H⁻¹` | `−Ad(H)` | — |
| eye-in-hand, base | `(P H)⁻¹` | `−Ad(P H)` | `−Ad(P)` |
| eye-to-hand, base | `H` | `I` | — |
| eye-to-hand, gripper | `H P` | `Ad(P⁻¹)` | `I` |

The hand-eye then enters as one more calibration parameter, correlated with
the cameras, planes and extrinsics through the calibration's covariance, and
each robot pose as an independent source:

- `triangulate_laser` (closed form) adds the columns `[x]ₓ, −I` times `A`
  under the hand-eye's labels to `J_c`, and `[x]ₓ, −I` times `B` per robot
  pose with its own covariance;
- `triangulate` (the builder) adds each capture's pose with
  `Problem::add_rig_pose(name, frame, robot)`: a fixed SE3 block whose
  tangent joins the considered coordinates with `Cov(ξ, c) = A Cov(h, c)`
  and `Cov(ξ, ξ) = A Σ_hh Aᵀ + B Σ_r Bᵀ`, so `S Σ_c Sᵀ` carries every cross
  term.

Rejected: a frame change applied to a finished measurement (points and
their covariance in, points in the gripper or base frame out). It needs the
points' cross-covariance with the hand-eye, which their covariance alone
does not hold; with it dropped, the laser points of a joint rig hand-eye
laserline calibration in the base frame state 0.41 mm against a 0.10 mm
scatter (the calibration pins the board in the base frame, so the hand-eye
errs with the planes and cameras it was estimated with). A frame option on
each measurement puts the hand-eye into the same consider machinery.

Rejected: the robot pose's error as a calibration parameter. It is the
robot's at the measurement, not the calibration's, and independent of it;
a robot's systematic error across poses (its kinematic calibration) is not
modelled.

A staged laser export (`rig_laserline_device`) does not list the hand-eye:
its upstream maps the hand-eye into per-view board poses it does not
report. A rig built from it with `with_handeye` counts the hand-eye exact;
the joint `rig_handeye_laserline` export carries it, and the staged route
gains it when a use needs it (the upstream would list the hand-eye with its
cross terms to the planes).

The pipeline's `pixel_to_gripper_point` (a point without σ, from a hand-eye
export and planes in the rig frame) is deleted: `triangulate_laser` with
`Captures` replaces it.

### What the σ promises

Each propagated σ is gated by Monte-Carlo, redrawing both the calibration
images and the measurement images, as in "Compose Your Own Pipeline". The
calibration's share is only as good as the calibration's own stated
covariance, which is why that one is scaled by the view scatter on real
images (ADR 0026); MVG-DEMO checks the whole chain against known lengths
on real data.

## Consequences

- New DAG edge `vision-mvg → vision-calibration-optim`; `vision-mvg`
  loses its `refine` feature and its tiny-solver dependency (ADR 0006
  amended).
- `optim::problem` gains:
  - `add_calibrated_rig` with `Use::{Consider, Prior}`;
  - point-block and rig-chain reprojection;
  - whitened reprojection.
- ADR 0027's "more chains are added when a need is shown" is that need.
- Core gains `CalibratedRig` and `LaserPlane` (moved from optim), and the
  robot vocabulary `Handeye`, `RobotPose`, `RobotFrame`; the pipeline's
  `CalibratedCameras` gains `calibrated_rig` and, for the laser exports,
  `laser_planes_cam`, for the hand-eye exports `handeye`.
- `triangulate` and `triangulate_laser` take `Captures` (the frame of the
  measurement); `LaserObservation` names its capture. `optim::problem` gains
  `add_rig_pose`.
- Every MVG function that takes `K` changes signature to take a
  `CalibratedRig`: a breaking release.
- MVG results carry covariances. Bundle adjustment needs an explicit
  scale anchor for a single camera.
- Python gets the MVG surface with the `CalibratedRig` from an export.
