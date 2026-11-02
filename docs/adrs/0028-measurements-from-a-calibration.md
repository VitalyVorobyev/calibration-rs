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
  variance, like `Noise::Known`). This is for bundle adjustment that
  refines the cameras with the new observations.

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

`optim::problem` also gains, as typed constructors (ADR 0027):

- reprojection of a **point block** (a free 3D point), next to the
  constant point it takes today;
- reprojection through a **rig chain**: `cam_se3_rig · rig_se3_world`;
- a **whitened reprojection** for an observation's 2×2 covariance.

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
- Core gains `CalibratedRig`; the pipeline's `CalibratedCameras` gains
  `calibrated_rig`.
- Every MVG function that takes `K` changes signature to take a
  `CalibratedRig`: a breaking release.
- MVG results carry covariances. Bundle adjustment needs an explicit
  scale anchor for a single camera.
- Python gets the MVG surface with the `CalibratedRig` from an export.
