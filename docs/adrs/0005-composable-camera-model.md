# ADR 0005: Composable Camera Model

## Context

A calibration library has to cover these lens and sensor types:
- perspective lenses with polynomial distortion;
- fisheye and wide-angle lenses (Kannala-Brandt, EUCM, double sphere);
- telecentric metrology lenses;
- tilted (Scheimpflug) sensors.

It must do so without a camera type per combination and without a second
implementation of each model for the optimizer. Every problem type must
accept every model. New models must land without touching the problem
types.

Telecentric lenses break the usual assumption that every pixel's ray passes
through one centre: a telecentric camera projects orthographically, and its
rays are parallel. Wide-angle lenses break a second one: points at or
behind the image plane (`z ≤ 0`) can still be visible.

## Decision

### Four slots plus pixel intrinsics

```
pixel = K(sensor(distortion(projection(p_c))))
```

| Slot | Maps | Kinds | Own parameters |
|---|---|---|---|
| Projection | camera-frame point → normalized coordinates | `Pinhole`, `Equidistant`, `Eucm`, `DoubleSphere`, `Telecentric` | EUCM `alpha, beta`; double sphere `xi, alpha`; none for the others |
| Distortion | normalized → distorted normalized | `None`, `BrownConrady5`, `Rational8`, `ThinPrism9`, `Division1`, `KannalaBrandt4` | the coefficients |
| Sensor | distorted normalized → sensor plane | `Identity`, `Scheimpflug` | `tilt_x, tilt_y` |
| K | sensor plane → pixels | one model, `fx, fy, cx, cy` | — |

The code has the `Pinhole` and `Equidistant` projections, all six
distortion kinds and both sensors. `Eucm`, `DoubleSphere` and `Telecentric`
are the design's planned kinds (backlog CAM-EUCM-DS, CAM-TELECENTRIC); the
sections on them below are what those tasks implement. Camera validation
rejects a model without an initializer (an equidistant camera with a
Scheimpflug sensor).

- **The projection takes the point, not a direction.**
  - Central projections use only its direction.
  - `Telecentric` uses its `x, y`.
  - Each projection defines its own validity domain (`in_domain`;
    `project` itself is total, so the optimizer's kernels stay branch-free).
    There is no global `z > 0` test, so a fisheye camera sees beyond 90°:
    the equidistant domain is every direction but straight back.
- **Back-projection returns a ray** `{ origin, direction }`. The
  direction is not normalized: for a pinhole camera it is the ray's point
  on the `z = 1` plane, so pinhole back-projection is exact.
  - Central projections return the origin at the camera centre.
  - `Telecentric` returns a ray parallel to the optical axis from `(x, y, 0)`.
  - `Projection::is_central()` tells the two apart.
  - Algorithms that assume a single centre (epipolar geometry, essential
    and homography decomposition, Zhang's method) reject a non-central
    camera with a typed error. Ray-based algorithms (triangulation, laser
    plane intersection) serve both.
- **Normalized coordinates carry the projection's unit:**
  - pinhole: tangent plane, unitless;
  - equidistant: angle in radians;
  - EUCM and double sphere: their unit plane;
  - telecentric: target units, i.e. metres for metre targets.

  Distortion coefficients are therefore in that unit. `fx`/`fy` are pixels
  per normalized unit, which for a telecentric lens is magnification over
  pixel pitch.
- **Fisheye is a composition.** Kannala-Brandt (OpenCV `fisheye`) is
  `Equidistant` + `KannalaBrandt4`: an odd polynomial in the incidence
  angle. UCM is `Eucm` with `beta` fixed at 1, so it needs no kind of its
  own.
- **No skew.** Calibration never estimated it, and current sensors have
  none, so intrinsics are `fx, fy, cx, cy`.

### One implementation per model

Each model is implemented once, in `vision-calibration-core`. The same code
must run on two scalar types:
- the f64 used by core and the pipeline;
- the dual numbers the optimizer differentiates through.

So it is generic over `T: RealField` with no `Copy` bound.

The optimizer's kernels are adapters that read parameter blocks into the
core model. They hold no math of their own.

Each kind also declares its metadata once, in core:
- name;
- parameter names;
- dimension;
- bounds (EUCM `alpha ∈ [0, 1]`, `beta > 0`; added with the first kind
  that has them);
- parameters it fixes by default (`CameraModelSpec::default_fix`, which
  also holds the cross-slot rules).

Covariance labels, fix-list validation and the optimizer's block layout
derive from that metadata; the JSON schema and the Python types carry the
kind and parameter names.

### A camera model is a spec plus values

`CameraModelSpec { projection, distortion, sensor }` names the kinds, and
`CameraParams` holds the values. `CameraParams::spec()` recovers the spec.
The kind enums live in core; the optimizer's factors carry the same
`CameraModelSpec`.

Every problem config that calibrates a camera takes a `camera` group
(planar intrinsics, single-camera hand-eye, the laserline device, the rigs,
and through its rig stage the joint rig laser problem). The rig laserline
device calibrates no camera: it takes each camera's `CameraParams` from the
upstream rig.

```rust
camera: CameraConfig {
    model: CameraModelSpec,
    fix: CameraFix,   // named per slot
}
```

`CameraFix` holds lists of parameter names per slot, for example:

```json
{ "intrinsics": ["cx", "cy"], "distortion": ["k3"], "sensor": ["tilt_y"] }
```

Rules for the lists:
- Names are validated against the chosen kinds before any work. An unknown
  name is a typed error.
- An absent list means the spec's default: the union of each kind's own
  defaults, plus the cross-slot rules in `CameraModelSpec::default_fix`. A
  Scheimpflug sensor holds every coefficient but the low-order radial ones
  (`k3, p1, p2` for Brown-Conrady), because tangential and higher-order
  distortion absorb tilt. A telecentric lens without distortion holds
  `cx, cy`, because they are unobservable.
- A stage that holds more than the model's default names its own default
  per model the same way: the joint rig laser solve holds every
  higher-order coefficient and the tilt of each camera unless its
  `joint_optimize.fix` lists say otherwise.

Rigs take one `camera` for all cameras, plus optional per-camera overrides
keyed by camera index, so lenses can be mixed.

Exports carry `CameraParams` for every camera. `Camera<S, P, D, Sm, K>`
remains the statically typed form for library code. `CameraParams::build`
gives the runtime-polymorphic form.

### Initialization is per projection

Each projection kind names its linear initializer:
- `Pinhole`: Zhang plus the distortion fit.
- `Equidistant`: a search over focal length, principal point and the
  first two Kannala-Brandt coefficients, each candidate scored by pixel
  error through per-view poses recovered from bearings
  (`b × H p = 0`, which holds beyond 90°). It needs no image size.
- `Telecentric`: affine factorization.
- Other central wide-angle kinds: their own initializer; the bearing pose
  serves all of them.

Seeds are values in the camera's own model: a distortion seed is a
`DistortionParams` whose kind must equal the configured one, used as given.

A kind without an initializer requires a seed (ADR 0011). That is a typed
error at validation, never a silent fallback.

### Telecentric gauge

An orthographic camera cannot see depth along its optical axis:
- **Per-view depth:** each view's depth (`tz` of `camera_se3_target`) is
  held at its initial value.
- **Hand-eye:** the hand-eye translation along the optical axis is
  likewise held.
- **Laser:** results that depend on the axial origin are reported
  relative to the first view.
- **Planar targets:** a planar target seen orthographically has a two-fold
  rotation ambiguity per view (reflection about the image plane). It does
  not change the projection, so intrinsics are unaffected. Hand-eye
  motion selects the consistent branch.

These held components are part of the spec's default fix, so the
covariance reports them as fixed rather than unobservable.

## Consequences

- **Adding a model** needs:
  - a core type implementing its slot trait, plus its metadata;
  - a kind variant;
  - one arm in that slot's optimizer dispatch;
  - an initializer, or a documented seed requirement.

  No problem type, config struct, fix-mask type or Python class changes,
  because those derive from the metadata.
- **Optimizer dispatch is per slot**, never one row per combination
  (ADR 0008).
- **Every problem type accepts every model.** A problem that cannot use a
  model, such as epipolar geometry with a non-central camera, rejects it
  with a typed error.
- **Numerical tests:** each model ships a math note and a synthetic matrix
  test across every problem type.
