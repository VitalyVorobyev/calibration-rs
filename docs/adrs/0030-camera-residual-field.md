# ADR 0030: A Residual Field Is a Camera Slot Evaluated at the Model's Own Projection

Status: **accepted, partly implemented** (backlog CAM-RESIDUAL-FIELD).
Designed and prototyped in the bench example `scheimpflug_model_study`. The
camera slot is in the library (core kernel, wire, every consumer, held
fixed); the fit — gauge, penalty, selection, covariance — is not. The
evidence is [`notes/camera-residual-field.md`](../notes/camera-residual-field.md).

## Context

On two six-camera Scheimpflug rigs the Brown-Conrady + tilt model leaves a
smooth, camera-fixed displacement that no parametric term describes while
keeping the tilt and principal point identified: freeing the thin-prism or
tangential terms, a sensor-plane height, a distortion centre or an elliptic
radial metric either gains little or lets the tilt and principal point
wander by tens of pixels between folds. A generic displacement field is the
remaining tool, and it has four ways to go wrong, each measured:

- **Evaluated at the observed pixel** it is not a camera model: the cost
  then depends on the observation through the basis, and a field whose
  Jacobian nears the identity contracts the measurement noise instead of
  explaining it. On synthetic data from the plain model it lowers the
  held-out residual 11 % below the injected noise while `k1` runs from
  −0.4 to −4; on the real rigs it produced most of the gain first
  attributed to the field.
- **A free basis spans the camera's own parameters** (a cubic B-spline
  reproduces every polynomial of degree ≤ 3, so `fx, fy, cx, cy` and the
  tilt's keystone exactly): the intrinsics become unobservable.
- **A polynomial extrapolates**: held-out points outside the training
  region land tens of pixels off.
- **A smoothness prior counted as known information** overstates every σ
  near the field: the sandwich adds the prior's stated variance to the
  gradient's scatter, although a fixed field has none.

## Decision

### A fifth slot after `K`

```
pixel = F(K(sensor(distortion(projection(p_c)))))      F(u) = u + Σ_k β_k(u)·c_k
```

- `F` is a **`FieldKind`** slot of `CameraModelSpec` (ADR 0005): `None`
  (default) or `BSpline { cells: [nx, ny], image_size: [w, h] }`, a
  uniform bicubic B-spline over the image rectangle `[0, w] × [0, h]`.
  Parameters: `2·(nx + 3)·(ny + 3)` control displacements in pixels, `u`
  components then `v`, each row-major (`v` index outer).
- **The domain is part of the kind** (amended at implementation). A
  library camera carries no image size, and the knot grid *is* the
  basis: a field over 1280 × 1024 is a different function space from one
  over 640 × 480. With the size in the kind the spec stays `Copy`/`Eq`,
  fully describes the block layout and the kernel, and every factor that
  carries a spec evaluates the field with no extra data. The size is
  validated with the kind (`1..=1024` cells per axis, no more cells than
  pixels); where a field is built from data (the fit), the size comes
  from the dataset and is a typed error when absent.
- **The basis is evaluated at the pixel the rest of the model produces**,
  never at a detection. Every projection, residual and Jacobian goes
  through the same kernel, generic over `T: RealField`.
- **Bounded outside**: the argument is clamped to the image rectangle, so
  a point outside it takes the border's displacement; nothing grows.
- Pixel space, not the normalized plane: the field is then independent of
  `fx, fy, cx, cy`, its control points are in pixels whatever the lens,
  and a coverage mask is the image's.

### Gauge: orthogonal to what the camera already spans

At the parametric solution, the field is held to `A c = 0`, where row `j`
of `A` is `Σ_cells J_jᵀ B(u)` over the covered image (one sample per
occupied cell of a fixed grid, uniform weight): the pixel direction `J_j`
of each free camera parameter (intrinsics, free distortion terms, tilt)
and of a rotation common to every view (a gauge the poses undo) against
the basis there. Uniform over the coverage, not weighted by the points,
so the camera/field split follows what was seen, not where detections
cluster. A common translation is not a row: it moves each point by its
inverse depth, which no image field can copy, and constraining it removes
real freedom (measured).

- In the IR this is a **subspace manifold**: the block stores `c`, its
  tangent is `z` with `c = c₀ + N z`, `N` an orthonormal basis of
  `null(A)` (from the small `n_c × n_c` eigenproblem of `AᵀA`). Both
  backends see an ordinary manifold block; covariance comes back in `z`
  and is mapped to `c` by `N Σ_z Nᵀ`. A stiff penalty does the same in
  the prototype, but the covariance then reads it as known information
  with its stated σ, which leaks into `cx, cy` (three times their scatter
  at σ 1e-4 px; honest at 1e-6, where a solve from the route accepts no
  step: no penalty σ serves both).
- **Control points without support are fixed at zero**: a control whose
  basis mass over the covered cells is below a threshold carries no data,
  and freeing it only lets the penalty extrapolate (border controls reach
  7 px in the prototype while the field on the points is 0.1 px).
- To first order the field then cannot move the camera: the camera's
  parameters, covariance and meaning are those of the plain model plus
  what the orthogonal field explains.

### Smoothness, chosen on held-out views

- A second-difference penalty on the control grid along each axis, weight
  `λ` relative to the reprojection residuals (dimensionless, so its
  meaning does not change with the pixel noise).
- **It is a penalty, not a prior**: the covariance treats it as part of
  the objective (`H`) but not as a measurement (`G`), so `σ` is the
  estimate's scatter under the measurement noise. The penalty's bias is
  not in `σ`; it is what the selection rule bounds.
- `λ` and `cells` are chosen by K-fold held-out views (poses refitted,
  camera held) from a short ladder, the same rule CAM-SELECT applies to
  distortion models: the field is kept only if its held-out gain beats
  the fold scatter; otherwise the result is the plain model and the
  report says why.

### Unprojection

`u_par = u − F(u_par)`, fixed-point from `u_par = u` until the step is
below 1e-11 px (at most 100 steps), closed by one Newton step on
`u_par + F(u_par) = u`. The Newton step leaves the value at rounding level
and, under dual numbers, gives the converged point the derivative of the
exact inverse, `−(I + ∂F/∂u)⁻¹ ∂F/∂c`, however few fixed-point steps ran
(a laser residual back-projects its observed pixel, so this derivative
matters once the field is estimated). Validation requires `L ≤ 0.5`, with
`L` the Frobenius norm of the four per-entry bounds of `∂F/∂u` (each the
largest control difference along that axis over the knot spacing, since a
cubic B-spline's derivative is a convex combination of them): an upper
bound on the spectral norm, so the iteration is a contraction and `F`
cannot fold. A field above the bound is a typed error wherever a camera is
built or packed (`CameraParams::build`, the problem builders), not a
silent unprojection failure.

### Reach

- **Opt-in, per camera**: `camera.field` (default `None`); a rig adds it
  per camera through `camera_overrides`. It is fitted as the last stage of
  any workflow that refines a camera, after the parametric model has
  converged (that solution defines `A`), and it travels with the camera
  into every downstream workflow (hand-eye, rig, laser) held fixed or
  refined with the same constraint.
- **Wire**: `CameraParams` gains `field` (`{"type": "b_spline", cells,
  image_size, du, dv}`), omitted when absent, so every existing file
  reads unchanged and the export shapes and versions stay (an added
  optional member keeps the version). A 1.0.0-rc.1 reader ignores
  `field` and so reads a different camera, which the changelog states.
  Rejecting unknown camera members was tried and dropped: committed sample
  exports carry stray members (`_phantom` from an old serialization of the
  runtime camera), so it would break files users already hold. Once a field is estimated, the export's camera
  covariance includes its block. Python: the `BSplineField` dataclass and
  the `CameraParams.field` member. Book: a chapter on when a residual field
  is warranted, how to read its held-out gain, and why it must not be
  extrapolated (with the fit, when the field becomes a user capability).
- **Held fixed** (implemented). Until the fit lands, a field travels as a
  fixed camera block: `add_camera_blocks` and the public problem builder
  add a `field` block (`field/<i>` in a rig) fully fixed, the covariance
  lists it among the fixed parameters by its name, and both backends take
  a fully fixed field block out of the factor's variables and splice its
  values back in as constants (tiny-solver would otherwise differentiate
  through hundreds of fixed coordinates; factrs fuses a camera into one
  static variable of at most 17). A field block that is not fully fixed is
  a typed error on both backends. Projection derivatives
  (`CameraJacobian`, projection uncertainty, view information) take the
  field as part of the model, not a coordinate of `θ`.
- **Config** (implemented): a workflow's `camera.model.field` is a typed
  error, since no workflow estimates a field; the fit adds its own
  options (`camera.field`).

## Consequences

- One more camera slot, as cheap as the others for every workflow; no
  workflow-specific code.
- A field changes no default and no existing result.
- On both rigs the field removes 40–50 % of the held-out residual
  variance and leaves the rest per view, so it does not replace work on
  the detector, the target or the view design. The notes say how much is
  left and where.
- On the display rig the field doubles the fold-to-fold scatter of `fx`
  and `tilt_x` beyond what σ states; the implementation measures whether
  the view-scatter scaling (ADR 0026) covers it before the slot ships.

## Alternatives rejected

- **A structured physical term.** Distortion centre apart from the
  principal point, tilt before distortion, a field-dependent tilt, depth-
  dependent `k1`, an elliptic metric, a sensor surface height centred on
  the foot of the sensor normal or on a free centre, the thin-prism and
  tangential terms: each either gains under 6 % or trades with the tilt
  and principal point (folds disagree by 10–100 px).
- **A polynomial field.** Spans the intrinsics, extrapolates, and was the
  carrier of the observation-evaluated artefact.
- **A field with the camera held** (two-stage). Identifiable, simpler, and
  equal on held-out views (the route's residual is already orthogonal to
  the camera's directions), but it carries camera-like terms into the
  uncovered border (controls of 20–96 px against 2–7 px). Rejected
  because a later workflow that refines the camera jointly (rig,
  hand-eye) would let the free field trade with it again; the constraint
  is what keeps the decomposition when the camera moves. The two-stage fit is a valid first implementation step
  (constraint-free, no IR change) if the downstream workflows hold the
  field fixed.
- **A field in the normalized plane.** Couples the field to `fx, fy`, so
  every intrinsics change rescales it.
- **The smoothness term as known information.** Measured: the control
  points' σ five to six times their Monte-Carlo scatter.
