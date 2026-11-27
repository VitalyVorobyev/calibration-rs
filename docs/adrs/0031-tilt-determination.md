# ADR 0031: A Scheimpflug Tilt Is Estimated Only When the Data Determine It

Status: **proposed** (backlog CAL-TILT-DETERMINE). Prototyped in the bench
example `tilt_identifiability` (parts `determine`, `prior`, `rule`); not in
the library. Evidence: [`notes/tilt-determination.md`](../notes/tilt-determination.md),
building on [`notes/tilt-identifiability.md`](../notes/tilt-identifiability.md).

## Context

A tilted sensor's homography factors exactly as `K·H(τ) = K'·R(τ)`: without
lens distortion a tilted camera *is* a pinhole `K'` whose poses are rotated
by `R(τ)`, in a single camera, a rig with free extrinsics and a hand-eye
with known robot poses alike. The tilt is the rotation from the sensor
normal to the lens axis, and only the lens distortion, centred on the lens
axis, observes it. So its σ runs from 0.006° to beyond 5° with the same
views, and to first order the tilt is a decentring of the distortion
(`p1 ≈ −k1·tan τx`), so a real decentring is reported as tilt
(`Δτx = −p1/k1`).

Today a Scheimpflug camera always estimates the tilt, with `p1, p2` held.
Where the distortion is weak that tilt is a weak claim about the camera
frame, and it is what every pose, extrinsic and hand-eye rotation is
expressed in. A mandatory prior was rejected by the maintainer: most users
have none. The alternative measured here — decide from the data, use a
prior when there is one, otherwise fix the tilt — has one trap the study
found: fixing the tilt at zero *with `p1, p2` held* costs a median 2× and
up to 31× the noise floor downstream where the tilt is undetermined,
because the tilt's decentring is then left in the residual.

## Decision

### The procedure

For a camera whose sensor is Scheimpflug, under `TiltPolicy::Determine`:

1. **Plain fit.** The camera with the tilt held at zero and `p1, p2`
   free (the untilted Brown-Conrady camera, always identifiable), from the
   untilted linear init or the intrinsics seed.
2. **Tilted fit.** The tilt free, `p1, p2` held (today's free set), seeded
   from the plain fit by the decentring identity — `τx = atan(−p1/k1)`,
   `τy = atan(p2/k1)`, `K` and the poses moved from the sensor-normal to the
   lens frame — or from the user's tilt seed when one is given. Unbounded,
   with the bounded solve only as a runaway fallback (CAL-TILT-BOUNDS).
3. **Decide.** `s` = the larger semi-axis of the tilt's 2×2 covariance as
   the tilted fit states it, **view-scatter scaling included** (ADR 0026;
   without it `s` is 2.3–5× short whenever a view's corners share an
   error, and the rule calls 40 % of undetermined tilts determined).
   Determined iff `s ≤ max_sigma` (default 0.5°).
4. **Determined** → the tilted fit: `source = Estimated`.
5. **Not determined, a prior given** → the tilted fit with the prior and
   `p1, p2` **free**: the tilt comes from the prior and the data, the
   decentring from the data. `source = Prior`. This solve runs in the
   gauge-aligned coordinates (`K'`, sensor-frame poses) or under the
   σ-unit stopping rule (the loop's default, `notes/lm-stopping.md`):
   under the old decrease rule, in the raw ones, it stopped a σ short of
   the posterior mean.
6. **Not determined, no prior** → if the data reject the plain camera
   against tilt-plus-decentring (`LR = (cost_plain − cost_both)·dof/cost_both`
   above the 99 % point of χ²₂, 9.21), the tilted fit with a warning
   (`source = EstimatedImprecise`); otherwise the plain fit, the tilt
   zero: `source = FixedUndetermined`.

`TiltPolicy::Estimate` is today's behaviour and stays the default until the
procedure has passed the real-data gate (below); then `Determine` becomes
the default with `max_sigma = 0.5°`.

### The camera frame each branch reports

- `Estimated`, `EstimatedImprecise`, `Prior`: **lens-axis frame** — `z`
  along the lens axis, `(cx, cy)` where it meets the sensor (the
  distortion centre). Poses, extrinsics and hand-eye rotations compare with
  a drawing's lens axis; their σ includes the tilt's.
- `FixedUndetermined`: **sensor-normal frame** — `z` along the sensor
  normal through the projection centre, `(cx, cy)` its foot; the lens
  axis's offset is in `p1, p2`. Measured within 0.01–0.05° of the true
  sensor normal on weak lenses, 0.05–0.5° on moderate ones. Extrinsics
  compare with the sensor (camera body) mounting, never with the lens
  axis.
- A fit with the tilt fixed and `p1, p2` held ends 0.06–9° from the
  sensor normal and 5–29° from the lens axis; it is never reported.

### Priors on camera parameters

- `CameraConfig.prior`: Gaussian priors named per slot like `fix`
  (`{"sensor": {"tilt_x": {"mean": …, "sigma": …}}}`), radians for angles.
  One mechanism for every camera parameter; the tilt is its first user.
  `CameraConfig` is `#[non_exhaustive]`, so the field (and the tilt policy)
  is an additive change: a minor release, `serde(default)`, no config
  version bump.
- A prior is a library residual (a `FactorKind` on chosen coordinates of a
  block), weighed against the measured pixel σ̂ by one re-solve exactly as
  robot-pose priors are (ADR 0026), and counted as known information in
  the covariance. The design σ is the user's, not estimated.
- `ScheimpflugMountSpec` gains `tilt_sigma_deg` next to `tilt_x_deg`,
  `tilt_y_deg`; `device_seed` turns the pair into a seed and a prior. The
  type is `#[non_exhaustive]` and the field optional, so it is additive and
  keeps `DEVICE_SPEC_VERSION`.
- A prior on a fixed parameter, a non-positive σ, or a tilt policy or tilt
  prior on a non-Scheimpflug camera is `Error::InvalidInput` before any
  work.

### What the result records

Per Scheimpflug camera, in the export and the session, as an optional
field (`serde(default)`, skipped when absent): an additive wire change that
keeps the export's `SCHEMA_VERSION` and the session's schema version (ADR
0012). A version bump would make every existing reader reject every export,
since readers accept only their own `schema_version`:

```text
tilt_determination: {
  source: estimated | estimated_imprecise | prior | fixed_undetermined,
  frame: lens_axis | sensor_normal,
  sigma: s (rad), max_sigma (rad),
  likelihood_ratio: LR, separable_sigma: s with p1, p2 free (rad)
}
```

`separable_sigma` says whether the tilt is told apart from a decentring at
all (only strongly distorting lenses: ≤ 0.5° in 8 of 20 cameras of the
sweep). The quality report adds `TiltDetermination`: `Warn` for
`EstimatedImprecise` (supply a prior), and for a prior–data conflict
`|τ̂ − τ₀| / √(s² + σₚ²) > 3`.

The quality report's `ParameterConditioning` check (shipped ahead of the
procedure) explains *why* a tilt's σ is large: its σ inflation
`1/√(1 − R²)` by the camera's other parameters (the camera's marginal
block of the covariance: trade-offs with poses, rig extrinsics, a hand-eye
or a laser plane do not count, since no change to the camera model or its
capture moves them; with them counted, 17 of the suite's 22 camera checks
warned, most on the principal point and focal length against a rig's
extrinsics, a hand-eye or a laser plane, inflation up to 98), the
partners behind it, and a
named cause (free tangential or thin-prism terms when holding them would
at least halve its σ, else weak or unobserved distortion). It does not
decide *whether* the tilt is determined: an inflation measures a
trade-off, not a precision. A narrow field with the default free set is
not flagged (inflation 1.2 at 20°: the distortion centre is the
well-determined coordinate there, so nothing trades off), although its
σ(tilt) is larger than at 50°; that loss is a precision matter for the
σ-based `TiltDetermination`. The cause is read from the group gain, not
from partial correlations: with `p1, p2` free at the study's baseline the
tilt's partner by partial correlation is the principal point alone, while
holding `p1, p2` cuts its σ 12× (two solves: 11.6×). A radial
coefficient's inflation leaves the camera's other radial coefficients out
of its regressors (as `fx` and `fy` leave each other out): the terms of
one polynomial trade off by construction (Kannala–Brandt `k1..k4` on the
suite's fisheye cameras at 10–76, a 90° Brown–Conrady `k2` at 7.1), the
pairwise check names a pair that does, and the remedy is model comparison,
not a capture change. They stay regressors of every other parameter,
since the tilt is determined through them. Under both rules no suite
camera warns (largest inflation 6.0, the synthetic Scheimpflug entry's
`tilt_y`), and the tilt cells keep their statuses and numbers.

### Where it runs

In `planar_family::optimize_camera`, so planar intrinsics, each rig
camera's own planar stage and the single-camera hand-eye's intrinsics
stage share it. A rig's joint stage keeps each camera's branch: a fixed
tilt stays fixed with `p1, p2` free, a prior stays a prior factor. The
seeded route's tilt seed becomes step 2's start; the cold multi-start tilt
sweep is replaced by the decentring seed, which reached the reference
minimum in 512 of 512 synthetic draws, once that is measured on the
private Scheimpflug captures (open). Step 1's cold linear init fails on
strong barrel distortion (35° field, 15 % at the corner); there an
intrinsics seed is required, as the seeded route already asks.

## Consequences

- A tilted camera whose tilt the data do not determine is reported in the
  sensor-normal frame with a zero tilt and its decentring, and says so.
  Its downstream predictions (held-out reprojection, rig transfer and
  triangulation, hand-eye) are within 2× of the estimated tilt's in
  1592 of 1600 synthetic draws (99th percentile 1.9×, max 4.9×, the
  excess where 8 central views cannot show the misfit); a zero tilt with
  `p1, p2` held exceeds 1.5× in 24–28 % of the draws at 5–20° (max 47×).
- A real lens decentring on a determined tilt is still reported as tilt
  (`Δτ = −p1/k1`): the statistic is conditional on no decentring, and no
  statistic from the data alone removes that (it is the same first-order
  signal). A prior with `p1, p2` free removes it; `separable_sigma` shows
  when nothing can.
- Python: `TiltPolicy`, `CameraPrior` and `tilt_determination` in
  `types.py` and `__init__.pyi`; the book's Scheimpflug chapter states the
  frames and the procedure next to what determines the tilt. Migration:
  none while `Estimate` is the default; the default flip changes what a
  result means (a camera may then report a zero tilt and a different
  frame), so it is a major release with migration notes (AGENTS.md §8).
- Gate before the default flips: on the private Scheimpflug rigs, the
  procedure's branch per camera, the held-out error against `Estimate`
  within 5 %, and the decentring seed reaching the seeded route's minimum.

## Alternatives rejected

- **Always estimate (today).** Never worse downstream on a tilt-only lens,
  but reports a lens-axis frame the data cannot fix where the distortion
  is weak, and one biased by any decentring by `p1/k1`, a bias that grows
  as the distortion weakens.
- **A mandatory prior.** Most users have no mechanical tolerance to give.
- **Fix the tilt at zero with `p1, p2` held.** Costs a median 2× and up
  to 31× the noise floor in undetermined cameras with a 5–20° tilt, and
  its frame is neither the lens axis nor the sensor normal.
- **Decide on the tilt's σ with `p1, p2` free** ("is the tilt told apart
  from decentring"). The honest question, but only strongly distorting
  lenses pass, so nearly every Scheimpflug lens would need a
  prior; it is recorded as `separable_sigma` instead.
- **Decide on the likelihood ratio alone.** It answers "is there a tilt
  beyond a decentring", not "how well is the frame known", and has no
  power at small tilts where the fix still costs (up to 21× the estimated
  tilt's error at 2°).
- **A threshold on the unscaled σ.** Under per-view errors it is 2.3–5×
  short of the Monte-Carlo scatter and calls those tilts determined.
