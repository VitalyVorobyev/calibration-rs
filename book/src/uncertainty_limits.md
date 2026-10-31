# Limits and Unavailable Results

A covariance is an honest statement about noise, not about everything that can be wrong with a calibration. This chapter lists the cases where none can be given, and where one is given but answers a narrower question than it seems to.

## When It Is Unavailable

| `reason` | Meaning | What to do |
|---|---|---|
| `not_converged` | The solve stopped on its iteration budget | Raise `solver.max_iters` |
| `rank_deficient` | The data does not determine the parameters listed in `unobservable` (alone or in combination) | Add views that differ in the way the listed parameters need, e.g. tilted boards for `fx`, `cx`; or fix one of them |
| `at_bound` | A parameter ended on a bound (`labels`) | Widen or remove the bound, or fix the parameter |
| `no_redundancy` | No more observations than parameters | Add views |
| `not_computed` | The result was produced or stored without uncertainty | — |

`rank_deficient` is a statement about the data, not a failure of the
calibration: the solve still returns its best parameters. A Scheimpflug
camera whose views do not separate the sensor tilt from a principal-point
shift, for example, reports `tilt_y`, `cx` and the board translations as
unobservable together.

## Scope

Every workflow reports the uncertainty of its final solve. When a hand-eye
solve refines the robot poses (the default), it also estimates the robot's
orientation and position error from the corrections, weighed against the
corner noise the residuals show (`robot_poses.estimate_sigma`). The
estimate is then the maximum-likelihood one and its covariance holds: on
synthetic eye-in-hand scenes the normalized errors had an RMS of 1.02–1.13,
from priors that matched the robot's error and from priors ten times too
tight or too loose. With the estimate off, the configured
`robot_poses.rot_sigma` and `trans_sigma` are taken as the robot's true
accuracy: too loose, and the hand-eye and target standard deviations are
pessimistic; too tight, and they are too small, with nothing in the
residuals to say so.

Without refinement the robot poses are taken as exact. Their errors then
show up in the corner residuals, but as an error shared by every corner of a
view rather than independent noise, and the covariance treats it as
independent: with 476 corners a view, it was 16–17 times too small on the
same scenes. Refine the robot poses whenever their error is not well below
the corner noise.

## What a Covariance Does Not Cover

- **Systematic error.** The covariance measures the scatter caused by noise. A wrong board cell size, a board that is not flat or the wrong distortion model leaves it small and the estimate biased; watch the [per-feature residuals](per_feature_residuals.md) for structure (see [Parameter Uncertainty](uncertainty.md)). On real images this dominates: calibrating `stereo`, `kuka_1` and `ds8` from two disjoint halves of their views, the two projections differed by 2 to 5 times their predicted σ, with or without the detector's per-corner covariance. Read the projection σ as a lower bound.
- **Fixed parameters.** It is conditional on the parameters the final solve held fixed (`fixed` in the [result](uncertainty_reading.md)). Fixing a parameter the data cannot pin down makes the others look better determined than they are: a rig with `rig.refine_intrinsics: false` holds each camera's intrinsics, and its extrinsics' σ leaves their uncertainty out.
- **Robot poses.** They are taken as exact unless the hand-eye solve refines them, as described in the Scope section above.
- **The shape of stated point covariances.** [Per-point covariances](uncertainty_point_covariance.md) are trusted up to one common factor, which `σ̂₀` estimates. Covariances that are right for some points and wrong for others (one image overstated, say) are not detected by `σ̂₀` and weigh the points wrongly.
- **The camera's own pose.** The projection map varies only the camera's parameters ([Projection Uncertainty](uncertainty_projection.md)).
