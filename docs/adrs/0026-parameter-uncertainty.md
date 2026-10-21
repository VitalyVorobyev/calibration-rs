# ADR 0026: Parameter Uncertainty as a Standard Result

## Context

Point estimates and reprojection statistics cannot say whether `fx` is known
to 0.1 px or to 10 px, which parameters trade off, or how well a pixel's
ray is determined: a fit to too few or too similar views reprojects well
and extrapolates badly. The LM's own linearization is not a usable source,
because it is loss-corrected differently by each backend.

## Decision

### Estimator

At the optimum, over the free tangent coordinates, the covariance is the
sandwich `Σ = H⁻¹ G H⁻¹` of Huber's M-estimator covariance, with its
moments pooled per residual family (reprojection px, laser point-to-plane
m, laser line px) and the priors' `sqrt_info` as known information. It is
built from a **loss-free** linearization at the optimum, once, with the
same normal-equation assembly for both backends. With one family, no priors
and no loss it is the classical `σ̂² (JᵀJ)⁻¹`. The derivation, its
identifiability analysis and the Monte-Carlo gates are in
`docs/notes/uncertainty.md`.

The weights are the solve's: `Σ` is the sandwich for the weights the solver
used, not the inverse information of an optimally weighted solve.

Each result also reports, per family, the robust noise level
`σ̂_f = 1.4826 · median |r|` as a diagnostic.

### What a result carries

- The full labelled covariance of the global parameters (every block that is
  not per view), and 6×6 marginals for per-view blocks; their
  cross-covariance with the globals is left out so the size stays linear in
  the views.
- Labels come from one source per block kind (`cam0.fx`, `handeye.rx`,
  `plane.nx`, distortion names from `DistortionKind::param_names()`).
  Coordinates follow ADR 0009; laser planes are reported over
  `[nx, ny, nz, d]`.
- The covariance is **conditional on everything fixed** — fixed
  coordinates, fixed blocks and earlier stages the final solve holds — and
  the result lists the fixed labels. Only the final solve feeds it.
- Projection uncertainty (the σ of where a pixel's ray lands, per camera,
  over a grid) is derived on demand from the covariance through
  `CalibratedCameras`, not stored in the export.

### Availability is a type

```text
Uncertainty = Available { covariance, view_marginals, noise, dof, fixed }
            | Unavailable { NotConverged | RankDeficient { unobservable }
                          | AtBound { labels } | NoRedundancy }
```

A solve that stopped on its iteration budget, a rank-deficient
`H` (with the labels spanning the null space), a free coordinate on its
bound, or no redundancy makes the covariance unavailable with a reason; the
calibration itself still succeeds.

### Always on

No config knob: one extra linearization, assembly and factorization, about
1.2–1.6× the solve time.

## Consequences

- Every export carries `uncertainty` (ADR 0012); Python exposes `std_dev`,
  `correlation` and the projection map; the app shows `value ± σ`.
- Staged problems whose final stage holds intrinsics fixed report
  conditional, hence optimistic, extrinsic σ; the fixed list says so.
  Propagating earlier-stage covariance is open.
- Defaults do not fix an observable block for speed, since that would make
  the default covariance conditional on an arbitrary choice.
- The regression suite gates parameter changes in units of σ.

## Alternatives rejected

- **The LM's loss-corrected linearization** — differs between backends.
- **IRLS covariance with variance components** — overstates σ about twofold
  under Huber with outliers.
- **Per-residual sandwich** — calibrated on average, too noisy per run.
- **Bootstrap or Monte-Carlo at runtime** — tens of solves per result; it is
  the validation method instead.
- **Treating a parameter at its bound as fixed** — silently conditions the
  result.
