# ADR 0026: Parameter Uncertainty as a Standard Result

- Status: Accepted
- Date: 2026-10-19

## Context

A calibration result gives point estimates and reprojection statistics. It
cannot say whether `fx` is known to 0.1 px or to 10 px, which parameters trade
off against each other, or how well a pixel's ray is determined. Reprojection
error does not answer this: a fit to too few or too similar views reprojects
well and extrapolates badly.

The pieces for a first-order answer are in place. Both backends linearize
through one `LinearizationEngine` (ADR 0025), and the shared LM assembles the
normal equations once per solve with a symbolic sparse LLT
(`backend/normal_equations.rs`). The linearization the LM uses is not the
right input, though. It is loss-corrected, and the backends correct
differently: tiny-solver applies the Triggs correction, factrs `√ρ′`
reweighting. A covariance taken from it would depend on the backend.

## Decision

### Estimator

At the converged optimum `θ̂`, over the **free tangent coordinates** (`n` of
them), the covariance is Huber's asymptotic covariance of an M-estimator,
with its moments pooled per residual family `f`:

```
Σ = H⁻¹ G H⁻¹,    H = Σ_f κ_f A_f + A_p,    G = Σ_f s_f A_f + A_p
A_f = Σ_{i∈f} JᵢᵀJᵢ,    A_p = Σ_priors JᵢᵀJᵢ
κ_f = E_f[ψ′],    s_f = E_f[ψ²] · m / (m − n),    ψ = ρ′ r,   ψ′ = tr(ρ′ I + 2ρ″ r rᵀ)
```

`H` is the curvature of the objective the solve minimized, `G` the scatter
of its gradient. With one family and no priors, `Σ = σ_f² A_f⁻¹` with
`σ_f² = s_f / κ_f²`, and `G` is not formed.

- `Jᵢ` and `rᵢ` are the **raw** Jacobian and residual, from a loss-free
  linearization at `θ̂`. Residuals include their observation weight `w`.
  Without a robust loss the solve's own linearization is reused; with one,
  the IR is recompiled without its losses for this single linearization.
- `E_f` averages over the family's scalar residuals, and `ρ′`, `ρ″` are the
  loss derivatives at `‖rᵢ‖²` (`RobustLoss::rho_prime` / `rho_second`).
  Without a loss, `σ_f² = Σ‖r‖² / (m − n)`, and `Σ` is the classical
  `σ̂² (JᵀJ)⁻¹`.
- Families are grouped by factor kind: reprojection in pixels, laser
  point-to-plane in metres, laser line distance in pixels. Each family's
  `σ_f` comes from its own residuals, so mixed problems need no iteration.
  Observation weights are relative: `σ_f` absorbs their scale.
- `A_p` stacks the prior residuals (`Se3TangentPrior`), whose `sqrt_info` is
  known information.
- **The weights are the solve's.** The solver weighs each family's raw
  residuals and the priors as the IR states them; it does not rescale a
  family by its noise. So `Σ` is the sandwich for those weights, not the
  inverse information `(Σ_f A_f/σ_f² + A_p)⁻¹` of an optimally weighted
  solve. The two agree with one family and no priors; with robot-pose priors
  the inverse-information form understated σ by up to 36 % against
  Monte-Carlo scatter, the sandwich held 0.93–1.05.
- `H` is assembled with the existing `NormalEquations` (unit-diagonal
  normalization, sparse Cholesky) in one routine shared by both backends.

**Why this form.** A seeded Monte-Carlo study ran 200–300 noise realizations of
a planar scene, with and without 5 % outliers displaced by 10–30 px:

- The IRLS form `σ̂² (JᵀWJ)⁻¹`, with `σ̂² = Σρ′‖r‖²/(m − n)`, overstated σ
  2.1–2.2× under Huber. An outlier's `ρ′‖r‖² = k|r|` inflates `σ̂`, while its
  effect on the estimate is a bounded push `k`.
- The per-residual sandwich `Ĥ⁻¹ (Σ ρ′² Jᵀr rᵀJ) Ĥ⁻¹` was calibrated in σ.
  It was too noisy run to run, though: its mean NEES was 9.0 against 8.
- The pooled form above held σ within 0.91–1.05 of the scatter in all three
  cases (clean, Huber, Cauchy), with NEES 8.0–8.6.
- With priors or several families, pooling the moments per family inside
  the sandwich keeps that stability; see "The weights are the solve's"
  above.

The proof pack records the gates.

The first-order (Gauss–Newton) approximation ignores second-order residual
terms, which is valid at small residuals.

### Noise level

The result also reports, per family, `σ̂_f = 1.4826 · median |rᵢⱼ|`. That is
the Gaussian-equivalent noise of one scalar residual, which outliers do not
inflate. For the reprojection family it estimates the corner noise in pixels,
which is useful on its own. It is a diagnostic; the covariance uses `σ_f`
above.

### Coordinates and labels

- **Euclidean blocks** contribute their free coordinates; fixed indices are
  left out.
- **SE(3)** blocks use the right perturbation `T · Exp([δθ, δt])`: rotation
  first (radians), then translation (metres), in the source frame of
  `a_se3_b`. Both backends retract this way. Their exponentials differ beyond
  first order, which does not matter at `δ = 0`. ADR 0009 records the
  convention.
- **Laser planes** are reported over the ambient `[nx, ny, nz, d]`. Their
  covariance maps through `diag([b₁ b₂], 1)`, where `b₁, b₂` is the S² tangent
  basis. The 4×4 block is positive semi-definite with its null direction along
  `n`, and it does not depend on the internal tangent basis.
- **Cameras solved apart.** The rig laserline fits each camera's plane in
  its own solve, with the rig calibration fixed. Its covariance is the
  block-diagonal of the per-camera ones, and `ResidualNoise::camera` names
  the camera each noise level belongs to; joint solves report one level per
  family (`camera: None`).
- **Labels** come from one source per block kind:
  - intrinsics `fx, fy, cx, cy`;
  - distortion from `DistortionKind::param_names()`, next to its `dim()`;
  - Scheimpflug `tilt_x, tilt_y`;
  - poses `rx, ry, rz, tx, ty, tz`;
  - planes `plane.nx, plane.ny, plane.nz, plane.d`.

  Each label is prefixed by its block's name (`cam0.fx`, `handeye.rx`).
- The covariance is **conditional on everything fixed**: fixed coordinates,
  fixed blocks, and the outputs of earlier stages that the final stage holds
  constant. The result lists the fixed labels.

### What a result carries

- The **global parameters**, meaning every block that is not per view, with
  their full labelled covariance.
- **Per-view blocks** (target poses, robot-pose corrections) as 6×6 marginals.
  Their cross-covariance with the global parameters is left out to keep the
  size linear in the number of views.
- `σ̂_f` per family, the residual count and the degrees of freedom.
- Only the final solve of a staged pipeline feeds the result. Earlier stages
  are initializations.

### Availability is a type

```text
Uncertainty = Available { covariance, view_marginals, noise, dof, fixed }
            | Unavailable { reason }
reason      = NotConverged
            | RankDeficient { unobservable: [label] }
            | AtBound { labels: [label] }
            | NoRedundancy
```

- **NotConverged**: the LM stopped on its iteration budget, so `θ̂` is not a
  stationary point.
- **RankDeficient**: `H`, Jacobi-scaled, has eigenvalues below
  `1e-12 · λ_max`. The labels carrying the null vectors are reported, not
  failed on. A dense eigendecomposition runs only on this path.
- **AtBound**: a free coordinate sits on its bound (Scheimpflug bounds), where
  the Gaussian approximation does not hold. Treating it as fixed would
  silently condition on it.
- **NoRedundancy**: there are observation residuals but no more of them
  than free coordinates (`m ≤ n`).

Each reason is logged at `warn`, and the calibration itself succeeds.

### Always on

No config knob: uncertainty is part of every result. The cost is one
linearization at the solution (plus a recompile when a robust loss is set),
one assembly and factorization, and `n` triangular solve pairs. On the
solver benchmark's planar scenes that is 1.14–1.24× the solve time without
a loss and 1.2–1.6× with one.

## Consequences

- Every `*Export` gains `uncertainty` (`#[serde(default)]`, so older JSON
  loads). The Python package exposes it with `std_dev(label)` and
  `correlation(a, b)`. The app shows `value ± σ`.
- Some staged problems hold intrinsics fixed in their final stage, such as the
  Scheimpflug rig BA, which keeps intrinsics fixed. Their extrinsic σ is
  conditional, and therefore optimistic. The fixed list says so; propagating
  earlier-stage covariance is future work.
- A default that fixes an observable block for speed rather than for a gauge
  makes the default result's covariance conditional on an arbitrary view, so
  the laserline and Scheimpflug defaults hold no pose fixed.
- Regression-suite drift gates can be stated in units of `σ`.
- The LM loop reports whether a stopping rule ended it; the covariance is
  unavailable when the iteration budget did.

## Alternatives considered

- **Covariance from the LM's loss-corrected Jacobian.** Rejected: it differs
  between backends.
- **IRLS covariance `σ̂²(JᵀWJ)⁻¹` with Förstner variance components.**
  Rejected on the Monte-Carlo evidence above: it overstates σ about twofold
  under Huber with outliers.
- **Per-residual sandwich.** Rejected: calibrated on average but noisy per
  run.
- **Runtime bootstrap or Monte-Carlo.** Rejected: tens of solves per result.
  It is the validation method instead.
- **An opt-in flag.** Rejected: a standard result is always present, and the
  cost is small.
- **A user-supplied pixel noise instead of `σ̂`.** Rejected as the default: `σ̂`
  calibrates itself, and a known-noise mode can be added if needed.
- **Treating parameters at a bound as fixed.** Rejected: it silently
  conditions the result.
