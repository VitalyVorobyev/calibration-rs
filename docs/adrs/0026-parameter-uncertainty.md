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
them), the information matrix is

```
H = Σ_f  J_fᵀ W_f J_f / σ̂_f²  +  J_pᵀ J_p
Σ = H⁻¹
```

- `J_f` is the **raw** Jacobian of the residual family `f`, from one
  loss-free linearization at `θ̂`. Residuals include their observation weight
  `w`.
- `W_f = diag(ρ′(‖rᵢ‖²))` holds the IRLS weights of the family's blocks, from
  `RobustLoss::rho_prime`. Without a loss, `W = I`.
- `J_p` stacks the prior residuals (`Se3TangentPrior`). Their `sqrt_info` is
  known information, and they are not rescaled.
- `H` is assembled into the existing `NormalEquations` at `λ = 0` and
  factored with its symbolic LLT, in one routine shared by both backends.

The first-order (Gauss–Newton) approximation ignores second-order residual
terms, which is valid at small residuals. The robust "sandwich"
(Huber–White) form `H⁻¹ (Σᵢ Jᵢᵀ ρ′ᵢ² rᵢ rᵢᵀ Jᵢ) H⁻¹` is not shipped. The proof
pack evaluates it, and it replaces the simple form only if Monte-Carlo shows
the simple form miscalibrated under outliers.

### Noise level

Residual families are grouped by factor kind: reprojection in pixels, laser
point-to-plane in metres, laser line distance in pixels. Each family gets an
a posteriori variance

```
σ̂_f² = Ω_f / r_f,   Ω_f = Σ_{i∈f} ρ′ᵢ ‖rᵢ‖²,   r_f = m_f − tr(H_f H⁻¹)
```

- `m_f` counts the family's scalar residuals, and `H_f` is its term of `H`.
- The `r_f` are Förstner's redundancy shares; with one family and no priors,
  `r = m − n`.
- The estimate iterates from `σ̂_f = 1` until every `σ̂_f` changes by less than
  1 %, for at most five iterations.
- Observation weights `w` are relative: their scale is absorbed by `σ̂_f`.
- The result reports `σ̂_f` and `r_f` per family. `σ̂` of the reprojection
  family estimates the corner noise in pixels, which is useful on its own.

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
- **Labels** come from one source per block kind:
  - intrinsics `fx, fy, cx, cy`;
  - distortion from `DistortionKind::param_names()`, next to its `dim()`;
  - Scheimpflug `tilt_x, tilt_y`;
  - poses `rx, ry, rz, tx, ty, tz`;
  - planes `nx, ny, nz, d`.

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
- `σ̂_f`, `r_f`, the residual count and the degrees of freedom.
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
- **NoRedundancy**: some `r_f ≤ 0`.

Each reason is logged at `warn`, and the calibration itself succeeds.

### Always on

No config knob: uncertainty is part of every result. The cost is one extra
linearization, one numeric factorization and one triangular solve pair per
reported column. That is `n_global + 6` per view, small next to the LM's
dozens of factorizations; the bench measures it.

## Consequences

- Every `*Export` gains `uncertainty` (`#[serde(default)]`, so older JSON
  loads). The Python package exposes it with `std_dev(label)` and
  `correlation(a, b)`. The app shows `value ± σ`.
- Some staged problems hold intrinsics fixed in their final stage, such as the
  Scheimpflug rig BA, which keeps intrinsics fixed. Their extrinsic σ is
  conditional, and therefore optimistic. The fixed list says so; propagating
  earlier-stage covariance is future work.
- A default that fixes an observable block for speed rather than for a gauge
  makes the default result's covariance conditional on an arbitrary view.
  The laserline and Scheimpflug defaults hold pose 0 fixed this way. These
  defaults become free, which changes those results slightly.
- Regression-suite drift gates can be stated in units of `σ`.
- `SolveReport` gains `converged`, the residual count and the degrees of
  freedom; `LinearizationEngine` gains a raw linearization and a column
  layout (IR block → columns).

## Alternatives considered

- **Covariance from the LM's loss-corrected Jacobian.** Rejected: it differs
  between backends.
- **Runtime bootstrap or Monte-Carlo.** Rejected: tens of solves per result.
  It is the validation method instead.
- **An opt-in flag.** Rejected: a standard result is always present, and the
  cost is small.
- **A user-supplied pixel noise instead of `σ̂`.** Rejected as the default: `σ̂`
  calibrates itself, and a known-noise mode can be added if needed.
- **Treating parameters at a bound as fixed.** Rejected: it silently
  conditions the result.
