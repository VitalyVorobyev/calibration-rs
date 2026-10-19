# Parameter uncertainty — proof pack

Family: the covariance every calibration result carries (ADR 0026).
Implementation: `vision-calibration-optim::backend` (covariance at the LM
optimum, shared by both backends); labels and the export type in core;
wiring per problem in `problems/` and the pipeline exports.

Status: design fixed (ADR 0026); evidence is filled in as the implementation
lands. Sections marked *open* are what the evidence has to settle.

## Model

The LM minimizes `½ Σᵢ ρᵢ(‖rᵢ(θ)‖²)` over the free tangent coordinates `θ`.
Each residual block `rᵢ` belongs to one family `f` (reprojection, laser
point-to-plane, laser line distance) or is a prior. Within a family the
scalar residuals are modelled as i.i.d. with an unknown distribution, which
may be heavy-tailed: outliers are part of the noise. Priors are `N(0, I)`
after their `sqrt_info`.

## Estimator

At `θ̂`, with `Jᵢ = ∂rᵢ/∂θ` and `rᵢ` raw (loss-free) and `sᵢ = ‖rᵢ‖²`:

```
Σ = (Σ_f A_f / σ_f² + A_p)⁻¹,   A_f = Σ_{i∈f} JᵢᵀJᵢ
σ_f² = (Σ_{i∈f} ρ′ᵢ² sᵢ / m_f) / (Σ_{i∈f} (dᵢ ρ′ᵢ + 2ρ″ᵢ sᵢ) / m_f)² · m / (m − n)
```

`dᵢ` is the block's residual dimension. This is Huber's asymptotic
covariance of an M-estimator, `E[ψ²]/E[ψ′]² · (JᵀJ)⁻¹`, for i.i.d. residuals
within a family. `ψ = ρ′ r` is the influence of a residual and `ψ′` its
derivative, `ρ′ I + 2ρ″ r rᵀ` (trace per scalar). The factor `m/(m − n)`
accounts for the degrees of freedom the fit absorbs. Without a loss it
reduces to the classical `σ̂² (JᵀJ)⁻¹`.

Two alternatives were measured and rejected:
- **IRLS covariance** `σ̂² (JᵀWJ)⁻¹`, with `σ̂² = Σρ′s/(m − n)` and Förstner
  variance components. Under Huber with outliers it overstated σ about
  twofold: the outlier terms `ρ′s = k|r|` inflate `σ̂`, while an outlier only
  pushes the estimate with a bounded force `k`.
- **Per-residual sandwich** `Ĥ⁻¹ (Σ ρ′² Jᵀ r rᵀ J) Ĥ⁻¹`, with the exact robust
  Hessian as bread. It was calibrated in σ but noisy per run (NEES 9.0
  against 8).

The second-order term `Σ ρ′ r · ∇²r` is dropped; it vanishes for exact data
and is small at sub-pixel residuals. *Open*: its size on the real-data suite
entries (compare `H` with a finite-difference Hessian of the objective).

## Noise level

The result also reports `σ̂_f = 1.4826 · median |rᵢⱼ|`, the
Gaussian-equivalent noise of one scalar residual, robust to outliers. It is
a diagnostic only; the covariance uses `σ_f` above.

## Coordinates

- SE(3): right perturbation, `[δθ, δt]` in the source frame of `a_se3_b`.
  Converting to the left perturbation (a change of frame) is the adjoint,
  `Σ_left = Ad_T Σ Ad_Tᵀ`.
- Laser plane: ambient `[n, d]` with `∂(n, d)/∂(δ, δd) = diag([b₁ b₂], 1)`.
  The 4×4 block has rank 3, with its null direction along `n`.
- Fixed coordinates have no rows or columns in `H`. Inverting `H` over the
  free coordinates gives their covariance conditional on the fixed values,
  which is never larger than the marginal would be.

## Identifiability and rank deficiency

What each family leaves unobservable, and how the result reports it. The
gauge analysis is in the family's own note.

- **Planar and Scheimpflug intrinsics**: no gauge. Views of one board
  orientation, or all fronto-parallel views, leave `fx`, `fy` and `cx`, `cy`
  nearly unobservable. A tilt and the principal point trade off when the board
  covers a small part of the image. Expect large `σ` and strong correlations,
  not `RankDeficient`.
- **Rig extrinsics**: the reference camera is fixed (the rig-frame gauge), so
  extrinsic covariances are relative to it.
- **Hand-eye**: all motions about one axis leave the hand-eye translation
  along that axis and the spin about it unobservable (`hand-eye.md`). This is
  the expected `RankDeficient` case.
- **Robot-pose corrections**: view 0's correction is fixed (the common-mode
  gauge). The others are prior-regularized, so they stay observable.
- **Laser**: a plane seen from a single board pose is underdetermined. The
  S² sign is a discrete gauge and does not affect the covariance.

## Evidence

Each gate is a seeded Monte-Carlo test: `N ≥ 200` noise realizations of a
synthetic scene, solved by the standard pipeline. It compares the predicted
`σ` (the mean of the per-run predictions) with the empirical standard
deviation of the estimates, per labelled coordinate, and checks the
normalized estimation error squared against `χ²_n`.

| Family | Scene | Gate | Result |
|--------|-------|------|--------|
| Planar intrinsics | 8 views of an 8×6 board, ±0.5 px uniform noise, 300 runs | σ within ±10 %; NEES mean within 3 standard errors of χ²(8) | σ 0.94–1.04, NEES 8.07 |
| Planar intrinsics, outliers | 5 % displaced 10–30 px, Huber / Cauchy (1 px), 200 runs | σ within ±20 %; NEES as above | Huber σ 0.91–1.00, NEES 8.55; Cauchy σ 0.95–1.05, NEES 7.98 |
| Scheimpflug intrinsics | `scheimpflug_intrinsics/*` | σ within ±10 % | *pending* |
| Hand-eye | `single_cam_handeye/*`, with robot-pose refinement | σ within ±10 % | *pending* |
| Rig extrinsics / hand-eye | `rig_*` scenes | σ within ±10 % | *pending* |
| Laser | `laserline_device/*`, `rig_laserline_device/*` | σ within ±10 %, two families | *pending* |
| Backend parity | planar, with and without Huber | tiny-solver vs factrs `Σ` within 1e-6 relative | pass |
| Rank deficiency | one camera through two free SE(3) blocks (a 6-DOF gauge) | `RankDeficient` naming the gauge blocks | pass |
| Rank deficiency | hand-eye, single rotation axis | `RankDeficient` naming the axis translation and spin | *pending* |

## Limitations

- **Conditional covariance**: staged problems whose final stage holds earlier
  outputs fixed under-report `σ` for what depends on them.
- **Target geometry is exact**: board manufacturing and flatness errors are
  not modelled, and neither is robot-pose noise beyond the correction priors.
- **First order only**: strongly non-linear directions (wide-angle
  distortion at the image edge) are approximated by their linearization.
