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
point-to-plane, laser line distance) or is a prior. Observation residuals are
modelled as `rᵢ ~ N(0, σ_f² / ρ′ᵢ · I)`: an unknown per-family noise,
inflated where the robust loss down-weights. Priors are `N(0, I)` after their
`sqrt_info`.

## Estimator

First order at `θ̂`, with `Jᵢ = ∂rᵢ/∂θ` (raw, loss-free) and `ρ′ᵢ = ρ′(‖rᵢ‖²)`:

```
H = Σ_f (1/σ̂_f²) Σ_{i∈f} ρ′ᵢ Jᵢᵀ Jᵢ  +  Σ_{priors} Jᵢᵀ Jᵢ,      Σ = H⁻¹
```

`H` is the Gauss–Newton Hessian of the IRLS objective at its fixed point.
The second-order term `Σ ρ′ rᵢ · ∇²rᵢ` and the `ρ″` term are dropped. Both
vanish for exact data, and the first is small against `JᵀJ` at sub-pixel
residuals. *Open*: the size of the dropped terms on the real-data suite
entries (compare `H` with a finite-difference Hessian of the objective).

**Sandwich alternative.** `Σ_s = H⁻¹ B H⁻¹` with `B = Σᵢ ρ′ᵢ² Jᵢᵀ rᵢ rᵢᵀ Jᵢ`
is consistent when the weights are misspecified. *Open*: whether the Huber /
Cauchy Monte-Carlo cells need it (gate below).

## Variance components

`σ̂_f² = Ω_f / r_f` with `Ω_f = Σ_{i∈f} ρ′ᵢ ‖rᵢ‖²` and the redundancy share
`r_f = m_f − tr(H_f Σ)`, where `H_f` is family `f`'s term of `H`. The shares
of all families and priors sum to `m − n`. Iterate from `σ̂_f = 1` to a 1 %
change (at most five iterations). With one family and no priors this is the
classical `σ̂² = Ω / (m − n)` in one step.

`tr(H_f Σ)` needs only the diagonal blocks of `Σ` on the columns `H_f`
touches, and the solve that produces the reported covariance already
computes those.

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
| Planar intrinsics | `planar_intrinsics/pinhole`, 0.1 / 0.5 px | σ within ±10 %; NEES mean within the χ² 95 % band | *pending* |
| Planar intrinsics, outliers | 5 % displaced 10–30 px, Huber / Cauchy | σ within ±20 % on inliers | *pending* |
| Scheimpflug intrinsics | `scheimpflug_intrinsics/*` | σ within ±10 % | *pending* |
| Hand-eye | `single_cam_handeye/*`, with robot-pose refinement | σ within ±10 % | *pending* |
| Rig extrinsics / hand-eye | `rig_*` scenes | σ within ±10 % | *pending* |
| Laser | `laserline_device/*`, `rig_laserline_device/*` | σ within ±10 %, two families | *pending* |
| Backend parity | every scene above | tiny-solver vs factrs `Σ` within 1e-6 relative | *pending* |
| Rank deficiency | hand-eye, single rotation axis | `RankDeficient` naming the axis translation and spin | *pending* |

## Limitations

- **Conditional covariance**: staged problems whose final stage holds earlier
  outputs fixed under-report `σ` for what depends on them.
- **Target geometry is exact**: board manufacturing and flatness errors are
  not modelled, and neither is robot-pose noise beyond the correction priors.
- **First order only**: strongly non-linear directions (wide-angle
  distortion at the image edge) are approximated by their linearization.
