# Parameter uncertainty — proof pack

Family: the covariance every calibration result carries (ADR 0026).
Implementation: `vision-calibration-optim::backend` (covariance at the LM
optimum, shared by both backends); labels and the export type in core;
wiring per problem in `problems/` and the pipeline exports.

## Model

The LM minimizes `½ Σᵢ ρᵢ(‖rᵢ(θ)‖²)` over the free tangent coordinates `θ`.
Each residual block `rᵢ` belongs to one family `f` (reprojection, laser
point-to-plane, laser line distance) or is a prior. Within a family the
scalar residuals are modelled as i.i.d. with an unknown distribution, which
may be heavy-tailed: outliers are part of the noise. A prior's residual is
`N(0, σᵢ² I)` after its `sqrt_info`, with `σᵢ = 1` unless the prior is
weighed in another family's unit (the robot-pose priors, in pixels).

## Estimator

At `θ̂`, with `Jᵢ = ∂rᵢ/∂θ` and `rᵢ` raw (loss-free) and `sᵢ = ‖rᵢ‖²`:

```
Σ = H⁻¹ G H⁻¹,   H = Σ_f κ_f A_f + A_p,   G = Σ_f s_f A_f + B_p
A_f = Σ_{i∈f} JᵢᵀJᵢ,   A_p = Σ_priors JᵢᵀJᵢ,   B_p = Σ_priors σᵢ² JᵢᵀJᵢ
κ_f = Σ_{i∈f} (dᵢ ρ′ᵢ + 2ρ″ᵢ sᵢ) / m_f,   s_f = Σ_{i∈f} ρ′ᵢ² sᵢ / m_f · m / (m − n)
```

`dᵢ` is the block's residual dimension. With one family and no priors this
is Huber's asymptotic covariance of an M-estimator, `E[ψ²]/E[ψ′]² · (JᵀJ)⁻¹`,
for i.i.d. residuals within a family. With priors or several families the
moments stay pooled per family, inside the sandwich for the weights the
solve used, which need not be optimal (laser and corner families are each
weighed in their own unit). The robot-pose priors are weighed in pixels at
the reprojection `σ̂` (ADR 0026, "Robot-pose priors"), which makes the
hand-eye objective the maximum-likelihood one: there the sandwich equals the
inverse information `(A_f/σ_f² + A_p⁰)⁻¹`, `A_p⁰` the priors' information in
their own units. When the pixels were weighed at 1 px instead, the inverse
information put σ at 0.64× the scatter for some coordinates while the
sandwich held 0.93–1.05. `ψ = ρ′ r` is the influence of a residual and `ψ′` its
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
and is small at sub-pixel residuals. Open: its size on the real-data suite
entries (compare `H` with a finite-difference Hessian of the objective).

## Noise level

The result also reports `σ̂_f = 1.4826 · median |rᵢⱼ|`, the
Gaussian-equivalent noise of one scalar residual, robust to outliers. It is
a diagnostic only; the covariance uses `σ_f` above.

## Per-point covariance (whitening)

When the observations state `Σᵢ` per point, the reprojection residual is
`rᵢ = Lᵢ eᵢ`, `LᵢᵀLᵢ = Σᵢ⁻¹` (upper Cholesky factor of the inverse, computed
in closed form: `l₀₀ = √(yy/det)`, `l₀₁ = −xy/√(yy·det)`, `l₁₁ = 1/√yy`). If
the stated covariances are right up to a common factor, `Σᵢ_true = σ₀² Σᵢ`,
the whitened residuals are i.i.d. with variance `σ₀²`, which is exactly the
model above with one family: the sandwich, the dof factor and the
median-based diagnostic apply unchanged, and the diagnostic becomes `σ̂₀`
(reported as `NoiseLevel::Sigma0`). An input is all whitened or all pixel
(see ADR 0026): a pooled family needs one unit.

`σ̂₀` reads below 1 even with exact covariances: the residuals at the
optimum have variance `(m − n)/m` (the covariance's `m/(m − n)` corrects
for it; the diagnostic does not). With `m = 1620`, `n = 68`, that is 0.979,
against 0.976 measured.

Study (`optim/tests/feature_covariance.rs`, `--ignored`): per-point
covariances fixed across seeds, major-axis σ uniform in 0.1–1 px, minor axis
1–4× smaller, orientation uniform; noise `R diag(σ₁, σ₂) z`, `z ~ N(0, I)`,
SplitMix64-keyed Box–Muller per (seed, camera, view, point). Each seed
solves the same pixels twice, whitened and with no covariances (uniform).
1000 seeds per scene:

| Scene | σ̂₀ (mean ± sd) | predicted σ / MC σ | normalized-error RMS | within 2σ | RMSE whitened / uniform |
|---|---|---|---|---|---|
| Planar, Brown-Conrady, 10 views of a 9×9 board (8 parameters) | 0.976 ± 0.029 | 0.964–1.041 | 0.961–1.038 | 95.4 % | 0.326–0.411 |
| Rig extrinsics, 2 cameras, intrinsics free, cam0 fixed (22 parameters) | 0.986 ± 0.020 | 0.962–1.038 | 0.963–1.041 | 95.2 % | 0.313–0.398 |

At 200 seeds the planar ratios spread 0.82–1.09 (`cy`, `p1` low): the
Monte-Carlo σ alone has a 5 % standard error there, so the ±10 % gate runs
at 500 seeds by default. The efficiency gain is large because the stated σ
span a factor of 10 (variance 100): the uniform solve gives the 1 px points
the weight of the 0.1 px ones. The always-on test is a 12-seed planar run
with loose bounds.

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
- **Robot-pose corrections**: every view's correction is free. With the
  camera pose of each view fixed by its corners, a change of the hand-eye
  and target poses can be absorbed view by view by the corrections, so those
  12 directions (6 + 6) are determined by the priors alone: the hand-eye and
  target σ scale with the priors' σ, not with the corner noise. The target
  in the base frame inherits the robot's absolute rotation error, a common
  rotation of all corrections about the base origin: its translation σ is
  about `σ_rot / √views` times the target's distance from the base.
  Fixing view 0's correction (the earlier gauge choice) took that view's
  robot pose as exact and dropped its error from the covariance.
- **Laser**: a plane seen from a single board pose is underdetermined. The
  S² sign is a discrete gauge and does not affect the covariance.

## Evidence

Each gate is a seeded Monte-Carlo test: `N ≥ 200` noise realizations of a
synthetic scene, solved by the standard pipeline. It compares the predicted
`σ` (the mean of the per-run predictions) with the empirical standard
deviation of the estimates, per labelled coordinate, and checks the
normalized estimation error squared against `χ²_k`, `k` the covariance's
rank (a plane normal's three components have rank 2). The NEES is computed
on the correlation matrix, so the rank does not depend on units. The plug-in
covariance, evaluated at each run's estimate, puts the mean NEES a few
percent above `k`, an offset that more runs do not shrink, so the
`optim/tests/uncertainty.rs` gates accept the wider of three standard errors
and the σ tolerance.

| Family | Scene | Gate | Result |
|--------|-------|------|--------|
| Planar intrinsics | 8 views of an 8×6 board, ±0.5 px uniform noise, 300 runs | σ within ±10 %; NEES mean within 3 standard errors of χ²(8) | σ 0.94–1.04, NEES 8.07 |
| Planar intrinsics, outliers | 5 % displaced 10–30 px, Huber / Cauchy (1 px), 200 runs | σ within ±20 %; NEES as above | Huber σ 0.91–1.00, NEES 8.55; Cauchy σ 0.95–1.05, NEES 7.98 |
| Scheimpflug intrinsics | 10 views of a 9×9 board, tilt (0.05, −0.02) rad, ±0.5 px uniform noise, 200 runs (`optim/tests/uncertainty.rs`) | σ within ±10 %; NEES as above | σ 0.90–1.04 (fx…tilt_y), NEES 8.75 |
| Rig extrinsics | 2 cameras (intrinsics free), 10 views of a 9×9 board, ±0.5 px, 200 runs; `cam1.cam_se3_rig` through the inverse-pose adjoint | σ within ±10 %; NEES as above | σ 0.95–1.08 (22 parameters), NEES 22.37 |
| Hand-eye, exact robot poses | eye-in-hand, intrinsics free, 10 poses, ±0.5 px, 200 runs | σ within ±10 %; NEES as above | σ 0.93–1.07 (20 parameters), NEES 19.93 |
| Hand-eye, robot-pose priors | as above, robot poses perturbed with the priors' σ (0.5°, 1 mm; view 0 exact) and refined, 400 runs | σ within ±10 %; NEES as above | sandwich σ 0.93–1.05, NEES 20.91; inverse-information form 0.64–1.07 (rejected) |
| Laserline device | 10 views of a 9×9 board, laser sheet crossing the axis at 0.5 m, ±0.5 px uniform noise on corners and laser pixels, laser loss Huber 1 px (the default), 400 runs | σ within ±10 %, two families; NEES as above | σ 0.99–1.07 (12 parameters, rank 11), NEES 11.84; with Huber 0.01 px, nearly L1, `plane.d` 1.11 and the plane less precise (σ(d) 1.58e-4 against 1.45e-4) |
| Projection σ | the Scheimpflug scene above, 744 × 516 image; rays fixed by the true camera at the centre, two opposite corners (outside the boards' coverage) and an interior node, 200 runs | major-axis σ within ±10 % of the scatter of the projected pixel | ratio 0.94–0.97; σ ≈ 10.5–11.3 px everywhere, set by σ(cx) 9.3, σ(cy) 10.0 (ρ(cx, tilt_y) = −0.99) |
| Synthetic suite entries | six frozen scenes of the solver benchmark (0.5 px), one realization each, default configs | every parameter within 5σ of the truth (`suite check`) | planar 1.9σ, Scheimpflug 1.5σ, hand-eye 0.9σ, rig 2.1σ, laserline 2.2σ (Mahalanobis² 11.6 over rank 11; 17.8 with a nearly-L1 laser loss of 0.01 px), joint rig hand-eye laserline 4.5σ (`cam1.cam_se3_rig.rx`); hand-eye coordinates under robot-pose priors within 0.7σ |
| Quality verdict | 20 views of a 9×7 board on a 1280 × 960 camera, 0.1 px Gaussian noise, spread over the image or bunched in its top-left quadrant (`analysis/quality/tests.rs`) | spread passes every check; bunched fails coverage | spread: coverage 100 %, projection σ ≤ 0.70 px, verdict pass; bunched: coverage 25 % (fail), projection σ 3.88 px (warn) |
| Next-view prediction | 6 tilted views of a 9 × 7 board over the central part of a 1280 × 960 image, 0.1–0.3 px Gaussian noise, 5 seeds; Brown-Conrady, division, Scheimpflug; best-, median-, worst-ranked candidate and best three (`analysis/next_view/tests.rs`) | predicted σ per parameter and max projection σ against the re-solve at the current σ_eff, within 10 % on average; best candidate's real reduction above the worst one's | σ 0.5 %, 0.1 %, 2.1 % (worst case 13.5 %); max projection σ 0.5 %, 0.0 %, 1.2 %; best above worst in all 15 scenes |
| Whitened reprojection | planar and 2-camera rig, per-point covariances 0.1–1 px, anisotropy up to 4:1, 1000 runs (see "Per-point covariance") | σ̂₀ within 5 % of 1; σ within ±10 %; normalized RMS within ±10 %; within-2σ 95.4 ± 2 %; RMSE below the uniform solve | σ̂₀ 0.976, 0.986; σ 0.96–1.04; RMSE 0.31–0.41 of uniform |
| Backend parity | planar, with and without Huber | tiny-solver vs factrs `Σ` within 1e-6 relative | pass |
| Backend parity, whitened factor | every camera model × chain, `L` upper triangular with an off-diagonal term (`backend/parity_tests.rs`) | residual, Jacobian and cost equal in both backends | pass |
| Rank deficiency | one camera through two free SE(3) blocks (a 6-DOF gauge) | `RankDeficient` naming the gauge blocks | pass |
| Rank deficiency | joint rig hand-eye laserline, eye-to-hand, robot rotating about z only | `RankDeficient` naming the hand-eye and target translations along z | pass (`rig_se3_base.tz`, `gripper_se3_target.tz`) |

## Limitations

- **Conditional covariance**: staged problems whose final stage holds earlier
  outputs fixed under-report `σ` for what depends on them.
- **Target geometry is exact**: board manufacturing and flatness errors are
  not modelled, and neither is robot-pose noise beyond the correction priors.
- **First order only**: strongly non-linear directions (wide-angle
  distortion at the image edge) are approximated by their linearization.
- **Projection σ is camera-frame**: it includes the principal-point error a
  re-estimated camera pose would absorb. On `stereo_rig` with the intrinsics
  refined in the rig solve it is 0.74 px at the centre (σ(cx) 0.75) and
  2.4 px at the corners the boards did not reach.
