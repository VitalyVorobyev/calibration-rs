# ADR 0026: Parameter Uncertainty as a Standard Result

## Context

Point estimates and reprojection statistics cannot say whether `fx` is known
to 0.1 px or to 10 px, which parameters trade off, or how well a pixel's ray
is determined: a fit to too few or too similar views reprojects well and
extrapolates badly. The LM's own linearization is not a usable source,
because each backend loss-corrects it differently.

## Decision

**Estimator.** At the optimum, over the free tangent coordinates, the
covariance is the M-estimator sandwich `Σ = H⁻¹ G H⁻¹`.
- Its moments are pooled per residual family (reprojection px, laser
  point-to-plane m, laser line px), and the priors' square-root information
  counts as known information.
- It is built once, from a loss-free linearization with the same assembly in
  both backends.
- `Σ` is the covariance of the solve as weighted. It is not the inverse
  information of some optimally weighted solve.
- With one family, no priors and no loss it reduces to `σ̂² (JᵀJ)⁻¹`.

Derivation, identifiability and the Monte-Carlo gates are in
`docs/notes/uncertainty.md`.

**The covariance is scaled by the view scatter.**
- The residual noise model takes every scalar residual as independent. On
  real images much of the error is shared by a view's corners (lens model,
  target, detector bias), and the estimate varies between view sets 5–26
  times (in variance) more than predicted.
- A delete-one-view jackknife measures that scatter: one Gauss–Newton step
  per left-out view on the shared coordinates, each view's own coordinates
  profiled out. Its excess over its own expectation under the noise model
  (known from the views' leverages), pooled over the shared coordinates,
  gives `k`, which has mean 1 when the noise model holds. When `k` is above
  its null distribution's 97.5 % point (a scaled χ² with the exact mean and
  variance, the views' steps correlated), the whole covariance is scaled by
  `k`, keeping the model's shape, and `k` is reported as `view_scatter`.
  Derivation: `docs/notes/uncertainty.md`.
- A view is a residual's per-view parameter block (`pose/`, `target/`,
  `rig_pose/`, `robot_delta/<view>`). From six views up.
- Measured on the six public real cameras (stereo left and right, `kuka_1`,
  three `ds8`; 21 split halves each): split-half `r` (mean `z²/2`, 1 when
  honest) 0.66–1.53, geometric mean 1.03. One camera's `r` spans 0.5–1.5
  (5–95 %) even with an exact σ, so the gate is the pooled value. Synthetic
  truth: `r` 0.88 with independent noise and 0.93 with per-view errors;
  the Monte-Carlo gates hold within 10 %.
- Rejected: the view-clustered sandwich `V/(V−1) · H⁻¹ (Σ_v g_v g_vᵀ) H⁻¹`
  (the jackknife without leverage terms). Biased low where a few views
  carry most of the information: `r` 1.4–3.4 on the public cameras. The
  CR2 form (leverage at half power, unbiased under the noise model): 1.05–2.4.
  The raw jackknife: honest on real data, but 15–35 % high on data that fit
  the noise model, which fails the Monte-Carlo gates. Scaling by every
  `k > 1`: `k` has a standard deviation of about 0.6 with eight views, so
  the clamp inflates clean covariances by 10 %. Scaling per direction
  (eigenvalues of `Σ⁻¹ Σ_J` floored at 1): no closer on real data and more
  conservative on synthetic data. The jackknife in place of the model
  covariance: rank-deficient and noisy, so it supplies a scale but not a
  shape. A fixed documented inflation: the factor ranges from 3 to 25
  between datasets.

**Robot-pose priors are weighed against the measured pixel noise.**
- Every hand-eye view gets a free pose correction `δᵢ` in the gripper frame,
  `T_B_G · exp(δᵢ)`, with a zero-mean prior at the configured `rot_sigma`,
  `trans_sigma`: the tool's orientation and position error. A base-frame
  correction (`exp(δᵢ) · T_B_G`) was rejected: its rotation turns the
  gripper about the base origin, so the translation prior also bounded the
  rotation's lever arm and `trans_sigma` was not the tool's position error.
  On `kuka_1` the gripper frame halves the translation corrections and
  makes the hand-eye from disjoint halves of the views agree 39 % better in
  translation.
- The prior's residual is scaled by the reprojection σ̂, estimated by one
  re-solve. The objective is then the maximum-likelihood one, and the
  sandwich reduces to the inverse information.
- No correction is pinned; the priors alone separate the corrections from a
  common change of the transform the gripper carries (the hand-eye, or the
  target in eye-to-hand).
- By default the robot's σ are estimated, not taken as configured: each
  prior group's variance factor `ŝ² = Σ (δ̂ₖ/σ)² / Σ rₖ`, with the
  redundancy `rₖ = 1 − Var(δ̂ₖ)/σ²` of each prior row (Förstner's variance
  component estimate), rescales `σ` until it settles. The configured σ are
  starting values, or the robot's true accuracy with `estimate_sigma` off.
  A fixed prior was rejected as the default: on real arms the corrections
  are several times smaller than any safe default, so the hand-eye σ was
  pessimistic, and a tight prior made it too small.

**Observations may state a per-point 2×2 covariance, and the solve whitens
by it.**
- A stated covariance replaces the scalar per-point weight, which had no
  source and no meaning for the noise estimate.
- The residual is `L (u_obs − u_proj)` with `LᵀL = Σ⁻¹`.
- The family's noise estimate becomes the variance factor σ̂₀, and the export
  types which one it carries (σ̂ in pixels, or σ̂₀).
- Covariances are all or none per input. Whitened and pixel residuals cannot
  share one family, so an input that mixes views with and without them is a
  typed error before any work.
- There is no weighting switch: the solve uses what the observations carry.
- Reprojection statistics stay in raw pixels; a robust loss acts on the
  whitened residual.
- The corner detector's covariance is stated only on request, and weighting
  a calibration by it is advised against. On real chessboard and ChArUco
  images it did not improve held-out accuracy, and on some cameras it made
  it worse (`docs/notes/feature-covariance-study.md`, decided by its
  pre-registered rule). Per-point covariances remain the way to state
  noise that is known.

**What a result carries.**
- The full labelled covariance of the global parameters, plus 6×6 marginals
  per view. Cross-covariance between views and globals is dropped, so the
  size stays linear in the number of views.
- The covariance is **conditional on everything held fixed**, and the result
  lists the fixed labels.
- Projection uncertainty is derived on demand from the covariance, not
  stored.

**Availability is a type.**
- The covariance is either available, or unavailable with a reason: not
  converged; rank-deficient, naming the null-space labels; at a bound; or no
  redundancy.
- The calibration succeeds either way.
- It is always computed; there is no knob.

**A quality verdict reads the result alone.**
- `Pass | Warn | Fail` over typed checks (observability, convergence,
  variance factor, projection σ, feature coverage, view outliers, parameter
  correlation). Each check records its value, unit and the threshold that
  decided it.
- `Fail` means the data does not support the result. `Warn` names a weakness
  that leaves the result usable.
- A check without its inputs is *not applicable*, never a pass.
- Ambiguous inputs (for example image sizes that disagree) are errors.

**Next-view planning predicts, without a re-solve, the covariance a planar
calibration would have with candidate views added.**
- Each candidate's information is Schur-marginalized over its own pose and
  added to `Σ⁻¹`, at the noise scale the covariance itself implies.
- Candidates are ranked by the largest projection σ, the figure the quality
  check bounds.
- It is first order: it refines a calibration that already determines the
  camera roughly.

## Consequences

- Every export carries `uncertainty` (ADR 0012). Python exposes standard
  deviations, correlations and the projection map; the app shows `value ± σ`.
- Staged problems refine the intrinsics in their final stage by default, so
  rig covariances include them. Holding them fixed gives conditional, hence
  optimistic, extrinsic σ, and the fixed list says so.
- Defaults never fix an observable block for speed; that would make the
  default covariance conditional on an arbitrary choice.
- The regression suite gates parameter changes in units of σ, and
  `calib-bench accept` fails an entry whose verdict is `Fail`.
- Open:
  - propagating earlier stages' covariance;
  - robot-pose noise as a measured quantity;
  - target flatness;
  - cross-frame projection uncertainty;
  - next-view planning for rigs.

## Alternatives rejected

- **The LM's loss-corrected linearization.** It differs between backends.
- **IRLS covariance with variance components.** It overstates σ about
  twofold under Huber with outliers.
- **A per-residual sandwich.** Calibrated on average, but too noisy per run.
- **Bootstrap or Monte-Carlo at runtime.** It costs tens of solves per
  result; it is the validation method instead.
- **Pixels weighed at 1 px against robot-pose priors.** The priors become too
  strong and robot error leaks into the camera. Exact robot poses are worse:
  σ too small by about `√(corners per view)`.
- **Rescaling the reprojection family by `1/σ̂` instead of scaling the
  priors.** Same estimate, but residuals, σ̂ and a robust loss's scale would
  leave pixel units.
- **A `Uniform | DetectorCovariance` weighting switch.** It is a second way
  to drop covariances the caller chose to state.
- **Treating a parameter at its bound as fixed.** That silently conditions
  the result.
