# Stopping rules of the shared Levenberg–Marquardt loop

Family: the LM loop every backend runs (`vision-calibration-optim`,
`backend/lm.rs`) and the scale its stopping rules measure against. Each
claim is marked *measured* (with the test or command that measured it) or
*hypothesis* (with what would decide it).

The loop stops on the first of: the Gauss–Newton step left to the minimum
within 0.01 of the solution's standard deviation (`SigmaStep`, the
section "Stopping on the distance to the minimum" below); a step that
moved the solution by under 0.001σ short of that (`StalledStep`); the cost
below
`min_error` (1e-10); a first-attempt step `‖dx‖ ≤ ε(‖x‖ + ε)`,
`ε = 1e-8` (Ceres' `parameter_tolerance`, `RelativeStep`); no step
accepted after 32 damping retries; the iteration budget. The decrease
rules `min_abs_decrease` and `min_rel_decrease` (an accepted step lowering
`Σρ` by less than a threshold, or a fraction of it) are off by default and
kept for tests that set them; until the change below they were the
default, both at 1e-5.

## The relative-step scale (fixed)

**Defect.** `‖x‖` summed every parameter block of the problem, the held
ones too. In a solve of a few small free parameters under a held block of
large values the held block set the scale: the rig laserline workflow's
plane solve (`optimize_rig_laserline`, one `optimize_laserline` per camera
with the camera and every target pose held) has `‖x‖ ≈ 1441`, dominated by
the held intrinsics in pixels, so any step below `1.44e-5` (radians of the
normal's tangent, metres of the distance) counted as no step. Against the
plane's σ (normal tangent ≈ 1.1e-5, distance ≈ 1.7e-6 m on the synthetic
scenes below) that is about one σ of the normal and eight of the distance:
a start that close stopped where it began.

Evidence, *measured*:

- The seeded-run scene of a two-camera rig laserline device
  (`dataset_runner/seed/tests.rs`): the unseeded solve of both cameras
  stopped on `RelativeStep` before its first step, returning the linear fit
  unchanged; from the true planes the solve of camera 0 also took no step.
  The sibling seed (planes 1° and 2 mm off) took two accepted steps and
  stopped when the third was below the scale: `plane_z` 1.81 against the
  unseeded run, on both backends.
- On one camera of that kind (`noisy_one_camera_scene` in
  `problems/laserline_rig_bundle.rs`, 1670 laser residuals, 0.1 px noise),
  started at the true plane: the plane Jacobian's singular values are
  7.66e3, 1.57e4, 1.25e5 (1.25e5/7.66e3 = 16; with the loop's Jacobi column
  scaling 0.42, 0.99, 1.35), so the problem is well conditioned and no flat
  valley is involved. The gradient norm is 1.2e4 there against 0.54 at the
  minimum; the Gauss-Newton step is `‖dx‖ = 1.31e-5` (normal tangent
  `(−1.9e-6, 1.29e-5)`, distance `−9.7e-7`), below the old threshold, and
  it lowers `Σr²` from 16.2016 to 16.1791, the minimum. One LM iteration
  reaches it. From a plane 1° and 2 mm off the cost is 61212 and the solve
  converges to the same point.
- The point-to-plane residual stalled the same way (a 2 mm shift ended
  1.08σ off): the scale, not the residual, was at fault.

**Fix.** `‖x‖` counts only the blocks the solve can move: used by a
residual and with a free coordinate (`LinearizationEngine::norm`, both
backends), as Ceres leaves constant blocks out of the program it
minimizes. The plane solve's scale is then `‖(n, d)‖ ≈ 1.1`, the threshold
1.1e-8.

After the fix, *measured*:

- `a_plane_under_a_held_camera_converges_from_a_near_start`
  (`problems/laserline_rig_bundle.rs`): from the true plane, from a 2 mm
  shift and from a 1° turn plus 2 mm, for both laser residuals, on both
  backends, the solves end within 0.01σ of one another, and the one from
  the truth moves by more than 0.1σ. Before the fix it failed: the solve
  from the true plane took no step.
- The seeded-run cell `RigLaserlineDevice sibling` is gated now:
  `plane_z` 6.4e-5 on both backends (gate 0.05), the laser cost identical.
- Monte Carlo over 60 noise draws of the rig laserline scene (two cameras,
  120 plane solves): before, 79 of 120 solves took no step from the
  linear fit; after, none. Against the true planes the RMS normal error
  falls from 1.635e-5 to 1.310e-5 (camera 0) and from 1.966e-5 to
  1.926e-5 (camera 1), the RMS distance error from 1.686e-6 to 1.379e-6 m
  and from 3.685e-6 to 3.659e-6 m; the median laser-cost decrease from
  the old result is 0.5 %, the largest 33 %.
- A second Monte Carlo, written without the first one's code (two cameras
  of different focal length, 10 views, 200 draws, both backends agreeing
  to every printed digit): before, 103 and 185 of 200 solves per camera
  took no step; after, none of 400. The RMS normal error falls by 3 % and
  17 %, the RMS distance error by 11 % and 19.5 %. After the fix the RMS
  of error/σ over the four plane coordinates is 0.96 to 1.09 (and 0.93 to
  1.15 at 0.03 px, 0.3 px and 25 views); before, at 25 views, the stated
  σ was 3.5 to 4.5 times the RMS error. Over the regression suite's 204
  solves the iteration total is 748 before and after, and none stops at
  the iteration limit.
- The covariance moves with it: at a point that is not the minimum the
  per-view scores do not average to zero, so the view-scatter check
  inflated the stated σ (camera 1's normal σ 4.8e-5 at the linear fit,
  7.5e-6 at the minimum, against camera 0's 1.1e-5 on a like geometry).
- Unchanged: the single-camera laserline device and the joint rig hand-eye
  laserline workflow (their seeded-run cells and suite entries are
  identical to before; the joint workflow's warm start uses the
  point-to-plane residual by default, whose minimum the linear plane fit
  already is, so those plane solves took no step before and take none
  now), and every
  calibration parameter of the regression suite (`suite check`:
  tiny-solver's largest change per entry is what it was before, at most
  4.7e-7σ; factrs' largest change against the references is the same as
  before, entry by entry). The suite's board measurement moved: its
  triangulation holds the calibrated cameras, so its points now converge
  past the old scale (`opencv_stereo` up to 4.7e-6σ, `opencv_fisheye_stereo`
  up to 1.65e-3σ in a length's mean error, and that entry's RMS length
  errors all fall), on both backends. Those two references were re-blessed
  for that reason. The calibration export in the `opencv_fisheye_stereo`
  reference also changed, by at most 3.8e-9 relative: that is an older
  drift of the stored reference from what the code computes, which the
  re-bless took in; the code's calibration exports are byte-identical
  before and after on every suite entry and both backends.

Open (*hypothesis*): the rule still measures a step against every free
block, so a small free block (a plane) beside a large free one (free
intrinsics, in pixels) could stop early. The seeded single-camera and
joint cells show no sign of it; a joint laserline solve with free
intrinsics and the plane started about 1σ off would decide it. Also seen
on one scene and not explained: the point-to-plane residual's stated σ is
30 to 40 % too large (RMS of error/σ 0.70 to 0.76), before and after.

## Stopping on the distance to the minimum (fixed)

**Defect.** The decrease rules measured the last step's gain, not the
distance left, and on no scale of the solution's uncertainty. With `m`
residuals, `n` free parameters and `σ̂² = Σρ/(m − n)`, a step that lowers
`Σρ` by `ΔC` moves the solution by about `√(ΔC/σ̂²)` standard deviations
(Mahalanobis), so `min_rel_decrease = 1e-5` lets the last step be
`√(1e-5 (m − n))` σ long: 0.10σ at 1000 residuals, 0.30σ at 9000, 0.53σ
at 28 000. Two things made the distance left larger than the last step:

- **Damping.** The damping starts at `1e-4` of the Jacobi-scaled diagonal
  and falls at most threefold per good step (Ceres' schedule). A solve
  started near its minimum (a restart, or a warm-started pass of an outer
  iteration) takes damped steps that cover a third to a half of the
  undamped one, with a gain ratio of 1, and the rule fires on the first.
- **A robust loss.** Where `ρ″ < 0` (Huber beyond its scale, Cauchy
  everywhere) both engines use the reweighted curvature `ρ′JᵀJ` (Ceres'
  clamped correction in tiny-solver, `√ρ′` scaling in factrs), larger than
  the objective's. The step then covers only `a/b` of the way (objective
  curvature `a`, model `b`), the gain ratio is `2 − a/b`, and the
  convergence is linear with ratio `1 − a/b`.

Evidence, *measured* (a scratch probe in the shared loop that ran every
solve on past its stop to the minimum, `d = √((Σρ − Σρ*)/σ̂²)`; and the
plane scene of the Python test `test_planes_fit_against_an_upstream_rig`,
rebuilt in a bench example):

- The rig laser plane solve under its held camera, the scene the backlog
  entry reported: per iteration the damping is `1e-4 … 4e-6`, negligible
  next to the scaled Hessian's eigenvalues, the gain ratio 1.30, 1.34,
  1.43, 1.52, 1.55, 1.57 (laser residuals of RMS about 1.3 px under Huber
  at 1 px), so `a/b ≈ 0.43` and the restarts' ratio 0.57 is `ρ − 1`.
  Restarted from its own result the distance moved 1.56e-6, 0.89e-6,
  0.50e-6, 0.29e-6, 0.16e-6 m (the reported sequence), one iteration each.
- Over the regression suite's 102 solves per backend (204 in all), 39 of
  tiny-solver's and 33 of factrs' stopped more than 0.01σ from their
  minimum, 9 more than 0.1σ; the farthest 0.845σ (`kuka_1`, a pass of the
  robot-prior variance iteration that stopped after one iteration). In the
  six farthest the stopping step was damped to 0.33–0.72 of the undamped
  one; in the rest the step was undamped within 10 % and the scale alone
  let it be long.
- `kuka_1`'s variance-component iteration (`robot_poses::solve_weighted`)
  re-estimates the prior weights from each solve and stops when they
  change by less than 1 %. Its passes changed the weights by 0.89, 0.56,
  0.27, 0.13, 0.069, then 0.0024: the last pass's inner solve took one
  damped step and stopped, so the weights it fed back barely moved, a
  false fixed point. With the solves converged the sequence is 0.89, 0.56,
  0.27, 0.13, 0.072, 0.044, 0.028, 0.019, 0.012, 0.0080: geometric, ratio
  about 0.65, stopping four passes later.

**Measure.** At `x` with gradient `g` and Gauss–Newton matrix `H` of `Σρ`,
the undamped step to the minimum of the local model lowers it by
`gᵀH⁻¹g`, and the solution's covariance is about `σ̂²H⁻¹`, so
`d_N² = gᵀH⁻¹g/σ̂²` (the Newton decrement in units of `σ̂²`) is the
squared Mahalanobis length of the step still to make, a bound on every
parameter's distance in units of its σ. On the suite's iterates (426 with
a true distance between 0.001 and 30σ) `d_N / d` has median 0.999, 5th
percentile 0.95, range 0.86–1.015, on both backends. Under the reweighted
curvature it understates: 0.65–0.77 on the plane scene, `√(a/b)`.

**Rule.** On the first iteration, and whenever the last accepted step
lowered `Σρ` by at most `σ̂²` (moved the solution at most one σ), the loop
also solves the normal equations at damping `1e-12` (the covariance's
rank tolerance). When `d_N ≤ 0.01` it takes that step if it lowers the
cost and stops (`SigmaStep`) when what the step leaves,
`|ρ − 1|·d_N/√(2 − ρ)` from its gain ratio `ρ`, is within 0.01 too;
otherwise it iterates on from there. A step that does not lower the cost
at that distance is the minimum to working precision, and the solution
stays. Without redundancy (`m ≤ n`) or at zero cost the rule is off and
the other rules end the solve. The measure is the step still to make, so
damping that shortens the loop's own steps cannot end a solve early. A
coordinate on a bound that the step would push through is held while the
step is solved (an active set; both engines report the bounded coordinates
on their bound, `LinearizationEngine::on_bounds`), and with one held the
step is taken whenever it lowers the cost: the loop's own steps push such
a coordinate into the bound and the clamp leaves the rest to crawl.
Without that, the covariance test `coordinate_on_a_bound_is_reported` (fx
held on a lower bound) ran to its 50-iteration budget at `d ≈ 6.7`, the
loop gaining about 0.03 % of the cost per step (*measured*); with it, it
stops on `SigmaStep` and reports the coordinate at its bound. No suite
entry has a coordinate on a bound: the suite's results are the same with
and without this part.

A stall rule backs it: an accepted step that lowered `Σρ` by at most
`(0.001)²σ̂²` (moved the solution under a tenth of the tolerance) while
`d_N > 0.01` ends the solve (`StalledStep`); converging linearly with ratio
`q` the solve is then within `0.1q/(1 − q)` of the tolerance. It fires on
none of the suite's solves (their results are bit-identical with and
without it, *measured*) and exists for long curved valleys, where the
local model keeps placing the minimum out of reach.

**Degenerate valleys** (*measured*): where a parameter is undetermined the
new rule no longer stops wherever the last step's gain ran out. A
Scheimpflug camera without lens distortion (the tilt trades with the
principal point; `quality_tilt_conditioning`) crawls 0.03–0.1σ per
iteration along a curved valley: at the budget of 120 it ends on
`max_iters` with no covariance (before: a covariance with the tilt's σ
inflated sevenfold, wherever the decrease rule had fired), and with
2000 iterations it stalls after 245 with the principal point on its
bound. The degree-4 residual field of the model study trades with `k1`
the same way. The result now says it has not converged instead of
reporting a covariance for an arbitrary point; the quality report's
conditioning check is not applicable there. The tilt policy that holds an
undetermined tilt (ADR 0031) is the remedy for that case.

Alternatives measured on the same 204 solves (largest `d` at the stop,
total iterations, tiny-solver; factrs agrees within a few iterations):

| Rule | Largest `d` | Solves > 0.01σ | Iterations |
|---|---|---|---|
| decrease rules at 1e-5 (the old default) | 0.845 | 42 | 370 |
| relative decrease 1e-6 (Ceres' `function_tolerance` default) | 0.040 | 13 | 443 |
| relative decrease 1e-8 | 0.0023 | 0 | 528 |
| last step's decrease ≤ (0.01)² σ̂² | 0.0023 | 0 | 514 |
| geometric tail of the decreases ≤ 0.01σ | 0.019 | 3 | 465 |
| `d_N ≤ 0.01` (chosen, replayed before the final step) | 0.0099 | 0 | 514 |

(Replayed on the recorded trajectories; the probe's count of 42 differs
from the 39 of the runs below by solves within rounding of 0.01σ.)
Tightening the relative decrease to 1e-8 or scaling the decrease by `σ̂²`
end as close on the suite, but they measure the step taken: a step damped
to a fraction of the undamped one, or a reweighted one, reads as nearer
the minimum than it is, and only the scale of the suite's problems keeps
them inside 0.01σ. On the plane scene the decrease rule at `(0.01)²σ̂²`
ends at 0.0046σ and the Newton test without the gain-ratio correction at
0.014σ (`√(a/b)` short); with it, inside 0.01σ.

What established solvers do: Ceres stops on `function_tolerance` (1e-6:
`|Δcost|/cost` of the iteration), `gradient_tolerance` (1e-10:
`‖x − Π⊞(x, −g)‖∞`) and `parameter_tolerance` (1e-8: `‖Δx‖ ≤
(‖x‖ + tol)·tol`), with the same initial trust region (1e4, so damping
1e-4) and the same radius update, `radius /= max(1/3, 1 − (2ρ − 1)³)`
(Ceres documentation, "Solving Non-linear Least Squares", and
`levenberg_marquardt_strategy.cc`). MINPACK's `lmder` stops when both the
actual and the predicted relative reductions of the sum of squares are at
most `ftol`, when the trust region is at most `xtol·‖Dx‖`, or when the
scaled gradient cosine is at most `gtol` (`lmder1` sets `ftol = xtol = tol`,
`gtol = 0`). None of them measures the distance in the estimate's own σ;
each is a relative tolerance whose meaning in σ changes with `m`.

After the change, *measured* (`suite check` with a scratch switch back to
the old rules; per solve the distance `d` at the stop and after three
restarts from it, against a run to `d_N ≤ 1e-7`):

| Entry | Solves | Iterations old → new | Largest `d` old → new | Restart drift (3) old → new |
|---|---|---|---|---|
| kuka_1 | 8 → 12 | 41 → 99 | 0.845 → 4.7e-5 | 0.097 → 4.7e-5 |
| stereo_left | 1 | 6 → 8 | 0.023 → 6.1e-4 | 0.0095 → 6.1e-4 |
| stereo_right | 1 | 6 → 8 | 0.014 → 1.7e-4 | 0.0070 → 1.7e-4 |
| stereo_rig | 3 | 18 → 23 | 0.023 → 0.0015 | 0.0095 → 0.0015 |
| ds8 | 11 | 50 → 83 | 0.610 → 4.7e-4 | 0.064 → 4.7e-4 |
| opencv_stereo | 12 | 46 → 50 | 0.0074 → 2.9e-4 | 0.0048 → 2.9e-4 |
| opencv_fisheye_left | 1 | 4 → 6 | 0.022 → 3.0e-5 | 0.0084 → 3.0e-5 |
| opencv_fisheye_right | 1 | 5 → 6 | 0.0035 → 3.5e-5 | 0.0016 → 3.5e-5 |
| opencv_fisheye_stereo | 23 | 62 → 73 | 0.022 → 1.2e-4 | 0.0084 → 1.2e-4 |
| synth_planar | 1 | 4 → 5 | 0.0058 → 5.4e-5 | 0.0041 → 5.4e-5 |
| synth_scheimpflug | 7 | 29 → 34 | 0.016 → 1.2e-4 | 0.0051 → 1.2e-4 |
| synth_handeye | 8 | 23 → 35 | 0.036 → 7.5e-5 | 0.026 → 7.5e-5 |
| synth_rig | 3 | 10 → 16 | 0.109 → 1.0e-4 | 0.070 → 1.0e-4 |
| synth_laserline | 1 | 5 → 7 | 0.023 → 1.5e-4 | 0.0064 → 1.5e-4 |
| synth_rig_handeye_laserline | 21 | 61 → 88 | 0.097 → 1.8e-4 | 0.048 → 1.8e-4 |
| all, tiny-solver | 102 → 106 | 370 → 541 | 0.845 → 0.0015 | |

(tiny-solver; factrs: 378 → 543 iterations, largest `d` 0.845 → 0.0015.)
Every solve ends on `SigmaStep`; none reaches its budget (the most
iterations of one solve 10, factrs 11, against a budget of 50 or 100).
The "restart drift" column is how much three restarts lowered `d`.

- The plane scene: the solves take 9 and 8 iterations (4 and 5 before);
  restarted, the distance moves 9.2e-8, 5.3e-8, 3.0e-8 m (0.0005σ, before
  0.0079σ for the first restart), each restart one iteration.
- Results against the stored references, in the references' σ (both
  backends alike to the printed digits): `kuka_1` 1.03σ
  (`base_se3_target.rx`, through the variance iteration above), `ds8`
  0.093σ, `synth_rig` 0.094σ, `synth_rig_handeye_laserline` 0.045σ,
  `synth_laserline` 0.022σ, `synth_scheimpflug` 0.012σ, the rest below
  0.01σ. The references were re-blessed. factrs now ends within 2.3e-4σ
  of tiny-solver's references (`synth_scheimpflug`; 3.7e-5σ elsewhere);
  before, up to 0.0085σ (`synth_scheimpflug`, `stereo_left`): the two
  linearizations reach one minimum, and the old stop left each short of
  it by a different amount.
- An outside check (`suite compare`, OpenCV run to its own convergence on
  the entries where the models are the same): the largest difference falls
  from 0.09σ to 0.00σ on `synth_rig` and from 0.01σ to 0.00σ on
  `stereo_left`, `synth_scheimpflug` and `opencv_fisheye_left`; on
  `kuka_1` (a different model, robot-pose priors) from 7.32σ to 6.25σ.
- The weak-distortion valley of `tilt-identifiability.md` (its bench
  example, `--parts mc,solver`; "tight" there now also turns the σ rule
  off): from the coarse seed the default ends at most 0.0020σ from the
  tight solve (median 0.0002σ; before, 0.66σ), 23 iterations against 20
  before and 28 tight, 16 of 16 starts reached (before 15). The
  Monte-Carlo ratio scatter / stated σ in that cell is 1.03–1.15 with the
  default, as with the tight solve (before 0.69–0.82), and 0.88–1.01 with
  `p1, p2` free (before 0.76–0.89); every other cell matches the tight
  solve to the printed digits, both backends. This closes the backlog's
  stopping-rule-in-σ entry, which proposed this rule from that study.
- Tests that run a solve "to precision" through `min_rel_decrease` now
  also set `sigma_tolerance: 0.0`, so they keep running to that precision.
- Two tests passed only because a solve stopped early, and were rebuilt:
  `synthetic_planar_intrinsics_with_outliers_robust_better_than_l2` had
  three views turned about one axis, a flat valley the robust solve left
  after 14 iterations (on `MinRelDecrease`, 241 px of intrinsics error)
  and now crawls along to its budget (4244 px); with six views turned
  about both axes it converges. `public_api_converges_with_deterministic_noise`
  (Scheimpflug, no lens distortion) asserted the tilt within 0.03 rad of
  the truth, where the tilt is undetermined without distortion (stated σ
  0.25–0.35 rad): the old solve stopped after one iteration
  (`MinAbsDecrease`, the absolute rule on a cost of 3) at its seed; the
  new one walks the valley to the principal point's bound. With
  `k1 = −0.1` the tilt is determined (σ 0.008–0.013 rad) and both rules
  end at the same tilt to 4e-5 rad.
- Cost: CPU time of `suite check` 45.9 → 71.0 s (tiny-solver), 43.4 →
  74.2 s (factrs); `kuka_1` the bulk (four more solves of the variance
  iteration, each converged).
- `restart_stability` (`vision-calibration-optim/tests`): a planar
  calibration of 40 views under Huber at half the pixel noise, both
  backends. Under the old rules a restart moved a camera parameter
  0.0073σ, five restarts 0.015σ, and a solve started at the truth ended
  0.039σ away; under the new rule 6.3e-4σ, 0.0019σ and 0.0027σ.

Open (*hypothesis*): `σ̂²` pools every residual. In a joint solve whose
families have different units and noise (a point-to-plane laser residual
in metres beside reprojection in pixels), a parameter seen by the small
family alone has its distance understated by the ratio of the noises, as
it was under the old relative rule. Measuring `d_N` with each family
whitened by its own `σ̂_f` (the covariance's pooling) would remove it; a
joint laserline solve with the point-to-plane residual, its plane started
a σ off, would show whether it matters. `kuka_1`'s variance iteration
converges linearly (ratio 0.65) against a 1 % tolerance on the weights,
so it still stops about 1.5 % of a weight from its fixed point; a stop on
the extrapolated remainder would decide whether that moves results.
