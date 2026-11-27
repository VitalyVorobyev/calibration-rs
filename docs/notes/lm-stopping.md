# Stopping rules of the shared Levenberg–Marquardt loop

Family: the LM loop every backend runs (`vision-calibration-optim`,
`backend/lm.rs`) and the scale its stopping rules measure against. Each
claim is marked *measured* (with the test or command that measured it) or
*hypothesis* (with what would decide it).

The loop stops on the first of: the cost below `min_error` (1e-10); an
accepted step that lowers the robust cost `Σρ` by less than
`min_abs_decrease` (1e-5) or by less than `min_rel_decrease` (1e-5) of it;
a first-attempt step `‖dx‖ ≤ ε(‖x‖ + ε)`, `ε = 1e-8` (Ceres'
`parameter_tolerance`, `RelativeStep`); no step accepted after 32 damping
retries; the iteration budget.

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
