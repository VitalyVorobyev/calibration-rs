# Planning the Next View

A planar calibration's covariance says where its camera is poorly
determined; the projection map shows where in the image that matters. The
next question during capture is which view to take next. `predict_with_views`
answers it without a re-solve: it predicts the covariance the calibration
would have with candidate views of its target added, and `rank_candidates`
orders candidate target poses by how much each would reduce the largest
projection σ.

```rust
use vision_calibration::analysis::{CandidateGrid, candidate_poses, predict_with_views, rank_candidates};

let export = session.export()?;
let board = vision_calibration::synthetic::planar::grid_points(9, 7, 0.03);
let image = [1280, 960];

// 3 × 3 image regions, the board 0.55 m away, fronto-parallel and tilted
// by ±0.4 rad about each of its axes: 45 candidates.
let grid = CandidateGrid::new(0.55, 0.4, [3, 3]);
let candidates = candidate_poses(&export.camera, &board, image, &grid)?;

let ranked = rank_candidates(&export, &candidates, &board, image)?;
let best = &ranked[0];
println!(
    "take candidate {}: max projection σ {:.2} px (−{:.2} px)",
    best.index, best.max_projection_sigma_px, best.projection_sigma_reduction_px
);

// The covariance with the best three taken together.
let picks: Vec<_> = ranked[..3].iter().map(|r| candidates[r.index]).collect();
let predicted = predict_with_views(&export, &picks, &board, image)?;
println!("σ(cx) {:.3} px", predicted.covariance.std_dev("cx").unwrap());
```

```python
board = [(0.03 * i, 0.03 * j, 0.0) for j in range(7) for i in range(9)]
image = (1280, 960)

candidates = vc.candidate_poses(result.camera, board, image, vc.CandidateGrid(0.55, 0.4, (3, 3)))
ranked = vc.rank_candidates(result, candidates, board, image)
best = ranked[0]
print(f"take candidate {best.index}: max σ {best.max_projection_sigma_px:.2f} px")

predicted = vc.predict_with_views(result, [candidates[r.index] for r in ranked[:3]], board, image)
print(f"σ(cx) {predicted.covariance.std_dev('cx'):.3f} px")
```

A candidate is a target pose, `cam_se3_target`; any pose works, so an
application can propose its own (a pose the operator can reach, a robot
station) instead of the grid. `target_points` are the target's points in its
own frame, as the calibration observed them.

## How the prediction works

Each candidate's target points are projected through the current estimate;
points behind the camera or outside the image are dropped. What the view
would teach the camera is its Fisher information `JᵀJ`, over the camera's
parameters and the view's own pose. A re-solve would estimate that pose too,
so it is marginalized out (a Schur complement), and the rest is added to the
calibration's information:

```text
Σ′ = (Σ⁻¹ + Σ_c S_c / σ²)⁻¹
```

`Σ` is the result's covariance of its estimated parameters, so the
prediction is conditional on the same fixed parameters and every existing
view's pose is already accounted for. `σ` is the pixel noise that covariance
implies, read off the result itself (`effective_noise_px`): the existing
views' information is rebuilt from the export's observed corners and poses,
and `σ²` is the scalar that relates it to `Σ`. A robust loss enters the
covariance as that one scalar, so the prediction is on the same scale as the
result, whatever loss the calibration used.

A candidate that sees fewer than four target points, or whose points do not
determine its pose, adds nothing: a calibration would reject the view.

## Reading the ranking

- **The score is the largest projection σ** over the image after adding the
  candidate alone, on the same 16 × 12 grid as the projection map and the
  quality check's limit. `projection_sigma_reduction_px` is how much it drops.
- **`log_det_reduction`** is `ln det Σ − ln det Σ′`, the shrinkage of the
  whole covariance's volume. Unlike a trace, it does not add pixels² of the
  focal length to squared distortion coefficients, and a change of units
  does not change it. A candidate can reduce it a lot while leaving the
  worst corner where it was.
- **Candidates are scored one at a time.** Two candidates that each fix the
  same corner are not worth twice one of them. To plan several views, call
  `predict_with_views` with the views already chosen plus each further
  candidate and compare `projection.max`, or capture, calibrate and rank
  again.
- **Ties keep the order of the candidate list**, and the result is
  deterministic: the same export and candidates give the same ranking.

## How far to trust it

On synthetic scenes (a 9 × 7 board, six views over the central part of a
1280 × 960 image, 0.1–0.3 px noise), the predicted σ of each parameter and
the predicted largest projection σ match a re-solve with the view actually
added to within 0.1–2.1 % on average, for Brown-Conrady, the division model
and a Scheimpflug camera, and for one or three views at once.

The prediction is first order around the current estimate. It holds while
the current uncertainty is small against the distortion's curvature; a
calibration whose corners are extrapolated to tens of pixels of σ is far
enough from the truth that its Jacobians are not the re-solve's, and the
prediction can then be off by tens of percent. Take a few spread views
first; plan from there.

## Planning the next robot pose

A rig hand-eye calibration estimates more than its cameras: the rig, the
hand-eye and the target pose. Its next view is a robot pose, and the target
pose follows from the calibration's own chain. `predict_with_robot_poses`
predicts the covariance with views at candidate robot poses added, and
`rank_robot_poses` orders the candidates.

```rust
use vision_calibration::analysis::{predict_with_robot_poses, rank_robot_poses};

let export = session.export()?; // a rig hand-eye export
let board = vision_calibration::synthetic::planar::grid_points(9, 7, 0.03);
let image_sizes = [[1280, 960], [1280, 960]]; // one per camera

// Robot poses the arm can reach (`base_se3_gripper`), from the application.
let ranked = rank_robot_poses(&export, &reachable, &board, &image_sizes)?;
let best = &ranked[0];
println!(
    "move to pose {}: ln det Σ −{:.2}, max projection σ {:.2} px",
    best.index, best.log_det_reduction, best.max_projection_sigma_px
);

let picks: Vec<_> = ranked[..3].iter().map(|r| reachable[r.index]).collect();
let predicted = predict_with_robot_poses(&export, &picks, &board, &image_sizes)?;
println!(
    "σ(hand-eye tz) {:.3} mm",
    1e3 * predicted.covariance.std_dev("gripper_se3_rig.tz").unwrap()
);
```

```python
ranked = vc.rank_robot_poses(result, reachable, board, [(1280, 960), (1280, 960)])
best = ranked[0]
print(f"move to pose {best.index}: ln det Σ −{best.log_det_reduction:.2}")

picks = [reachable[r.index] for r in ranked[:3]]
predicted = vc.predict_with_robot_poses(result, picks, board, [(1280, 960), (1280, 960)])
print(f"σ(hand-eye tz) {1e3 * predicted.covariance.std_dev('gripper_se3_rig.tz'):.3f} mm")
```

A candidate is a `base_se3_gripper` pose. Each camera sees the target
through the result's chain at that pose: eye-in-hand, the rig on the gripper
and the target in the base; eye-to-hand, the rig in the base and the target
on the gripper. Points behind a camera or outside its image are dropped, and
a camera that keeps fewer than four adds nothing.

**What the view adds.** Its information is exact over every parameter the
result estimated: each camera's intrinsics and distortion, each camera's
`cam_se3_rig`, the hand-eye and the target pose, in the coordinates the
covariance reports them. A robot pose adds no unknown when the calibration
took the robot poses as exact. A calibration that corrected them
(`robot_poses.refine`, the default) corrects the candidate's too: the
correction is marginalized under the result's robot pose error, weighed
against its reprojection noise as the solve weighed it.

**The scale** is read off the result as for a planar calibration: the
existing views' information is rebuilt from the export's observed target
points at the robot poses it solved with, and `σ²` relates it to `Σ`. With
exact robot poses that relation is exact. With corrected ones it holds when
the corner noise is Gaussian: the solve weighs its robot pose priors at a
median-based estimate of that noise, which overstates a heavy-tailed noise's
σ.

**The ranking** is by `log_det_reduction`, the shrinkage of the whole
covariance's volume. A view that pins down the hand-eye counts as much as one
that pins down a lens, and the score does not depend on units. The largest
projection σ over every camera's image is reported alongside, for the lenses
alone.

**How far to trust it.** On synthetic two-camera rigs (a 9 × 7 board, six
base robot poses, 0.2 px Gaussian noise, both modes, with and without robot
pose corrections), the predicted σ of each parameter matches a re-solve with
the views actually added to within 0.1–1.2 % on average, for the best-,
median- and worst-ranked candidate and the best three at once; the largest
single-parameter difference is 9.6 %. The best-ranked candidate's real
reduction of `ln det Σ` beats the worst one's in every scene.

A rig hand-eye laserline calibration is not planned this way: its laser
residuals are not part of the prediction.
