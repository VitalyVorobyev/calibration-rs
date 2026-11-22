# ADR 0032: Camera Models Are Compared on Held-Out Views

## Context

Choosing a camera model (a distortion kind and which of its coefficients
are free) was guidance in the book, backed by training reprojection errors
on two private rigs. A training error cannot choose: a richer model nests
the simpler one, so it fits the views it saw at least as well whether its
extra terms describe the lens or the noise. Users need a number that
falls only when a term predicts something, with an error bar, from the
library rather than from a study harness (backlog CAM-SELECT).

Three measures were on the table: held-out reprojection error, an
information criterion (AIC/BIC from the training cost and the parameter
count), and the structure of the residuals. Held-out error needs no model
of the noise and no count of effective parameters; it costs refits.

## Decision

- **One function per problem type, beside the problem, not a
  `ProblemType` method.** Planar intrinsics first:
  `analysis::compare_camera_models(input, config, alternatives, opts,
  control) -> ModelComparison`. Rig and hand-eye comparisons will be
  sibling functions with their own fold unit (a rig view, a robot pose);
  a trait method would force one fold and scoring shape on problems that
  need different ones.
- **A candidate is a `CameraConfig`, not a whole config.** `config` is the
  estimator (initialization, refinement, solver, robust loss) and its
  `camera` is candidate 0; alternatives replace only the camera. Comparing
  different estimators at once would confound the model with the solver.
- **K-fold over views.** A SplitMix64 Fisher–Yates shuffle from
  `CrossValidationOptions::seed` (value-stable, the library's own sequence,
  not a dependency's), view `i` of the shuffle to fold `i mod K`, so fold
  sizes differ by at most one; K ≥ 2, default 5, and every fit keeps
  Zhang's three views. `fix_poses` is rejected: it indexes views of the
  whole input.
- **Fold fits are the ordinary workflow.** A fresh session runs
  `run_calibration` on the training views with the candidate substituted:
  what is scored is what a user would get.
- **Held-out scoring uses the fold's camera only.** Each held-out view's
  pose is seeded from the camera's bearings and solved alone with every
  camera parameter fixed (the config's loss and backend), so no other view
  and no full-data pose enters it. The score is the Euclidean pixel error
  of every point, nothing trimmed; the robust loss shapes only the pose.
  Each view is scored once, by the fold that held it out.
- **The result is scores, not cameras.** Per candidate: training and
  pooled held-out statistics, per-fold and per-view RMS, the fold scatter,
  the difference from the best with its fold-paired standard error, and
  the free parameter count resolved from the fix lists (not the model's
  slot count). Ranking is by pooled held-out RMS; a candidate with any
  unprojected held-out point is listed, not ranked. The user calibrates the
  chosen configuration as usual. The guidance is to prefer the fewest free
  parameters within about two standard errors of the best.
- **A failing candidate does not fail the comparison.** A fit or held-out
  solve error makes that candidate `Failed { fold, reason }`; invalid
  inputs and candidates are typed errors before any fit or progress
  report; cancellation is `Cancelled`.

## Consequences

- The book's model-selection guidance is a runnable comparison instead of
  numbers from one rig.
- Cost is `(K + 1)` planar fits per candidate plus one small pose solve
  per view and fold.
- **The planar blind spot.** The held-out pose is re-solved per view, so a
  camera term that a rigid pose of a plane can partly imitate (a
  principal-point shift against a rotation, a focal length against
  distance) is partly forgiven on held-out views. The comparison is
  sharpest for distortion terms. Rig and hand-eye siblings, where the pose
  is constrained by other cameras or by the robot, see more of the camera.

## Alternatives rejected

- **An information criterion now.** AIC/BIC need the noise level and the
  effective parameter count, both uncertain with a robust loss, fixed
  parameters and correlated corners; it is deferred until its ranking has
  been compared with the held-out one on the regression suite.
- **Scoring held-out views with poses from the full-data fit.** Those poses
  were fitted with the held-out view's own points: leakage.
- **Returning the fold or full-data cameras.** They are not the result of
  any workflow configuration a user would deploy, and would invite using a
  fold's camera.
- **Leave-one-view-out by default.** `V + 1` fits per candidate; available
  by setting `folds` to the number of views.
