# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §11); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the v1.0 criteria these serve.

## App

- [ ] B-UX2-ELEVATION - Workspace-by-workspace polish: empty states, error
  surfaces (fail-fast errors from ADR 0019 shown well), manifest-sniff UX,
  and three exploration views — a pre-calibration per-camera/pose image
  grid, a detection-cache overlay, and a board coverage map. Gate: a
  frontend review with no high-severity findings.
- [ ] B-QUAL-TS7 - Move to TypeScript 7. **Blocked**: `typescript-eslint`
  rejects TS 7 (typescript-eslint#10940), and adopting it would drop the
  type-aware lint config. Revisit when typescript-eslint supports TS 7.

## v1.0

- [ ] D4-RELEASE - Cut v1.0 once the ROADMAP exit criteria hold.
- [ ] D4-NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 /
  `faer-ext` 0.8. **Blocked on `tiny-solver`**, still built against
  0.34 / 0.23 / 0.7: `vision-calibration-optim` passes nalgebra and faer
  types across that boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`), so a bump fails
  to compile. `vision-calibration-detect` must stay nalgebra-free (it
  converts to plain arrays at its boundary); that is what lets
  `calib-targets`' nalgebra 0.35 coexist. Re-check on each tiny-solver
  release.

## Deferred and parked

- [ ] C-PYO3-MVG - Python bindings for the MVG surface. Deferred: the
  Python package binds the calibration facade only, and nothing in Python
  needs two-view/N-view geometry yet.
- [ ] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Schedule only
  if the acceptance run for the six-camera rtv3d rig exceeds ~5 min.
- [~] P3-BACKEND-COST - Profile tiny-solver's autodiff/assembly/solve split
  and evaluate analytic Jacobians, caching or rayon for the `ReprojPoint`
  factor. Parked until after 1.0.
- [~] M4-FISHEYE - Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`. Parked until after 1.0; no fisheye dataset exists in
  the acceptance set.
- [~] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by user
  decision — **do not reopen unprompted.**
