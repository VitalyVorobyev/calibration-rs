# calibration-rs Tutorials

Hands-on, copy-pasteable walkthroughs for the public surface of
`vision-calibration` (the facade crate). Each tutorial pairs a narrative with
a runnable example checked into the workspace; the example is the
authoritative reference and stays green in CI.

## When to use what

| Goal | Tutorial | Runnable example |
|---|---|---|
| Calibrate intrinsics + multi-camera rig from scratch | [Five-minute calibration](./five-minute-calibration.md) | [`stereo_charuco_session.rs`](../../crates/vision-calibration/examples/stereo_charuco_session.rs) |
| Seed prior knowledge into the pipeline (datasheet K, factory tilts, mechanical drawings) | [Manual initialization](./manual-init.md) | [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs) |
| Drill into per-corner reprojection errors (diagnose mode) | [Per-feature residuals](./per-feature-residuals.md) | [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs) (Run A printout) |
| Calibrate a Scheimpflug rig + laser device end-to-end on real data | [Puzzle 130×130 walkthrough](./puzzle-130x130-walkthrough.md) | [`puzzle_130x130_rig.rs`](../../crates/vision-calibration-examples-private/examples/puzzle_130x130_rig.rs) (private dataset) |
| Describe a laser dataset in `dataset.toml` and run it through the app | [Laser dataset manifest](./laser-dataset-manifest.md) | `rtv3d_laser_end_to_end` test in [`app/src-tauri/src/run.rs`](../../app/src-tauri/src/run.rs) (private dataset) |
| Recover relative pose, triangulate, bundle-adjust, and rectify a calibrated stereo pair | [Multiple-view geometry](./multiple-view-geometry.md) | [`mvg_two_view.rs`](../../crates/vision-calibration/examples/mvg_two_view.rs) |

## Target detectors

All four calibration-target detectors are wired end-to-end (as of 2026-06-14);
the target type is chosen by the `[target]` table's `kind` in `dataset.toml`
(or the Run workspace's schema-driven manifest form). Detection is cached and
dispatched server-side — no per-detector code path to learn.

| `kind` | Manifest fields | Notes |
|---|---|---|
| `chessboard` | `rows`, `cols` (interior corners), `square_size_m` | Plain checkerboard. |
| `charuco` | `rows`, `cols` (squares), `square_size_m`, `marker_size_m`, `dictionary` | Sparse — only decoded cells contribute corners. |
| `puzzleboard` | `layout` (`"puzzle_<R>x<C>"`), `cell_size_m` | Self-identifying; a single partial view is globally consistent. |
| `ringgrid` | `pitch_m`, `rows`, `long_row_cols`, `marker_outer_radius_m`, `marker_inner_radius_m`, `marker_ring_width_m` | Coded **hex-lattice** of ring markers — `long_row_cols` is the longest (even) row; shorter rows derive from the lattice. |

Optional ChESS corner-stage overrides (`[detector.chess_corners]`,
`threshold_mode` / `threshold_value`) apply to the chess-based detectors
(chessboard / charuco) and are hashed into the detection-cache key.

## Structure

Each tutorial is self-contained and follows the same outline:

1. **Why** — what problem the feature solves, who it is for.
2. **Mental model** — one or two paragraphs of necessary background. No
   geometry derivations — those live in the ADRs.
3. **Walkthrough** — a minimum-viable code path with explanations between
   each step.
4. **Common variations** — small recipes for the obvious follow-on questions.
5. **What to read next** — pointers into ADRs, source files, and other
   tutorials.

## How tutorials relate to ADRs

ADRs in [`docs/adrs/`](../adrs/) explain *why* the API is shaped a particular
way. Tutorials explain *how to use* the resulting API. When the two
diverge, the ADR is authoritative for design intent and the tutorial is
authoritative for the working code path; please file an issue when you spot
drift.
