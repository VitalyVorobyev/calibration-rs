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
