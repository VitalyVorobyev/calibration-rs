# Puzzleboard Full Mode Failure Handoff

Date: 2026-04-25

## Context

This handoff is for investigating a puzzleboard detection failure seen during the
private `puzzle_130x130_rig` end-to-end calibration run.

The dataset is available locally to the next agent through the same images:

- default path: `privatedata/130x130_puzzle`
- override: `PUZZLE_DATA_DIR=/path/to/130x130_puzzle`

The current calibration example deliberately avoids the failure by using
`PuzzleBoardSearchMode::FixedBoard` in:

`crates/vision-calibration-examples-private/src/lib.rs`

```rust
let mut params = PuzzleBoardParams::for_board(&spec);
params.decode.search_all_components = false;
params.decode.search_mode = PuzzleBoardSearchMode::FixedBoard;
```

## Failure Summary

`PuzzleBoardSearchMode::Full` can return a geometrically plausible detection
with the wrong master-board origin on this dataset. The corner localization looks
reasonable, but the decoded `target_position` values are assigned to the wrong
absolute puzzleboard coordinates.

This poisons calibration because the 2D image corners are good while the 3D
target coordinates are wrong. The downstream symptom is not a sparse-detection
failure; it is a coherent but inconsistent target frame.

Observed symptoms from the e2e investigation:

- Stage 1 puzzleboard features were visually and numerically detected.
- With Full-mode-style decoding, at least one run produced a camera target range
  far outside the physical 130 x 130 board. A concrete example was cam 2 with
  target `y` extending to about `0.229 m` after centering.
- The physical board is 130 x 130 cells with `cell_size_mm = 1.014`, so valid
  centered coordinates must lie within approximately:
  - `x in [-0.0654, +0.0654] m`
  - `y in [-0.0654, +0.0654] m`
- Once detections were forced through `FixedBoard` and board-range filtering, the
  target coordinate ranges became physically plausible and the rig reprojection
  dropped from the earlier ~20 px basin to the low-single-pixel range.

## Why Full Mode Is Suspect

The detector documentation says:

- `Full` scans all `(D4, master_row, master_col)` candidates against the full
  501 x 501 master code.
- `FixedBoard` matches observations against the declared board bit pattern.

For this e2e calibration, the printed target is known: a 130 x 130 board. The
calibration pipeline needs coordinates in that board's canonical frame, not an
arbitrary global master-board placement recovered from a noisy partial view.

The suspected failure mode is:

1. A tile contains only a partial view of the printed board.
2. Full search finds a different master origin that scores better under real
   image noise, partial visibility, or component selection.
3. The decoded corner IDs / positions remain internally consistent enough to
   pass the detector and homography/PnP steps.
4. The calibration receives wrong 3D target coordinates for otherwise good
   image points.

This is especially damaging in a multi-camera rig because each camera may decode
a different global origin for a different visible fragment of the same physical
board. Per-camera intrinsics can still look acceptable, but the rigid rig fit
cannot reconcile the inconsistent target coordinate frames.

## Reproduction Plan

Start from the current branch and use the private e2e example:

```bash
PUZZLE_DATA_DIR=privatedata/130x130_puzzle \
cargo run --manifest-path crates/vision-calibration-examples-private/Cargo.toml \
  --example puzzle_130x130_rig --release \
  2>&1 | tee /tmp/puzzle_fixedboard_baseline.log
```

This baseline should use `FixedBoard` and print stable detection diagnostics:

```text
detection diagnostics:
  cam N: target_views=... target_pts=... target_x=[...,...]m target_y=[...,...]m ...
```

Then temporarily switch the detector back to Full mode in
`crates/vision-calibration-examples-private/src/lib.rs`:

```rust
let mut params = PuzzleBoardParams::for_board(&spec);
params.decode.search_all_components = true; // or test false as a second case
params.decode.search_mode = PuzzleBoardSearchMode::Full;
```

Important: do not let the current board-range filter silently hide the problem.
For the Full-mode test, instrument the raw `corner.target_position` values before
this guard:

```rust
if !(0.0..=max_x_mm).contains(&x_mm) || !(0.0..=max_y_mm).contains(&y_mm) {
    eprintln!("out-of-board target_position: x_mm={x_mm:.3} y_mm={y_mm:.3}");
    continue;
}
```

Run again:

```bash
PUZZLE_DATA_DIR=privatedata/130x130_puzzle \
cargo run --manifest-path crates/vision-calibration-examples-private/Cargo.toml \
  --example puzzle_130x130_rig --release \
  2>&1 | tee /tmp/puzzle_full_mode_repro.log
```

Expected repro signals:

- Any raw `target_position` outside `[0, 129 * 1.014] mm` in either axis.
- Any centered target diagnostic outside approximately `+/-0.0654 m`.
- Large discontinuity in target ranges between cameras or poses.
- Rig reprojection returning to a much worse basin than the FixedBoard run.

If the range guard drops many corners, compare detection counts as well. A
Full-mode wrong-origin result may become either:

- a bad accepted view if the guard is disabled, or
- an unexpectedly sparse/rejected view if the guard is enabled.

Both outcomes are useful evidence.

## Minimal Detector-Level Test To Add

Prefer adding a focused regression in `calib-targets-puzzleboard` or a private
integration test that loads a small set of failing tiles from this dataset.

Suggested fixture selection:

- use the target image/pose/camera that reproduces the largest out-of-board
  `target_position` under Full mode;
- include at least one neighboring camera/pose that decodes correctly;
- keep the test deterministic and assert decoded coordinate bounds.

Suggested assertion:

```text
For a known 130 x 130 printed board, every decoded target_position from a
successful detection must satisfy:

0 <= x_mm <= 129 * 1.014
0 <= y_mm <= 129 * 1.014
```

For the same fixture, compare:

- `Full`
- `Full` with `search_all_components=false`
- `FixedBoard`
- both scoring modes if useful: `SoftLogLikelihood` and `HardWeighted`

The expected result for this application is not necessarily that `Full` must
always win. The important contract is that a known printed board should have a
safe mode that returns positions in the declared board frame, and that Full mode
should not be used by this calibration pipeline unless its global-origin
ambiguity is explicitly handled.

## Current Calibration-Side Workaround

The calibration example now uses the safer known-board path:

1. `PuzzleBoardSearchMode::FixedBoard`
2. `search_all_components=false`
3. hard board-range filtering before converting millimeters to meters
4. fixed board-centered coordinates:
   - `origin_x_mm = 0.5 * (cols - 1) * cell_size_mm`
   - `origin_y_mm = 0.5 * (rows - 1) * cell_size_mm`

This workaround is correct for the private e2e pipeline because the printed
board specification is known. It should remain in place unless the detector
learns to expose and validate the recovered master origin in a way the pipeline
can use safely.

## Open Questions For The Detector Investigation

- Does Full mode pick the wrong `(D4, master_row, master_col)` only when
  `search_all_components=true`, or also on the primary component?
- Does the wrong Full hypothesis have a small score margin? If so, the detector
  should expose/reject low-margin full-origin matches.
- Are the bad hypotheses correlated with small visible fragments, edge-of-tile
  crops, glare, or one specific camera?
- Can the detector report the chosen D4 transform and master origin so the e2e
  harness can print them per camera/pose?
- Should `PuzzleBoardParams::for_board(&spec)` default to `FixedBoard` when the
  caller supplied an explicit printed board spec?

## Success Criteria

The handoff is complete when the detector-side agent can:

1. reproduce at least one Full-mode wrong-origin decode on the shared images, or
   prove the issue was caused by example-side coordinate handling;
2. identify the affected image, pose, camera, chosen mode, scoring mode, and raw
   decoded coordinate range;
3. add a regression or diagnostic that prevents this exact failure from being
   invisible in future e2e runs;
4. confirm that `FixedBoard` remains stable on the same tiles.
