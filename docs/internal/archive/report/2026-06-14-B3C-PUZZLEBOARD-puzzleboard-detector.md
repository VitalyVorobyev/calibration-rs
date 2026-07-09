# B3C-PUZZLEBOARD - PuzzleBoard detector

## Scope

Wire the PuzzleBoard target detector into `vision-calibration-detect` and
the dataset runner, the first of the two B3c-remainder detectors that
complete the "4 target detectors" goal. PuzzleBoard detection already
ships in `calib-targets` 0.9 (`detect_puzzleboard`, already a transitive
dependency), so this is a wrap + dispatch wiring, not algorithm work.

## Files Changed

- `crates/vision-calibration-detect/src/puzzleboard.rs` (new) — `PuzzleboardConfig`
  + `PuzzleboardDetector` implementing the sealed `Detector` trait. Mirrors
  the proven bench recipe (`bench/src/detect.rs::detect_puzzleboard_view`):
  2× upscale before detection, `FixedBoard` search, in-board filter. Emits
  metric `Feature`s in the board's top-left-origin frame (same convention as
  the chessboard/charuco detectors, not the centred bench-example frame).
- `crates/vision-calibration-detect/Cargo.toml` — `puzzleboard` feature
  (`["dep:calib-targets"]`), added to `default`.
- `crates/vision-calibration-detect/src/lib.rs` — module + re-exports.
- `crates/vision-calibration-pipeline/src/dataset_runner/mod.rs` —
  `detector_config_for_target` now resolves `TargetSpec::Puzzleboard`
  (was `UnsupportedTarget`); `pick_detector` gains the `"puzzleboard"` arm;
  new `parse_puzzleboard_layout` resolves the manifest's `"puzzle_<R>x<C>"`
  layout name to dimensions, failing fast on a typo (ADR 0019).

## Design Notes

- The detector is **parametric** (`rows`/`cols`/`cell_size_m`); the manifest's
  named-layout convention (`"puzzle_130x130"`) is resolved at the pipeline
  boundary. This keeps the detect crate free of manifest-naming concerns —
  the layout→dimensions map lives where `TargetSpec`→config translation
  already lives.
- Sealed `Detector` trait + per-detector feature flag keep this **open for
  extension** (OCP): only the two `dataset_runner` match arms are shared edits.

## Validation Run

- PASS: `cargo test -p vision-calibration-detect --all-features` (20 tests,
  incl. `detects_synthetic_board` — a rendered PuzzleBoard decoded back to
  ≥8 in-board saddles).
- PASS: `cargo test -p vision-calibration-pipeline --lib` (238 tests, incl.
  `puzzleboard_target_maps_to_detector_config`,
  `puzzleboard_bad_layout_fails_before_io`,
  `parse_puzzleboard_layout_accepts_known_forms`).
- PASS: `cargo clippy -p vision-calibration-detect -p vision-calibration-pipeline --all-targets --all-features -- -D warnings`.
- PASS: `cargo fmt --all -- --check`.

## Follow-Ups

- Ringgrid detector (`B3C-RINGGRID`) is the second remaining detector.
- App Run-workspace preset for puzzleboard is enabled in a later commit
  (the `ds8-scheimpflug` preset was disabled pending this detector).
