# B3C-RINGGRID - Coded ring-grid detector

## Scope

Wire the coded ring-grid detector — the last of the four target detectors —
into `vision-calibration-detect` and the dataset runner, wrapping the
published `ringgrid` 0.6 crate. With this, all four target types
(chessboard / charuco / puzzleboard / ringgrid) calibrate end-to-end, which
unblocks the B3d "sniff folder" manifest UX (it must dispatch to a detector
for every `TargetSpec` variant).

## Spec realignment (breaking, pre-1.0)

The speculative `TargetSpec::Ringgrid` modeled a rectangular `rows × cols`
grid (`spacing_m`, `inner_radius_m`, `outer_radius_m`). The real `ringgrid`
board is a **hex-lattice** parameterized by `(pitch, rows, long_row_cols,
outer_radius, inner_radius, ring_width)`. Rather than silently reinterpret
the rectangular fields (which ADR 0019 forbids), `TargetSpec::Ringgrid` is
redefined to mirror `ringgrid::BoardLayout`:

```text
pitch_m, rows, long_row_cols,
marker_outer_radius_m, marker_inner_radius_m, marker_ring_width_m
```

The JSON schema (`app/src/schemas/dataset_spec.json`) is regenerated to match.

## Files Changed

- `Cargo.toml` — `ringgrid = "0.6"` workspace dependency.
- `crates/vision-calibration-dataset/src/spec.rs` — `TargetSpec::Ringgrid`
  realigned to the hex-lattice model.
- `crates/vision-calibration-detect/src/ringgrid.rs` (new) — `RinggridConfig`
  + `RinggridDetector` (sealed `Detector`). Builds a `BoardLayout` from the
  config and runs `Detector::detect_adaptive` (auto scale selection — the
  marker pixel size is unknown for an arbitrary dataset image). Each decoded
  marker with a `board_xy_mm` becomes a metric `Feature`.
- `crates/vision-calibration-detect/Cargo.toml` — `ringgrid` feature +
  optional dep, added to `default`.
- `crates/vision-calibration-detect/src/lib.rs` — module + re-exports.
- `crates/vision-calibration-pipeline/src/dataset_runner/mod.rs` —
  `detector_config_for_target` maps `TargetSpec::Ringgrid` (was
  `UnsupportedTarget`); `pick_detector` gains the `"ringgrid"` arm.
- `crates/vision-calibration-pipeline/src/dataset_runner/planar.rs` —
  dropped the obsolete `rejects_unsupported_target` test (no `TargetSpec`
  variant is unsupported anymore; ringgrid support is covered without I/O by
  `ringgrid_target_maps_to_detector_config`).
- `app/src/schemas/dataset_spec.json` — regenerated.

## Validation Run

- PASS: `cargo test -p vision-calibration-detect --all-features` — incl.
  `ringgrid::tests::detects_synthetic_board` (renders a board via
  `BoardLayout::render_target_png`, decodes ≥4 markers) and
  `invalid_board_geometry_rejected`.
- PASS: `cargo test -p vision-calibration-pipeline --lib` — incl.
  `ringgrid_target_maps_to_detector_config` (field-name contract guard).
- PASS: `cargo test -p vision-calibration-dataset`.
- PASS: `cargo clippy -p vision-calibration-detect -p vision-calibration-pipeline -p vision-calibration-dataset --all-targets --all-features -- -D warnings`.
- PASS: `cargo fmt --all -- --check`; `cargo xtask emit-schemas` (schema in sync).

## Follow-Ups

- Charuco dedup (`B3C-CHARUCO-DEDUP`) consolidates the three charuco paths.
- App Run-workspace ringgrid preset + form is enabled in the app-wiring commit.
- The synthetic ringgrid test runs adaptive multi-scale detection (~20 s); if
  CI time matters, a lower-DPI render or `detect()` with a scale hint would cut it.
