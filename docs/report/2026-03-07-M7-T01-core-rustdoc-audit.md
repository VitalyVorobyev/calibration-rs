# M7-T01: Core Crate Rustdoc Audit

Date: 2026-03-07
Commit: pending

## Scope

- Audited `vision-calibration-core` public API with `missing_docs` lint.
- Added rustdoc for all previously undocumented public items and fields in:
  - top-level core helpers/types (`PinholeCamera`, conversion helpers, `TargetPose`, reprojection helper)
  - serialized model parameter fields (`models/params.rs`)
  - RANSAC estimator associated types
  - planar observation convenience method (`planar_points`)
  - view/dataset structures and constructors (`view.rs`)
- Re-ran the `missing_docs` audit to confirm zero warnings for `vision-calibration-core`.

## Files changed

- `crates/vision-calibration-core/src/lib.rs`
- `crates/vision-calibration-core/src/models/params.rs`
- `crates/vision-calibration-core/src/ransac.rs`
- `crates/vision-calibration-core/src/types/observation.rs`
- `crates/vision-calibration-core/src/view.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M7-T01-core-rustdoc-audit.md`

## Validation run

- `cargo rustc -p vision-calibration-core --lib -- -W missing-docs` -> pass (no warnings)
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-core --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-core --all-features` -> pass

## Follow-ups / risks

- M7-T02..M7-T05 remain and likely benefit from the same `missing_docs`-driven audit workflow.
