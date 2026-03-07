# M7-T02: Linear Crate Rustdoc Audit

Date: 2026-03-07
Commit: pending

## Scope

- Audited `vision-calibration-linear` public API with `missing_docs` lint.
- Added missing rustdoc for:
  - `prelude` module in `lib.rs`
  - `MetaHomography` struct and field in `distortion_fit.rs`
  - `ExtrinsicPoses` fields in `extrinsics.rs`
  - `MotionPair` fields in `handeye.rs`
- Re-ran `missing_docs` audit to ensure zero warnings for the linear crate.

## Files changed

- `crates/vision-calibration-linear/src/lib.rs`
- `crates/vision-calibration-linear/src/distortion_fit.rs`
- `crates/vision-calibration-linear/src/extrinsics.rs`
- `crates/vision-calibration-linear/src/handeye.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M7-T02-linear-rustdoc-audit.md`

## Validation run

- `cargo rustc -p vision-calibration-linear --lib -- -W missing-docs` -> pass (no warnings)
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-linear --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-linear --all-features` -> pass

## Follow-ups / risks

- M7-T03 (optim crate rustdoc audit) remains next.
