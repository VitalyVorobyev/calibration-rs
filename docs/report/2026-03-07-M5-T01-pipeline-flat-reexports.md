# M5-T01: Remove Pipeline Flat Re-exports

**Date**: 2026-03-07
**Task**: M5-T01

## Scope

Removed all flat `pub use` re-export blocks from `vision-calibration-pipeline/src/lib.rs`.
Fixed the facade prelude to use module-segmented import paths.

## Files Changed

- `crates/vision-calibration-pipeline/src/lib.rs` — deleted 99 lines of `pub use` blocks; file now has only `pub mod` declarations
- `crates/vision-calibration/src/lib.rs` — fixed 3 prelude import groups to use `vision_calibration_pipeline::<module>::` paths

Also changed in this session (separate from M5-T01):
- `CLAUDE.md` — trimmed from 450 to 91 lines
- `docs/backlog.md` — new milestones M5-M9
- `docs/adrs/` — added retroactive ADRs 0005-0009, updated README

## Validation

- `cargo fmt --all -- --check`: PASS
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: PASS
- `cargo test --workspace --all-features`: PASS (all suites)
- `cargo doc --workspace --no-deps`: PASS (no warnings)

## Follow-ups

- M5-T02: Clean up facade core glob re-exports
- M5-T03: Standardize option type naming
