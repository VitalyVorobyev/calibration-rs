# M7-T03: Optim Crate Rustdoc Audit

Date: 2026-03-07
Commit: pending

## Scope

- Audited `vision-calibration-optim` public API with `missing_docs` lint.
- Added missing rustdoc across backend/IR/problem surfaces, including:
  - `SolveReport` and public report fields
  - `RobustLoss` variants/fields
  - `FactorKind` variant payload fields
  - IR structure fields (`ParamBlock`, `ResidualBlock`, `ProblemIR`)
  - hand-eye dataset/option/estimate public fields
  - laserline metadata/params/estimate methods and fields
  - planar/rig estimate and parameter docs
- Re-ran `missing_docs` audit to confirm the optim crate is clean.

## Files changed

- `crates/vision-calibration-optim/src/backend/mod.rs`
- `crates/vision-calibration-optim/src/ir/types.rs`
- `crates/vision-calibration-optim/src/problems/handeye.rs`
- `crates/vision-calibration-optim/src/problems/laserline_bundle.rs`
- `crates/vision-calibration-optim/src/problems/planar_intrinsics.rs`
- `crates/vision-calibration-optim/src/problems/rig_extrinsics.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M7-T03-optim-rustdoc-audit.md`

## Validation run

- `cargo rustc -p vision-calibration-optim --lib -- -W missing-docs` -> pass (no warnings)
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-optim --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-optim --all-features` -> pass

## Follow-ups / risks

- M7-T04 (pipeline rustdoc audit) is the next remaining crate-level rustdoc item.
