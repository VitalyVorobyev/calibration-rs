# M8-T05: Enforce `deny(missing_docs)` in Facade Crate

Date: 2026-03-07
Commit: pending

## Scope

- Enabled strict rustdoc coverage enforcement for the facade crate by adding:
  - `#![deny(missing_docs)]` in `crates/vision-calibration/src/lib.rs`
- Verified the existing facade API docs are complete enough to satisfy the lint without adding compatibility shims or suppressions.

## Files changed

- `crates/vision-calibration/src/lib.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M8-T05-facade-deny-missing-docs.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- Any new public facade item now requires docs by default; this is intentional and should prevent future API doc regressions.
