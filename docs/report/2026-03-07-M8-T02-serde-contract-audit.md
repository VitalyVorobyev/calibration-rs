# M8-T02: Serde Contract Audit for JSON-Facing API Types

Date: 2026-03-07
Commit: pending

## Scope

- Audited pipeline JSON-facing API contracts and enforced them with compile-time trait checks.
- Added integration test coverage that asserts `Serialize` + `Deserialize` for all session/problem contract types exposed to users.
- Removed serde derives from `FilterOptions` in planar step functions, treating it as a runtime-only step override type (not part of persistent JSON contract).

## Files changed

- `crates/vision-calibration-pipeline/src/planar_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/tests/json_contract_traits.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M8-T02-serde-contract-audit.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- This is a narrow hardening step for serde contracts; schema-version behavior is still tracked by `M8-T04`.
