# M8-T04: Session Schema Version Pinning and Validation

Date: 2026-03-07
Commit: pending

## Scope

- Hardened session JSON metadata behavior in `CalibrationSession`.
- `to_json` now pins metadata identity/version on write:
  - `metadata.problem_type = P::name()`
  - `metadata.schema_version = P::schema_version()`
- `from_json` now validates both values strictly on read:
  - rejects mismatched `problem_type`
  - rejects any schema version mismatch (older or newer)
- Updated schema-version documentation in `ProblemType` to match strict validation semantics.
- Added/extended session tests for:
  - newer schema rejection
  - older schema rejection
  - problem-type mismatch rejection
  - metadata pinning during serialization

## Files changed

- `crates/vision-calibration-pipeline/src/session/calibsession.rs`
- `crates/vision-calibration-pipeline/src/session/problem_type.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M8-T04-session-schema-version-pinning.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- Strict mismatch rejection is intentionally conservative and may require explicit migration tools for legacy session JSON if schema changes in future releases.
