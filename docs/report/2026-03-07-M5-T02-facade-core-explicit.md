# M5-T02: Facade Core Re-export Cleanup

Date: 2026-03-07
Commit: pending

## Scope

- Replaced `vision_calibration::core` glob re-export with an explicit curated list of core symbols.
- Removed `vision_calibration::handeye` escape-hatch module from facade.
- Updated rig hand-eye example import to use `vision_calibration::optim::RobotPoseMeta`.

## Files changed

- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration/examples/rig_handeye_synthetic.rs`
- `docs/backlog.md`

## Validation run

- `cargo fmt --all -- --check` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- External users importing previously leaked `core::*` items not in the new explicit list will need to import from `vision_calibration_core` directly.
