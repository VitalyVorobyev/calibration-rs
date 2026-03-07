# M5-T07: Trim Facade Prelude to Planar Hello-World Surface

Date: 2026-03-07
Commit: pending

## Scope

- Reduced `vision_calibration::prelude` to a minimal planar hello-world surface.
- Removed non-planar problem types, non-planar run aliases, `ProblemType`, and common optimization options from prelude.
- Kept only session + planar problem + planar runner + core planar dataset/types needed for basic usage.
- Fixed examples and facade doctests that previously relied on broad prelude imports by adding explicit module imports.
- Updated workspace/facade/book documentation text and snippets to describe the new prelude contract.

## Files changed

- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration/examples/laserline_device_session.rs`
- `crates/vision-calibration/examples/handeye_session.rs`
- `crates/vision-calibration/examples/rig_handeye_synthetic.rs`
- `crates/vision-calibration/examples/stereo_session.rs`
- `crates/vision-calibration/examples/stereo_charuco_session.rs`
- `crates/vision-calibration/README.md`
- `README.md`
- `book/src/architecture.md`
- `docs/backlog.md`
- `docs/report/2026-03-07-M5-T07-prelude-trim.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- This is a breaking change for users relying on broad prelude imports; explicit module imports are now required for non-planar workflows.
