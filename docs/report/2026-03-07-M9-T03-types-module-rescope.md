# M9-T03: Re-scope `types.py` to Low-Level Compatibility Surface

Date: 2026-03-07
Commit: pending

## Scope

- Re-scoped package surface so `types.py` is no longer re-exported via top-level aliases.
- Removed top-level re-exports from `vision_calibration` package:
  - `HandEyeMode`
  - `LaserlineResidualType`
  - `RobustLoss`
- Updated top-level stubs to avoid exporting these low-level aliases, while keeping robust helper return types typed internally.
- Strengthened `types.py` module docs with explicit low-level compatibility warning.
- Updated README note to steer new code to typed dataclasses/models instead of `types.py` contracts.
- Added Python test coverage to assert low-level `types` is still importable but not top-level re-exported.

## Files changed

- `crates/vision-calibration-py/python/vision_calibration/__init__.py`
- `crates/vision-calibration-py/python/vision_calibration/__init__.pyi`
- `crates/vision-calibration-py/python/vision_calibration/types.py`
- `crates/vision-calibration-py/README.md`
- `crates/vision-calibration-py/tests/test_typed_models.py`
- `docs/backlog.md`
- `docs/report/2026-03-07-M9-T03-types-module-rescope.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_typed_models.py` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py` -> pass
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass

## Follow-ups / risks

- The `vision_calibration.types` module remains public for interop compatibility by design.
- Full docs/examples migration to typed-first usage is tracked in `M9-T05`.
