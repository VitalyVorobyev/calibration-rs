# M9-T01: Typed Camera/Result Models for Python High-Level API

Date: 2026-03-07
Commit: pending

## Scope

- Implemented typed camera model contracts in Python bindings:
  - `PinholeIntrinsics`
  - `BrownConradyDistortion`
  - `ScheimpflugSensor` (with legacy alias parse support: `tau_x`/`tau_y`)
  - `PinholeBrownConradyCamera`
  - `PinholeBrownConradyScheimpflugCamera`
- Added typed laserline result payload models:
  - `LaserlinePlane`
  - `LaserlineEstimateParams`
  - `LaserlineEstimate`
  - `LaserlineStats`
- Converted high-level result dataclasses to typed fields and removed dict/raw fields:
  - `PlanarCalibrationResult.camera`
  - `SingleCamHandeyeResult.camera`
  - `RigExtrinsicsResult.cameras`
  - `RigHandeyeResult.cameras`
  - `ScheimpflugIntrinsicsResult.camera`
  - `LaserlineDeviceResult.estimate` + `stats`
  - removed `raw` from all public high-level result dataclasses
- Exported new typed models via package surface (`__init__.py`, `__init__.pyi`).
- Updated Python runtime test assertion for typed Scheimpflug camera result.
- Added new parser-focused Python tests in `test_typed_models.py`.
- Locked hard-break policy wording for M9 migration strategy in backlog/TODO docs.

## Files changed

- `crates/vision-calibration-py/python/vision_calibration/models.py`
- `crates/vision-calibration-py/python/vision_calibration/__init__.py`
- `crates/vision-calibration-py/python/vision_calibration/__init__.pyi`
- `crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py`
- `crates/vision-calibration-py/tests/test_typed_models.py`
- `crates/vision-calibration-py/README.md`
- `docs/backlog.md`
- `docs/python-bindings-dictless-todo.md`
- `docs/report/2026-03-07-M9-T01-typed-camera-result-models.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_typed_models.py` -> pass
- Rebuilt local extension artifact for runtime test sync:
  - `cargo build -p vision-calibration-py --all-features`
  - `cp target/debug/lib_vision_calibration.dylib crates/vision-calibration-py/python/vision_calibration/_vision_calibration.abi3.so`
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py` -> pass

## Follow-ups / risks

- High-level runner signatures still accept mapping inputs; this is intentionally deferred to `M9-T02` for hard-break removal at API entrypoint level.
- Several Python examples still use dict-style access patterns; cleanup is tracked in `M9-T05`.
