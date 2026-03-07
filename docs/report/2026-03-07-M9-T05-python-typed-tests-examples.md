# M9-T05: Typed Python Tests and Example Migration

Date: 2026-03-07
Commit: pending

## Scope

- Added high-level Python API contract tests covering all six runners:
  - dict-style high-level input rejection (`TypeError`)
  - dict-style high-level config rejection (`TypeError`)
  - typed input/config path reaches runtime validation (`RuntimeError` on intentionally invalid datasets)
- Migrated Python examples away from dict-style result access to typed attribute access:
  - planar synthetic + real
  - single-cam hand-eye synthetic + real
  - stereo rig session
  - laserline device synthetic session
- Verified synthetic examples execute successfully with typed result models.

## Files changed

- `crates/vision-calibration-py/tests/test_high_level_api_contract.py`
- `crates/vision-calibration-py/examples/planar_synthetic.py`
- `crates/vision-calibration-py/examples/planar_real.py`
- `crates/vision-calibration-py/examples/handeye_synthetic.py`
- `crates/vision-calibration-py/examples/handeye_session.py`
- `crates/vision-calibration-py/examples/stereo_session.py`
- `crates/vision-calibration-py/examples/laserline_device_session.py`
- `docs/backlog.md`
- `docs/report/2026-03-07-M9-T05-python-typed-tests-examples.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_high_level_api_contract.py` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_typed_models.py crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py` -> pass
- Synthetic example runs (all pass):
  - `planar_synthetic.py`
  - `handeye_synthetic.py`
  - `rig_handeye_synthetic.py`
  - `laserline_device_session.py`
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass

## Follow-ups / risks

- Real-image examples (`planar_real.py`, `stereo_session.py`, `stereo_charuco_session.py`, `handeye_session.py`) still depend on external datasets/tooling availability for runtime execution.
