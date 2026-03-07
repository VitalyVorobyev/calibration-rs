# M9-T02: Typed-Only High-Level Python Runner Signatures

Date: 2026-03-07
Commit: pending

## Scope

- Applied hard API break at Python high-level entrypoints in `_api.py`:
  - `run_*` functions now accept typed dataset/config objects only.
  - mapping/list-based high-level inputs were removed from signatures and runtime behavior.
- Added explicit runtime type checks for high-level args with clear `TypeError` messages.
- Added low-level raw helper runners (prefixed `_run_*_raw`) in `_api.py` for interop/advanced serde payload use.
- Updated package stub signatures (`__init__.pyi`) to typed-only high-level API.
- Updated package module docstring to reflect low-level raw helper path.
- Updated Scheimpflug Python tests:
  - invalid config test now uses typed config object.
  - added negative tests asserting high-level mapping input/config rejection.

## Files changed

- `crates/vision-calibration-py/python/vision_calibration/_api.py`
- `crates/vision-calibration-py/python/vision_calibration/__init__.py`
- `crates/vision-calibration-py/python/vision_calibration/__init__.pyi`
- `crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py`
- `docs/backlog.md`
- `docs/report/2026-03-07-M9-T02-typed-high-level-signatures.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_typed_models.py` -> pass
- `PYTHONPATH=crates/vision-calibration-py/python python3 -m unittest crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py` -> pass
- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass

## Follow-ups / risks

- `types.py` remains publicly importable and still documented as low-level serde schema; scope tightening is tracked in `M9-T03`.
- Python examples still need complete typed-usage migration (`M9-T05`).
