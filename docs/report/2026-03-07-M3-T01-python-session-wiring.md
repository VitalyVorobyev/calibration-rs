# M3-T01: Scheimpflug Python Session Wiring (Bundled with M3-T03)

Date: 2026-03-07
Commit: pending

## Scope

- Switched Rust/PyO3 Scheimpflug entrypoint to generic `run_problem` session path.
- Removed Scheimpflug-specific Rust binding payload parsing/dispatch path.
- Added Python runtime tests that verify:
  - success path still works,
  - invalid config maps to `failed to set config`,
  - invalid input maps to `failed to set input`.

Bundling note:

- `M3-T01` and `M3-T03` were implemented together because wiring change and error-mapping tests
  must land atomically to enforce the contract.

## Files changed

- `crates/vision-calibration-py/src/lib.rs`
- `crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py`
- `docs/backlog.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass
- `source .venv-codex/bin/activate && cd crates/vision-calibration-py && maturin develop` -> pass
- `source .venv-codex/bin/activate && python -m unittest crates/vision-calibration-py/tests/test_scheimpflug_intrinsics.py` -> pass

## Follow-ups / risks

- `M3-T02` remains pending for final contract synchronization check across
  `models.py`, `types.py`, and `__init__.pyi` once remaining pipeline/facade changes settle.
