# Python Bindings Dictless API TODO

## Goal
Eliminate dict-like public result and model APIs from `vision_calibration` Python bindings, and make typed dataclasses/models the default and only supported high-level interface.

## Current Gaps
- Result models expose dict payloads (`camera`, `cameras`, `raw`) in `crates/vision-calibration-py/python/vision_calibration/models.py`.
- Public API functions in `_api.py` still accept raw `Mapping[...]` payloads/configs as first-class inputs.
- `types.py` (TypedDict raw schema) is part of the public package surface and encourages dict-centric usage.
- Downstream users (for example calibration examples) cannot rely on stable typed camera fields.

## Work Plan

### 1. Define typed camera model contracts
- [ ] Add explicit Python dataclasses for camera payloads:
- [ ] `PinholeIntrinsics`, `BrownConradyDistortion`, `ScheimpflugSensor`.
- [ ] `PinholeBrownConradyCamera`, `PinholeBrownConradyScheimpflugCamera`.
- [ ] Add `from_payload(...)` constructors for each typed camera model.
- [ ] Keep parsing tolerant to legacy key aliases (`tilt_x`/`tau_x`) internally only.

### 2. Convert result models to typed fields
- [ ] Replace dict fields in result dataclasses:
- [ ] `PlanarCalibrationResult.camera: PinholeBrownConradyCamera`
- [ ] `RigExtrinsicsResult.cameras: list[PinholeBrownConradyCamera]`
- [ ] `RigHandeyeResult.cameras: list[PinholeBrownConradyCamera]`
- [ ] `ScheimpflugIntrinsicsResult.camera: PinholeBrownConradyScheimpflugCamera`
- [ ] For laserline result, replace `estimate`/`stats` dicts with typed dataclasses.
- [ ] Remove or quarantine `raw: dict[str, Any]` from public results (internal/debug-only).

### 3. Tighten high-level `_api.py` signatures
- [ ] Make high-level runners accept typed dataset/config objects only.
- [ ] Move raw mapping-based entry points behind clearly named low-level helpers (for example `_run_*_raw`).
- [ ] Update docstrings to remove “raw serde mapping is also accepted” from high-level APIs.

### 4. Re-scope `types.py`
- [ ] Keep `types.py` only as low-level compatibility schema.
- [ ] Remove `types` from top-level public docs/examples/import patterns.
- [ ] Add explicit warning in module docs: low-level/internal, not recommended for new code.

### 5. Add migration and compatibility strategy
- [ ] Provide one transition release where mapping input is accepted but emits `DeprecationWarning`.
- [ ] In next release, remove mapping input acceptance from high-level APIs.
- [ ] Publish migration notes with before/after snippets.

### 6. Update package exports and stubs
- [ ] Update `__init__.py` and `__init__.pyi` to export typed camera/result models.
- [ ] Ensure stubs do not expose dict payload shapes as primary result fields.
- [ ] Keep type checker experience clean (`pyright`/`mypy` friendly).

### 7. Test coverage
- [ ] Unit tests for typed camera parsers (`from_payload`) including legacy key aliases.
- [ ] Binding tests for each runner asserting typed result objects (no dict result fields).
- [ ] Negative tests asserting high-level API rejects dict-like camera payload/model construction.
- [ ] End-to-end test for `run_scheimpflug_intrinsics` returning typed Scheimpflug camera.

### 8. Docs and examples
- [ ] Update README and python examples to use typed fields only (attribute access, no key indexing).
- [ ] Remove dict-style snippets from docs and reports.
- [ ] Add one “strict typed usage” example per major runner.

## Definition of Done
- [ ] No dict-typed camera/result fields in public high-level dataclasses.
- [ ] No mapping-based high-level API signatures in `_api.py`.
- [ ] All Python tests pass, including new typed API tests.
- [ ] Calibration examples can consume binding results without any dict-like fallback code.
