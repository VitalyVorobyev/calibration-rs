# Implementation Plan (Prioritized)

1. **Fix public API correctness first (breaking changes allowed)**
- Redesign `RigHandeyeExport` to be mode-explicit and unambiguous, aligned with `SingleCamHandeyeExport`.
- Add `handeye_mode` to `RigHandeyeExport`.
- Replace overloaded transform fields with mode-specific optional fields:
  - `gripper_se3_rig` / `rig_se3_base`
  - `base_se3_target` / `gripper_se3_target`
- Update rig hand-eye docs and comments to use explicit transform semantics.
- Update affected tests for new export contract.

2. **Reduce API ambiguity in facade docs/surface**
- Update high-level crate docs to reflect explicit hand-eye export semantics.
- Keep API cohesive by documenting which symbols are stable/high-level vs advanced.
- Ensure README examples and explanations match the new mode-explicit contract.

3. **Create Python binding crate `crates/vision-calibration-py`**
- Add a new PyO3 + maturin crate to workspace.
- Expose high-level native functions for all workflows:
  - `run_planar_intrinsics`
  - `run_single_cam_handeye`
  - `run_rig_extrinsics`
  - `run_rig_handeye`
  - `run_laserline_device`
- Implement serde-based Python<->Rust conversion so Python receives/returns native dict/list structures.
- Add clear Python-visible docstrings and predictable error mapping.

4. **Design Python-native typed wrapper package**
- Add `python/vision_calibration` package in `crates/vision-calibration-py`.
- Provide typed public API (`typing` aliases + `TypedDict`-based contracts for configs/IO payloads).
- Provide ergonomic helper constructors for enum-like options (e.g., robust loss helpers).
- Include `py.typed` marker and `.pyi` typing stubs.

5. **Adapt PyPI release workflow**
- Rework `.github/workflows/release-pypi.yml` from ringgrid paths to `vision-calibration-py`.
- Keep the same multi-platform wheel build/publish structure.
- Add version synchronization check for:
  - git tag
  - workspace version
  - `crates/vision-calibration-py/pyproject.toml`
  - Python extension crate version source (workspace-backed).
- Replace ringgrid-specific typing-generation checks with project-specific package checks.

6. **Update repository documentation and agent guidance**
- Update root `README.md` with Python package scope, installation, and usage.
- Update `AGENTS.md` with new crate role and dependency/layering guidance for `vision-calibration-py`.
- Add a README for `crates/vision-calibration-py` with usage and release notes.

7. **Validation and quality gates**
- Run `cargo fmt --all`.
- Run targeted checks:
  - `cargo test -p vision-calibration-pipeline --all-features`
  - `cargo test -p vision-calibration --all-features`
  - `cargo test -p vision-calibration-py --all-features` (if tests exist)
- Verify `release-pypi.yml` syntax and path consistency.
