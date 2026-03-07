# M4-T04: Release Notes and Version Synchronization

Date: 2026-03-07
Commit: pending

## Scope

- Added changelog entry for `0.2.0` capturing Scheimpflug API additions and workflow hardening.
- Applied workspace-wide minor version bump to `0.2.0` in Rust workspace metadata.
- Synchronized Python package version in `pyproject.toml`.
- Updated lockfile package versions after workspace version bump.

## Files changed

- `CHANGELOG.md`
- `Cargo.toml`
- `Cargo.lock`
- `crates/vision-calibration-py/pyproject.toml`
- `docs/backlog.md`

## Validation run

- Version fields manually cross-checked across workspace and Python package metadata.
- Release workflow (`release-pypi.yml`) includes tag/version sync verification.

## Follow-ups / risks

- Publish step still depends on tagged release and CI passing on target branch.
