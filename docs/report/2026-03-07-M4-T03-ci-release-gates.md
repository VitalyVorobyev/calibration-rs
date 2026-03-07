# M4-T03: CI and Release Gate Enforcement

Date: 2026-03-07
Commit: pending

## Scope

- Updated CI workflow to enforce hard release gates:
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - Python runtime job with extension build + runtime tests.
- Updated PyPI release workflow verification job to run Python runtime tests before publish.

## Files changed

- `.github/workflows/ci.yml`
- `.github/workflows/release-pypi.yml`
- `docs/backlog.md`

## Validation run

- Workflow syntax and command review performed locally.
- Runtime test command validated locally in dev venv:
  - `maturin develop`
  - `python -m unittest discover -s crates/vision-calibration-py/tests -p "test_*.py"`

## Follow-ups / risks

- Full workflow execution requires GitHub Actions run on push/PR/tag.
