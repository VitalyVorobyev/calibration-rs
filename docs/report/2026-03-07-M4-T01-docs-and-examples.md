# M4-T01: Scheimpflug Documentation and Examples (Bundled with M4-T02)

Date: 2026-03-07
Commit: pending

## Scope

- Updated documentation surfaces to include Scheimpflug workflow coverage.
- Added/updated minimal Rust and Python examples showing Scheimpflug usage.
- Added ADR set and planning-process docs (`docs/adrs/`) and removed legacy `IMPLEMENTATION_PLAN.md`.

Bundling note:

- `M4-T01` and `M4-T02` were implemented together because documentation shape and example snippets
  are published in the same README/doc updates.

## Files changed

- `README.md`
- `crates/vision-calibration/README.md`
- `crates/vision-calibration-py/README.md`
- `docs/adrs/README.md`
- `docs/adrs/0001-pipeline-problem-module-shape.md`
- `docs/adrs/0002-planar-intrinsics-family-and-sensor-modes.md`
- `docs/adrs/0003-facade-and-python-api-consistency.md`
- `docs/adrs/0004-planning-process-adr-backlog.md`
- `IMPLEMENTATION_PLAN.md` (removed)
- `docs/backlog.md`

## Validation run

- Documentation-only updates; examples were sanity-checked against exported API names.

## Follow-ups / risks

- If API names or signatures change during M2 work, snippets must be re-synced before release.
