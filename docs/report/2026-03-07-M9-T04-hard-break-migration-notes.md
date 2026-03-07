# M9-T04: Hard-Break Migration Notes for Typed Python API

Date: 2026-03-07
Commit: pending

## Scope

- Finalized hard-break migration documentation for Python API dictless transition.
- Published before/after usage snippets and explicit break notice in Python README.
- Added explicit compatibility guidance to use low-level raw helpers in `vision_calibration._api` when raw serde control is required.
- Updated changelog breaking section with typed-only high-level API note.
- Synced migration checklist progress in `docs/python-bindings-dictless-todo.md`.

## Files changed

- `crates/vision-calibration-py/README.md`
- `CHANGELOG.md`
- `docs/python-bindings-dictless-todo.md`
- `docs/backlog.md`
- `docs/report/2026-03-07-M9-T04-hard-break-migration-notes.md`

## Validation run

- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass

## Follow-ups / risks

- Examples still need full typed-usage cleanup and expansion across all runners (`M9-T05`).
