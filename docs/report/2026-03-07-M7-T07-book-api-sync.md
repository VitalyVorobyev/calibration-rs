# M7-T07: Book API Surface Synchronization

Date: 2026-03-07
Commit: pending

## Scope

- Updated book content to match current finalized API and workflow shape.
- Applied API-surface corrections:
  - `quickstart.md`: dependency version `0.2`, fixed camera constructor call to current import path (`make_pinhole_camera`)
  - `step_functions.md`: replaced stale pipeline function names with current module-scoped `run_calibration` naming and added Scheimpflug pipeline row
  - `architecture.md`: updated workflow count to six problem types and clarified workspace includes Python bindings
  - `new_pipeline.md`: aligned prelude guidance with current minimal-prelude policy

## Files changed

- `book/src/quickstart.md`
- `book/src/step_functions.md`
- `book/src/architecture.md`
- `book/src/new_pipeline.md`
- `docs/backlog.md`
- `docs/report/2026-03-07-M7-T07-book-api-sync.md`

## Validation run

- `mdbook build book` -> pass
  - warning noted: `mdbook-katex` version mismatch (`built against mdbook v0.4.48`, runtime `v0.4.52`)

## Follow-ups / risks

- Consider aligning `mdbook-katex` plugin/tooling versions to remove build warning noise.
