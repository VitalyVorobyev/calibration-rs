# M0-T02: Coupled-Task Bundling Rule

Date: 2026-03-07
Commit: pending

## Scope

- Updated `AGENTS.md` backlog workflow to allow bundled commits only when tasks are tightly coupled and cannot land independently without breaking build/API continuity.
- Required explicit coupling documentation in backlog/report when using this exception.

## Files changed

- `AGENTS.md`
- `docs/backlog.md`

## Validation run

- Documentation/process change only.

## Follow-ups / risks

- Bundling remains exception-only; future backlog tasks should default to one task per commit.
