---
description: Full architect → implementer → reviewer pipeline with automated retry and commit
argument-hint: <backlog-id>
---

# /implement $ARGUMENTS

Full architect → implementer → reviewer → commit pipeline for vitavision backlog task `$ARGUMENTS`.
Runs automatically without pausing unless Open Questions arise.

---

## Phase 1 — Architect (read-only planning)

Act as an experienced RUST and computer vision Architect. Do NOT write or modify any code or handoff files in this phase until the blueprint is complete.

**Reading order (always do this first):**
1. `docs/backlog.md` — find task `$ARGUMENTS`, read its notes, milestone, and acceptance scenarios
2. `CLAUDE.md` — architecture overview, commands, boundaries
3. `docs/handoffs.md` — handoff workflow and file ownership rules
4. All files directly relevant to the task (read before referencing them)

**Mint the task ID:**
- Scan `docs/handoffs/` for existing directories to find the highest `TASK-N` number
- If a directory already exists whose `01-architect.md` contains `Backlog ID: $ARGUMENTS`, reuse that task ID
- Otherwise mint the next ID: `TASK-<N+1>-<slug>` where slug is the first 4–5 meaningful words of the task title, lowercased, spaces→hyphens
- All subsequent files use this task ID

**Produce a blueprint with these sections:**

- **Task Summary** — what it does and why it matters
- **Current State** — what already exists (code, tests, types), what is complete, what is partial, what is missing or incorrect; read every referenced file before assessing
- **Affected Files** — exact list to create or modify, one-line description each
- **Implementation Steps (phased)** — numbered phases with file + line-level guidance; keep each function ≤ 40 lines, each module ≤ 300 lines, no logic duplicated across backend and frontend
- **API Contract** — for any new or changed endpoint: method, path, request model, response model, error cases
- **Test Plan** — for every new pure function and every endpoint: test name, file, inputs, expected output; backend tests in `backend/tests/test_api.py`; frontend build smoke only (no jest tests exist)
- **Invariants Checklist** — for each item, state whether this task touches it:
  - CV logic only in `backend/routers/cv.py`
  - Storage logic only in `backend/services/storage_service.py`
  - Frontend API calls only in `src/lib/api.ts`
  - Algorithm adapters only in `src/components/editor/algorithms/<name>/adapter.ts`
  - `bun` (not npm) for all JS operations
  - All `HTTPException` raises include `from exc`
  - Algorithm features carry `readonly: true`, `algorithmId`, `runId`
  - Storage keys match `uploads/<sha256-64-hex>`
- **Docs to Update** — which of `CHANGELOG.md` / `README.md` need updating
- **Quality Gate Plan** — which gates are relevant (backend / frontend / both)
- **Open Questions** — anything that cannot be resolved from the codebase alone

**Write `docs/handoffs/<task-id>/01-architect.md`** using `docs/templates/task-handoff-report.md`.

Required metadata header:
```
Title: <task title>
Task ID: <task-id>
Backlog ID: $ARGUMENTS
Role: Architect
Date: <today>
Status: ready_for_implementer
```

Required sections: Summary · Decisions · Files/Modules Affected · API Contract · Test Plan · Validation · Risks/Open Questions · Next Handoff.

**Show the Summary, Current State, and Open Questions to the user.**

If Open Questions cannot be resolved from the codebase, **stop and ask the user** before proceeding to Phase 2.

---

## Phase 2 — Implementer

Act as the vitavision Implementer. Follow the Phase 1 blueprint exactly. Read every file you plan to modify before touching it.

**Repo boundaries (non-negotiable):**
| Boundary | Rule |
|----------|------|
| CV algorithm logic | Only in `backend/routers/cv.py` (Pydantic models + algorithm calls) |
| Storage logic | Only in `backend/services/storage_service.py` |
| Auth | Only in `backend/auth.py` |
| Frontend API layer | Only in `src/lib/api.ts` (fetch calls + TS types) |
| Editor state | Only in `src/store/editor/useEditorStore.ts` |
| Feature schema | Only in `src/store/editor/featureSchema.ts` |
| Algorithm adapters | Only in `src/components/editor/algorithms/<name>/adapter.ts` |
| JS/TS tooling | Always `bun`, never `npm` |

**Code quality invariants:**
- Functions ≤ 40 lines; modules ≤ 300 lines; extract helpers when needed
- No logic duplication between backend and frontend (types must mirror; conversion helpers in one place)
- All `HTTPException` raises include `from exc`
- No bare `except` that silences errors
- Algorithm features: `readonly: true`, `algorithmId`, `runId` always set
- Storage keys always `uploads/<sha256-64-hex>` — validate with `_KEY_PATTERN`
- `bun` not `npm`

**Implementation workflow:**
1. Implement phases in order from the blueprint
2. Backend changes: `cd backend && source .venv/bin/activate`
   - `ruff check .` — fix all warnings
   - `ruff format --check .` — fix formatting
   - `mypy . --ignore-missing-imports` — fix type errors
   - `STORAGE_MODE=local LOCAL_STORAGE_ROOT=/tmp pytest tests/ -v` — fix failures
3. Frontend changes: `bun run lint && bun run build` — fix all errors
4. Append one line to `CHANGELOG.md` under `## [Unreleased]`
5. Update `README.md` only if a user-visible capability was added

**Write `docs/handoffs/<task-id>/02-implementer.md`** using `docs/templates/task-handoff-report.md`.

Required metadata:
```
Title: <task title>
Task ID: <task-id>
Backlog ID: $ARGUMENTS
Role: Implementer
Date: <today>
Status: ready_for_review
```

Required sections: Summary · Files Changed · Tests Added · Deviations from Spec · Validation Results · Risks/Open Questions · Next Handoff.

**Do NOT:** refactor outside blueprint scope · add docstrings to unchanged code · create one-use abstractions · use `npm` · silently skip a quality gate.

---

## Phase 3 — Review Loop (automated, max 3 retries)

Act as the vitavision Reviewer. Run all quality gates. Do NOT write production code.

**Review checklist:**

**A. Repo Boundary Violations (Blocking)**
- CV algorithm logic outside `backend/routers/cv.py`
- Storage logic outside `backend/services/storage_service.py`
- Frontend fetch calls outside `src/lib/api.ts`
- Algorithm adapter logic outside `src/components/editor/algorithms/<name>/adapter.ts`
- `npm` used instead of `bun`

**B. Code Safety (Blocking)**
- Bare `except` that silences errors in Python
- `HTTPException` raised without `from exc`
- `unwrap()` / `.unwrap_or_else(panic)` pattern (if any Rust ever appears)

**C. Code Quality (Important)**
- Any function > 40 lines — flag by name and line count
- Any module > 300 lines — flag by name and line count
- Logic duplicated between backend model and frontend type
- API response fields that are undocumented

**D. Test Coverage (Important)**
- Every new endpoint has at least one happy-path test and one error-path test in `backend/tests/test_api.py`
- Every new pure Python function has a unit test
- Flag every untested function/endpoint by name

**E. API Contract Stability (Blocking if API changed)**
- No removal or rename of existing response fields
- New optional fields only — never change a required field to a different type
- `CHANGELOG.md` has an entry for this task

**F. Feature Model Invariants (Blocking if algorithm adapter changed)**
- Algorithm features have `readonly: true`, `algorithmId`, `runId`
- Feature `meta` fields match `FeatureMeta` interface in `useEditorStore.ts`

**G. Quality Gates (run all, report pass/fail):**
```bash
# Backend (if any backend file changed)
cd backend && source .venv/bin/activate
ruff check .
ruff format --check .
mypy . --ignore-missing-imports
STORAGE_MODE=local LOCAL_STORAGE_ROOT=/tmp pytest tests/ -v

# Frontend (if any frontend file changed)
bun run lint
bun run build
```

Only report findings with confidence ≥ 80%.

**Write `docs/handoffs/<task-id>/03-reviewer.md`** using `docs/templates/task-handoff-report.md`.

Required metadata:
```
Title: <task title>
Task ID: <task-id>
Backlog ID: $ARGUMENTS
Role: Reviewer
Date: <today>
Status: approved | approved_with_minor_followups | changes_requested
```

Required sections: Summary · Gate Results (table) · Issues Found (Blocking / Important / Minor) · Unit Test Coverage · Verdict.

Gate results table:
```markdown
| Gate | Result |
|------|--------|
| ruff check | PASS / FAIL / SKIP |
| ruff format | PASS / FAIL / SKIP |
| mypy | PASS / FAIL / SKIP |
| pytest | PASS / FAIL / SKIP |
| bun lint | PASS / FAIL / SKIP |
| bun build | PASS / FAIL / SKIP |
```

**If Blocking Issues found:** switch back to Implementer, fix ONLY the listed issues, re-run gates, then re-review. Max 3 total attempts. If still blocking after attempt 3:
> "Could not resolve blocking issues after 3 attempts. Manual intervention needed."
> List remaining issues and stop. Do not commit.

---

## Phase 4 — Important Issues

If only Important/Minor issues remain (no Blocking), present them and ask:
> "Review complete. Important issues found: [list]. Fix before committing? (yes / no / list which ones)"

Proceed based on user answer.

---

## Phase 5 — Architect Closeout

Write `docs/handoffs/<task-id>/04-architect.md` using `docs/templates/task-handoff-report.md`.

Required metadata:
```
Title: <task title>
Task ID: <task-id>
Backlog ID: $ARGUMENTS
Role: Architect
Date: <today>
Status: ready_for_human
```

Required sections: Outcome Summary · Residual Risks · Human Decision / Validation Needed · Follow-up Tasks (if any).

---

## Phase 6 — Commit

When review is clean (zero Blocking Issues):

1. `git diff --name-only HEAD` — list changed files
2. Stage all changed source files plus:
   - `docs/handoffs/<task-id>/` (all files)
   - `CHANGELOG.md`
   - `README.md` (if modified)
3. Commit with Co-Authored-By trailer:
   ```
   git commit -m "feat($ARGUMENTS): {task title from backlog}"
   ```
   using a HEREDOC for the message
4. Update `docs/backlog.md`: change task status to `done`, add `~~strikethrough~~` on ID and task name, move row to `## Done` table with today's date
5. Commit the backlog update:
   ```
   git commit -m "chore: mark $ARGUMENTS done in backlog"
   ```

---

## Phase 7 — Summary

Print:
- What was implemented (one paragraph)
- Handoff location: `docs/handoffs/<task-id>/`
- Next suggested task: first `todo` P0 or P1 task from `docs/backlog.md`
