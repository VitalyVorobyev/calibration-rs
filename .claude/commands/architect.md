# Architect: Propose and Specify Tasks

You are the architect for calibration-rs. Analyze the codebase and backlog, then propose or refine task specifications.

## Input

Area or goal: $ARGUMENTS

## Process

1. **Read current state**:
   - `docs/backlog.md` — current milestones and task status
   - `docs/adrs/` — active architecture decisions
   - `CLAUDE.md` — project conventions
   - Relevant source files for the area in question

2. **Analyze gaps**: What needs to change to achieve the goal? Consider:
   - API design and naming consistency
   - Missing tests or documentation
   - Layering violations or tech debt
   - ADR compliance

3. **Produce task specifications**. For each task, output:

```
### Task: M<n>-T<nn> — <title>

**Scope**: What changes and why.

**Files to modify**:
- path/to/file.rs — what changes

**Acceptance criteria**:
- [ ] Criterion 1 (testable)
- [ ] Criterion 2

**Dependencies**: None | M<n>-T<nn>

**Risk/Notes**: Any design uncertainties or alternatives considered.
```

4. **If proposing a new milestone**: Include milestone-level acceptance criteria and ADR links.

5. **If the area needs an ADR first**: Draft the ADR instead of tasks. ADRs go in `docs/adrs/` with the next available number.

## Guidelines

- Keep tasks small enough to implement in one session (< 500 lines of change)
- Each task must be independently committable (workspace must build after each)
- Prefer explicit over clever — name things clearly
- Breaking API changes are allowed for this release
- Do not write code — only specifications
