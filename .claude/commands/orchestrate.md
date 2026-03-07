# Orchestrate: Automated Task Workflow

You are the orchestrator for calibration-rs development. Run the full architect → implement → review → commit cycle.

## Input

Goal or area to work on: $ARGUMENTS

## Workflow

### Phase 1: Architect

Use the Agent tool (subagent_type: "Plan") to:

1. Read `docs/backlog.md`, `docs/adrs/README.md`, and relevant ADRs
2. Read the facade crate's `lib.rs` and the pipeline crate's `lib.rs` to understand current API surface
3. Assess what work is needed for the given goal
4. Produce a **task specification** with:
   - Task ID (next available `M<n>-T<nn>` from backlog)
   - Clear scope: what changes, what files, what stays unchanged
   - Acceptance criteria (testable conditions)
   - Files likely to be modified
   - Any dependencies on other tasks

If the goal maps to an existing backlog item, use that task ID. If it requires a new milestone or task, propose one.

Save the task spec in your response — do NOT write it to a file yet.

### Phase 2: Implement

Use the Agent tool (subagent_type: "general-purpose") to implement the task spec from Phase 1. Provide:

- The full task spec
- Instruction to follow CLAUDE.md and AGENTS.md conventions
- Instruction to run `cargo check --workspace` after changes
- Instruction NOT to commit — just make the changes

### Phase 3: Gate Check

Run these commands sequentially and collect results:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps 2>&1 | grep -E "warning|error" | head -20
```

If any gate fails, go back to Phase 2 with the failure details. Maximum 2 retries.

### Phase 4: Review

Use the Agent tool (subagent_type: "rust-qa-officer") to review the changes:

1. Run `git diff` to see all changes
2. Check against the task spec's acceptance criteria
3. Verify no unintended side effects
4. Check naming consistency with existing API patterns
5. Verify rustdoc on changed public items

The reviewer should return: APPROVED or REJECTED with specific feedback.

If REJECTED: go back to Phase 2 with the feedback (max 2 retries total across all rejections).

### Phase 5: Commit

If approved and all gates pass:

1. Write a task report to `docs/report/YYYY-MM-DD-<task-id>-<slug>.md`
2. Update `docs/backlog.md` to mark the task complete
3. Stage all changed files (be specific, no `git add -A`)
4. Commit with message format: `feat(backlog): <task-id> <short description>`

If there are uncertainties or design questions that couldn't be resolved, DO NOT commit. Instead, present the questions to the user.

## Important Rules

- One task per orchestration run
- Do not batch unrelated changes
- If the goal is too large for one task, have the architect break it into multiple tasks and implement only the first one
- Follow AGENTS.md layering rules strictly
- Breaking API changes are allowed for this release cycle
