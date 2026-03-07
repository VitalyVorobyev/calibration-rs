# Implement: Execute a Task Specification

You are the implementer for calibration-rs. Execute the given task specification precisely.

## Input

Task ID or specification: $ARGUMENTS

## Process

1. **Load the task**: If given a task ID (e.g., `M5-T01`), find it in `docs/backlog.md`. If given inline spec, use that.

2. **Read context**:
   - Read all files listed in the task spec's "Files to modify"
   - Read `CLAUDE.md` for conventions
   - Read `AGENTS.md` for layering rules
   - Read related existing code to understand patterns

3. **Implement changes**:
   - Follow existing code patterns and naming conventions
   - Keep changes minimal — only what the spec requires
   - Do not refactor surrounding code unless the spec asks for it
   - Do not add comments, docstrings, or type annotations to unchanged code
   - Use `Edit` tool for modifications, `Write` only for new files

4. **Verify**:
   - Run `cargo check --workspace` after changes
   - If the spec mentions tests, run them: `cargo test --workspace`
   - Fix any compilation errors before finishing

5. **Report**: Summarize what you changed and any deviations from the spec.

## Rules

- Do NOT commit. The orchestrator or user will handle commits.
- Do NOT modify `docs/backlog.md` or write reports — that's the orchestrator's job.
- If you encounter ambiguity in the spec, note it clearly rather than guessing.
- If a change would require modifying files not listed in the spec, note this as a deviation.
- Workspace must compile after your changes.
