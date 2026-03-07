# Review: Check Implementation Quality

You are the reviewer for calibration-rs. Review recent changes against the task specification and quality gates.

## Input

Task ID or description of what to review: $ARGUMENTS

## Process

1. **Inspect changes**: Run `git diff` to see all uncommitted changes. If reviewing a recent commit, use `git show HEAD`.

2. **Check against spec**: If a task ID is given, read the spec from `docs/backlog.md` or the task report. Verify each acceptance criterion.

3. **Quality checks**:

   a. **Gates** — run and report results:
   ```
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   cargo doc --workspace --no-deps 2>&1 | grep -E "warning|error" | head -20
   ```

   b. **API design**:
   - Are public type/function names consistent with existing patterns?
   - Are new public items documented with rustdoc?
   - Do new types derive the expected traits (Debug, Clone, Serialize, Deserialize)?
   - Is the facade re-export clean and module-first?

   c. **Architecture**:
   - Do changes respect crate layering (AGENTS.md)?
   - Are there unnecessary dependencies added?
   - Is code in the right crate?

   d. **Testing**:
   - Do new algorithms have synthetic ground-truth tests?
   - Do new config/export types have JSON roundtrip tests?
   - Are edge cases covered?

   e. **No unintended changes**:
   - Are there files modified that shouldn't be?
   - Are there debug prints or TODO comments left in?

4. **Verdict**: Output one of:

   **APPROVED** — All criteria met, gates pass, code is clean.

   **APPROVED WITH NOTES** — All criteria met but minor suggestions (list them). OK to commit.

   **REJECTED** — Criteria not met or gates fail. Provide specific feedback:
   - What failed
   - What needs to change
   - Suggested fix if obvious

## Guidelines

- Be specific in feedback — reference file:line
- Don't nitpick style if `cargo fmt` passes
- Focus on correctness, API design, and spec compliance
- Breaking changes are allowed this release cycle
