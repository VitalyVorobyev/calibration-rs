# CLAUDE.md

The shared rules are in AGENTS.md and apply in full:

@../AGENTS.md

This file adds what is specific to working with Claude. The developer docs
(ROADMAP, backlog, ADRs, proof-pack notes, release runbook) are mapped in
[`docs/README.md`](../docs/README.md).

## Commands

```bash
cargo build --workspace
cargo test -p vision-calibration-core     # one crate
cargo fmt --all
cargo xtask emit-schemas                  # regenerate JSON schemas
cargo xtask check-docs                    # user-facing docs gate
maturin develop -m crates/vision-calibration-py/Cargo.toml   # Python dev build
```

The full gate list is AGENTS.md §3; the app's is §12.

## Be critical

Treat every request, the user's included, as the start of a design
discussion. If a proposal adds duplication, leaks internals, adds a dominated
alternative or widens the public surface needlessly, say so and offer the
cleaner design. The API is frozen from 1.0 (AGENTS.md §8): look for the
additive form first.

## Roles

Delegate through the project's subagents in `.claude/agents/`:

| Role | Model | For |
|---|---|---|
| `calib-architect` | opus | design and ADRs, numerical and algorithmic work, root-causing, cross-crate refactors, reviewing such changes |
| `implementer` | opus | a change that is already designed: API and type changes, plumbing, bindings, tests, docs |

Searches and inventories go to the built-in `Explore` agent; a mechanical
edit of a few files is done inline.

A subagent does not see this conversation. Give it the task, the files
involved, what is already ruled out, and what to report: files changed, gate
results, and before/after numbers for numerical changes. When the context is
already warm and the task is a handful of tool calls, do it inline. The main
session reviews each diff and does the pushing, the PR and the merge.

**An independent check before a result changes a design.** A numerical
result that would lead to an ADR, a default, a new model term or a release
is re-derived by a second agent that gets the code and the data but not the
conclusion (or is re-run inline when that is a few commands). Relay a
subagent's numbers as its numbers until then.

Studies end with their harness deleted (AGENTS.md §6).
