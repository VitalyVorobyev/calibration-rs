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
cleaner design. Pre-1.0, prefer the better design even when it breaks the
API, and record the break in the CHANGELOG with migration notes (§8).

## Roles

Delegate through the project's subagents in `.claude/agents/`:

| Role | Model | For |
|---|---|---|
| `calib-architect` | opus | design and ADRs, numerical and algorithmic work, root-causing, cross-crate refactors, reviewing such changes |
| `implementer` | sonnet | a change that is already designed: plumbing, serde types, bindings, tests, docs |
| `sweeper` | haiku | searches, inventories, mechanical edits with an exact spec |

A subagent does not see this conversation. Give it the task, the files
involved, what is already ruled out, and what to report: files changed, gate
results, and before/after numbers for numerical changes. When the context is
already warm and the task is a handful of tool calls, do it inline.
