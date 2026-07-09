# O3-CERES: drop the dead `BackendKind::Ceres` stub

**Date:** 2026-06-15
**Task:** `O3-CERES` (docs/backlog.md → Track O)
**Scope:** Remove the unused `BackendKind::Ceres` placeholder backend and the
helper it orphaned. Pure dead-code removal — no behavior change on any live
path.

## Scope (what changed)

`BackendKind::Ceres` was a never-constructed placeholder variant whose only use
was a dispatch arm in `solve_with_backend` that returned
`Err(Error::numerical("Ceres backend not implemented"))`. No caller ever
selected it (the whole codebase only ever uses `BackendKind::TinySolver`), so the
arm was unreachable and the variant was dead API surface.

- Removed the `BackendKind::Ceres` variant and its doc comment.
- Removed the unreachable `BackendKind::Ceres => Err(...)` arm in
  `solve_with_backend`. `BackendKind` is now a single-variant enum and the
  `match` is exhaustive with one arm.
- Removed the now-orphaned `Error::numerical` convenience constructor
  (`pub(crate)`, its only caller was the deleted arm). The `Error::Numerical`
  variant itself stays — it is still constructed by `From<anyhow::Error>`.
- Trimmed the stale "or the requested backend is not available" clause from the
  `solve_with_backend` `# Errors` doc, since that failure path no longer exists.

This is the independent half of Track O. O1/O2 (the apex-solver second backend)
are **parked** on an autodiff API mismatch — see
`docs/report/2026-06-14-O1-apex-solver-preverify.md` and the Track O section of
`docs/ROADMAP.md` / `docs/backlog.md`.

## Files changed

- `crates/vision-calibration-optim/src/backend/mod.rs` — drop variant + doc +
  dispatch arm; trim `# Errors` doc.
- `crates/vision-calibration-optim/src/error.rs` — drop the orphaned
  `Error::numerical` helper.
- `docs/ROADMAP.md`, `docs/backlog.md` — mark O1/O2 PARKED, O3 DONE.
- `docs/report/2026-06-14-O1-apex-solver-preverify.md` — committed (the O1
  pre-verify findings; previously untracked).

## Validation run

- `cargo build -p vision-calibration-optim` — pass.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — pass.
- `cargo fmt --all -- --check` — pass.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p vision-calibration-optim --all-features --no-deps`
  — pass (no broken intra-doc links from the trimmed doc).
- `cargo test --workspace --all-features` — pass (45 suites, 0 failures).

## Follow-ups / remaining risks

- **None for O3.** It is pure removal; no live path changes.
- `BackendKind` being a single-variant enum is intentional. If/when a second
  backend lands (a revived Track O with an autodiff-capable optimizer), it
  re-grows a variant and the dispatch arm comes back. Reviving Track O is a user
  call (autodiff-capable optimizer vs. tiny-solver as sole backend).
