# MSRV

Workspace MSRV: **rustc 1.93** (`rust-version = "1.93"` in `Cargo.toml`).

The MSRV is enforced by the `MSRV (1.93)` CI job
(`.github/workflows/ci.yml`) which runs `cargo build` and `cargo test`
with `--locked`. `--locked` means CI uses the exact versions committed
to `Cargo.lock`.

## History

- **2026-05-23 — raised 1.88 → 1.93.** The previous 1.88 floor required
  pinning `fixed` to 1.30.0 and `kiddo` to 5.2.4 in `Cargo.lock`
  because their newer releases (`fixed 1.31.0` MSRV 1.93, `kiddo 5.3.0`
  uses non-const `f64::fract()` requiring 1.89) outran our toolchain.
  Both crates were dev-deps only, but every workspace-wide
  `cargo update` re-broke the pins, so we bumped MSRV to 1.93 instead.
  See commit history for `Cargo.toml` and `Cargo.lock`.

## When to raise MSRV again

We are pre-1.0 and `release_goal_v1.md` does **not** commit to a
specific floor; raise the MSRV when:

- a transitive dep we want to track ships behind a newer toolchain, or
- a stable language feature would meaningfully simplify the code.

Bump `rust-version` in `Cargo.toml` *and* the `MSRV (…)` CI job in
`.github/workflows/ci.yml` together, in the same PR. The release
workflow (`release.yml`) and PyPI workflow (`release-pypi.yml`) both
publish with `--locked`, so the versions validated by CI are the
versions that ship.
