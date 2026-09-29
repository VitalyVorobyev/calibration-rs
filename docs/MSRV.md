# MSRV

Workspace MSRV: **rustc 1.93**. The `rust-version` field in the root
`Cargo.toml` is the single source of truth.

The MSRV is enforced by the `MSRV (…)` CI job (`.github/workflows/ci.yml`),
which runs `cargo build` and `cargo test` with `--locked`, so CI uses the exact
versions committed to `Cargo.lock`.

## When to raise MSRV

We are pre-1.0 and do not commit to a specific floor; raise the MSRV
when:

- a transitive dep we want to track ships behind a newer toolchain, or
- a stable language feature would meaningfully simplify the code.

Pinning a dependency in `Cargo.lock` to stay under the floor is fragile (every
workspace-wide `cargo update` re-breaks it), so prefer raising the MSRV.

Bump `rust-version` in `Cargo.toml` *and* the `MSRV (…)` CI job in
`.github/workflows/ci.yml` together, in the same PR. The release
workflow (`release.yml`) and PyPI workflow (`release-pypi.yml`) both
publish with `--locked`, so the versions validated by CI are the
versions that ship.
