# MSRV and pinned transitive dependencies

Workspace MSRV: **rustc 1.88** (`rust-version = "1.88"` in `Cargo.toml`).

The MSRV is enforced by the `MSRV (1.88)` CI job
(`.github/workflows/ci.yml`) which runs `cargo build` and `cargo test`
with `--locked`. `--locked` means CI uses the exact versions committed
to `Cargo.lock`. Running `cargo update` can quietly bump a transitive
dep past our MSRV — the entries below are the pins that exist **solely**
to stay on 1.88.

## Pinned transitive deps

| Crate | Pinned to | Latest | Reason |
|-------|-----------|--------|--------|
| `fixed` | `1.30.0` | `1.31.0` | `fixed 1.31.0` declares `rust-version = "1.93"`. `kiddo ^5` accepts `fixed ^1`, so `1.30.0` (MSRV 1.85) works. Pulled via `kiddo → calib-targets-chessboard` (dev-dep only). |
| `kiddo` | `5.2.4` | `5.3.0` | `kiddo 5.3.0` calls non-const `f64::fract()` inside a const context (`kiddo/src/float/distance.rs:265`), which requires rustc ≥ 1.89. Pulled via `calib-targets-chessboard` (dev-dep only). Pinning also transitively holds `cmov` at `0.4.6` (not `0.5.3`). |

Both crates are reachable only as **dev-dependencies** of
`vision-calibration`, so these pins do not affect downstream consumers
of the published crates.

## Before running `cargo update`

```bash
# Safe: runs with the committed lockfile — cannot drift.
cargo build --workspace --locked
cargo test --workspace --locked

# Dangerous: may bump transitive deps past MSRV silently.
cargo update

# If you must update, re-pin the known-bad versions afterwards:
cargo update
cargo update -p fixed --precise 1.30.0
cargo update -p kiddo --precise 5.2.4

# Then verify on MSRV:
rustup toolchain install 1.88.0   # if needed
cargo +1.88.0 test --workspace --locked --no-run
```

If `cargo +1.88.0 test --locked --no-run` passes, CI will too.

## When to drop a pin

A pin becomes unnecessary once **either** of these holds:

1. The upstream crate releases a newer version whose MSRV is ≤ our MSRV.
2. We raise the workspace MSRV past the problematic version.

At that point, drop the precise pin with `cargo update -p <crate>` and
update this table. The 0.3.0 cycle intentionally stays on MSRV 1.88 to
avoid churning consumers; revisit when planning the next MSRV bump.

## Hardening recap

- Workspace declares `rust-version = "1.88"` so `cargo` refuses to
  compile on older toolchains.
- The `MSRV (1.88)` CI job uses `--locked` so it fails loudly on
  lockfile drift.
- `release.yml` publishes with `--locked` so the versions we validated
  are the versions that ship.
