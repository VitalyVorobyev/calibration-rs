# P4-CRITERION — criterion benchmarks for the hot paths (2026-06-16)

## Context

The [from-scratch profiling report](2026-06-16-perf-from-scratch-rig-profiling.md)
fixed two dense-SVD hangs (P1) but the workspace had **zero** criterion
infrastructure — no `benches/`, no `[[bench]]` targets, no criterion dep — so
nothing guards against a regression reinstating the pathological path, and the
P1 gains were never quantified. A homography-DLT bench would have caught the
original hang outright.

## What changed

- `criterion = "0.5"` added to `[workspace.dependencies]` and as a **dev**-
  dependency of `vision-calibration-linear` and `vision-calibration-optim` (so
  it never enters the published dependency tree). Each crate gains a
  `[[bench]] … harness = false` target.
- **`vision-calibration-linear/benches/linear_init.rs`** — the linear-init hot
  paths, all on deterministic synthetic data:
  - `homography_dlt_225pts` — 15×15 grid through a known perspective `H` (the
    dense case that once hung >15 min).
  - `zhang_intrinsics_from_15h` — `estimate_intrinsics_from_homographies` over
    15 synthetic views.
  - `distortion_fit_12views_70pts` — `estimate_distortion_from_homographies`
    over a 10×7 board × 12 views (~840-row design — the F2 normal-equations
    path).
- **`vision-calibration-optim/benches/ba_iter.rs`** — `planar_intrinsics_ba_10views_70pts`,
  one full per-camera `optimize_planar_intrinsics` LM solve (10 views, ~700
  reprojection residuals) — the per-camera BA stage and the natural place to
  measure P3 backend work.

Benches follow the `criterion-bench` skill conventions: representative sizes,
deterministic (non-random) input, named by operation + size.

## Baseline (local, Apple M-series, quick-sample run)

| bench | time |
|-------|------|
| `homography_dlt_225pts` | **~8.6 µs** (was a >15 min hang on the old SVD path) |
| `zhang_intrinsics_from_15h` | ~1.1 µs |
| `distortion_fit_12views_70pts` | ~22.5 µs |
| `planar_intrinsics_ba_10views_70pts` | ~16.9 ms |

These are indicative (reduced warm-up/sample-size); `cargo bench` records the
authoritative baseline per machine.

## Verification

- `cargo bench -p vision-calibration-linear -p vision-calibration-optim
  --no-run` compiles both bench binaries (the CI-friendly guard).
- `cargo clippy --all-targets -D warnings` clean on both crates (benches
  included).
- `cargo build --workspace` green; `Cargo.lock` carries criterion. Benches are
  dev-only, so the published crates' dependency tree is unchanged.

## Follow-ups

- A criterion-based hot-path guard could be added to CI (`cargo bench --no-run`
  at minimum, or a regression-threshold run); deferred to the CI ratchet.
- The joint rig/hand-eye BA iteration is the other interesting target for P3 but
  needs the full rig dataset fixtures; the per-camera BA bench here is the
  representative single-stage proxy until then.
