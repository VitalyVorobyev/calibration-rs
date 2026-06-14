# C1: land vision-geometry + vision-mvg (fresh additive port)

**Date:** 2026-06-14
**Scope:** Track C / C1 — bring multiple-view geometry into the workspace.

## Decision

The `mvg` branch (PR #28) had diverged ~136 commits behind `main`: it predates
`vision-calibration-detect`, `-dataset`, `-bench`, and the Tauri app, and
carried a `vision-calibration-linear` refactor (moving solvers out of `linear`)
that conflicts with the many `main` callers added since. Merging it was deemed
high-risk. Instead, the two crates were **ported fresh and additively** from the
branch source — `git checkout mvg -- crates/vision-geometry crates/vision-mvg` —
onto current `main`, with **no** `vision-calibration-linear` change. See
[ADR 0015](../adrs/0015-mvg-ceiling.md).

## What landed

- **`vision-geometry`** (deterministic solvers; deps: core + nalgebra + anyhow):
  `math`, `epipolar/{fundamental,essential,decomposition,polynomial}`,
  `homography`, `triangulation`, `camera_matrix`. **20 unit tests pass.**
- **`vision-mvg`** (pipelines/robust over the solvers; deps: vision-geometry +
  core + optional tiny-solver): `pose_recovery`, `robust`, `cheirality`,
  `degeneracy`, `triangulation`, `residuals`, `homography`, `types`, optional
  nonlinear `refine` (feature `refine` → tiny-solver). **31 unit tests pass**
  (incl. `--all-features`).
- Workspace wiring: both added to `[workspace] members` + `[workspace.dependencies]`.
- Both `publish = false` (internal-first; promote deliberately later).

## Verification

The crates built against current `core` with **zero API drift** (the only core
symbols used are the stable `Pt3`/`Vec3` aliases). Gates: `cargo build`,
`cargo test -p vision-geometry -p vision-mvg --all-features` (51 tests),
`cargo clippy -p vision-geometry -p vision-mvg --all-targets --all-features -D
warnings`, and the full `cargo test --workspace --all-features` (no regression)
all green. (A transient `cc` linker error appeared once when two
`cargo test --workspace` ran concurrently; a clean single run passed.)

## Deferred (C1-FOLLOWUP)

- De-duplicate `vision-calibration-linear` onto `vision-geometry` (have `linear`
  depend on the new crate, drop the duplicated epipolar/homography/triangulation
  code). Kept out of C1 to stay additive / zero-regression.
- Promote both crates to the crates.io publish set + release version-lockstep
  (CLAUDE.md) once the API stabilises.
- PyO3 bindings (deferred per Track A5).
- Close the stale `mvg` branch / PR #28.
