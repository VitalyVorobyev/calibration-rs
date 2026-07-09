# P1-SVD-SWEEP — finish the dense-SVD hang sweep + centralize math helpers (2026-06-16)

## Context

The [from-scratch profiling report](2026-06-16-perf-from-scratch-rig-profiling.md)
identified **systemic root-cause #1**: `nalgebra::svd(true, true)` is used
pervasively for null-space extraction and least-squares. On a tall/dense design
matrix it accumulates the U factor across thousands of rows *and* runs an
**unbounded** Golub-Kahan QR iteration (`max_niter = 0`) that can fail to
converge — hanging for minutes on real detected-corner data (homography DLT
>15 min, distortion fit >11 min before the in-session fixes). Crucially, the
earlier `svd(true, true) → svd(false, true)` ("skip U") mitigation is
**insufficient**: the non-convergence is in the QR sweep itself, which runs
regardless of `compute_u`.

Two confirmed hangs (homography DLT, distortion fit) were fixed in place in a
prior session. This task finishes the sweep across the remaining sites in
`vision-calibration-linear` and `vision-geometry` and centralizes the pattern so
future solvers reach for the hang-safe helper by default.

## What changed

### Centralized helpers (`math.rs` in both crates)

- **`null_space(&DMatrix) -> NullSpaceSolution`** — solves `A x = 0` as the
  smallest-eigenvalue eigenvector of the `k×k` normal matrix `AᵀA` (`k = cols`,
  ≤ 12 here) via `symmetric_eigen`, never `A.svd(...)`. `AᵀA` is small regardless
  of the row count and its symmetric eigensolve converges in a handful of sweeps.
  Returns the unit null vector **and** the descending singular values
  (`σ = √λ`) so rank guards (`dlt_rank_ok`) are preserved. NaN-safe: a
  non-finite `AᵀA` returns an error instead of panicking/hanging.
- **`ridge_lstsq(&DMatrix, &DVector, λ) -> DVector`** (linear crate) — solves the
  overdetermined `A x ≈ b` via the ridge normal equations `(AᵀA + λI) x = Aᵀb`
  (LU). Algebraically identical to an augmented `[A; √λ I]` least-squares but
  without a dense SVD.
- **`project_to_so3(&Mat3) -> Mat3`** (linear crate) — the polar-decomposition
  idiom (`R = U Vᵀ`, last-column sign flip for `det = +1`) that was copy-pasted
  across 5 sites. The SVD here is a fixed 3×3 (no hang risk).
- **`mat3_from_vec` / `mat34_from_vec`** — vector-valued reshapers for the
  null vector (analogous to the existing `mat*_from_svd_row`).

### Hang-risk null-space sites converted to `null_space`

| Site | Shape | Crate |
|------|-------|-------|
| `camera_matrix.rs` (`dlt_camera_matrix`) | `2N×12` | linear + vision-geometry |
| `pnp/dlt.rs` (`dlt`) | `2N×12` | linear |
| `pnp/epnp.rs` (`epnp`) | `2N×12` | linear |
| `epipolar/fundamental.rs` (`fundamental_8point`) | `N×9` | linear + vision-geometry |
| `epipolar/essential.rs` (`essential_linear`) | `N×9` | vision-geometry |
| `zhang_intrinsics.rs` (`from_homographies`) | `2M×6` | linear |
| `handeye.rs` (`estimate_rotation_allpairs`) | `4N×4` | linear |
| `homography.rs` (`dlt_homography`, was `svd(false,true)`) | `2N×9` | vision-geometry |

The wide `n == 8` row-padding in the fundamental/essential solvers is now
unnecessary (`AᵀA` is always 9×9) and was removed. Rank guards (`dlt_rank_ok`,
and homography's `sv[7]/sv[0]` check) now read `null_space`'s `singular_values`.

### Least-squares + SO(3) dedup

- `handeye.rs::ridge_llsq` → delegates to `ridge_lstsq` (dropped the augmented
  SVD solve).
- `handeye.rs::project_to_so3`, `pnp/dlt.rs`, `pnp/pose_utils.rs`,
  `planar_pose.rs` → delegate to `math::project_to_so3`.

Left as-is (correctly): fixed 9×9/10×10 minimal solvers (7-point fundamental,
5-point essential — bounded size), all 3×3 SVDs (essential-manifold projection,
RQ decomposition), and `vision-mvg/homography.rs`'s SO(3) projection (uses a
different `r = -r` determinant convention paired with its ± sign-ambiguity
logic — not a safe dedup, and a 3×3 with no hang risk).

## Numerics

`AᵀA` squares the design's condition number; on Hartley-normalized inputs this
is harmless and these linear estimates are refined downstream by BA. One test
tolerance was relaxed to match the method: `essential_linear`'s minimal 8-point
epipolar residual went `<1e-6` → `~7e-6`, so its bound is now `1e-4` (>10×
margin) — still an excellent seed.

## Verification

- `vision-calibration-linear` (46) + `vision-geometry` (42) + `vision-mvg` (44)
  unit tests, plus stereo integration tests — green. New direct tests for
  `null_space` (recovery + NaN-rejection), `ridge_lstsq` (exact solution +
  NaN-rejection), and `project_to_so3` (rotation recovery + reflection fix).
- Downstream `vision-calibration-optim` + `vision-calibration-pipeline` (incl.
  the 241-test pipeline suite with golden-value pins) — green, confirming the
  public API and production numerics are unchanged.
- `cargo clippy --all-targets -D warnings` and `RUSTDOCFLAGS=-D warnings cargo
  doc --no-deps` clean on both crates.

## Remaining / out of scope

The `triangulation.rs` `2N×4` sites in both crates are low-risk (`N` = view
count, typically ≤ a dozen) and were left on their existing SVD path; they can
adopt `null_space` opportunistically. The `linear`→`vision-geometry`
de-duplication (C1-FOLLOWUP) would later collapse the two parallel `math`
modules — and their now-duplicated `null_space`/reshaper helpers — into one.
