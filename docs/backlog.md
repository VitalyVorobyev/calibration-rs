# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §9); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the v1.0 criteria these serve.

## App

- [ ] B-UX2-ELEVATION - Workspace-by-workspace polish: empty states, error
  surfaces (fail-fast errors from ADR 0019 shown well), manifest-sniff UX,
  and three exploration views — a pre-calibration per-camera/pose image
  grid, a detection-cache overlay, and a board coverage map. Gate: a
  frontend review with no high-severity findings.
- [ ] B-QUAL-TS7 - Move to TypeScript 7. **Blocked**: `typescript-eslint`
  rejects TS 7 (typescript-eslint#10940), and adopting it would drop the
  type-aware lint config. Revisit when typescript-eslint supports TS 7.

## Solver backends

- [~] O-FACTRS-PARALLEL - Parallel linearization for factrs, proposed
  upstream as rpl-cmu/fact-rs#55 with PR rpl-cmu/fact-rs#56: `Sync` next
  to the `Send` bounds from fact-rs #47, and rayon in `Graph::linearize` /
  `Graph::error` under factrs' existing `rayon` feature.
  - **Measured** on 2026-10-10 at `00a8ab00`, M4 Pro, 12 threads:
    `calib-bench solver run --preset full --backend factrs`, with factrs
    patched to the fork's `calib-rs/parallel-on-0.3.0`.
    - 3.9× over serial factrs (geometric mean; 2.6–5.1× by workflow).
    - Bit-identical final costs and iteration counts.
    - +0.4 % on one thread, about +5 % on the smallest scenes: rayon's
      handoff costs about 7 µs per call.
  - **Method and tables**: `REPORT.md` on the fork's
    `notes/parallel-linearization` branch.
  - **Blocked on factrs upstream**: the merge, then a release. fact-rs #55
    asks for 0.4.0 instead of the pending 0.3.1, because fact-rs #47, #51
    (faer 0.24) and #56 are all breaking.
  - **Trigger**: that release. Then, in one commit:
    - Require it in the root `Cargo.toml` with `features = ["rayon"]`. The
      feature is off today, and without it nothing runs in parallel.
    - Take its Jacobian as a faer-0.23 view (D4-NALGEBRA-035).
    - Rerun `calib-bench solver` for both backends.
- [ ] O-TINYSOLVER-PERF - Adopt the tiny-solver speed-ups once released.
  On the tiny-solver fork, on 0.18.3:
  - `perf/assemble-without-mutex` (A): residuals and Jacobian values
    assembled without the `Mutex`. Internal, fits 0.18.x, 1.18× on 12
    threads.
  - `perf/symbolic-structure` (C, on A): the Jacobian pattern laid out by
    a counting sort, with one lookup per variable instead of one per entry
    plus a sort, and the values scattered in place. Internal, fits 0.18.x.
    `build_symbolic_structure` was 18 % of a 12-thread solve with A+B; C
    makes A+B solves 1.20× faster.
  - `perf/stride4-stack-duals` (B): stack duals in passes of 4 directions,
    as in Ceres' `DynamicAutoDiffCostFunction`. It adds supertraits to
    `FactorImpl` and `Manifold`, so it is a 0.19 change upstream.

  With A+B+C (local branch `calib-rs/perf-abc`), solves are 3.9× faster
  than with 0.18.3 on 12 threads (1.8–5.8× per scene) and 1.5× faster than
  parallel factrs, with bit-identical results and no code change here.
  **Trigger**: the tiny-solver release that carries them. Bump the floor in
  the root `Cargo.toml`, then rerun `calib-bench solver` to confirm.

## Calibration quality

Found by `calib-bench solver` (well-conditioned synthetic scenes; the named
scenes are the gate).

- [ ] Q-LASERLINE-OUTLIERS - With 5 % of target corners displaced 10–30 px,
  `laserline_device` under Huber or Cauchy (scale 1 px) settles off the
  minimum: inlier RMS 0.16–0.30 px at σ 0.1 px (floor 0.14), focal error
  0.4–1.8 %, laser plane up to 0.29° / 7.6 mm. The other problems (the
  Scheimpflug rig aside, below) reach the floor on the same contamination,
  and the robust-cost LM did not change it, so the init is the suspect. Gate:
  `laserline_device/pinhole/*/n0.1/{huber,cauchy}` at the noise floor.
- [ ] Q-SCHEIMPFLUG-RIG-PERCAM - The Scheimpflug rig's per-camera stage
  (staged init, radial-only BA) lands off the minimum on clean data, even
  seeded with each camera's nominal tilt: inlier RMS 0.21–0.26 px at
  σ 0.1 px, principal point 26–44 px and tilt about 2.5° off. The rig BA
  keeps intrinsics fixed, so it cannot recover; the single-camera Scheimpflug
  problem reaches the floor on comparable data (principal point 1.4 px).
  Gate: `rig_extrinsics/scheimpflug/*/clean` at the noise floor.

## v1.0

- [ ] D4-RELEASE - Cut v1.0 once the ROADMAP exit criteria hold.
- [ ] D4-NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 /
  `faer-ext` 0.8. **Blocked on both backends**, still built against
  0.34 / 0.23 / 0.7: `vision-calibration-optim` passes nalgebra and faer
  types across the tiny-solver boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`) and the factrs
  one (its variables and residual traits), so a bump fails to compile.
  `vision-calibration-detect` must stay nalgebra-free (it converts to plain
  arrays at its boundary); that is what lets `calib-targets`' nalgebra 0.35
  coexist. Re-check on each tiny-solver and factrs release (tiny-solver
  0.18.3 and factrs 0.3.0 are on 0.34 / 0.23 / 0.7).
  - factrs `main` is already on faer 0.24 / faer-ext 0.8 (nalgebra 0.34).
    The shared LM owns its sparse solve (faer 0.23 directly) and reads a
    Jacobian only as its pattern and values (`backend/normal_equations.rs`).
    So the next factrs release needs the factrs engine to hand those slices
    across as a faer-0.23 matrix: a zero-copy view, not a blocker.

## Deferred and parked

- [ ] BENCH-DETECT-DEDUP - `vision-calibration-bench`'s Tier-B adapters
  (`src/detect.rs`: chessboard, ChArUco, puzzleboard) duplicate the
  detectors in `vision-calibration-detect`; route them through the detect
  crate so detector changes land once. Unpublished crate, so not urgent.

- [ ] C-PYO3-MVG - Python bindings for the MVG surface. Deferred: the
  Python package binds the calibration facade only, and nothing in Python
  needs two-view/N-view geometry yet.
- [ ] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Schedule only
  if the acceptance run for the six-camera rtv3d rig exceeds ~5 min.
- [~] P3-BACKEND-COST - Close the remaining gap to Ceres. Planar
  intrinsics, against Ceres 2.2 at matched tolerances (2026-10-06):
  - today we are 7–9× slower on 12 threads and about 21× slower on one
    thread;
  - with tiny-solver A+B+C we are 2.6× slower on 12 threads and 4.5×
    slower on one.

  After O-TINYSOLVER-PERF, what remains is linearization:
  per-block library overhead, num-dual against Ceres' `Jet`, and
  differentiation through the SE(3) retraction. An analytic `ReprojPoint`
  Jacobian is the next lever. Parked until after 1.0. The harness and the IR
  export are on the local branch `exp/solver-perf`.
- [~] M4-FISHEYE - Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`. Parked until after 1.0; no fisheye dataset exists in
  the acceptance set.
- [~] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by user
  decision — **do not reopen unprompted.**
