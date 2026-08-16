# Backlog

Open and parked implementation tasks. One task per commit; on completion,
mark it `[x]` with a dated one-paragraph note, then move the entry to
[`backlog-archive.md`](backlog-archive.md)
so this file stays a picture of what is left rather than what was done
(AGENTS.md §11). `[~]` means parked — a deliberate decision not to pursue it
now, with the reason recorded.

Strategy and phase gates live in the [ROADMAP](ROADMAP.md); design decisions
in [ADRs](adrs/).

## B-QUAL / B-UX / B-DIST — app to production grade (Phase III)

- [ ] B-QUAL-HOOKS7 - `eslint-plugin-react-hooks` 7 (adopted 2026-08-11
  with the eslint 10 bump) ships two new rules, both set to `warn` in
  `app/eslint.config.js` rather than blocking CI:
  - `react-hooks/set-state-in-effect` — five real sites
    (`useImageData.ts` ×2, `DiagnoseWorkspace/index.tsx` ×3) reset derived
    state from an effect when their input changes. Correct, but one render
    pass more than keying the component would cost. Rewrite when touching
    those components anyway.
  - `react-hooks/purity` — one report, a **false positive**: the
    `Date.now()` stamping a run's start time in RunWorkspace's async
    submit handler. The rule cannot distinguish an event handler from
    render. Re-check on plugin updates; drop the override if it learns to.
- [ ] B-QUAL-TS7 - TypeScript 7 is **blocked upstream**: `typescript-eslint`
  hard-errors on it (`typescript-eslint does not support TS 7.0`, tracking
  issue typescript-eslint#10940). Adopting it today means dropping the
  type-aware lint config B-QUAL1-LINT-CI deliberately built. Pinned at
  TypeScript 6; revisit when typescript-eslint ships TS 7 support.
- [ ] B-UX2-ELEVATION - Workspace-by-workspace elevation: empty states, error
  surfaces (ADR 0019 fail-fast shown well), progress for long calibrations,
  manifest-sniff UX polish. Absorbs **B-LASER** (laser-pixel overlay in
  Diagnose compare mode, point-to-plane mm panel, single-cam laser plane in
  3D) and **B-EXPLORE** (pre-calibration image grid, detection-cache overlay,
  coverage map) as scoped sub-items. Gate: frontend-review pass with no
  high-severity findings.
  Shipped so far: stage-progress streaming and cancellation for long solves,
  a sortable per-pose residual table, a cross-camera residual matrix, and the
  single-camera laser plane in the 3D viewer. Still open: the pre-calibration
  image grid, detection-cache overlay and coverage map, plus empty-state,
  error-surface and manifest-sniff polish.

## V — rtv3d validation

- [~] V7-RTV3D-INTRINSICS-FLOOR - **PARKED** (user call 2026-06-14;
  disposition confirmed 2026-07-02). Drive the rtv3d from-scratch
  reprojection floor below 0.4 px, or prove the blocking term — isolated to
  detector/target/model, not rig-chain. Seeded init (ADR 0022) is the
  official acceptance path. **Do not reopen unprompted.**

## P — Performance & profiling

- [ ] P2-BA-DENSITY - **Conditional (2026-07-02):** schedule only if the S4
  acceptance-runner wall time for the 6-camera rtv3d rig is painful (>~5 min);
  otherwise stays parked. Principled corner budget for the joint rig +
  hand-eye BA (spatially-distributed subsample preserving coverage, or
  per-stage decimation knobs). Extrinsics/hand-eye converge on a fraction of
  the corners; the per-camera intrinsics stage already uses a cheap subsampled
  tilt sweep + a full-data refine (`optimize_scheimpflug_intrinsics_staged`).
- [~] P3-BACKEND-COST - **Parked post-1.0 (2026-07-02)** — Track O's
  second-backend premise is dead. Would profile the tiny-solver
  autodiff/assembly/solve split and evaluate analytic Jacobians / caching /
  rayon parallelism for the hot `ReprojPoint` factor.

## M — camera models (gated on M0)

- [~] M4-FISHEYE - **Parked post-1.0 (2026-07-02)** — no fisheye dataset in
  the acceptance set. Would add Kannala-Brandt equidistant k1–k4 as a new
  `ProjectionModel`.

## C — MVG (multiple-view geometry)

- [ ] C-PYO3-MVG - PyO3 bindings for the MVG surface. **Deferred** — the
  Python crate binds the calibration facade only, and no Python consumer
  for two-view/N-view geometry exists. Revisit if one appears.

## D — Earn v1.0

- [ ] D4-NALGEBRA-035 - **Blocked on `tiny-solver`.** `nalgebra` 0.35,
  `faer` 0.24 and `faer-ext` 0.8 are all unadoptable while `tiny-solver`
  0.18 (latest) is built against 0.34 / 0.23 / 0.7:
  `vision-calibration-optim` passes both nalgebra and faer types straight
  across that boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`), so a bump
  produces ~33 trait-mismatch errors. `calib-targets` 0.12 already uses
  nalgebra 0.35 internally — harmless, because `vision-calibration-detect`
  is nalgebra-free and converts to plain arrays at its boundary. **Keep
  that property**: it is what lets the two nalgebra versions coexist.
  Re-check on each `tiny-solver` release; the alternative is replacing the
  solver backend (see O-track).
- [ ] D4-RELEASE - v1.0 gate — **checklist replaced 2026-07-02** by the
  production-grade program exit criteria (ROADMAP): S4 acceptance command
  green on all on-disk datasets, Q proof packs complete, R-track API/config
  freeze done, app CI green (B-QUAL), docs current, plus the standing
  requirement that the API has been stable across two minor releases.

## B — app (extend; sequencing serves V-track)

- [~] B-EXPLORE - **Re-scoped into B-UX2-ELEVATION** 2026-07-02: per-camera/
  pose image grid, detection-cache overlay, board coverage map.

## Closed tracks

Fully delivered; notes in the
[archive](backlog-archive.md).

- **S — Device-spec → seed initialization (Phase I)** — 4 tasks.
- **Q — Algorithmic soundness: proofs + regression (Phase I)** — 8 tasks.
- **R — API/config/design revision (Phase II)** — 7 tasks.
- **O — apex-solver backend (O1/O2 won't-do 2026-07-04; O3 done)** — 3 tasks.
- **M — camera models** — the wiring; only fisheye is parked.
- **C — MVG** — everything but the deferred PyO3 bindings above.
- **Benchmark** — 6 tasks.
