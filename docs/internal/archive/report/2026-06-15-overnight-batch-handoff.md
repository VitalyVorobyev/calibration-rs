# Overnight autonomous batch — handoff (2026-06-14 → 2026-06-15)

**Mandate.** Work the open roadmap tracks autonomously overnight; pick the
order; merge to `main` **sequentially** (few branches, no parallel sprawl);
gate every merge on **CI green + codex-reviewed with findings addressed**; do
not wait for the user's review. Pre-1.0, breaking changes acceptable; quality
over velocity.

**Order chosen:** M → C → O → B-INFRA (additive/low-risk first; the apex-solver
pre-verify and the app test slice last). All work is **strictly additive** or
**dead-code removal** — no validated production path (rtv3d / puzzle rig) was
touched.

## What landed

| Track | PR | Squash | Outcome |
|-------|----|--------|---------|
| **M** — camera-model distortion expansion | [#61](https://github.com/VitalyVorobyev/calibration-rs/pull/61) | `74263be` | 3 additive distortion models: rational (k4–k6), thin-prism (s1–s4), Fitzgibbon division. Core runtime + optim IR/backend layers; ZST kernels; synthetic-GT tests. Brown-Conrady paths byte-identical. |
| **C** — MVG crates | [#62](https://github.com/VitalyVorobyev/calibration-rs/pull/62) | `314705f` | Fresh additive port (NOT the stale `mvg` branch): `vision-geometry` (deterministic solvers — epipolar/F/E, homography, triangulation, camera DLT) + `vision-mvg` (robust, cheirality, degeneracy, two-view triangulation). Both `publish = false`. ADR 0015 caps the ceiling. |
| **O** — optimization backend | [#63](https://github.com/VitalyVorobyev/calibration-rs/pull/63) | `e7ef5ad` | apex-solver **parked** (decision below); dead `BackendKind::Ceres` stub + orphaned `Error::numerical` helper removed (O3). |
| **B-INFRA** — app test infra | [#64](https://github.com/VitalyVorobyev/calibration-rs/pull/64) | `e0e63e1` | Vitest harness + 18 pure-logic tests (`inferExportKind` / `exportKindLabel` / `mergeConfig`). Safe subset; rest of B-INFRA deferred. |

Per-task detail reports live alongside this one in `docs/report/`
(`…-M-distortion-models.md`, `…-C1-mvg-crates.md`,
`…-O1-apex-solver-preverify.md`, `…-O3-CERES-drop-ceres-stub.md`,
`…-B-INFRA-vitest-unit-slice.md`).

## The one real judgment call — Track O / apex-solver (PARKED)

The roadmap premise was "apex-solver gets a second real `OptimBackend`." The O1
pre-verify gate **failed on the load-bearing requirement**: apex-solver 1.3 is a
**hand-Jacobian factor-graph library** (`Factor::linearize` returns a
caller-supplied Jacobian; no generic scalar, no dual numbers, no autodiff). Our
IR (ADR 0008) and the M0 factor generification (ADR 0020) are **autodiff-first**
(`fn residual<T: RealField>()`, monomorphized per `CameraModelDesc`). Wiring
apex-solver would mean hand-deriving Jacobians for every factor family — which
defeats M0 and adds correctness risk at exactly the geometry we most need to
trust. Secondary gaps: no S2/unit-vector manifold (we use one for laser-plane
normals), no documented robust losses, undocumented SE3 quaternion order.

**No backend code was written.** O1/O2 are parked. **Reviving Track O is a user
call:** pick an autodiff-capable Rust optimizer (e.g. a `factrs`/`num-dual`
stack), or keep tiny-solver as the sole backend and drop Track O. Full findings:
`docs/report/2026-06-14-O1-apex-solver-preverify.md`.

## Codex review summary

Every PR was codex-reviewed (auto on push + `@codex review` re-trigger) and
merged only after findings were addressed and a clean re-review.

- **C (#62)** was the deep one: codex ran a methodical file-by-file
  degenerate-input audit — **7 rounds, 16 P2s, all fixed**, converging to a
  clean round 7. The recurring class (homogeneous-DLT rank deficiency) was
  root-caused with a single shared `math::dlt_rank_ok` guard applied uniformly
  across fundamental 7/8-pt, `essential_linear`, triangulation, and camera DLT.
- **O (#63)**: 1 P2 — a process finding (AGENTS.md §11 requires a per-task
  report for the completed O3). Added the O3 report; clean re-review.
- **M (#61), B-INFRA (#64)**: findings (if any) addressed; see each PR.

## B-INFRA — what's deferred (B-INFRA stays open)

The Vitest unit slice landed. Deferred, with sub-items in `docs/backlog.md`:
- **ts-rs/specta codegen + export discriminator tag (F6)** — risky wire change
  that replaces the `inferExportKind` shape-probe with a Rust-emitted tag. The
  new tests are its conformance spec. Needs supervised design.
- **`resource_dir` presets** — `RunWorkspace/presets.ts:16` still hard-codes the
  repo root.
- **Playwright smoke tests** — need a Tauri/webview harness.
- **App CI job** — the Rust CI workflow does not build `app/`, so the Vitest
  tests are a local/dev safety net (`bun run test`), not yet a CI gate.

## State after the batch

- `main` carries M + C + O + B-INFRA (all four merged).
- The stale `mvg` branch is **untouched** (the user flagged it as far behind;
  Track C was done fresh instead of merging it). It can be deleted at leisure.
- Roadmap/backlog updated: M1–M3 done, C1 landed, O1/O2 parked + O3 done,
  B-INFRA partially done.

## Recommended next steps (for the user)

1. **Decide Track O's fate** — autodiff-capable second backend vs. tiny-solver
   as sole backend (then close or re-scope Track O).
2. **M-WIRE** — the user-facing pipeline-selection plumbing for the new
   distortion models (config → builder, fix-mask generalization,
   init/pack/export). Deliberately left for a supervised slice.
3. **B-INFRA discriminator** — the deferred ts-rs wire change, now backed by a
   test spec.
4. Optionally delete the stale `mvg` branch.
