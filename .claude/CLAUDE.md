# CLAUDE.md

Shared rules — layering, where code goes, quality gates, testing, docs,
backlog workflow, desktop app — live in AGENTS.md and apply here in full:

@../AGENTS.md

This file adds what is specific to working with Claude.

## Commands

```bash
cargo build --workspace
cargo test -p vision-calibration-core     # one crate
cargo fmt --all
cargo xtask emit-schemas                  # regenerate JSON schemas
cargo xtask check-docs                    # user-facing docs gate
maturin develop -m crates/vision-calibration-py/Cargo.toml   # Python dev build
```

App commands and the bun-only rule: AGENTS.md §10.

## Engineering principles & critical review

Hold every change to a high design bar, and keep the workspace from
drifting into undisciplined, copy-paste code:

- **SOLID** — one responsibility per type/module; extend behavior through
  traits (`Detector`, `ProblemType`, the camera-model kernels, the solver
  backends), not by editing parallel match arms; depend on abstractions,
  not concretions.
- **DRY / single source of truth** — one canonical definition per concept.
  Lower config to core params in exactly one place; never let a
  threshold's or parameter's meaning diverge across crates.
- **KISS & YAGNI** — the simplest design that meets the actual
  requirement; no config knobs, enum variants or abstraction layers for
  hypothetical needs.
- **Make illegal states unrepresentable** — enum-with-payload over a
  discriminator plus parallel fields; `Option` / newtypes over magic
  sentinels.
- **Least astonishment & orthogonality** — APIs do what their names say;
  independent concerns stay independent.
- **Minimal, honest public surface** — expose only what callers need;
  diagnostics and internals stay out of the stable API (`pub(crate)`,
  sealed traits, curated re-exports).

**Be critical of every proposal, including the user's.** Treat a request
as the start of a design discussion. If it would add duplication, leak
internals, add a dominated alternative or widen the public surface
needlessly, say so and offer the cleaner design. We are pre-1.0: when the
better design needs a breaking change, prefer it and record it in the
CHANGELOG with migration notes.

## Design in brief

- **Camera model** ([ADR 0005](../docs/adrs/0005-composable-camera-model.md)):
  `pixel = K(sensor(distortion(projection(dir))))`, as
  `Camera<S, P, D, Sm, K>`.
- **Sessions** ([ADR 0007](../docs/adrs/0007-session-framework.md)): every
  workflow is `CalibrationSession<P: ProblemType>` driven by free step
  functions (`set_input` → `step_init` → `step_optimize` → `export`).
  Eight problem types: `PlanarIntrinsics`, `ScheimpflugIntrinsics`,
  `SingleCamHandeye`, `LaserlineDevice`, `RigExtrinsics`, `RigHandeye`,
  `RigLaserlineDevice`, `RigHandeyeLaserline`; the rig types cover pinhole
  and Scheimpflug rigs through `SensorMode`.
- **Optimization IR** ([ADR 0008](../docs/adrs/0008-backend-agnostic-optimization-ir.md)):
  problems are `ProblemIR` (parameter + residual blocks) compiled to a
  solver backend; factors are generic over `T: RealField` for autodiff
  (`.clone()` freely, `T::from_f64().unwrap()` for constants).
- **Conventions** ([ADR 0009](../docs/adrs/0009-coordinate-and-pose-conventions.md)):
  `frame_se3_frame` naming (`T_C_W` = world → camera); SE3 stored as
  `[qx, qy, qz, qw, tx, ty, tz]`.
- **Numerics**: Hartley normalization for DLT, robust losses for outliers,
  Lie-group manifolds for rotations; `fix_k3: true` by default.
- **Tests**: synthetic ground truth for algorithms, JSON round-trips for
  config/export types; ~5 % tolerance for linear init, < 1 % after
  optimization.

## Where things are

- [`docs/backlog.md`](../docs/backlog.md) — what is left;
  [`docs/ROADMAP.md`](../docs/ROADMAP.md) — v1.0 exit criteria and standing
  decisions.
- [`docs/adrs/`](../docs/adrs/README.md) — design decisions;
  [`docs/notes/`](../docs/notes/README.md) — proof packs.
- [`docs/RELEASE-RUNBOOK.md`](../docs/RELEASE-RUNBOOK.md) — read before
  touching any version string; [`docs/MSRV.md`](../docs/MSRV.md) — MSRV
  policy.
- `.claude/commands/` — `/orchestrate`, `/architect`, `/implement`,
  `/review`, `/gate-check`.
