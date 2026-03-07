# ADR 0002: Planar Intrinsics Family and Sensor Modes

- Status: Accepted
- Date: 2026-03-07
- Last Updated: 2026-03-07

## Context

Planar calibration exists as two sensor-specific workflows:

- Standard planar intrinsics session pipeline:
  [planar_intrinsics](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs)
- Scheimpflug planar intrinsics session pipeline:
  [scheimpflug_intrinsics](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/mod.rs)

Both workflows share the same high-level phases (initialization, pose recovery, non-linear
refinement), but they differ in camera model and optimized parameter blocks
(identity sensor vs Scheimpflug sensor tilt).

By design policy (ADR 0001 + layered crate rules), these workflows must remain `ProblemType`
modules with session semantics, while shared math/optimizer plumbing should be deduplicated.

## Decision

Treat planar intrinsics as a workflow family with explicit sensor-model variants.

Keep separate public workflow modules/contracts:

- `planar_intrinsics` for pinhole + Brown-Conrady + identity sensor output contract.
- `scheimpflug_intrinsics` for pinhole + Brown-Conrady + Scheimpflug sensor output contract.

Unify shared implementation internals:

- Shared initialization bootstrap in pipeline:
  [planar_family.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/planar_family.rs)
  - computes view homographies
  - runs iterative intrinsics/distortion initialization
  - recovers per-view planar poses
- Shared optimization IR setup in optim:
  [planar_family_shared.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-optim/src/problems/planar_family_shared.rs)
  - shared parameter-block construction (intrinsics/distortion/poses + optional sensor)
  - shared residual construction for planar reprojection factors

Step-function routing requirement:

- `planar_intrinsics` and `scheimpflug_intrinsics` steps must call these shared helpers
  instead of maintaining private duplicated setup code.

## Consequences

Positive:

- Preserves clear semantics: identity sensor and Scheimpflug are explicit, not hidden flags.
- Avoids forcing incompatible output types into current `PlanarIntrinsicsEstimate`.
- Enables shared testing patterns while keeping model-specific constraints visible.

Negative:

- Two workflow modules remain long-term at API level.
- Internal helper contracts now become a maintenance point that must stay aligned across both flows.

## Rejected Alternative

Embedding Scheimpflug as an ad-hoc branch in existing `planar_intrinsics` without
updating core types/contracts was rejected due to high risk of API ambiguity and
internal conditional complexity.
