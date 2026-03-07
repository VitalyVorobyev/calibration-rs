# ADR 0002: Planar Intrinsics Family and Sensor Modes

- Status: Accepted
- Date: 2026-03-07

## Context

Planar calibration now exists in two parallel paths:

- Standard planar intrinsics session pipeline:
  [planar_intrinsics](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs)
- Scheimpflug-specific direct function:
  [scheimpflug_intrinsics.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/scheimpflug_intrinsics.rs)

Both share core algorithm phases (initialization, pose recovery, non-linear refinement),
but differ in camera model and optimization parameterization.

## Decision

Treat planar intrinsics as a workflow family with explicit sensor-model variants.

Near-term:

- Implement a first-class `scheimpflug_intrinsics` `ProblemType` module in pipeline
  (same structural contract as other workflows).

Mid-term:

- Unify shared logic between standard and Scheimpflug planar flows.
- Keep sensor-model-specific behavior explicit in type/config contracts.

## Consequences

Positive:

- Preserves clear semantics: identity sensor and Scheimpflug are explicit, not hidden flags.
- Avoids forcing incompatible output types into current `PlanarIntrinsicsEstimate`.
- Enables shared testing patterns while keeping model-specific constraints visible.

Negative:

- Two workflow modules remain in short term.
- Requires careful API stabilization for eventual unification.

## Rejected Alternative

Embedding Scheimpflug as an ad-hoc branch in existing `planar_intrinsics` without
updating core types/contracts was rejected due to high risk of API ambiguity and
internal conditional complexity.
