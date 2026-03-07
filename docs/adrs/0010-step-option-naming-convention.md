# ADR 0010: Step Option Naming Convention

- Status: Accepted
- Date: 2026-03-07

## Context

Pipeline step options in `vision-calibration-pipeline` used mixed naming:

- generic names in some modules (`InitOptions`, `OptimizeOptions`)
- stage-qualified names in others (`IntrinsicsInitOptions`, `RigOptimOptions`)
- abbreviated action words (`Optim`)

This made the API inconsistent and harder to scan in docs/facade exports, especially when
multiple problem modules are used together.

## Decision

Standardize all step option types to:

`<Stage><Action>Options`

Rules:

1. `Stage` is mandatory for all modules (no bare `InitOptions` / `OptimizeOptions`).
2. `Action` uses full words (`Init`, `Optimize`, `Filter`, etc.); abbreviations like `Optim` are not allowed.
3. Names should align with the step family in the same module (for example, `step_intrinsics_*` uses `Intrinsics*Options`).
4. Migration is a hard rename with no temporary compatibility aliases.

Examples:

- `IntrinsicsInitOptions`, `IntrinsicsOptimizeOptions`
- `RigOptimizeOptions`
- `HandeyeOptimizeOptions`
- `DeviceInitOptions`, `DeviceOptimizeOptions`

## Consequences

Positive:

- Uniform API naming across all problem modules.
- Cleaner facade re-exports and easier type discovery in rustdoc.
- Fewer ambiguities when importing multiple modules in one crate.

Negative:

- Breaking API change for downstream users relying on old option type names.
- Minor migration work in docs/examples and external code.

## Required Follow-up

- Keep future option types compliant with this convention.
- Reject new abbreviated action names in review (`Optim*`, etc.).
