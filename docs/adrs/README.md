# ADR Index

This directory stores architecture decision records for `calibration-rs`.

Process:

1. Capture design decisions in ADRs first.
2. Track implementation work in `docs/backlog.md`.
3. Keep `IMPLEMENTATION_PLAN.md` removed; use ADR + backlog as the single planning flow.

Status legend:

- `Accepted`: decision is active and should guide implementation.
- `Superseded`: replaced by a newer ADR.
- `Proposed`: under discussion and not yet binding.

## Foundational (retroactive)

- [0005 - Composable Camera Model Pipeline](0005-composable-camera-model.md)
- [0006 - Layered Crate Architecture](0006-layered-crate-architecture.md)
- [0007 - Session Framework with External Step Functions](0007-session-framework.md)
- [0008 - Backend-Agnostic Optimization IR](0008-backend-agnostic-optimization-ir.md)
- [0009 - Coordinate and Pose Conventions](0009-coordinate-and-pose-conventions.md)

## Process & Migration

- [0001 - Pipeline Problem Module Shape](0001-pipeline-problem-module-shape.md)
- [0002 - Planar Intrinsics Family and Sensor Modes](0002-planar-intrinsics-family-and-sensor-modes.md)
- [0003 - Facade and Python API Consistency](0003-facade-and-python-api-consistency.md)
- [0004 - Planning Process: ADR + Backlog](0004-planning-process-adr-backlog.md)
- [0010 - Step Option Naming Convention](0010-step-option-naming-convention.md)

## Workflow & Schema

- [0011 - Manual Parameter Initialization Workflow](0011-manual-initialization-workflow.md)
- [0012 - Per-Feature Reprojection Residuals on Export Types](0012-per-feature-reprojection-residuals.md)
