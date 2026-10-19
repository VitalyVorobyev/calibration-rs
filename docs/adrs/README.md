# ADR Index

This directory stores architecture decision records for `calibration-rs`.

Design decisions are recorded here; the work they imply is tracked in
`docs/backlog.md` (AGENTS.md §9). An ADR states the decision as it stands
today: when a decision changes, edit the ADR (or supersede it), do not append
history.

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

## Conventions

- [0002 - Planar Intrinsics Family and Sensor Modes](0002-planar-intrinsics-family-and-sensor-modes.md)
- [0010 - Step Option Naming Convention](0010-step-option-naming-convention.md)

Numbers 0001, 0003 and 0004 are withdrawn: their rules (module shape, the
generic Python binding, the ADR + backlog process) live in ADR 0007 and
AGENTS.md. Numbers are never reused.

## Workflow & Schema

- [0011 - Manual Parameter Initialization Workflow](0011-manual-initialization-workflow.md)
- [0012 - Per-Feature Reprojection Residuals on Export Types](0012-per-feature-reprojection-residuals.md)
- [0013 - rig_family Sensor-Axis Refactor (Pinhole + Scheimpflug Unification)](0013-rig-family-sensor-axis-refactor.md)
- [0014 - Tauri 2 Desktop App for Calibration Diagnose](0014-tauri-desktop-app.md)
- [0015 - Multiple-View Geometry Crates and Scope Ceiling](0015-mvg-ceiling.md)
- [0016 - Canonical Dataset Manifest (`DatasetSpec`)](0016-dataset-manifest.md)
- [0017 - Content-Addressed Detection Cache](0017-detection-cache.md)
- [0018 - Schema-Driven UI for Configs and Dataset Manifests](0018-schema-driven-ui.md)
- [0019 - Fail Fast on Ambiguity (`AskUser` runtime contract)](0019-fail-fast-on-ambiguity.md)
- [0020 - Camera Model as Data in the Factor IR](0020-camera-model-as-data-factor-ir.md)
- [0021 - Laser-Frame Dataset Manifest](0021-laser-frame-manifest.md)
- [0022 - Scheimpflug Intrinsics: User-Seeded Initialization is the Supported Default](0022-scheimpflug-intrinsics-seeded-default.md)
- [0023 - `DeviceSpec`: Device Specification Schema and Seed Derivation](0023-device-spec-seed-derivation.md)
- [0024 - One Config Vocabulary Across the Eight Problem Types](0024-config-vocabulary.md)
- [0025 - A Second Backend (factrs) Under One Levenberg–Marquardt Loop](0025-factrs-backend-shared-lm.md)
- [0026 - Parameter Uncertainty as a Standard Result](0026-parameter-uncertainty.md)
