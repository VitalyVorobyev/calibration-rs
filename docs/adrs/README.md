# Architecture Decision Records

An ADR records a decision that shapes the design across modules, had real
alternatives, and whose reasoning the code does not show. Implementation
detail belongs in rustdoc, algorithms and evidence in
[`docs/notes/`](../notes/README.md), rules in `AGENTS.md`, and work in
[`docs/backlog.md`](../backlog.md).

Every ADR listed here is in force. When a decision changes, edit its ADR to
state the new decision; do not append history. Numbers are stable
identifiers and are not reused, so the sequence has gaps.

Shape: **Context** (the forces, briefly), **Decision** (what holds, stated
as rules), **Consequences**, **Alternatives rejected** (each with its
reason). Aim for about 120 lines or fewer. Measurements behind a choice go to
`docs/notes/`; type and function names only where the decision is about
them; layouts, commands and component wiring go to rustdoc or
`app/DEVELOPING.md`.

## Structure

- [0006 — Layered crate architecture](0006-layered-crate-architecture.md)
- [0007 — Session framework with external step functions](0007-session-framework.md)
- [0024 — One API vocabulary](0024-api-vocabulary.md)

## Models and numerics

- [0005 — Composable camera model](0005-composable-camera-model.md)
- [0009 — Coordinate and pose conventions](0009-coordinate-and-pose-conventions.md)
- [0008 — Optimization IR and solver backends](0008-optimization-ir-and-backends.md)
- [0013 — Lens and sensor variants are camera slots](0013-sensor-variants.md)
- [0011 — Seeded initialization](0011-seeded-initialization.md)
- [0026 — Parameter uncertainty as a standard result](0026-parameter-uncertainty.md)

## Data and tools

- [0016 — Dataset ingestion](0016-dataset-ingestion.md)
- [0012 — The export is the result's wire contract](0012-export-contract.md)
- [0014 — Desktop app](0014-desktop-app.md)
