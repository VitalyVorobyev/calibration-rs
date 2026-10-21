# ADR 0013: Sensor Variants of Workflows

## Context

Planar and rig workflows each come in a pinhole and a Scheimpflug flavour.
The flavours share their step skeleton but differ in the sensor parameter
block, a few config fields and the optim entry point. Workflows themselves
(intrinsics, rig extrinsics, hand-eye, laser) differ fundamentally in
state, steps, residuals and exports.

## Decision

The sensor model is a variant axis; the workflow is a module.

- **Rigs: sensor is a config axis.** `rig_extrinsics` and `rig_handeye`
  each serve both flavours through `SensorMode { Pinhole, Scheimpflug {…} }`
  (`#[non_exhaustive]`, defined once in `rig_family`). Step functions match
  on it and call the pinhole or Scheimpflug optim entry point. Outputs are
  enums with accessors (`cameras()`, `cam_to_rig()`, …) so callers rarely
  match; exports carry `sensors: Option<Vec<ScheimpflugParams>>`. The shared
  per-camera bootstrap and helpers live in `rig_family` as free functions.
  The per-camera Scheimpflug distortion mask defaults to radial-only,
  because tangential distortion absorbs tilt-like signal.
- **Single-camera planar: two modules, shared internals.**
  `planar_intrinsics` and `scheimpflug_intrinsics` stay separate public
  workflows, because they differ in more than a block: the Scheimpflug
  workflow has its own seeded route (ADR 0011), its own distortion-model
  choice and fix masks, and its own output contract. Their initialization
  bootstrap (`planar_family.rs`) and IR construction
  (`optim::problems::planar_family_shared`) are shared, and both workflows
  call them instead of keeping private copies.
- Laser rigs (`rig_laserline_device`) accept either flavour from their
  upstream calibration; a pinhole upstream is exactly zero tilt.

## Consequences

- A new sensor variant for rigs is a `SensorMode` variant and an inner
  branch, not a new module; a cross-cutting change touches one file per
  workflow.
- `sensors: Option<…>` on rig exports carries an invariant the types do not
  enforce ("`Some` iff Scheimpflug"); helpers that need sensors return an
  error for a pinhole export.

## Alternatives rejected

- **A trait matrix (sensor × workflow)** — workflows differ in state shape
  and step sequence; a generic `RigWorkflow` hides the control flow a
  calibration engineer needs to read, and the shared code is small free
  functions.
- **One module per (sensor × workflow)** — every cross-cutting change then
  edits sibling configs, states and exports.
- **Scheimpflug as an ad-hoc branch inside `planar_intrinsics`** — makes the
  single-camera output contract ambiguous.
