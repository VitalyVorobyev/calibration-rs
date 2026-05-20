# ADR 0016: Canonical Dataset Manifest (`DatasetSpec`)

- Status: Accepted
- Date: 2026-05-02

## Context

After B0/B0.5/B0.6/B1/B2 shipped (Tauri 2 viewer with Diagnose / Viewer3D /
Epipolar workspaces), the user committed to making the Tauri app the
primary calibration tool — full workflow inside the UI: foreign datasets,
all 8 problem types, all 4 target detectors, every config knob exposed.
That commitment forces an answer to a question we deferred during B0: how
do we ingest data in arbitrary on-disk layouts without making users
copy / rename / reshape their files to fit a workspace convention?

Two non-options established up-front:

1. **Folder convention** — pick a canonical layout (`cam_<i>/<frame>.png`,
   `poses.csv`) and require users to mirror their data into it. Rejected
   because every external dataset already has its own layout, and copying
   gigabytes of images for every new dataset is not how engineers want
   to work.
2. **Per-problem `*Input` JSON as the wire format** — freeze the existing
   `RigDataset<RobotPoseMeta>` / `PlanarDataset` shapes as user-facing
   files. Rejected because those structs were never designed for human
   authoring, drift pre-1.0, and embed type-erased nalgebra serde shapes
   that aren't friendly to forms.

What we _do_ want: a single _descriptive_ manifest format that points at
the user's data in place. The manifest is the only on-disk wire format
the user touches; the existing `*Input` types become internal IR fed by
per-problem converters and free to break pre-1.0.

## Decision

### 1. Single canonical schema

`DatasetSpec` (`crates/vision-calibration-dataset/src/spec.rs`) is the
single on-disk manifest type. It carries:

- `cameras: Vec<CameraSource>` — per-camera glob or explicit path list,
  optional ROI.
- `target: TargetSpec` — closed enum tagged on `kind` covering
  Chessboard, Charuco, Puzzleboard, Ringgrid.
- `robot_poses: Option<RobotPoseSource>` — `{ path, format, columns }`
  with a flexible column mapping so vendor-specific CSV / JSON layouts
  map onto canonical fields.
- `topology: Topology` — selects one of the 8 problem types
  (`PlanarIntrinsics` … `RigLaserlineDevice`).
- `pose_pairing: Option<PosePairing>` — `ByIndex` or
  `SharedFilenameToken { regex, group }` for cross-camera /
  pose-to-image alignment.
- `pose_convention: Option<PoseConvention>` — three orthogonal closed
  enums (`transform`, `rotation_format`, `translation_units`), no
  defaults. Required whenever `robot_poses` is set.
- `_unresolved: Vec<String>` — paths to fields the AI manifest generator
  could not determine. Validation rejects manifests with non-empty
  `_unresolved`.

The schema derives `JsonSchema` under the `schemars` feature; the
generated schemas are emitted into `app/src/schemas/dataset_spec.json`
by `cargo xtask emit-schemas` for the React form generator (ADR 0018).

### 2. `*Input` types become internal IR

The 8 existing `*Input` types stop being a stable on-disk format. Per-
problem converters (e.g. `pipeline::dataset_runner::build_planar_input`)
take `(DatasetSpec, *Config, &dyn DetectionCache) → *Input`. Pre-1.0
breaking changes to `*Input` are explicitly OK; the user-facing wire
format is `DatasetSpec` only.

### 3. Closed-enum frame conventions

`PoseConvention` declares conventions as three closed enums with
**no defaults**:

- `transform: T_base_tcp | T_tcp_base | T_world_tcp | T_tcp_world`
- `rotation_format: quat_xyzw | quat_wxyz | euler_*_(deg|rad) |
  axis_angle_rad | matrix_4x4_row_major`
- `translation_units: m | mm`

The validator refuses manifests where any component is missing while
`robot_poses` is set. This is the central anti-footgun decision —
silently defaulting to `T_base_tcp / quat_xyzw / m` would silently
produce plausible-looking but wrong calibrations on roughly half of
real datasets (KUKA / ABB / UR / Fanuc all differ on at least one
axis).

### 4. AI manifest generator (PR 3)

The runtime contract is that an external generator (CLI binary in PR 3,
in-app command in PR 5) inspects a foreign folder and emits a
`dataset.toml` populating fields it can determine and listing the rest
in `_unresolved`. v0 of the generator is heuristic-only (regex / file
extension / vendor signature / README scraping). LLM-backed inference
for hard cases is deferred to its own ADR.

## Consequences

- A single schema means one form to render, one validator, one
  reproducibility primitive. Datasets become commit-able artefacts.
- The user never edits `*Input` JSON by hand. Pre-1.0 churn in those
  types is a workspace-internal concern only.
- Frame conventions become explicit. The cost is a richer manifest;
  the benefit is silent-failure mitigation on the most common
  calibration footgun.
- Adding a new convention (e.g. ABB's `T_base_tcp` with millimetre
  translation but degree Euler) is a code change, not a manifest
  change. This is a deliberate trade-off: predictable behaviour over
  unbounded extensibility.

## Status of work

- ✅ `vision-calibration-dataset` crate landed with `DatasetSpec`,
  validator, and 7 unit tests.
- ✅ Schema emitted to `app/src/schemas/dataset_spec.json` via
  `cargo xtask emit-schemas`.
- ✅ `pipeline::dataset_runner::build_planar_input` consumes the
  manifest end-to-end (validation → detection → IR) for the
  PlanarIntrinsics topology with the Chessboard target.
- ⏳ Coverage extension to the other 7 topologies + 3 detectors
  (PR 2).
- ⏳ AI manifest generator (PR 3).
