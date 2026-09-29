# ProblemType Trait

The `ProblemType` trait defines the interface for a calibration problem. It specifies the types for configuration, input, output, and export, along with validation and lifecycle hooks. The trait is intentionally minimal — behavior is implemented in external step functions, not trait methods.

`ProblemType` is **sealed**: it can only be implemented by the eight problem types that ship with the pipeline crate, so downstream crates use it as a bound (`CalibrationSession<P: ProblemType>`) but do not implement it. Contributors adding a problem type inside the pipeline crate should read [Adding a New Pipeline Problem Type](new_pipeline.md).

## Definition

```rust
pub trait ProblemType: ProblemState + Sized + 'static {
    type Config: Clone + Default + Serialize + DeserializeOwned + Debug;
    type Input: Clone + Serialize + DeserializeOwned + Debug;
    type Output: Clone + Serialize + DeserializeOwned + Debug;
    type Export: Clone + Serialize + DeserializeOwned + Debug;

    fn name() -> &'static str;
    fn schema_version() -> u32 { 1 }

    fn validate_input(input: &Self::Input) -> Result<(), Error> { Ok(()) }
    fn validate_config(config: &Self::Config) -> Result<(), Error> { Ok(()) }
    fn validate_input_config(
        input: &Self::Input, config: &Self::Config
    ) -> Result<(), Error> { Ok(()) }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }
    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        input: &Self::Input, output: &Self::Output, config: &Self::Config
    ) -> Result<Self::Export, Error>;
}
```

`ProblemState` is a crate-private supertrait that carries the internal `State` associated type (the intermediate results between steps). It is what seals the trait; `State` is not part of the public API.

## Associated Types

| Type | Purpose | Requirements |
|------|---------|-------------|
| `Config` | Algorithm parameters | `Default` (sensible defaults) |
| `Input` | Observation data | Validated on set |
| `Output` | Final calibration result | Set by last step |
| `Export` | User-facing result | Created from Output + Config |

The `Serialize + DeserializeOwned` bounds enable JSON checkpointing (the internal state is serialized as well). `Clone` enables snapshot operations.

## Required Method: `name()`

Returns a stable string identifier for the problem type:

```rust
fn name() -> &'static str { "planar_intrinsics_v2" }
```

This is stored in session metadata and used for deserialization. Changing the name breaks compatibility with existing checkpoints.

## Required Method: `export()`

Converts the internal output to a user-facing export format:

```rust
fn export(
    input: &Self::Input, output: &Self::Output, config: &Self::Config,
) -> Result<Self::Export, Error>;
```

The export may transform, filter, or enrich the output. The input is provided so exports can attach per-feature reprojection residuals; for example, `PlanarIntrinsicsProblem::export()` computes per-feature residuals and reprojection error metrics from the raw optimization output.

## Validation Hooks

Three optional validation hooks run at specific points:

| Hook | When it runs |
|------|-------------|
| `validate_input(input)` | On `session.set_input(input)` |
| `validate_config(config)` | On `session.set_config(config)` or `update_config()` |
| `validate_input_config(input, config)` | When both input and config are present |

Example: `PlanarIntrinsicsProblem` validates that input has at least 3 views with at least 4 points each.

## Invalidation Policies

Control what happens when input or config changes:

```rust
pub struct InvalidationPolicy {
    pub clear_state: bool,
    pub clear_output: bool,
    pub clear_exports: bool,
}

impl InvalidationPolicy {
    pub const KEEP_ALL: Self = ...;       // Nothing cleared
    pub const CLEAR_COMPUTED: Self = ...; // State + output cleared
    pub const CLEAR_ALL: Self = ...;      // Everything cleared
}
```

Defaults:
- Input change → `CLEAR_COMPUTED` (re-running steps is needed)
- Config change → `KEEP_ALL` (output is still valid, but may not reflect new config)

## Problem Types

| Type | `name()` | Steps |
|------|----------|-------|
| `PlanarIntrinsicsProblem` | `"planar_intrinsics_v2"` | init → optimize |
| `ScheimpflugIntrinsicsProblem` | `"scheimpflug_intrinsics_v1"` | init → optimize |
| `SingleCamHandeyeProblem` | `"single_cam_handeye_v2"` | intrinsics init → intrinsics optimize → hand-eye init → hand-eye optimize |
| `LaserlineDeviceProblem` | `"laserline_device_v1"` | init → optimize |
| `RigExtrinsicsProblem` | `"rig_extrinsics_v2"` | intrinsics init (all) → intrinsics optimize (all) → rig init → rig optimize |
| `RigHandeyeProblem` | `"rig_handeye_v2"` | 6 steps (intrinsics + rig + hand-eye) |
| `RigLaserlineDeviceProblem` | `"rig_laserline_device_v1"` | init → optimize |
| `RigHandeyeLaserlineProblem` | `"rig_handeye_laserline_v1"` | `run_calibration` only |

## Design Philosophy

The trait is minimal by design:

- **No step methods**: Steps are free functions, not trait methods. This allows flexible composition and avoids trait object limitations.
- **No algorithm logic**: The trait defines data types and validation, not computation.
- **Serialization-first**: All types are serializable, enabling checkpointing from day one.
