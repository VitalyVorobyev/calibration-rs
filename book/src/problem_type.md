# ProblemType Trait

The `ProblemType` trait defines the interface for a calibration problem. It specifies the types for configuration, input, state, output, and export, along with validation and lifecycle hooks. The trait is intentionally minimal — behavior is implemented in external step functions, not trait methods.

## Definition

```rust
pub trait ProblemType: Sized + 'static {
    type Config: Clone + Default + Serialize + DeserializeOwned + Debug;
    type Input: Clone + Serialize + DeserializeOwned + Debug;
    type State: Clone + Default + Serialize + DeserializeOwned + Debug;
    type Output: Clone + Serialize + DeserializeOwned + Debug;
    type Export: Clone + Serialize + DeserializeOwned + Debug;

    fn name() -> &'static str;
    fn schema_version() -> u32 { 1 }

    fn validate_input(input: &Self::Input) -> Result<()> { Ok(()) }
    fn validate_config(config: &Self::Config) -> Result<()> { Ok(()) }
    fn validate_input_config(
        input: &Self::Input, config: &Self::Config
    ) -> Result<()> { Ok(()) }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }
    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        output: &Self::Output, config: &Self::Config
    ) -> Result<Self::Export>;
}
```

## Associated Types

| Type | Purpose | Requirements |
|------|---------|-------------|
| `Config` | Algorithm parameters | `Default` (sensible defaults) |
| `Input` | Observation data | Validated on set |
| `State` | Intermediate results | `Default` (empty initial state) |
| `Output` | Final calibration result | Set by last step |
| `Export` | User-facing result | Created from Output + Config |

The `Serialize + DeserializeOwned` bounds enable JSON checkpointing. `Clone` enables snapshot operations.

## Required Method: `name()`

Returns a stable string identifier for the problem type:

```rust
fn name() -> &'static str { "planar_intrinsics_v2" }
```

This is stored in session metadata and used for deserialization. Changing the name breaks compatibility with existing checkpoints.

## Required Method: `export()`

Converts the internal output to a user-facing export format:

```rust
fn export(output: &Self::Output, config: &Self::Config) -> Result<Self::Export>;
```

The export may transform, filter, or enrich the output. For example, `PlanarIntrinsicsProblem::export()` computes reprojection error metrics from the raw optimization output.

## Validation Hooks

Three optional validation hooks run at specific points:

| Hook | When it runs |
|------|-------------|
| `validate_input(input)` | On `session.set_input(input)` |
| `validate_config(config)` | On `session.set_config(config)` or `update_config()` |
| `validate_input_config(input, config)` | When both input and config are present |

Example: `PlanarIntrinsicsProblem` validates that input has ≥ 3 views with ≥ 4 points each.

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

## Existing Problem Types

| Type | Name | Steps |
|------|------|-------|
| `PlanarIntrinsicsProblem` | `"planar_intrinsics_v2"` | init → optimize |
| `SingleCamHandeyeProblem` | `"single_cam_handeye"` | intrinsics_init → intrinsics_optimize → handeye_init → handeye_optimize |
| `RigExtrinsicsProblem` | `"rig_extrinsics"` | intrinsics_init_all → intrinsics_optimize_all → rig_init → rig_optimize |
| `RigHandeyeProblem` | `"rig_handeye"` | 6 steps (intrinsics + rig + handeye) |
| `LaserlineDeviceProblem` | `"laserline_device"` | init → optimize |

## Design Philosophy

The trait is minimal by design:

- **No step methods**: Steps are free functions, not trait methods. This allows flexible composition and avoids trait object limitations.
- **No algorithm logic**: The trait defines data types and validation, not computation.
- **Serialization-first**: All types are serializable, enabling checkpointing from day one.
