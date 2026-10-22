# Adding a New Pipeline Problem Type

This chapter is a guide for contributors working inside the `vision-calibration-pipeline` crate. `ProblemType` is sealed, so new problem types are added to the crate itself rather than from downstream code. The laserline device module (`laserline_device/`) is a good template.

## Module Structure

Create a new folder under `crates/vision-calibration-pipeline/src/`:

```
my_problem/
├── mod.rs         # Module re-exports
├── problem.rs     # ProblemType implementation + Config/Export
├── state.rs       # Intermediate state type (crate-private)
└── steps.rs       # Step functions + pipeline function
```

## Step 1: Define the State (`state.rs`)

The state holds intermediate results between steps. It stays crate-private; users see typed step results instead.

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct MyState {
    // Initialization results
    pub initial_intrinsics: Option<FxFyCxCy<Real>>,
    pub initial_poses: Option<Vec<Iso3>>,

    // Optimization results
    pub final_cost: Option<f64>,
    pub mean_reproj_error: Option<f64>,
}
```

The state must implement `Default` (empty state) and `Clone + Serialize + Deserialize` (for checkpointing).

## Step 2: Define the Problem Type (`problem.rs`)

```rust
use crate::Error;
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

pub struct MyProblem;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MyConfig {
    pub solver: SolverConfig,
    pub init: IntrinsicsInitConfig,
}

// `MyInput`, `MyEstimate` and `MyExport` are ordinary
// `Clone + Serialize + DeserializeOwned + Debug` types. The output is the
// final solve's estimate; an input that is a shared data type is an alias
// (`pub type MyInput = PlanarDataset;`).

impl ProblemState for MyProblem {
    type State = MyState; // defined in state.rs
}

impl ProblemType for MyProblem {
    type Config = MyConfig;
    type Input = MyInput;
    type Output = MyEstimate;
    type Export = MyExport;

    fn name() -> &'static str { "my_problem_v1" }
    fn schema_version() -> u32 { 1 }

    fn validate_input(input: &MyInput) -> Result<(), Error> {
        if input.views.len() < 3 {
            return Err(Error::InsufficientData { need: 3, got: input.views.len() });
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy { InvalidationPolicy::CLEAR_COMPUTED }
    fn on_config_change() -> InvalidationPolicy { InvalidationPolicy::KEEP_ALL }

    fn export(
        input: &MyInput, output: &MyEstimate, _config: &MyConfig,
    ) -> Result<MyExport, Error> {
        // Build the user-facing export (attach per-feature residuals, etc.).
        todo!()
    }
}
```

Shared config groups (`SolverConfig`, `IntrinsicsInitConfig`, `RobotPoseConfig`, `HandeyeInitConfig`, `RigConfig`) live in `common/config.rs`; embed them as named fields rather than re-declaring the same settings flat.

## Step 3: Implement Step Functions (`steps.rs`)

Each step takes `Option<<Stage><Action>Options>` (reuse the shared ones in `common/mod.rs`; `None` keeps the session config) and returns a typed `<Stage><Action>Result`, so callers never need to read the state:

```rust
use crate::Error;
use crate::common::{IntrinsicsInitOptions, IntrinsicsOptimizeOptions};
use crate::session::CalibrationSession;
use super::problem::MyProblem;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IntrinsicsInitResult {
    pub intrinsics: FxFyCxCy<Real>,
    pub poses: Vec<Iso3>,
}

pub fn step_init(
    session: &mut CalibrationSession<MyProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    let opts = opts.unwrap_or_default();
    let input = session.require_input()?;
    let config = &session.config;

    // Run linear initialization
    let intrinsics = /* ... */;
    let poses = /* ... */;

    // Store the results in the problem's crate-private state
    // (the session's `state` field, in-crate access only):
    //   initial_intrinsics = Some(intrinsics);
    //   initial_poses = Some(poses.clone());

    session.log_success("init");
    Ok(IntrinsicsInitResult { intrinsics, poses })
}

pub fn step_optimize(
    session: &mut CalibrationSession<MyProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<IntrinsicsOptimizeResult, Error> {
    let input = session.require_input()?;

    // Require initialization
    // Read the initialization results back from the crate-private state,
    // failing with a typed error when `step_init` has not run:
    //   .ok_or_else(|| Error::not_available("initial intrinsics (call step_init first)"))?
    let init_k = /* state.initial_intrinsics */;

    // Build and solve the optimization problem, then store the output
    let output: MyEstimate = /* ... */;
    session.set_output(output);

    session.log_success_with_notes("optimize", "cost=..., reproj_err=...");
    Ok(/* IntrinsicsOptimizeResult { .. } */)
}

/// Convenience pipeline function: the same signature in every problem.
pub fn run_calibration(session: &mut CalibrationSession<MyProblem>) -> Result<(), Error> {
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}
```

Add `step_*_with_seed` variants when the problem supports manual seeds, following `planar_intrinsics/steps.rs`: the seed is a `<Stage>ManualInit` and comes before the options.

## Step 4: Module Re-exports (`mod.rs`)

```rust
mod problem;
mod state;
mod steps;

pub use problem::{MyConfig, MyEstimate, MyExport, MyInput, MyProblem};
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsInitResult, IntrinsicsOptimizeOptions,
    IntrinsicsOptimizeResult, run_calibration, step_init, step_optimize,
};
```

## Step 5: Register in the Pipeline Crate

In `crates/vision-calibration-pipeline/src/lib.rs`:

```rust
pub mod my_problem;
```

## Step 6: Wire into the Facade Crate

In `crates/vision-calibration/src/lib.rs`, list the public items explicitly (the facade curates its surface):

```rust
pub mod my_problem {
    pub use vision_calibration_pipeline::my_problem::{
        IntrinsicsInitOptions, IntrinsicsInitResult, IntrinsicsOptimizeOptions,
        IntrinsicsOptimizeResult, MyConfig, MyEstimate, MyExport, MyInput, MyProblem,
        run_calibration, step_init, step_optimize,
    };
}
```

Do not add new workflows to the prelude by default; keep the prelude minimal for planar hello-world usage.

Add the Python binding in `vision-calibration-py`: one `session_class!` invocation in `src/session.rs` (the problem type, its steps with their option and result types), a `python/vision_calibration/<module>.py` with the session class and the seed and result dataclasses, a `run_<module>` runner, and an entry per new facade item in `scripts/binding_parity.toml`. The parity check fails until every facade item is bound or marked Rust-only with a reason.

## Testing

Write a test next to the module (it can read the crate-private state) or an integration test using only the public API:

```rust
#[test]
fn my_problem_session_workflow() -> Result<(), Error> {
    let input = make_synthetic_input(); // local test helper

    let mut session = CalibrationSession::<MyProblem>::new();
    session.set_input(input)?;

    let init = step_init(&mut session, None)?;
    assert!(!init.poses.is_empty());

    step_optimize(&mut session, None)?;
    assert!(session.output().is_some());

    // Test JSON round-trip
    let json = session.to_json()?;
    let restored = CalibrationSession::<MyProblem>::from_json(&json)?;
    assert!(restored.output().is_some());

    Ok(())
}
```
