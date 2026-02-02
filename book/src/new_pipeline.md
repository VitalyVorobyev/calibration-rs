# Adding a New Pipeline Problem Type

This chapter describes how to create a new session-based calibration workflow in `vision-calibration-pipeline`, using the laserline device module as a template.

## Module Structure

Create a new folder under `crates/vision-calibration-pipeline/src/`:

```
my_problem/
├── mod.rs         # Module re-exports
├── problem.rs     # ProblemType implementation + Config
├── state.rs       # Intermediate state type
└── steps.rs       # Step functions + pipeline function
```

## Step 1: Define the Problem Type (`problem.rs`)

```rust
use crate::session::{ProblemType, InvalidationPolicy};

pub struct MyProblem;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MyConfig {
    pub max_iters: usize,
    pub fix_k3: bool,
    // ... other configuration
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyInput {
    pub views: Vec<MyView>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyOutput {
    pub calibrated_params: CameraParams,
    pub mean_reproj_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyExport {
    pub params: CameraParams,
    pub mean_reproj_error: f64,
}

impl ProblemType for MyProblem {
    type Config = MyConfig;
    type Input = MyInput;
    type State = MyState;  // defined in state.rs
    type Output = MyOutput;
    type Export = MyExport;

    fn name() -> &'static str { "my_problem_v1" }

    fn validate_input(input: &MyInput) -> Result<()> {
        ensure!(input.views.len() >= 3, "Need at least 3 views");
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy { InvalidationPolicy::CLEAR_COMPUTED }
    fn on_config_change() -> InvalidationPolicy { InvalidationPolicy::KEEP_ALL }

    fn export(output: &MyOutput, _config: &MyConfig) -> Result<MyExport> {
        Ok(MyExport {
            params: output.calibrated_params.clone(),
            mean_reproj_error: output.mean_reproj_error,
        })
    }
}
```

## Step 2: Define the State (`state.rs`)

```rust
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MyState {
    // Initialization results
    pub initial_intrinsics: Option<FxFyCxCySkew<f64>>,
    pub initial_distortion: Option<BrownConrady5<f64>>,
    pub initial_poses: Option<Vec<Iso3>>,

    // Optimization results
    pub final_cost: Option<f64>,
    pub mean_reproj_error: Option<f64>,
}
```

The state must implement `Default` (empty state) and `Clone + Serialize + Deserialize` (for checkpointing).

## Step 3: Implement Step Functions (`steps.rs`)

```rust
use crate::session::CalibrationSession;
use super::problem::MyProblem;

pub fn step_init(
    session: &mut CalibrationSession<MyProblem>,
    _opts: Option<&()>,
) -> Result<()> {
    let input = session.require_input()?;
    let config = &session.config;

    // Run linear initialization
    let intrinsics = /* ... */;
    let poses = /* ... */;

    // Update state
    session.state.initial_intrinsics = Some(intrinsics);
    session.state.initial_poses = Some(poses);

    // Log
    session.log_success("step_init", Some("Initialization complete"));
    Ok(())
}

pub fn step_optimize(
    session: &mut CalibrationSession<MyProblem>,
    _opts: Option<&()>,
) -> Result<()> {
    let input = session.require_input()?;
    let config = &session.config;

    // Require initialization
    let init_k = session.state.initial_intrinsics.as_ref()
        .context("Run step_init first")?;

    // Build and solve optimization problem
    let result = /* ... */;

    // Update state and output
    session.state.final_cost = Some(result.cost);
    session.state.mean_reproj_error = Some(result.reproj_error);
    session.set_output(MyOutput {
        calibrated_params: result.params,
        mean_reproj_error: result.reproj_error,
    })?;

    session.log_success("step_optimize", Some("Optimization complete"));
    Ok(())
}

/// Convenience pipeline function
pub fn run_calibration(
    session: &mut CalibrationSession<MyProblem>,
) -> Result<()> {
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}
```

## Step 4: Module Re-exports (`mod.rs`)

```rust
mod problem;
mod state;
mod steps;

pub use problem::{MyProblem, MyConfig, MyInput, MyOutput, MyExport};
pub use state::MyState;
pub use steps::{step_init, step_optimize, run_calibration};
```

## Step 5: Register in the Pipeline Crate

In `crates/vision-calibration-pipeline/src/lib.rs`:

```rust
pub mod my_problem;
```

## Step 6: Wire into the Facade Crate

In `crates/vision-calibration/src/lib.rs`:

```rust
pub mod my_problem {
    pub use vision_calibration_pipeline::my_problem::*;
}
```

And add to the prelude as needed.

## Testing

Write an integration test in `crates/vision-calibration-pipeline/tests/`:

```rust
#[test]
fn my_problem_session_workflow() -> Result<()> {
    let input = generate_synthetic_input();

    let mut session = CalibrationSession::<MyProblem>::new();
    session.set_input(input)?;

    step_init(&mut session, None)?;
    assert!(session.state.initial_intrinsics.is_some());

    step_optimize(&mut session, None)?;
    assert!(session.output().is_some());

    // Test JSON round-trip
    let json = session.to_json()?;
    let restored = CalibrationSession::<MyProblem>::from_json(&json)?;
    assert!(restored.output.is_some());

    Ok(())
}
```
