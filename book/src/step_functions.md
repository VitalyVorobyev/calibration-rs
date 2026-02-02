# Step Functions vs Pipeline Functions

calibration-rs offers two ways to run calibration workflows: **step functions** for granular control and **pipeline functions** for convenience.

## Step Functions

Step functions are free functions that operate on a mutable session reference:

```rust
pub fn step_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<&InitOptions>,
) -> Result<()>
```

Each step:
1. Reads input and/or state from the session
2. Performs one phase of the calibration (e.g., initialization or optimization)
3. Updates the session state (and possibly output)
4. Logs the operation

### Advantages

**Intermediate inspection**: Examine the state between steps.

```rust
step_init(&mut session, None)?;

// Inspect initialization quality before committing to optimization
let init_k = session.state.initial_intrinsics.as_ref().unwrap();
if (init_k.fx - expected_fx).abs() / expected_fx > 0.5 {
    eprintln!("Warning: init fx={:.0} is far from expected {:.0}",
              init_k.fx, expected_fx);
}

step_optimize(&mut session, None)?;
```

**Per-step configuration**: Override options for individual steps.

```rust
// Use more iterations for optimization
let opts = OptimOptions { max_iters: 200, ..Default::default() };
step_optimize(&mut session, Some(&opts))?;
```

**Selective execution**: Skip steps or re-run specific steps.

```rust
// Re-run optimization with different settings (without re-initializing)
session.update_config(|c| c.robust_loss = RobustLoss::Cauchy { scale: 3.0 })?;
step_optimize(&mut session, None)?;
```

**Checkpointing**: Save and restore between steps.

```rust
step_init(&mut session, None)?;
let checkpoint = session.to_json()?;
std::fs::write("after_init.json", &checkpoint)?;

step_optimize(&mut session, None)?;
```

## Pipeline Functions

Pipeline functions chain all steps into a single call:

```rust
pub fn run_calibration(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
) -> Result<()> {
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}
```

### When to Use

- **Quick prototyping**: Get results with minimal code
- **Default settings**: When the defaults work and you don't need inspection
- **Scripts and automation**: When human inspection is not needed

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
run_calibration(&mut session)?;
let export = session.export()?;
```

## Available Pipeline Functions

| Problem | Pipeline function | Steps |
|---------|------------------|-------|
| `PlanarIntrinsicsProblem` | `run_calibration()` | init → optimize |
| `SingleCamHandeyeProblem` | `run_single_cam_handeye()` | 4 steps |
| `RigExtrinsicsProblem` | `run_rig_extrinsics()` | 4 steps |
| `RigHandeyeProblem` | `run_rig_handeye()` | 6 steps |
| `LaserlineDeviceProblem` | `run_calibration(session, config)` | init → optimize |

## Recommendation

**Use step functions** for production calibration where you need to:
- Verify initialization quality
- Adjust parameters between steps
- Handle failures gracefully
- Log and audit the process

**Use pipeline functions** for:
- Examples and tutorials
- Batch processing with known-good settings
- Quick experiments
