# Step Functions vs Pipeline Functions

calibration-rs offers two ways to run calibration workflows: **step functions** for granular control and **pipeline functions** for convenience.

## Step Functions

Step functions are free functions that operate on a mutable session reference:

```rust
pub fn step_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<PlanarInitResult, Error>
```

Each step:
1. Reads the input, config, and the results of earlier steps from the session
2. Performs one phase of the calibration (e.g., initialization or optimization)
3. Updates the session (and possibly its output)
4. Logs the operation
5. Returns a typed result value summarizing what it computed

### Advantages

**Intermediate inspection**: Examine the typed result of each step.

```rust
let init = step_init(&mut session, None)?;

// Inspect initialization quality before committing to optimization
if (init.intrinsics.fx - expected_fx).abs() / expected_fx > 0.5 {
    eprintln!("Warning: init fx={:.0} is far from expected {:.0}",
              init.intrinsics.fx, expected_fx);
}

let opt = step_optimize(&mut session, None)?;
println!("{:.3} px after {} iterations", opt.mean_reproj_error, opt.iterations);
```

**Per-step configuration**: Override options for individual steps.

```rust
// Use more iterations for optimization
let mut opts = IntrinsicsOptimizeOptions::default();
opts.max_iters = Some(200);
step_optimize(&mut session, Some(opts))?;
```

**Selective execution**: Skip steps or re-run specific steps.

```rust
// Re-run optimization with different settings (without re-initializing)
session.update_config(|c| c.solver.robust_loss = RobustLoss::Cauchy { scale: 3.0 })?;
step_optimize(&mut session, None)?;
```

**Checkpointing**: Save and restore between steps.

```rust
step_init(&mut session, None)?;
let checkpoint = session.to_json()?;
std::fs::write("after_init.json", &checkpoint)?;

step_optimize(&mut session, None)?;
```

## Manual Seeds

Every problem with an initialization step also has a `*_with_seed` variant (for example `step_init_with_seed` taking a `PlanarManualInit`) that accepts a coarse prior for some parameters and auto-estimates the rest. The plain `step_init` is the same call with an empty seed.

## Pipeline Functions

Pipeline functions chain all steps into a single call:

```rust
pub fn run_calibration(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
) -> Result<(), Error> {
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

## Available Steps and Pipeline Functions

Paths are relative to the facade module named after the problem (`vision_calibration::<module>`).

| Problem | Module | Steps | Pipeline function |
|---------|--------|-------|-------------------|
| `PlanarIntrinsicsProblem` | `planar_intrinsics` | `step_init`, `step_optimize`, `step_filter` | `run_calibration(session)`, `run_calibration_with_filtering(session, filter_opts)` |
| `ScheimpflugIntrinsicsProblem` | `scheimpflug_intrinsics` | `step_init`, `step_optimize` | `run_calibration(session, config)` |
| `SingleCamHandeyeProblem` | `single_cam_handeye` | `step_intrinsics_init`, `step_intrinsics_optimize`, `step_handeye_init`, `step_handeye_optimize` | `run_calibration(session)` |
| `LaserlineDeviceProblem` | `laserline_device` | `step_init`, `step_optimize` | `run_calibration(session, config)` |
| `RigExtrinsicsProblem` | `rig_extrinsics` | `step_intrinsics_init_all`, `step_intrinsics_optimize_all`, `step_rig_init`, `step_rig_optimize` | `run_calibration(session)` |
| `RigHandeyeProblem` | `rig_handeye` | the four rig steps above plus `step_handeye_init`, `step_handeye_optimize` | `run_calibration(session)` |
| `RigLaserlineDeviceProblem` | `rig_laserline_device` | `step_init`, `step_optimize` | `run_calibration(session)` |
| `RigHandeyeLaserlineProblem` | `rig_handeye_laserline` | (none; the pipeline is a single joint solve) | `run_calibration(session)` |

The `*_with_seed` variants exist for `step_init` / `step_intrinsics_init` / `step_intrinsics_init_all` / `step_rig_init` / `step_handeye_init` of the problems above.

`step_filter` removes high-residual points from the session input, which invalidates the computed state; run `step_init` and `step_optimize` again afterwards. `run_calibration_with_filtering` does exactly that: solve, filter, solve.

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
