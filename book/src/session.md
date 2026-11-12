# CalibrationSession

`CalibrationSession<P: ProblemType>` is the central state container for calibration workflows. It holds all data for a calibration run — configuration, input observations, intermediate state, final output, and an audit log — with full JSON serialization for checkpointing.

## Structure

A session exposes its data through methods:

| Member | Access | Purpose |
|--------|--------|---------|
| config | `config()`, `set_config()`, `update_config()` | Algorithm parameters (iterations, fix masks, loss functions) |
| exports | `exports()` | Timestamped `ExportRecord`s created by `export()` |
| input | `input()`, `require_input()`, `set_input()`, `clear_input()` | Observation data |
| output | `output()`, `require_output()` | Final calibration result, set by the last step |
| `metadata()` | method | `SessionMetadata`: problem type name, schema version, timestamps, description |
| `log()` | method | Audit trail of operations, `&[LogEntry]` |

Intermediate state is internal to the pipeline. Read intermediate results from the typed values returned by each step function (for example `IntrinsicsInitResult`), from `log()`, or from `export()`.

The public methods of `CalibrationSession<P>`:

| Group | Methods |
|-------|---------|
| Construction | `new()`, `with_description(..)`, `with_input(..)` |
| Input | `set_input`, `input`, `require_input`, `has_input`, `clear_input` |
| Config | `config`, `set_config`, `update_config` |
| Output | `output`, `require_output`, `has_output`, `clear_output` |
| Export | `export`, `export_with_notes`, `export_peek`, `exports` |
| Validation | `validate` |
| Log | `log` |
| Metadata | `metadata` |
| Reset | `reset` |
| Serialization | `to_json`, `from_json` |

## Lifecycle

### 1. Create

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
// Or with a description:
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::with_description(
    "Lab camera calibration"
);
```

### 2. Set Input

```rust
session.set_input(dataset)?;
```

Input is validated via `ProblemType::validate_input()`. Setting input clears computed state (per the invalidation policy).

### 3. Configure

```rust
session.update_config(|c| {
    c.solver.max_iters = 50;
    c.optimize.reprojection_loss = RobustLoss::Huber { scale: 2.0 };
})?;
```

Configuration is validated via `ProblemType::validate_config()`.

### 4. Run Steps

```rust
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
```

Step functions are free functions operating on `&mut CalibrationSession<P>`. Each step reads input/state, performs computation, and updates state (or output).

### 5. Export

```rust
let export = session.export()?;
```

Creates an `ExportRecord` with the current timestamp and output. Multiple exports can be created (e.g., after re-optimization with different settings).

## JSON Checkpointing

The entire session can be serialized and restored:

```rust
// Save
let json = session.to_json()?;
std::fs::write("calibration.json", &json)?;

// Restore
let json = std::fs::read_to_string("calibration.json")?;
let mut restored = CalibrationSession::<PlanarIntrinsicsProblem>::from_json(&json)?;

// Resume from where we left off
step_optimize(&mut restored, None)?;
```

This enables:
- **Interruption recovery**: Save progress and resume later
- **Reproducibility**: Share exact calibration state
- **Debugging**: Inspect the session at any point

## Invalidation Policies

When input or configuration changes, computed state may need to be cleared:

| Event | Default policy |
|-------|---------------|
| `set_input()` | Clear state and output (`CLEAR_COMPUTED`) |
| `update_config()` | Keep everything (`KEEP_ALL`) |
| `clear_input()` | Same as `set_input()` (`CLEAR_COMPUTED`), then the input is removed |

Each problem type may override these policies. A kept output still means
what it meant when it was solved: a hand-eye result records the mode it was
solved in, and its export reads that mode, not the current config's.

## Audit Log

Every step function appends to the session log:

```rust
pub struct LogEntry {
    pub timestamp: u64,
    pub operation: String,
    pub success: bool,
    pub notes: Option<String>,
}
```

The log records what was done and when, useful for tracking calibration history.

## Reading Results

```rust
// Input (required before steps)
let input = session.require_input()?;  // Returns error if no input

// Intermediate results: the typed value returned by each step
let init = step_init(&mut session, None)?;
println!("Init fx={:.1}", init.camera.intrinsics.fx);

// Output (available after optimize)
let output = session.require_output()?;

// Session bookkeeping
println!("{} ({} log entries)", session.metadata().problem_type, session.log().len());
```

## Progress and Cancellation

A long run reports its progress, and stops on request, through a
`RunControl` (in `vision_calibration::control`): an optional `CancelToken`
and observers that receive each `Progress` report.

```rust,ignore
use vision_calibration::control::{CancelToken, Progress, RunControl};

let token = CancelToken::new();
let control = RunControl::new()
    .with_cancel(token.clone())
    .with_observer(|progress| match progress {
        Progress::Step { name, .. } => eprintln!("{name}"),
        Progress::Iteration { iter, cost, .. } => eprintln!("  {iter}: {cost:.3e}"),
        _ => {}
    });
session.set_control(control);
// From another thread (a Ctrl-C handler, a Cancel button): token.cancel();
match run_calibration(&mut session) {
    Err(vision_calibration::Error::Cancelled) => eprintln!("stopped"),
    other => other?,
}
```

| Report | When |
|---|---|
| `Detection { done, total }` | a target image was detected or read from the cache |
| `LaserExtraction { done, total }` | a laser image was extracted |
| `Step { name }` | `run_calibration` begins a step (`"intrinsics_init_all"`, …) |
| `Iteration { iter, max_iters, cost }` | a solver iteration finished; `iter` restarts at 1 for each solve |

The token is checked before every solver iteration, between the steps of
`run_calibration` and before every image, so a run stops within one
iteration or one image of the request. A stopped call returns
`Error::Cancelled` (`CalibrateDatasetError::Cancelled` from
`calibrate_dataset`, `RunError::Cancelled` from a `build_*_input`
converter), never a partial result: a stopped step stores nothing in the
session. Progress is a side channel: a run with observers returns exactly
what it returns without them, bit for bit, and nothing it reports enters a
result.

The same control goes to `dataset_runner::calibrate_dataset` and the
`build_*_input` converters as their last argument, and to
`optim::problem::solve` as `SolveOptions::control`. Observers run on the
calibrating thread at every report, so keep them cheap; one that must
redraw a screen can drop reports that arrive too fast.

## In Python

Each problem type has its session class, named after it:
`PlanarIntrinsicsSession`, `SingleCamHandeyeSession`, `RigExtrinsicsSession`,
`RigHandeyeSession`, `LaserlineDeviceSession`, `RigLaserlineDeviceSession`
and `RigHandeyeLaserlineSession`. The step functions are its methods, with
the Rust names; each takes the step's options and returns the step's result
as a dataclass. Long steps release the GIL.

```python
import vision_calibration as vc

session = vc.PlanarIntrinsicsSession("Lab camera calibration")
session.set_input(dataset)  # a PlanarIntrinsicsInput, or its payload dict
session.set_config(vc.PlanarIntrinsicsConfig(solver=vc.SolverConfig(max_iters=50)))

init = session.step_init(vc.IntrinsicsInitOptions(iterations=3))
print(f"Init fx={init.camera.intrinsics.fx:.1f}")
session.step_optimize()
result = session.export(notes="first pass")  # a PlanarIntrinsicsExport

for entry in session.log:  # the audit log, as LogEntry dataclasses
    print(entry.operation, entry.success, entry.notes)

# Checkpoint and resume
restored = vc.PlanarIntrinsicsSession.from_json(session.to_json())
```

`input` and `output` return the serde payloads (dicts), `config` the config
dataclass, `exports` the `ExportRecord`s and `metadata` the
`SessionMetadata`. `run_calibration()` runs the problem's default steps. An
input or config the problem rejects, a step out of order and a seed that does
not fit raise `ValueError`; a numerical failure raises `RuntimeError`.

Ctrl-C stops a running step with `KeyboardInterrupt`. `set_progress` gives
the session a callable that receives every report as a dict
(`{"kind": "iteration", "iter": 3, "max_iters": 50, "cost": 12.5}`); raising
from it stops the step, and the exception propagates. The `run_*` functions,
`calibrate_dataset` and the `build_*_input` converters take the callable as
their `progress` argument:

```python
def show(report):
    if report["kind"] == "detection":
        print(f"detected {report['done']}/{report['total']}")

result = vc.calibrate_dataset("data/kuka_1/dataset.toml", progress=show)
```
