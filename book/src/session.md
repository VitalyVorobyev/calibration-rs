# CalibrationSession

`CalibrationSession<P: ProblemType>` is the central state container for calibration workflows. It holds all data for a calibration run — configuration, input observations, intermediate state, final output, and an audit log — with full JSON serialization for checkpointing.

## Structure

A session has two public fields and otherwise exposes its data through methods:

| Member | Access | Purpose |
|--------|--------|---------|
| `config` | `pub` field | Algorithm parameters (iterations, fix masks, loss functions) |
| `exports` | `pub` field | Timestamped `ExportRecord`s created by `export()` |
| input | `input()`, `require_input()`, `set_input()`, `clear_input()` | Observation data |
| output | `output()`, `require_output()` | Final calibration result, set by the last step |
| `metadata()` | method | `SessionMetadata`: problem type name, schema version, timestamps, description |
| `log()` | method | Audit trail of operations, `&[LogEntry]` |

Intermediate state is internal to the pipeline. Read intermediate results from the typed values returned by each step function (for example `PlanarInitResult`), from `log()`, or from `export()`.

The public methods of `CalibrationSession<P>`:

| Group | Methods |
|-------|---------|
| Construction | `new()`, `with_description(..)`, `with_input(..)` |
| Input | `set_input`, `input`, `input_mut`, `require_input`, `require_input_mut`, `has_input`, `clear_input` |
| Config | `set_config`, `update_config` |
| Output | `output`, `output_mut`, `require_output`, `set_output`, `has_output`, `clear_output` |
| Export | `export`, `export_with_notes`, `export_peek` |
| Validation | `validate` |
| Log | `log`, `log_success`, `log_success_with_notes`, `log_failure` |
| Metadata | `metadata` |
| Reset | `reset_state`, `reset_output`, `reset` |
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
    c.solver.robust_loss = RobustLoss::Huber { scale: 2.0 };
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

Each problem type may override these policies.

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
println!("Init fx={:.1}", init.intrinsics.fx);

// Output (available after optimize)
let output = session.require_output()?;

// Session bookkeeping
println!("{} ({} log entries)", session.metadata().problem_type, session.log().len());
```
