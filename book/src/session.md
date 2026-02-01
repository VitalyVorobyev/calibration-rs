# CalibrationSession

`CalibrationSession<P: ProblemType>` is the central state container for calibration workflows. It holds all data for a calibration run — configuration, input observations, intermediate state, final output, and an audit log — with full JSON serialization for checkpointing.

## Structure

```rust
pub struct CalibrationSession<P: ProblemType> {
    pub metadata: SessionMetadata,
    pub config: P::Config,
    pub state: P::State,
    pub exports: Vec<ExportRecord<P::Export>>,
    pub log: Vec<LogEntry>,
    // input and output are private — access via methods
}
```

| Field | Access | Purpose |
|-------|--------|---------|
| `metadata` | `pub` | Problem type name, schema version, timestamps |
| `config` | `pub` | Algorithm parameters (iterations, fix masks, loss functions) |
| input | `input()`, `set_input()` | Observation data (set once per calibration run) |
| `state` | `pub` | Mutable intermediate results (modified by step functions) |
| output | `output()`, `set_output()` | Final calibration result (set by the last step) |
| `exports` | `pub` | Timestamped export records |
| `log` | `pub` | Audit trail of operations |

## Lifecycle

### 1. Create

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
// Or with a description:
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::with_description(
    "Lab camera calibration 2024-01-15"
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
    c.max_iters = 50;
    c.robust_loss = RobustLoss::Huber { scale: 2.0 };
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
let restored = CalibrationSession::<PlanarIntrinsicsProblem>::from_json(&json)?;

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
| `clear_input()` | Clear everything (`CLEAR_ALL`) |

These defaults can be overridden per problem type.

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

## Accessing State

```rust
// Input (required before steps)
let input = session.require_input()?;  // Returns error if no input

// Intermediate state (available after init)
if let Some(k) = &session.state.initial_intrinsics {
    println!("Init fx={:.1}", k.fx);
}

// Output (available after optimize)
let output = session.require_output()?;
```
