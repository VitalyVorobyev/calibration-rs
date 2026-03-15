# ADR 0011: Manual Parameter Initialization Workflow

- Status: Accepted
- Date: 2026-03-15

## Context

In industrial calibration environments, users often have prior knowledge of camera parameters —
focal lengths from datasheets, sensor tilt angles from factory measurements, laser plane geometry
from mechanical drawings. The current pipeline always runs automatic linear initialization (Zhang's
method, Tsai-Lenz DLT), which may produce poor starting points for non-linear optimization when
the auto-init quality is limited (e.g., narrow baseline, few views, large tilt angles). There is no
mechanism to seed specific parameter groups with expert knowledge while letting the library
auto-initialize the rest.

## Decision

Add a `step_set_init()` function to each problem module, alongside the existing `step_init()`:

```rust
// Automatic path (unchanged):
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Manual path:
step_set_init(&mut session, PlanarManualInit {
    intrinsics: Some(nominal_camera_k),
    distortion: None,  // auto-initialized
    poses: None,       // auto-initialized from homographies
}, None)?;
step_optimize(&mut session, None)?;
```

**Behavioral contract**:

1. `step_set_init(session, manual, opts)` seeds any provided fields into the session state, then
   auto-initializes all remaining fields using the same routines as `step_init`.
2. After `step_set_init` returns, the session state is fully initialized — same postcondition as
   `step_init`. `step_optimize` requires no changes.
3. `step_init(session, opts)` is refactored to call `step_set_init(session, ManualInit::default(), opts)`,
   preserving its existing signature and behavior.
4. For problem types with multiple init stages (SingleCamHandeye, RigExtrinsics, RigHandeye), each
   stage gets its own `step_set_*` function with a corresponding typed `ManualInit` struct.

**ManualInit struct rules**:

- All fields are `Option<T>`. `None` means "auto-initialize this group".
- Derive `Debug`, `Clone`, `Default`.
- Do NOT apply `#[non_exhaustive]` — users construct these with `..Default::default()` and
  `#[non_exhaustive]` would break that pattern.
- Placed in `steps.rs` alongside the step functions, re-exported from `mod.rs`.

**Partial initialization semantics**:

- When `intrinsics` is provided manually but `poses` is `None`, poses are auto-recovered using
  the **manually provided intrinsics** (not auto-estimated ones). This ensures geometric consistency.
- The log entry records which fields were manual vs auto, e.g.:
  `"init (manual: intrinsics, distortion; auto: poses)"`.

**RigExtrinsics coupling rule**:

`cam_se3_rig` and `rig_se3_target` are geometrically coupled; providing one without the other is
ambiguous. `RigExtrinsicsManualInit` requires both or neither, enforced at runtime with a clear
error message.

## Consequences

Positive:

- Expert users can bypass unreliable auto-init for specific parameter groups.
- `step_init` backward compatibility is maintained (same signature, same behavior).
- `step_optimize` requires no changes.
- Typing is per-problem and discoverable via rustdoc and IDE completion.

Negative:

- Each problem module grows by one struct and one function.
- Users who provide incorrect manual seeds may get worse results — no automatic validation of
  physical plausibility beyond basic cardinality checks.

## Required Follow-up

- Python bindings for `ManualInit` types (initially: PlanarIntrinsics, SingleCamHandeye).
- Keep `ManualInit` structs out of `#[non_exhaustive]` enforcement in future API hardening passes.
