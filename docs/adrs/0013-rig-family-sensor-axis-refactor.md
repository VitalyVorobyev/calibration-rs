# ADR 0013: rig_family Sensor-Axis Refactor (Pinhole + Scheimpflug Unification)

- Status: Accepted
- Date: 2026-05-01

## Context

The calibration pipeline had grown five sibling **rig** modules
that calibrated multi-camera rigs in structurally similar but flavour-specific
ways:

| Module | Sensor | Workflow | LoC |
|--------|--------|----------|----:|
| `rig_extrinsics` | Pinhole | extrinsics-only | 1,760 |
| `rig_scheimpflug_extrinsics` | Scheimpflug | extrinsics-only | 933 |
| `rig_handeye` | Pinhole | hand-eye | 2,198 |
| `rig_scheimpflug_handeye` | Scheimpflug | hand-eye | 1,798 |
| `rig_laserline_device` | (Scheimpflug, frozen upstream) | laser-plane | 542 |
| **Total** | | | **~7,200** |

The four extrinsics + hand-eye modules were natural pairs (pinhole ↔
Scheimpflug). A duplication audit found ~30–40% structural overlap:

- `state.rs` files mirrored each other ~90% (the Scheimpflug variant added
  one `per_cam_sensors` field).
- `*Config` structs duplicated ~60–75% (Scheimpflug added 3–4 tilt-related
  fields).
- `*Steps.rs` files shared a near-identical six-step skeleton (intrinsics ×2,
  rig ×2, hand-eye ×2 for the hand-eye flavour) with the same per-camera
  bootstrap loop and view-extraction helpers re-implemented in each.

The per-feature-residuals work touched all six
exports independently, demonstrating that the duplication was an active tax
on every cross-cutting change rather than a one-time copy.

## Decision

Factor the rig family along a **single axis — sensor model (pinhole vs
Scheimpflug)** — and only that axis. Workflow remains per-module because each
workflow has fundamentally different state, step sequences, residual blocks,
and exports.

The factoring uses **composition over traits**:

1. A new internal helper module `crate::rig_family` owns shared
   sensor-axis-aware primitives:

   - `RigSensorBundle` — a `pub(crate)` payload carrying
     `cameras: Vec<PinholeCamera>` plus an optional
     `scheimpflug: Option<Vec<ScheimpflugParams>>`.
   - `RigIntrinsicsSeeds` — manual-init seeds that carry both pinhole
     intrinsics/distortion and Scheimpflug sensors as `Option<Vec<…>>`.
   - `SensorFlavour` — a `pub(crate)` enum tagging which kind of bootstrap
     the helper should run.
   - `bootstrap_rig_intrinsics(num_cameras, num_views, |cam_idx| views, …)` —
     the per-camera bootstrap loop, sensor-aware: for `Pinhole` it produces
     plain `PinholeCamera`s; for `Scheimpflug` it also threads per-camera
     tilt params (seeded or default).
   - `views_to_planar_dataset`, `estimate_target_pose`,
     `intrinsics_k_matrix`, `format_init_source` — small pure helpers used
     across workflow modules.

2. A user-facing `SensorMode` enum (also defined in `rig_family` and
   re-exported by both `rig_extrinsics::SensorMode` and
   `rig_handeye::SensorMode`):

   ```rust
   #[non_exhaustive]
   #[serde(tag = "kind")]
   pub enum SensorMode {
       Pinhole,
       Scheimpflug {
           init_tilt_x: f64,
           init_tilt_y: f64,
           fix_scheimpflug: ScheimpflugFixMask,
           refine_scheimpflug_in_rig_ba: bool,
       },
   }
   ```

   Workflow configs (`RigExtrinsicsConfig`, `RigHandeyeConfig`) carry
   `sensor: SensorMode`. Workflow step functions match on it and dispatch
   to either the pinhole or Scheimpflug optim entry point.

3. Workflow `Output` types become enums to preserve both flavours' optim
   estimates without losing structural fidelity:

   ```rust
   pub enum RigExtrinsicsOutput {
       Pinhole(vision_calibration_optim::RigExtrinsicsEstimate),
       Scheimpflug(vision_calibration_optim::RigExtrinsicsScheimpflugEstimate),
   }
   ```

   (and analogously `RigHandeyeOutput`). Public accessor methods
   (`cam_to_rig()`, `cameras()`, `sensors()`, `mean_reproj_error()`, …)
   provide a uniform read surface so downstream code rarely needs to match
   on the variant.

4. Workflow `Export` types gain optional Scheimpflug-only fields:

   - `RigExtrinsicsExport.sensors: Option<Vec<ScheimpflugParams>>`
   - `RigHandeyeExport.sensors: Option<Vec<ScheimpflugParams>>`

   `None` for pinhole rigs, `Some(_)` for Scheimpflug rigs.

The four workflow modules collapse to three:
`rig_extrinsics`, `rig_handeye`, `rig_laserline_device`. The two
Scheimpflug siblings (`rig_scheimpflug_extrinsics`,
`rig_scheimpflug_handeye`) are deleted; their integration tests are
rewritten to drive the unified problems with `SensorMode::Scheimpflug`.

`rig_laserline_device` keeps its own existence — it is the only laser-plane
sibling and there is no second variant to collapse with. It now consumes
`RigHandeyeExport` directly via
`RigHandeyeExport::to_upstream_calibration(...)`, which accepts both pinhole
and Scheimpflug exports (a pinhole upstream is exactly zero tilt).

## Considered alternatives

### Two-axis trait matrix (sensor × workflow)

Defining a `RigWorkflow` trait parameterised by sensor model would maximise
generic reuse. Rejected:

- Hand-eye and extrinsics-only workflows have genuinely different state
  shapes (hand-eye carries `initial_handeye`, `final_cost`, …); making
  state generic adds 2–3 layers of associated types that future readers
  have to re-derive every time.
- Step sequences differ (4 vs 6 steps); abstracting the sequence into a
  trait obscures the per-workflow control flow that a calibration engineer
  needs to reason about.
- Composition over traits is more idiomatic Rust here — the common code
  is small free functions, not behaviour worth an interface.

### Keep all five modules, share a helper module only

A `rig_family.rs` helper module consumed by the sibling modules removes pure
duplication with no API change. But leaving the sibling modules in place
preserves the pattern that makes every cross-cutting change cost 4×. The
collapse to three modules is the load-bearing simplification, not the helper
extraction.

### Lift Scheimpflug-only advanced fields into the unified config

The former `RigScheimpflugHandeyeIntrinsicsConfig` carried six
Scheimpflug-only knobs (`initial_cameras`, `initial_sensors`,
`fallback_to_shared_init`, `fix_intrinsics_when_overridden`,
`fix_intrinsics_in_percam_ba`, `fix_distortion_in_percam_ba`) that the
pinhole config did not. We considered preserving them in either a nested
struct under `SensorMode::Scheimpflug` or a new
`RigHandeyeAdvancedScheimpflugConfig` section. Rejected: those
fields were workarounds for narrow-FOV / Zhang-failure cases that have
not surfaced in practice on the unified pipeline; the most load-bearing
default — `DistortionFixMask::radial_only()` for Scheimpflug per-camera
intrinsics refinement — is preserved as a hard-coded constant in
`step_intrinsics_optimize_all`. Pre-1.0 breakage is acceptable; if a
real workflow needs them back, they can be re-added without disturbing
the `SensorMode` shape.

## Consequences

### Positive

- **Sibling count**: 5 rig modules → 3. Adding a new sensor variant
  (e.g., omnidirectional) is a `SensorMode` enum variant + an inner
  branch in 4 step functions, not a new ~1,500-line module.
- **Net LoC**: the collapse removed roughly 2,300 lines net.
- **Cross-cutting changes** (e.g., the next per-feature-residuals tweak)
  touch one file per workflow instead of two (multiplier 2× instead of 4×).
- **Single source of truth** for `SensorMode` (`rig_family.rs`) avoids
  the temptation to duplicate the enum across workflow modules.

### Negative

- **Breaking changes** at the public API: every rename, type-change, and
  module deletion is a downstream migration cost. Pre-1.0 absorbs this,
  but consumers (e.g., the removed Python `run_rig_scheimpflug_*` wrappers)
  had to migrate.
- **Output enum verbosity**: `RigExtrinsicsOutput::Pinhole(…)` and
  `RigHandeyeOutput::Scheimpflug(…)` matches at every internal use site.
  Mitigated by accessor methods so most call sites just write
  `output.cam_to_rig()`.
- **`Option<Vec<ScheimpflugParams>>` on exports** introduces a runtime
  invariant ("`Some(_)` iff sensor mode was Scheimpflug") that the type
  system does not enforce. Documented; downstream helpers that need
  sensors error explicitly when they receive a pinhole export.

### Risk: Scheimpflug calibration regression on real data

The two Scheimpflug integration tests (one for extrinsics, one for
hand-eye) preserve their exact convergence assertions (intrinsics within
5%, baseline within 2 cm, hand-eye within 2 cm + 0.02 rad, mean reproj
< 1 px). They cover both `EyeInHand` and `EyeToHand` for the hand-eye
variant. Real-data validation on the puzzle 130x130 rig is the further gate.

## Out of scope

- Workflow-axis abstraction (kept per-module).
- Adding new sensor variants beyond Pinhole/Scheimpflug.
- Python parity for the unified API (deferred).
- A `rig_scheimpflug_laserline_device` collapse — that variant did not
  exist as a separate module (`rig_laserline_device` was
  Scheimpflug-only by upstream contract); no work needed.
