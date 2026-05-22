# API Revision — `calibration-rs` workspace

> **Mode:** published `0.4.0` on crates.io → **0.x mode**. Breaking
> changes are allowed; batch them into a single `0.5.0` release with a
> migration section in CHANGELOG. Do not dribble them across multiple
> `0.x.y` bumps.

## Summary

Current public surface: **~440 top-level items** across 7 publishable
crates (`vision-calibration` facade, `-core`, `-linear`, `-optim`,
`-pipeline`, `-dataset`, `-detect`; `-py` is a PyO3 cdylib with no Rust
public surface).

**Verdict.** Three classes of contamination dominate. (1) Every problem
type exposes a public `*State` struct that is purely a session scratch
space with every field `Option<T>` — the classic optional-everything DTO,
read in examples via `session.state.foo.as_ref().unwrap()`. (2) The
facade re-exports `vision-calibration-linear` and `vision-calibration-optim`
via **glob `pub use *`**, so the facade's surface is "whatever the
algorithm crates happen to export, now and in every future version" —
the surface boundary is set by the implementation, not by design. (3)
The `linear` and `core` crates use the same pattern internally:
`pub use math::*; pub use models::*; pub use ransac::*; ...` flattens
every item in every algorithm-named module into the crate root, so the
top-level namespace doubles as the per-module namespace, and every
private module's contents are publicly reachable through two paths.

The Result/Diagnostic boundary has not been drawn. `*Export` types
*could* be the clean Result surface, but everything that should live
behind a diagnostic channel (the optional `*State` fields,
`SessionMetadata`, `LogEntry`, `ExportRecord`, `InvalidationPolicy`)
sits beside the contract types under `session`.

**Plan.** 4 phases / 22 changes, targeting `0.5.0`:

1. **Additive (non-breaking)** — introduce the diagnostics channel; add
   `#[non_exhaustive]` to growth-prone result/config types; add typed
   sub-results returned directly by `step_*` functions.
2. **Migrate in-repo callers** — examples + Python bindings + Tauri app
   onto the new surface; the workspace builds green after each
   migration.
3. **Tighten visibility (breaking)** — make `*State` fields and
   internal scratch space `pub(crate)`; replace glob re-exports with
   hand-picked lists; drop the duplicate `pub use math::*` patterns;
   hide `test_utils` behind a feature flag.
4. **Rename for honesty + close growth (breaking)** — fix
   double-pathed routes, naming inconsistencies, and dishonest names
   (`pixel_to_gripper_point` moved out of `lib.rs`,
   `make_pinhole_camera`'s call site review, the
   `step_set_*`/`step_*_init` naming pair).

Semver impact: minor in 0.x terms; **batch as 0.5.0**.

## The intended contract

The contract a normal consumer of this library *needs*, in full:

1. **Construct an input.** `PlanarDataset` (for single-camera planar
   calibration), `SingleCamHandeyeInput`, `RigExtrinsicsInput`,
   `RigHandeyeInput`, `LaserlineDeviceInput`,
   `ScheimpflugIntrinsicsInput`, `RigLaserlineDeviceInput` — one
   per problem type. With `CorrespondenceView` / `RigView` / `Pt2`
   `Pt3` / `Iso3` / `FxFyCxCySkew` / `BrownConrady5` /
   `make_pinhole_camera` / `PixelRect` / `FrameRef` / `ImageManifest`
   as the supporting math + image-metadata vocabulary.
2. **Run the calibration.** `CalibrationSession<P>::new()`,
   `.set_input(…)?`, `run_calibration(&mut session)?` (or the
   per-step `step_*` functions for inspection), `.export()?`.
3. **Read the result.** `PlanarIntrinsicsExport` / `RigExtrinsicsExport`
   / `RigHandeyeExport` / `SingleCamHandeyeExport` /
   `LaserlineDeviceExport` / `ScheimpflugIntrinsicsExport` /
   `RigLaserlineDeviceExport` — one typed result per problem type.
4. **Optional: per-feature residuals diagnostics.** Already on each
   `*Export` as `per_feature_residuals` (ADR 0012).

That is ~30 types + ~7 entry functions, against the present ~440
items.

Everything else — `*State`, `*ManualInit` (a niche advanced surface),
RANSAC internals at the crate root, every individual `Iter*`/`Solver*`
type from `linear`, every `*Estimate`/`*Params`/`*SolveOptions` from
`optim`, the `pub mod math` / `pub mod pnp` / `pub mod homography`
chain from `linear`, `test_utils`, the bare `Error` re-exports — is
either a diagnostic, an advanced/escape-hatch surface, or internal
scaffolding that leaked.

## Consumer workflow — before

Honest copy of what an example writes today (collapsed from
`examples/planar_real.rs` and `examples/handeye_session.rs`):

```rust
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

step_init(&mut session, None)?;

// Caller has to reach INTO the session's state to inspect what init produced —
// every field is Option<T>, every read is .as_ref().unwrap().
let init_k = session.state.initial_intrinsics.as_ref().unwrap();
let init_dist = session.state.initial_distortion.as_ref().unwrap();
println!("fx={}, fy={}, k1={}", init_k.fx, init_k.fy, init_dist.k1);

step_optimize(&mut session, None)?;

let state = &session.state;
println!("final cost: {:.2e}", state.final_cost.unwrap());
println!("mean reproj: {:.4}", state.mean_reproj_error.unwrap());

// Then the actual result:
let export = session.export()?;
```

For multi-stage problems (hand-eye, rig, rig-handeye), the friction
multiplies: 6-10 unwraps per script. From `manual_init_proof.rs`:

```rust
let cameras_a = session_a.state.per_cam_intrinsics.clone()
    .expect("Run A: per-cam intrinsics");
let cam_se3_rig_a = session_a.state.initial_cam_se3_rig.clone()
    .expect("Run A: initial cam_se3_rig");
let rig_se3_target_a = session_a.state.initial_rig_se3_target.clone()
    .expect("Run A: initial rig_se3_target");
```

The `Option<T>` does not represent a real "absent" outcome — `step_init`
*always* populates those fields. The optionality exists because the
struct exists *before* `step_init` runs. The type is lying.

## Consumer workflow — after

```rust
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

// Each step returns its own typed, non-optional output.
let init = step_init(&mut session, None)?;
println!("fx={}, fy={}, k1={}",
    init.intrinsics.fx, init.intrinsics.fy, init.distortion.k1);

let opt = step_optimize(&mut session, None)?;
println!("final cost: {:.2e}, mean reproj: {:.4}",
    opt.final_cost, opt.mean_reproj_error);

let export = session.export()?;
```

`session.state` becomes `pub(crate)` internal scratch space; consumers
read step outputs (typed, non-optional) and the final `Export`.

For the `manual_init_proof` pattern — re-injecting Run A's intermediate
values as seeds into Run B — the seed types (`*ManualInit`) are
already public and stay public. The example shrinks from "fish the
intermediate out of `session.state`" to "take the typed result of
`step_*` directly."

## Classification

Per-item classification is in **`docs/api-revision/classification.csv`**
(written alongside this report; see "Appendix" at end). The top-level
shape:

| Tier | Count | Sample |
|---|---|---|
| **Result** | ~35 | `PlanarIntrinsicsExport`, `RigExtrinsicsExport`, `RigHandeyeExport`, `SingleCamHandeyeExport`, `LaserlineDeviceExport`, `ScheimpflugIntrinsicsExport`, `RigLaserlineDeviceExport`, `PerFeatureResiduals` (ADR 0012), `FeatureResidualHistogram`, `ReprojectionStats`, `CorrespondenceView`, `Pt2`/`Pt3`/`Iso3`/`Vec2`/`Vec3`/`Mat3`, `FxFyCxCySkew`, `BrownConrady5`, `PinholeCamera`, `Camera`, `ScheimpflugParams`, `PlanarDataset`, `RigDataset`, `RigView`, `RigViewObs`, `View`, `NoMeta`, `FrameRef`, `ImageManifest`, `PixelRect`, `Real`, `ImagePattern`/`PoseConvention`/etc. from `dataset` |
| **Consumer entry** | ~50 | `CalibrationSession`, the 7 `*Problem` types, the ~30 `step_*` functions, the 7 `run_calibration` per-module facades |
| **Config** (stable) | ~25 | `PlanarIntrinsicsConfig`, `RigExtrinsicsConfig`, `RigHandeyeConfig`, `SingleCamHandeyeConfig`, `LaserlineDeviceConfig`, `ScheimpflugIntrinsicsConfig`, `RigLaserlineDeviceConfig`, `SensorMode`, `FilterOptions`, the per-problem `*ManualInit` variants, `DistortionFixMask`, `IntrinsicsFixMask` |
| **Config / step options** (likely tuning, candidate for advanced sub-config) | ~30 | `IntrinsicsInitOptions` ×5, `IntrinsicsOptimizeOptions` ×5, `HandeyeInitOptions` ×2, `HandeyeOptimizeOptions` ×2, `RigOptimizeOptions` ×2, `RigHandeyeBaConfig`, `RigHandeyeSolverConfig`, `LaserlineDeviceSolverConfig`, `LaserlineDeviceInitConfig`, `LaserlineDeviceOptimizeConfig`, `DeviceInitOptions`, `DeviceOptimizeOptions`, `StepOptions` |
| **Diagnostic** | ~40 | The 7 `*State` structs; `SessionMetadata`, `LogEntry`, `ExportRecord`, `InvalidationPolicy` from `session`; per-camera reprojection counters; `IterativeIntrinsicsTrace`, `IterStep` from `linear`; per-problem `optimize_*` raw entry points from `optim` |
| **Internal** (should be private / removed) | ~140 | Every individual `linear::<module>::Type` re-exposed at `linear::Type` (the duplicate path); every individual `pub mod math/models/ransac/types/view::Item` reachable both qua-module and at the crate root in `core`; the entire `vision-calibration-optim` crate-root surface (it's behind a facade, all the bare types are scaffolding the facade re-exports anyway); `vision-calibration-core::test_utils`; everything PAPI surfaces from inside `synthetic` beyond `grid_points` / `project_views_*` / pose helpers |
| **Algorithm primitives (legitimately exposed, but currently double-routed)** | ~120 | `dlt_homography`, `eight_point_fundamental`, `solve_pnp_*`, `estimate_intrinsics_from_homographies`, etc. — keep `pub mod homography::dlt_homography` but stop globbing them up to `linear::dlt_homography` |

## Leak deep-dives

### L1. The `*State` family — optional-everything DTO as a primary result (anti-pattern #2 + #3)

**What.** Seven public structs in `vision-calibration-pipeline`:
`PlanarState`, `SingleCamHandeyeState`, `RigExtrinsicsState`,
`RigHandeyeState`, `ScheimpflugIntrinsicsState`,
`LaserlineDeviceState`, `RigLaserlineDeviceState`. Every field is
`Option<T>`. Every external consumer (8 example files, none elsewhere
in repo) reads the fields via `.state.foo.as_ref().unwrap()` or
`.state.foo.unwrap_or(f64::NAN)` — the access pattern itself is the
contamination made concrete.

**Why it leaks.** These are session scratch space, not a result. The
"real contract" of which fields are populated when lives in the step
order (`step_init` populates `initial_*`, `step_optimize` populates
`final_*` and `mean_reproj_error`) and in `is_initialized()` /
`is_optimized()` helper methods on the State — none of which the type
encodes. A `step_init` *always* sets `initial_intrinsics`, so a
consumer who reads it after `step_init?` must unwrap a `None` that
cannot occur. The type lies.

The State also conflates two things: (a) post-step intermediate
results that a consumer actually wants to inspect, and (b) internal
pipeline plumbing (like the `homographies: Option<Vec<Mat3>>` field on
`PlanarState`) that no example reads.

**Surface violated.** Result (says "this is the result of running the
pipeline" when it is in fact a scratchpad) and Diagnostic (consumers
who read the *State* fields are doing so for diagnostic introspection,
without a clean channel for it).

**Fix.** Two-prong:

- Each `step_*` function returns a typed, non-`Option<>` value — its
  *own* result. `step_init -> PlanarInitResult { intrinsics, distortion,
  poses }`; `step_optimize -> PlanarOptimizeResult { final_cost,
  mean_reproj_error, iterations }`; analogous per problem. These
  become part of the contract.
- `CalibrationSession::state` becomes `pub(crate)`; the `*State`
  structs become `pub(crate)` types (or are folded into the session's
  internal representation entirely). Internal scratch fields that
  never make it into a step-result type (e.g., `homographies` in
  `PlanarState`) stay private.

**Consumer evidence.** All 8 in-repo consumers are example files. None
mutate. Migration target = "the `step_*` return value, plus the
existing `Export`."

**Before / after.** See "Consumer workflow — after" above. Each example
loses 6-10 lines of `.as_ref().unwrap()` boilerplate.

### L2. Facade glob re-exports of internal crates (anti-pattern #14)

**What.** In `crates/vision-calibration/src/lib.rs`:

```rust
pub mod linear { pub use vision_calibration_linear::*; }
pub mod optim  { pub use vision_calibration_optim::*; }
pub mod synthetic { pub use vision_calibration_core::synthetic::*; }
```

Plus the bare crate-root re-export `pub use vision_calibration_optim::{
compute_laserline_feature_residuals, compute_rig_laserline_feature_residuals,
handeye_observer_se3_target };` — at least *that* one is hand-picked.

**Why it leaks.** `pub use crate::*` across a crate boundary commits
the facade's surface to whatever the inner crate happens to expose,
*and in every future version*. Adding a new `pub` to `linear` is a
breaking change to the facade. Removing one is. Renaming one is. The
facade has surrendered control of its public API.

`linear` and `optim` are not single-purpose crates with curated
APIs — they're a kitchen sink (`linear`: 13 `pub mod`s + glob from
each one back into the crate root; `optim`: every problem-type
`optimize_*` function and its config/params/estimate structs). The
facade is currently committing to all of it.

**Surface violated.** Internal (a lot of what's in `linear` and
`optim` is algorithm primitive — useful but not curated; the facade
must choose what to expose).

**Fix.** Replace each glob with a hand-picked list. For `linear`,
re-export the public *modules* directly (so consumers write
`vision_calibration::linear::homography::dlt_homography`), and add a
small prelude module on top for the few primitives that genuinely
deserve top-level facade exposure. For `optim`, the rule is that
nothing inside it should be facade-reachable except the
`compute_*_feature_residuals` helpers already broken out — the rest is
the implementation that the per-problem `pipeline` module wraps.

### L3. `pub use module::*` flattening inside `linear` and `core` (anti-pattern #14 internal variant + #4)

**What.** In `crates/vision-calibration-linear/src/lib.rs`:

```rust
pub mod camera_matrix;
pub mod distortion_fit;
pub mod epipolar;
pub mod extrinsics;
pub mod handeye;
pub mod homography;
pub mod iterative_intrinsics;
pub mod laserline;
pub mod math;
pub mod planar_pose;
pub mod pnp;
pub mod triangulation;
pub mod zhang_intrinsics;

pub use camera_matrix::*;
pub use distortion_fit::*;
pub use epipolar::*;
// ... (13 of these)
```

In `crates/vision-calibration-core/src/lib.rs`:

```rust
mod math; mod models; mod ransac; mod types; mod view;
pub use math::*; pub use models::*; pub use ransac::*;
pub use types::*; pub use view::*;
```

**Why it leaks.** Two paths to every item — `linear::homography::dlt_homography`
*and* `linear::dlt_homography`. Either path is now a stable promise; we
own both. `core` is worse because the private modules `math`, `models`,
`ransac`, `types`, `view` aren't even namespaced — every item is at the
crate root *and* nowhere else. The crate's internal organization is
invisible to consumers, which makes it hard to navigate the docs and
also constrains future refactoring (renaming the private `view`
module is fine; renaming any of its public re-exports is a break).

**Surface violated.** Internal (the module structure is a *design*
tool — the implementer's view — not the consumer's view); the
re-export flattening defeats that.

**Fix.**

- `linear`: keep `pub mod homography`, etc. (these are algorithm
  primitives, each with a coherent surface). Drop the
  `pub use camera_matrix::*; pub use distortion_fit::*; …` block.
  Consumers write `linear::homography::dlt_homography`. Add an
  intentional `pub mod prelude` with a hand-picked handful of the
  most-used items.
- `core`: keep the items at the crate root (they're general utility:
  `Pt2`, `Iso3`, `BrownConrady5`, etc.) but make the private modules
  more honestly named (`mod math` →  `mod algebra`, `mod types` →
  `mod camera_models` or split). Drop the redundancy — pick *one* path
  per item.

### L4. Re-export of `core` types into `optim`'s namespace (anti-pattern #14 variant)

**What.** `crates/vision-calibration-optim/src/lib.rs` contains:

```rust
pub use vision_calibration_core::{RigDataset, RigViewObs, View};
```

A consumer can therefore write `optim::RigDataset` *or* `core::RigDataset`.

**Why it leaks.** The same item now has two stable paths owned by two
different crates. Either renaming or moving the item requires a
breaking change in both. The `optim` crate is committing to a type it
does not own.

**Fix.** Remove. Force `optim`'s API to take `core::RigDataset` (with
the type spelled out at the call site) — there is one Rust public-API
surface per item, owned by the originating crate.

### L5. `test_utils` is a top-level `pub mod` in `core` (anti-pattern #16)

**What.** `vision_calibration_core::test_utils` is `pub mod`, exposes
`CalibrationView`, `CornerInfo`, `ViewDetections`, `build_corner_info`,
`pixel_from_normalized`, `undistort_pixel_normalized`. Doc says "not
intended for production use" but the items are unconditionally `pub`.

**Why it leaks.** Test utilities exposed at the production API surface
look like product API. The "not for production use" caveat is in prose,
not in the type or feature system.

**Fix.** Feature-flag it: `cfg(feature = "test-utils")`, and document
the feature as unstable / internal. Or `#[doc(hidden)]` + add a
`#[deprecated]` note pointing at the proper path if any in-repo test
consumer needs it. The earlier inventory subagent found **zero**
external consumers; this can be made `pub(crate)` outright.

### L6. `pixel_to_gripper_point` floating at facade crate root

**What.** `vision-calibration/src/lib.rs` defines exactly one function
of original Rust code (everything else is `pub use`): a 110-line free
function `pixel_to_gripper_point` at the crate root.

**Why it leaks.** The crate-root surface mixes "the API" (`pub use`s)
with one piece of original implementation. The function semantically
belongs in `rig_laserline_device` — it computes a point given a rig
laserline calibration and a pixel — and that module already exists.

**Fix.** Move the function into `rig_laserline_device` (in
`vision-calibration-pipeline`), re-export through the facade's
`rig_laserline_device` module. The facade lib.rs returns to "all
re-exports."

### L7. Missing `#[non_exhaustive]` on growth-prone result + config types

**What.** None of the `*Export`, `*State`, `*Config`, `*ManualInit`
structs are `#[non_exhaustive]`. All have public fields. Once
published, every new field is a breaking change (struct-literal
construction breaks).

**Why it leaks.** Pre-publication is the only free moment to install
this. We are pre-1.0 still, in 0.x mode where breaking changes are
allowed but every one is disruptive.

**Fix.** Add `#[non_exhaustive]` (and constructors / builders where
struct-literal construction is the current idiom) to:
- All 7 `*Export` types.
- All 7 `*Config` types.
- All 7 `*ManualInit` types (per ADR 0011 these are an explicit
  extension/seed surface — almost guaranteed to grow).
- `RigHandeyeBaConfig`, `RigHandeyeInitConfig`,
  `RigHandeyeRigConfig`, `RigHandeyeIntrinsicsConfig`,
  `RigHandeyeSolverConfig`, `LaserlineDeviceSolverConfig`,
  `LaserlineDeviceInitConfig`, `LaserlineDeviceOptimizeConfig`.
- `SensorMode`, `HandEyeMode` enums (closed enums shouldn't be
  `#[non_exhaustive]`, but check whether new variants are anticipated:
  `Pinhole` + `Scheimpflug { … }` exhausts the design space per ADR
  0013, so this one stays as-is).
- `ReprojectionStats`, `FeatureResidualHistogram`,
  `PerFeatureResiduals`, `TargetFeatureResidual`,
  `LaserFeatureResidual` — these are diagnostic types that will gain
  fields.

### L8. `IntrinsicsInitOptions` / `IntrinsicsOptimizeOptions` × 5 — near-duplicate per-problem (anti-pattern #15)

**What.** Five copies of `IntrinsicsInitOptions` and five of
`IntrinsicsOptimizeOptions` — one per problem module
(planar/scheimpflug/single_cam_handeye/rig_extrinsics/rig_handeye).
Same for `HandeyeInitOptions`/`HandeyeOptimizeOptions` × 2, etc. Each
likely has the same fields or nearly so.

**Why it leaks.** Either they're identical and consumers reach into
the per-problem namespace for what should be one shared type; or they
diverged silently and there's no enforced contract. Both are
contamination — debugging artifacts crystallized as separate types.

**Fix.** Inspect the bodies (deferred to Phase B; needs an A-stage
diff). If identical, hoist to a single `vision_calibration::config`
or `vision_calibration_pipeline::common` module. If divergent, name
the divergence (`PlanarIntrinsicsInitOptions` vs
`HandeyeIntrinsicsInitOptions`, etc.) and document why; never name
the same thing differently across modules.

### L9. `step_set_*` versus `step_*` — naming pair (anti-pattern #13 mild)

**What.** Each problem has paired functions:

- `step_init` / `step_set_init`
- `step_intrinsics_init` / `step_set_intrinsics_init`
- `step_handeye_init` / `step_set_handeye_init`
- `step_rig_init` / `step_set_rig_init`
- `step_intrinsics_init_all` / `step_set_intrinsics_init_all`

`step_set_*` is the "manual seed" variant (ADR 0011): it accepts a
`*ManualInit` argument and writes it into the state instead of running
the linear init. The `set_` prefix is awkward — it reads as "set the
init result" rather than "skip init by providing a manual seed."

**Why it leaks.** A name that requires reading ADR 0011 to decode is a
dishonest name. The two functions are equally first-class.

**Fix.** Rename `step_set_*` to `step_*_with_seed` or
`step_seed_*_init`. Pick one and apply uniformly. Update ADR 0011
example code.

### L10. Re-exports inside `vision-calibration-py` go through one crate (the facade)

**What.** Cargo.toml of `vision-calibration-py` depends only on the
facade `vision-calibration`. So everything the bindings reach is
already constrained by the facade's surface.

**Why this is good.** This pins the Python contract to whatever we
decide the Rust contract is. No additional channel to keep aligned.

**No fix needed.** Mentioned here because Phase B subagents must
verify after each visibility change that the Python bindings (which
*do* reach into types like `RigHandeyeExport`'s fields via PyO3 wrap
code) still build.

## Diagnostics relocation

Where each currently-leaked diagnostic moves:

| Currently | Moves to |
|---|---|
| `PlanarState.homographies` | private to the implementation; not consumed externally |
| `PlanarState.initial_*` | return value of `step_init -> PlanarInitResult { intrinsics, distortion, poses }` |
| `PlanarState.final_cost`, `.mean_reproj_error`, `.iterations` | return value of `step_optimize -> PlanarOptimizeResult { final_cost, mean_reproj_error, iterations }` |
| `SingleCamHandeyeState.initial_camera`, `.optimized_camera`, etc. | return values of each of the 4 step functions; same pattern for rig variants |
| `session.metadata`, `session.log` | already accessor methods on `CalibrationSession`; promote to *the* introspection channel and document explicitly |
| `IterativeIntrinsicsTrace`, `IterStep` (in `linear::iterative_intrinsics`) | keep `pub` but move to a `linear::iterative_intrinsics::trace` submodule (anti-pattern #8 fix) |

There is *no* current `*Diagnostics` channel and no current diagnostic
surface needs to leave the workspace before 0.5.0 — the only consumers
that need richer information than `*Export` provides are in-repo
examples, and they migrate to the new `*Result` types from
`step_*` functions. A formal `Diagnostics` surface can wait.

## Breaking-change plan

Each row: change · semver-impact (in 0.x terms; minor bumps are breaking) · phase · in-repo migration note.

### Phase 1 — Additive (no breaks)

| # | Change | Semver | Migration note |
|---|--------|--------|----------------|
| 1 | Introduce typed `*StepResult` for each step function in pipeline. New struct per (problem × step), e.g. `PlanarInitResult { intrinsics, distortion, poses }`. Don't change `step_*` return type yet; have step functions write to BOTH the State and the new result. Expose under `vision_calibration_pipeline::<problem>::result`. | non-breaking | nothing yet |
| 2 | Add `#[non_exhaustive]` to all `*Export`, `*Config`, `*ManualInit` structs that currently lack it (see L7 list). Add `#[non_exhaustive]` constructors (`*::new(...)` or `*::default()`). | **breaking** (struct-literal construction) — but a Rust-level break only; serde round-trips unchanged. Bundle into 0.5.0. | callers using `Foo { a, b, ..Default::default() }` keep working; callers using bare `Foo { a, b }` need `..Default::default()` |
| 3 | Add `step_*_with_seed` aliases for the existing `step_set_*` functions. Keep both for one release. | non-breaking | docs: prefer `*_with_seed`; `step_set_*` deprecated in 0.5.0, removed in 0.6.0 |
| 4 | Add hand-picked re-export lists at facade for `linear`. Define `vision_calibration::linear::prelude` with curated items. Keep the existing `pub use linear::*` *for now* — old paths still work. | non-breaking | docs: prefer module paths and `prelude` |

### Phase 2 — Migrate in-repo consumers onto the new surface (no breaks)

| # | Change | Semver | Migration note |
|---|--------|--------|----------------|
| 5 | Migrate all 8 example files from `session.state.foo.as_ref().unwrap()` to the typed `step_*` return values introduced in #1. | non-breaking (per-file edits in examples/ only) | per-file diffs handled by Phase B subagents |
| 6 | Migrate any Python-binding code that touches `*State` fields onto `*Export` or the new `*Result` types. | non-breaking | Python bindings (verified by inventory subagent) don't actually touch `*State`, so this may be a no-op — but the subagent re-verifies during execution |
| 7 | Migrate any Tauri-app code (`app/src-tauri/`) that touches re-exported types from `optim` or `linear` to the explicit-path or facade-prelude variants. | non-breaking | re-verify under Phase B |
| 8 | Move `pixel_to_gripper_point` into `vision-calibration-pipeline::rig_laserline_device`, re-export through the facade's `rig_laserline_device` module. Keep an in-place `pub use` at the facade crate-root location for one release (deprecated). | breaking only via removal at 0.5.0 — the re-export ensures non-breaking during Phase 2 | callers update import path; the deprecated alias takes care of one cycle |

### Phase 3 — Tighten visibility (the major break batch)

| # | Change | Semver | Migration note |
|---|--------|--------|----------------|
| 9 | Make `CalibrationSession::state` accessor `pub(crate)` (or change visibility on the field). Add `CalibrationSession::log()`, `CalibrationSession::metadata()` accessors that return immutable views. | **breaking** | in-repo: migrated in #5 |
| 10 | Make all 7 `*State` types `pub(crate)`. **Paired adjustment (found in Phase B):** the public `ProblemType` trait carries `type State`, so demoting the structs alone triggers `E0446` (crate-private type in public interface). Fix: extract `type State` into a new `pub(crate)` supertrait `ProblemState` (`pub trait ProblemType: ProblemState + …`). The `impl ProblemState for XProblem` blocks are then crate-private. Side effect: this seals `ProblemType` (a crate-private supertrait blocks downstream `impl`) — intended, the 7 problem types are a closed set per ADR 0013. | **breaking** | in-repo: migrated in #5; the migration of L1 to per-step typed results obsoletes them. `ProblemType` becomes un-implementable downstream |
| 11 | Replace facade glob `pub use vision_calibration_linear::*` with a hand-picked list (or just `pub mod linear { pub use vision_calibration_linear::{Error, prelude, homography, ...}; }`). | **breaking** | docs: list of removed-from-facade items |
| 12 | Same for `pub use vision_calibration_optim::*` — hand-picked list; the typical consumer needs nothing here (they go via `pipeline`). Restrict to `LaserPlane`, `HandEyeMode`, `RobustLoss`, and the `compute_*_feature_residuals` already broken out. | **breaking** | docs: any consumer reaching `optim::*` from the facade rewrites to `optim::specific_item` or the pipeline equivalent |
| 13 | Remove `pub use vision_calibration_core::synthetic::*` glob at facade; replace with `pub mod synthetic { pub use vision_calibration_core::synthetic::{planar, noise, /* hand-pick */}; }`. | **breaking** | docs |
| 14 | Drop `pub use vision_calibration_core::{RigDataset, RigViewObs, View};` from `vision-calibration-optim`. Optim functions take these as parameters; they're spelled `vision_calibration_core::RigDataset` at the call site. | **breaking** | callers update one import path |
| 15 | Drop `pub use camera_matrix::*; pub use distortion_fit::*; …` block in `vision-calibration-linear/src/lib.rs`. Consumers use the module paths (`linear::homography::dlt_homography`). | **breaking** | callers update from `linear::dlt_homography` → `linear::homography::dlt_homography`. The earlier-introduced `prelude` covers the 5–10 most common items |
| 16 | Drop `pub use math::*; pub use models::*; pub use ransac::*; pub use types::*; pub use view::*` in `vision-calibration-core/src/lib.rs`. Rename the private modules (`math` → `algebra` is one option), expose the items at the crate root via a single explicit `pub use` block per logical group. | **breaking** | only Path changes inside `core`; downstream consumers see the same crate-root items. The current `pub use foo::*` simply becomes `pub use foo::{ListedItem1, ListedItem2, …}` |
| 17 | Feature-flag `vision-calibration-core::test_utils` behind `feature = "test-utils"` (off by default), `#[doc(hidden)]` the module. (Or: make it `pub(crate)` outright — inventory showed zero external consumers.) | **breaking** | tests that import `core::test_utils::*` add the feature in their dev-dep declaration; or migrate to the new replacement helpers in `core::synthetic` |

### Phase 4 — Rename + close growth

| # | Change | Semver | Migration note |
|---|--------|--------|----------------|
| 18 | Rename `step_set_*` → `step_*_with_seed` (or chosen alternative). Drop the deprecated alias added in #3. | **breaking** | callers rename; docs in ADR 0011 updated |
| 19 | Consolidate the 5× `IntrinsicsInitOptions` / `IntrinsicsOptimizeOptions` duplication. Either: (a) hoist a single `IntrinsicsInitOptions` to `vision_calibration_pipeline::common` and re-export from each problem module, or (b) rename to `<Problem>IntrinsicsInitOptions` if their bodies differ. **Decision deferred to body-diff inspection in Phase B**. | **breaking** | callers update one or both import paths |
| 20 | Audit `Error` re-exports. The facade has `pub use vision_calibration_pipeline::Error;` at crate root *and* via every per-problem module (transitively). Pick one canonical path (`vision_calibration::Error`), drop the per-module re-export confusion. | **breaking** | callers update one import path |
| 21 | Add `#[doc(hidden)]` to `vision_calibration_core::ransac::*` types that are not part of the curated surface (the inventory shows ~6 RANSAC trait/struct items at the crate root). | **breaking** | none if items are not used externally; inventory in Phase B subagent re-verifies |
| 22 | Run `cargo public-api --diff` against the published 0.4.0 baseline. Confirm the surface matches the plan. Update CHANGELOG with the full migration table. | non-breaking | none |

## Open questions

These need user judgment before Phase B; please respond inline (edit
this file) before approving the plan.

1. **`IntrinsicsInitOptions` × 5 — same body or different?** I have not
   diffed the bodies. If identical → hoist (one type, re-export). If
   different → rename. If "we don't know yet whether they will
   diverge" → keep current naming, document the per-module *intent* in
   each docstring. Which?

2. **`vision-calibration-linear`'s `pub mod prelude`** — what belongs?
   The current 7-item prelude is: `DistortionFitOptions`,
   `dlt_homography`, `IterativeIntrinsicsOptions`,
   `estimate_intrinsics_iterative`, `estimate_planar_pose_from_h`,
   `PlanarIntrinsicsLinearInit`,
   `estimate_intrinsics_from_homographies`. Is that the actual "90%
   path" through `linear`, or should it shrink/grow?

3. **`*ManualInit` types** — ADR 0011 made these the explicit seed
   surface. Per-problem they're: `PlanarManualInit`,
   `RigIntrinsicsManualInit`, `RigExtrinsicsManualInit`,
   `RigHandeyeIntrinsicsManualInit`, `RigHandeyeRigManualInit`,
   `RigHandeyeHandeyeManualInit`, `SingleCamIntrinsicsManualInit`,
   `SingleCamHandeyeManualInit`, `LaserlineDeviceManualInit`,
   `RigLaserlineDeviceManualInit`, `ScheimpflugManualInit`. Eleven
   types. Is this the intended granularity (one per problem × stage),
   or do you want a smaller, hoisted set?

4. **Diagnostic channel — design now, or defer?** This plan assumes
   "no formal `Diagnostics` struct ships in 0.5.0 — the `step_*`
   return values cover the in-repo consumers, and `session.log()` +
   the existing per-feature residuals on `*Export` cover anything
   beyond that." If you want a richer diagnostics surface before
   0.5.0 (e.g., a `PlanarDiagnostics` returned alongside the export),
   say so now — it changes the shape of L1's fix.

5. **`SensorMode` and `HandEyeMode` `#[non_exhaustive]`?** Closed
   enums per ADR 0013 (`Pinhole | Scheimpflug { fx_tilt, fy_tilt }`).
   Adding new variants in the future would only be for new sensor
   models. If you anticipate a 3rd variant within 1.0,
   `#[non_exhaustive]` now; otherwise keep them closed.

6. **`xtask` crate** — currently workspace-internal, not published.
   Does it need to re-export anything from the workspace public API,
   or is its `cargo xtask emit-schemas` entirely a private tool?

7. **`vision-calibration-detect::Detector` trait** — currently `pub
   trait Detector: Send + Sync`. Is this trait designed for downstream
   `impl` (i.e., consumers add their own detector)? If yes, it should
   stay open; if no, seal it.

8. **`vision-calibration-dataset`** — the dataset spec types
   (`DatasetSpec`, `CameraSource`, `Topology`, etc.) are themselves
   the contract this crate exists for. The classification slots them
   all under "Result / Consumer entry." No changes proposed. Confirm?

## Appendix — files and artifacts

- This report: `API_REVISION.md` at the repo root.
- Per-crate PAPI dumps (full verbatim, generated during the inventory
  pass): `/tmp/{core,linear,optim,pipeline,dataset,detect,facade}_papi.txt`.
  These are scratch files; they will not be committed. If you want
  them in-repo for posterity, say so and Phase B will move them under
  `docs/api-revision/`.

- Classification CSV: not committed yet; if useful (the report above
  summarizes the tier counts but doesn't list each item by row), say
  so and Phase B will produce one alongside the work.

- Phase B execution: each row of the plan above becomes one subagent
  dispatch. The user approves each, the subagent makes only that
  change, runs the workspace gates (`cargo fmt`, `cargo clippy
  --workspace --all-targets --all-features -- -D warnings`,
  `cargo test --workspace --all-features`, `cargo xtask emit-schemas
  --check`, `cargo +1.88.0 test --workspace --all-features --locked
  --no-run`, `cargo doc --workspace --no-deps`, app-side `cargo
  clippy` + `bun run build`), and reports concise results.
