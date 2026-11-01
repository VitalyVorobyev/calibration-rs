# API Revision — calibration-rs (for 0.11, ahead of v1.0)

Working document for the v1.0 API audit (ROADMAP Phase 1, criterion 9).
It drives the cleanup PRs and is deleted when they land. The inventory was
made with `cargo public-api --simplified` on `main` at b09a90dd.

## Summary

The current public surface is about 8,700 listing lines across 9 published
crates:

| crate | listing lines |
|---|---|
| pipeline | 3,455 |
| core | 1,480 |
| optim | 953 |
| dataset | 766 |
| mvg | 636 |
| detect | 523 |
| linear | 434 |
| facade | 365 |
| geometry | 102 |

The facade has 341 `pub use` entries in about 24 modules, and 43 symbol
names reachable at more than one path.

**Verdict.** The core is in good shape:
- The optimizer's IR, factors and backends are private.
- No solver crate (faer, tiny-solver, factrs) or detector crate leaks into
  a signature.
- Every problem type follows one session pattern.

The rot is in four places:
1. **Duplicate paths.** The same symbol is reachable at 2–5 facade paths.
   The pipeline's `session` exposes its submodules as well as their items.
2. **Parallel entry points.** There are 14 `step_*_with_seed` twins, 7
   `build_*_input` functions, and 7 homonymous `run_calibration` functions,
   one of which the prelude renames.
3. **Test and stage helpers on the published surface.** These include
   `core::synthetic`, `make_pinhole_camera`, `linear::handeye::MotionPair`,
   `mvg::cheirality` / `degeneracy`, and the `LaserPlane::*_dvec` packing.
4. **A determinism violation.** `session::current_timestamp` puts wall-clock
   values in exports, which AGENTS.md §4 forbids.

**Plan.** 19 changes in 6 phases. Every phase is a breaking (0.x minor)
change. The maintainer pre-authorized breaking changes, and there are no
downstream users.

## The intended contract

What a consumer needs, reached through the facade `vision_calibration`:

- **`session`:** `CalibrationSession<P>`, `ProblemType`, `Error`.
- **Seven problem modules:** each exposes `<Name>Problem`, `<Name>Config`,
  `<Name>Input`, `<Name>Export`, its `step_*` functions and one
  `run_calibration`.
- **`common`:** the shared config (`CameraConfig`, `CameraOverride`,
  `SolverConfig`, `RobotPoseConfig`, the init/optimize option structs) and
  `CalibratedCameras`.
- **`core`:**
  - the camera model (`CameraModelSpec`, `CameraParams`, `FxFyCxCy`, the
    distortion and sensor parameter types);
  - observations (`CorrespondenceView`, `PlanarDataset`, `RigDataset`,
    `View`, `PointCovariance`);
  - geometry aliases (`Pt2`, `Pt3`, `Iso3`, …);
  - `Uncertainty` / `ParameterCovariance`.
- **`dataset`:** `DatasetSpec`, with loading from a path.
- **`dataset_runner`:** manifest → problem input.
- **`detect`:** the detectors and the detection cache.
- **`analysis`:** the quality report, projection uncertainty and next-view
  planning.
- **`mvg`:** two-view geometry, triangulation, rectification and dense
  stereo, as tools in their own right.

The lower crates (`linear`, `optim`, `geometry`) stay published as sharp
tools: small, documented, and not on the facade's main path.

## Consumer workflow — before

```rust
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::build_planar_input;
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};
use vision_calibration::prelude::*;

let spec: DatasetSpec = serde_json::from_str(&std::fs::read_to_string("dataset.json")?)?; // own serde_json/toml
let cache = FsDetectionCache::new(".detect_cache");             // no "no cache" option
let run = build_planar_input(&spec, base, &cache, false)?;     // bare bool, one fn per topology
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(run.input)?;                                 // paths/rejections dropped by hand
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let export = session.export()?;
let k = export.params.intrinsics();
let d = export.params.distortion();                           // Option<BrownConrady5>, None for other models
```

## Consumer workflow — after

```rust
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::{build_input, Detection};
use vision_calibration::planar_intrinsics::{run_calibration, PlanarIntrinsicsProblem};
use vision_calibration::session::CalibrationSession;

let spec = DatasetSpec::from_path("dataset.toml")?;               // TOML or JSON
let input = build_input::<PlanarIntrinsicsProblem>(&spec, base, Detection::cached(".detect_cache"))?;
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::with_input(input)?;
run_calibration(&mut session)?;
let export = session.export()?;
let camera = &export.params.camera;                            // CameraParams: every model, no Option
```

## Classification (significant items; the full listings are in the audit scratch)

| Item | Crate | Tier | Smell | Target | In-repo consumers |
|---|---|---|---|---|---|
| `linear::handeye::{build_all_pairs, MotionPair}` | linear | Internal | stage state, nalgebra pub fields | `pub(crate)` | none outside linear |
| `LaserPlane::{to_dvec, from_dvec, from_split_dvec, normal_to_dvec, distance_to_dvec}` | optim | Internal | solver packing, `DVector` in API | `pub(crate)` | none |
| `mvg::{cheirality, degeneracy}`, `triangulate_two_view_partial`, `triangulate_point_two_view` | mvg | Internal | stage functions | private modules | none |
| `mvg::residuals::*` | mvg | Internal | metric helpers | `#[doc(hidden)]`, or move to geometry | optim factrs backend |
| `session::{types, problem_type, calibsession}` modules | pipeline | Internal | extra paths to one item | private modules, single `session::X` | none |
| `detect::cache::hash_image_bytes` (second path) | detect | Internal | duplicate path | one path | none |
| Shared `Intrinsics*Options`, `Handeye*Options`, `Rig*Options` re-exported per problem module | facade | Config-stable | 3–5 paths each | only at `common::` | py, app (via `common`) |
| `rig_handeye` re-exporting all of `rig_extrinsics` | facade | — | duplicate surface | only its own items | — |
| `prelude::run_planar_intrinsics` (renamed `run_calibration`) | facade | — | facade logic, rename | drop the rename | examples |
| `core::synthetic::*`, `make_pinhole_camera`, `pinhole_camera_params` | core | Internal (test data) | test generators in a published crate | behind a `synthetic` feature; facade `synthetic` behind the same feature | tests, bench, examples (21 + 41 files) |
| `session::current_timestamp`, `ExportRecord.timestamp`, `LogEntry` time | pipeline | Diagnostic | wall clock in results (AGENTS.md §4) | timestamps out of exports; log keeps them | py session, 1 example |
| `step_*_with_seed` (14) | pipeline | Result (duplicated) | twin entry points | seed in the step options (`IntrinsicsInitOptions::seed`, …) | examples-private, py |
| `run_calibration` ×7 | pipeline | Result | homonyms | keep: one per module is the pattern; drop the prelude rename | — |
| `dataset_runner::build_*_input` (7), `force_redetect: bool` | pipeline | Result | parallel fns, bare bool | `build_input::<P>()` via a `ProblemType` hook, `Detection` enum | py, bench |
| `DatasetSpec` has no loader | dataset | Result gap | — | `DatasetSpec::from_path` (TOML/JSON) | py reimplements it |
| `PlanarIntrinsicsParams::distortion() -> Option<BrownConrady5>` | pipeline/optim | Result | Option that lies for other models | remove; `params.camera.distortion` is the typed enum | examples |
| `SolveReport` as `report` on every Export and Estimate | optim/pipeline | Diagnostic | solver stats in the result | group under `Export::diagnostics` with `per_feature_residuals` | py, pipeline |
| `optim::{with_leading_radial, projection_covariance_at, view_information, ViewInformation, ViewPose}` | optim | Internal used cross-crate | pipeline-only helpers | `#[doc(hidden)]` "pipeline-internal", or move into pipeline | pipeline only |
| `core::{INTRINSICS_PARAM_NAMES, FixedParams, NoDistortion}` | core | Internal | model scaffolding | `#[doc(hidden)]` or `pub(crate)` + move | optim |
| `core::coordinate_utils` (duplicates root fns) | core | — | duplicate path | one path | app epipolar |
| `BrownConrady5::iters` etc. (pub field) | core | Config-tuning | solver knob on a model type | keep; document as the undistortion iteration count | many |
| Four `Error` types | all | Result | name clash | keep per crate; facade `Error` the only one at the facade root | — |
| `detect::Detector::detect_json(&DynamicImage, &Value)` | detect | Result | untyped config, image + serde_json | open question | pipeline, examples-private |

## Leak deep-dives

**Wall clock in exports.** `ExportRecord` stamps the export with the current
time, so two identical runs give different JSON. That breaks golden
comparisons and contradicts the determinism rule. The fix keeps time in the
session log only, which is not part of any export. A caller that wants a
time adds it.

**Seed twins.** For each step, `step_init_with_seed(session, seed, opts)`
mirrors `step_init(session, opts)`, 14 functions in all. The seed is an
option of the init step. Moving it into the init options
(`IntrinsicsInitOptions { seed: Option<IntrinsicsManualInit>, … }`) halves
the step API and makes seeding discoverable from the one function.

**Parallel `build_*_input`.** The topology is already in the manifest, and
each problem type knows its input. A `ProblemType::build_input` hook (or one
`build_input::<P>`) gives a single typed entry point. The bare
`force_redetect: bool` becomes `Detection::{Cached(cache), Fresh}`, and
`NoCache` becomes public instead of being reimplemented in Python.

## Diagnostics relocation

Every Export gains `diagnostics` (`#[serde(default)]`), holding the solve
report (`final_cost`, `num_iters`), the per-feature residuals, and the
image manifest. The result fields keep their names:
- `params`;
- `mean_reproj_error`, `per_cam_reproj_errors`;
- `uncertainty`;
- `robot_deltas`, `robot_pose_sigma`.

`analysis::ReprojReport` and its rig-stage levels stay in `analysis` as the
diagnostic report. They are documented there as diagnostics, with a looser
stability promise.

## Breaking-change plan

Each phase is one PR. Each leaves the workspace green, and each has its own
CHANGELOG migration notes.

### Phase 1 — Hide unused internals (breaking, no consumers)
1. `linear::handeye::{build_all_pairs, MotionPair}` → `pub(crate)`.
2. `optim::LaserPlane::*_dvec` → `pub(crate)`.
3. `mvg::{cheirality, degeneracy}` and the partial and two-view
   triangulation helpers → private.
4. `pipeline::session::{types, problem_type, calibsession}` → private
   modules, so `session::X` is the only path; `detect` gets one
   `hash_image_bytes` path.

### Phase 2 — One path per facade symbol (breaking)
5. Shared option and result types only at `common::`. Problem modules stop
   re-exporting them; `rig_handeye` stops re-exporting `rig_extrinsics`.
6. Drop `prelude::run_planar_intrinsics`. The prelude keeps only types, no
   renamed functions.
7. `core::coordinate_utils` → private; the root functions remain.
7b. The facade drops its `linear` and `geometry` modules; items a facade
    consumer needs are re-exported by name (decision 2).

### Phase 3 — Determinism and diagnostics (breaking JSON)
8. `current_timestamp` and the session bookkeeping constructors become
   private. (Revised after a closer look: exports carry no timestamp; the
   times live in the session's metadata, log and export records, which are
   session state, not results, so they stay.)
9. `Export::diagnostics` collects `per_feature_residuals` and
   `image_manifest`. Planar's `report` is dropped (it stays on the session
   output), and the laserline export gets `params` + `uncertainty` like the
   others instead of embedding the optimizer's estimate. Python and the app
   follow, and the schemas are regenerated.

### Phase 4 — One entry point per step (breaking)
10. ~~Seeds move into the init options.~~ Dropped: the intrinsics, rig and
    hand-eye init options are one type shared by several problems while
    their seeds differ per problem, so a seed field would split the shared
    options. `step_init` is literally `step_init_with_seed` with the default
    seed; the pair is an explicit, discoverable seeded route, not two
    competing entry points.
11. `PlanarIntrinsicsParams::distortion()` and similar model-specific
    `Option` accessors are removed in favour of `params.camera`.

### Phase 5 — The manifest path (additive first, then breaking)
12. Additive: `DatasetSpec::from_path` / `from_toml_str` /
    `from_json_str`, and a public `NoDetectionCache`.
13. The `Detection` enum replaces `cache, force_redetect: bool` in every
    converter, with an `Uncached` mode (Python's private `NoCache` goes).
14. ~~One `build_input::<P>`.~~ Dropped: the seven converters return seven
    different typed results and three need a laser extractor; a generic
    entry point would need a trait with associated types and a runtime
    error for a missing extractor. Seven typed functions named after their
    topology are the simpler, honest API.

### Phase 6 — Test helpers and cross-crate internals (breaking)
15. `core::synthetic`, `make_pinhole_camera` and `pinhole_camera_params` go
    behind a `synthetic` feature. The workspace's dev-dependencies enable
    it, and the facade's `synthetic` module sits behind the same feature.
16. The pipeline-only optim helpers (`with_leading_radial`,
    `projection_covariance_at`, `view_information`) and the core model
    scaffolding (`INTRINSICS_PARAM_NAMES`, `FixedParams`, `NoDistortion`)
    become `#[doc(hidden)]` and are documented as workspace-internal, as do
    optim's `optimize_*` functions and `*SolveOptions` (decision 3). The
    alternative is to move them into their only consumer.
17. Write down the public-dependency policy (criterion 9):
    - nalgebra is a deliberate public dependency, re-exported as
      `vision_calibration::nalgebra`, and its major version moves only with
      a minor release of ours.
    - `image` and `serde_json` appear only in `detect` and
      `dataset_runner`; see the open questions.
18. `cargo public-api` diff gate in CI. It runs against the committed
    listings and fails on an unreviewed change to the facade surface.
19. Re-run this inventory and record the final surface (after Phase 7).

### Phase 7 — Typed detector boundary (breaking)
20. `detect` takes a typed `DetectorConfig` enum (one variant per detector,
    carrying its config struct) and a grayscale image view, in place of
    `Detector::detect_json(&DynamicImage, &serde_json::Value)`. The cache
    key hashes the typed config's canonical JSON internally. `image` and
    `serde_json` leave detect's public signatures.

## Decisions (maintainer, 2026-11-01)

1. **Typed detector config: yes.** Phase 7 below.
2. **Lower crates on the facade: not as whole crates.** The facade's
   `linear` and `geometry` modules go away (change 7b). An item a facade
   consumer needs (book, facade examples) is re-exported by name in the
   module where it belongs; bench and the private examples depend on the
   lower crates directly.
3. **`optim::optimize_*` and `*SolveOptions`: workspace-internal.** At 1.0
   every published crate moves in lockstep, so a public `optimize_*` would
   become a semver promise on the most churn-prone API in the workspace
   (its options changed three times in 0.10). Every user need is met by the
   pipeline's step functions. They become `#[doc(hidden)]`, documented as
   workspace-internal and outside the semver promise (folded into change 16).
