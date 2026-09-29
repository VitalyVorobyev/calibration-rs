# ADR 0018: Schema-Driven UI for Configs and Dataset Manifests

- Status: Accepted
- Date: 2026-05-02

## Context

The "complete calibration workflow in the app" goal exposes every
`*Config` knob (8 problem types × ~10 fields each ≈ 80+ knobs), every
detector config (4 detectors × their own params), and the new
`DatasetSpec` (camera sources, target spec, robot poses, pose
conventions). Hand-writing React forms for ~120 fields across that
surface is not the actual cost — _keeping them in sync as configs
change pre-1.0 is_. Config shapes churn (ADR 0013, the manifest sweep, the move from `*Input`
JSON to `DatasetSpec`), and hand-written forms drift silently every time.

The grill session settled the principle ("schema-driven"); this ADR
pins the architecture.

## Decision

### 1. Source of truth: Rust types, with `JsonSchema` derives

Every user-editable config, input, and dataset type derives
`schemars::JsonSchema` under a `schemars` feature flag:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PlanarIntrinsicsConfig { … }
```

The feature flag keeps schemars optional for downstream consumers
(Python bindings, CI builds that don't need schemas) but the workspace
quality gate runs with `--all-features` so drift surfaces in CI.

Coverage: all problem-type configs, their sub-configs, the
shared option types (`RobustLoss`, `HandEyeMode`, `IntrinsicsFixMask`,
`DistortionFixMask`, `ScheimpflugFixMask`, `ScheimpflugParams`,
`SensorMode`, `LaserlineResidualType`), plus the new `DatasetSpec`
tree.

### 2. Schema emission via xtask, committed to the app

`cargo xtask emit-schemas` writes JSON Schemas to
`app/src/schemas/<name>.json` (one file per config family). The schemas are
committed to the repo so:

- Frontend builds don't need a Rust toolchain in their critical path.
- LSP / Vite read them as static assets (fast).
- `git diff` on a config change shows both the Rust change and the
  schema change in the same review.
- CI runs `cargo xtask emit-schemas --check` which fails the build
  if committed schemas drift from sources, enforcing developer
  discipline without a build-time regen.

`build.rs`-based emission was rejected: it slows clean builds,
confuses incremental tooling, and makes schema diffs invisible at
review time.

### 3. Form rendering: `<ConfigForm schema={…} value={…} onChange={…}/>`

The React side renders any of the emitted schemas through a single
component:

- v0: wrap `@rjsf/core` (well-known JSON Schema form library).
- Per-field overrides via the React-JSON-Schema-Form `uiSchema`
  mechanism for cases that need bespoke widgets (3-axis rotation
  widget, file pickers, etc.).
- If `@rjsf/core` proves too heavy or its rendering doesn't match the
  app's design tokens, fall back to a trimmed in-house generator
  (~300 LoC; sufficient because we only need to render the schema
  subset schemars produces, not arbitrary JSON Schema).

### 4. Foreign-type strategy

`*Config` types in this workspace are made entirely of internal types
plus primitives — no `nalgebra::Isometry3` or other foreign types in
the config tree. The feasibility check on `PlanarIntrinsicsConfig`
and `RigHandeyeConfig` confirmed schemars derives compile cleanly
without any `#[schemars(with = …)]` shims. If a foreign type _does_
appear in a config later (e.g. an `Iso3` initial-pose seed), the
escape hatch is a per-field shim mapping to a hand-written
`JsonSchema` impl.

`*Input` types contain foreign types (`Iso3`, `Pt2`, `Pt3`) but
**don't need schemars** — under ADR 0016 they're internal IR, never
edited by users.

## Consequences

- Adding a new config field is _free_ in the UI: derive picks it up,
  schemars regenerates, the form re-renders. No matching React
  change required.
- Removing a field forces the UI to follow: if a renamed field still
  appears in the form, the schema regen catches the drift.
- Pre-1.0 config churn is absorbed automatically — the cost the user
  was implicitly paying for "complete parameter control" goes to
  near-zero.
- The choice of `@rjsf/core` is reversible: the form-component
  abstraction is the boundary, swapping the underlying renderer is a
  single-file change.
- Schemars adds one transitive dep when the feature is enabled. The
  base build cost (no schemars) is unchanged.

## Wire types and export discriminator

The `schemars` surface covers every `*Config` and also every pipeline `*Export`
type (and its closure: `SolveReport`, `LaserlineEstimate`/`LaserlineStats`,
`CameraParams`, `PerFeatureResiduals`, `ImageManifest`, an `Iso3Schema` wire
proxy for nalgebra isometries, ...), behind the same feature gate. The desktop
app's TypeScript wire types are generated from these schemas
(`app/src-tauri/src/bin/emit_schemas.rs` -> `app/schemas-generated/diagnose_wire.json`
-> `app/src/types/generated/diagnose-wire.ts`, drift-checked in CI), so there
are no hand-written mirrors. Config schemas are the runtime source for
`<ConfigForm/>`; export schemas are build-time codegen inputs only.

Every pipeline `*Export` serializes a **required** `kind` discriminator: the
shared `ExportKind` unit enum (`vision_calibration::common::ExportKind`,
`#[serde(rename_all = "snake_case")]`, one variant per problem type), carried
as each export's first field. Consumers narrow on the single tag rather than
probing which fields are present; deserialize is strict (a missing `kind`
errors, no `serde(default)`). The enum derives `JsonSchema`, so it flows
through the same pipeline as a `"planar_intrinsics" | "scheimpflug_intrinsics"
| ...` union. The app's `detectExportKind` is a single validated read of
`data.kind` against that union, and its label `Record<ExportKind, string>` is
the single source of the label vocabulary and the recognised-kind set (a
renamed/added Rust variant fails the TypeScript build until the record is
updated). Python export result parsers read specific keys and ignore `kind`.
