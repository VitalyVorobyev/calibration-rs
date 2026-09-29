# ADR 0014: Tauri 2 Desktop App for Calibration Diagnose

- Status: Accepted
- Date: 2026-05-01

## Context

The calibration library exposes eight problem types, manual init (ADR 0011),
and per-feature reprojection residuals on every export (ADR 0012). What an
engineer cannot get from `cargo run -p vision-calibration --example …` is a
visual read of those residuals, which is how *root causes of suboptimal
calibration* (wrong pattern parameters, wrong distortion model, wrong camera
model, wrong initial state) are identified. A desktop app whose core is the
**diagnose mode** fills that gap. File pickers, runners, and 3D views are
enrichments on top of it, not prerequisites.

## Decision

### 1. Diagnose-first

The app's core is a viewer of a calibration export: per-image residual vector
fields overlaid on the source frames. Detection wrapping, the calibration
runner, and 3D views are layered on afterwards, prioritised by what the
diagnose workflow actually misses.

### 2. Framework: Tauri 2 + React + TypeScript

Tauri 2 earns the production posture (signed-app shell, native windowing)
while keeping the Rust backend authoritative: the webview is the rendering
surface; all calibration-domain logic lives in Rust through the
`vision-calibration` facade. Rejected:

- **rerun.io** — developer-tool aesthetic; gRPC IPC clashes with this
  project's zenoh-Rust stack.
- **egui** — less ecosystem leverage for production polish.

### 3. Diagnose scope

- **Input:** a calibration export JSON plus the images its manifest references.
- **UI:** the selected `(pose, camera)` image with an arrow per feature drawn
  `observed_px -> projected_px`, colour-coded by `error_px` against the
  `[1, 2, 5, 10]` px histogram bucket edges.
- **What it surfaces:** the residual vector field exposes "wrong distortion
  model" (radial residual pattern) and "wrong camera model" (systematic
  asymmetry); "wrong pattern parameters" is partially surfaced when scale
  errors are not absorbed by the optimizer.
- **What it does not address:** "wrong initial state". Diagnosing init failure
  requires driving the runner with perturbed inits and showing the
  cost-landscape signature, which is a separate backend feature.

### 4. Image-data contract: export-side, additive

`vision-calibration-core` owns an `ImageManifest`, an optional field on the
relevant `*Export` types:

```rust
pub struct ImageManifest {
    pub root: PathBuf,                       // resolved relative to export.json
    pub frames: Vec<FrameRef>,
}
pub struct FrameRef {
    pub pose: usize,
    pub camera: usize,
    pub path: PathBuf,                       // relative to ImageManifest.root
    pub roi: Option<PixelRect>,              // for tiled multi-camera frames
}
pub struct PixelRect { pub x: u32, pub y: u32, pub w: u32, pub h: u32 }
```

The contract is **viewer-side, not session-side**: the calibration pipeline
never reads the manifest; the runner (or a downstream tool that ships both
calibration and images) populates it on the way out. ROI supports tiled
multi-camera formats (e.g. a 4320x540 six-camera strip) without a new image
format: multiple `FrameRef`s point at the same `path` with disjoint ROIs.

The field is `Option<…>` with
`#[serde(default, skip_serializing_if = "Option::is_none")]`, so exports
without it stay byte-identical. `image_manifest` indexing matches
`per_feature_residuals` indexing (both pose-major over the same
`(pose, camera)` slots); the type system does not enforce this, so manifest
entries always come from the same loop that emits the residuals.

### 5. Fixture

`crates/vision-calibration/examples/planar_synthetic_with_images.rs`
generates a deterministic fixture: a 9x6 inner-corner checkerboard at 5
poses, rendered to 640x480 PNGs via the closed-form planar-to-image
homography, with sigma ~ 0.3 px observation noise. Outputs land in
`target/fixtures/planar_synthetic_with_images/` (gitignored). A regression test
(`crates/vision-calibration/tests/planar_synthetic_with_images.rs`) reuses the
generator and pins residual ceilings (mean < 0.5 px, max per-feature < 1.5 px,
all expected records present, manifest matches every PNG). A synthetic fixture
with known noise is what lets CI assert numeric properties, not just file
existence.

### 6. Layout: `app/` as a sibling to `crates/`

```
app/
  package.json                  # vite + react + tauri (bun-managed)
  vite.config.ts
  src/                          # React + TypeScript UI
  src-tauri/                    # Rust backend (its own crate, depends on
                                #   vision-calibration via path)
  dist/                         # Vite output; placeholder index.html committed
                                #   so `cargo check` succeeds before
                                #   `bun run build` has ever run
```

`app/` is **excluded from the root Cargo workspace** (`exclude = ["app"]`):
the Tauri build-time dependency tree (tauri-build, tauri-macros, the webview
FFI surface) has no business slowing `cargo build --workspace` for the
library. The committed `app/dist/index.html` placeholder is overwritten by
Vite on first build.

### 7. IPC surface

The webview receives the export as untyped JSON (`load_export`) and images as
`data:` URLs (`load_image`); the frontend narrows the JSON to TypeScript types
generated from the Rust schemas (ADR 0018). The current command set, workspace
layout, and dev workflow are documented in `app/README.md`.

## Considered alternatives

- **Build file loader, detection wrap, runner, and 3D viewer first.**
  Rejected: months of plumbing precede the user-facing value, and a
  residuals-shape mistake found late forces re-work across the whole stack.
- **Arrows on a bare coordinate grid (no image backdrop).** Rejected:
  diagnostic patterns (radial distortion, asymmetric Scheimpflug residuals)
  read meaningfully against the scene, not against an empty axis.
  `ImageManifest` is the cost of admission.
- **Session-input-side image contract** (a `CalibrationDataset` the session
  ingests and re-emits). Rejected: pulls dataset semantics (format
  negotiation, ROI conventions, pose provenance, missing cameras) into the
  viewer's contract. The export-side contract is additive and reversible.
- **A public real-data fixture.** Rejected: adds image-license review and lets
  CI assert nothing the synthetic fixture cannot.
- **Multi-platform signed installers.** Out of scope: each platform's signing
  and notarization carries operational overhead (keys, CI, renewal); the
  distribution model is `bun run tauri dev` / local builds.

## Consequences

- The residual contract is exercised by a fixture and regression test at the
  point the manifest schema changes.
- `vision-calibration-core` owns the manifest; every export pulls the same
  definition, so the app cannot define its own image-reference type.
- The workspace stays fast: the Tauri build only fires inside `app/`.
- Moving an `export.json` without its images breaks the viewer; the frontend
  surfaces this as an inline load error.
- Growing the manifest schema (timestamp, exposure, gain) migrates existing
  fixtures; the `Option` wrapper contains the blast radius.
- Tauri 2 is pinned (`tauri = "2"`, `@tauri-apps/api ^2`,
  `@tauri-apps/plugin-dialog ^2`); breaking changes within 2.x are accepted
  churn.
- A 640x480 image with a few hundred canvas arrows is well within HTML5 canvas
  capacity. If a real dataset (thousands of arrows per view) outgrows it, the
  canvas can move to WebGL driven by measurement.

## Out of scope

- Init-failure diagnosis (perturbed re-runs, cost-landscape signature).
- Image formats beyond PNG (no TIFF, RAW, 16-bit).
- Multi-OS signed installers, auto-update, code signing.
- Python parity for the manifest field.
