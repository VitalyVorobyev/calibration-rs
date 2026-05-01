# ADR 0014: Tauri 2 Desktop App for Calibration Diagnose (Track B Kickoff)

- Status: Accepted
- Date: 2026-05-01

## Context

Track A shipped 2026-05-01: eight problem types across the pinhole +
Scheimpflug rig family, manual init (ADR 0011), and per-feature
reprojection residuals on every export (ADR 0012). The remaining
load-bearing path on the four-track plan (`docs/ROADMAP.md`) is Track B
— a desktop app whose stated MVP is the **diagnose mode**: visualising
residuals so an experienced engineer can identify *root causes of
suboptimal calibration* (wrong pattern parameters, wrong distortion
model, wrong camera model, wrong initial state).

The pre-A6 roadmap framed Track B as B0–B6, with B5 ("diagnose mode")
listed sixth — behind file load (B1), detection wrap (B2), runner (B3),
and the 3D viewer (B4). A grill session against that ordering surfaced
the load-bearing problem: B0–B4 as scoped re-implements what
`cargo run -p vision-calibration --example …` already gives a
calibration engineer at the terminal — file picker instead of `argv`,
progress bar instead of stdout, frustums instead of JSON. The actual
new capability the engineer cannot get today is the residual
visualisation. Building 4–6 months of plumbing first risks discovering
in month 5 that the residuals contract is wrong for the diagnose UI.

This ADR records the resequencing and pins the v0 design.

## Decision

### 1. Sequencing — diagnose-first

Track B's first deliverable (B0) is a **passive viewer of one calibration
export**. Detection wrapping, calibration runner, 3D viewer, and the
remaining B1–B6 surface area are deferred to **post-B0 enrichments**.
Their priority is decided by what the user actually misses while using
v0 — not by a pre-committed phase order. The roadmap is updated to
reflect this.

### 2. Framework — Tauri 2 + React + TypeScript

Choice already grilled and settled in prior session memory: **Tauri 2 +
React + TypeScript + Three.js** (the Three.js piece is deferred out of
v0). Two finalists were rejected:

- **rerun.io** — developer-tool aesthetic, gRPC IPC clashes with this
  project's zenoh-Rust stack, posture-mismatched for a long-lived
  internal production tool.
- **egui** — less ecosystem leverage for production polish; pays a
  visual-quality cost the grill session deemed not worth ~6 months of
  ergonomic work.

Tauri 2 with a React frontend earns the production posture (signed-app
shell, native windowing, native menus once needed) while keeping the
Rust backend authoritative. The webview is the rendering surface; all
calibration-domain logic lives in Rust through the
`vision-calibration` facade.

### 3. v0 scope — passive viewer, headline overlay = 2D arrows on image

Concretely:

- **Input:** one `PlanarIntrinsicsExport` JSON file plus, alongside it,
  the PNG images the manifest references.
- **UI:** one window, one route. File-open button, `(pose, camera)`
  dropdown, a `<canvas>` rendering the selected image with an arrow per
  feature drawn `observed_px → projected_px`, color-coded by
  `error_px` against the existing `[1, 2, 5, 10]` px histogram bucket
  edges.
- **What this surfaces:** the per-image residual vector field is the
  primitive that exposes failure modes (2) "wrong distortion model"
  (radial residual pattern) and (3) "wrong camera model" (systematic
  asymmetry). Failure mode (1) "wrong pattern parameters" is partially
  surfaced when scale errors are not absorbed by the optimizer.
- **What v0 explicitly does not address:** failure mode (4) "wrong
  initial state". Diagnosing init failure properly requires the tool
  to *drive* the runner with perturbed inits and show the
  cost-landscape signature. That is a v2 backend feature; v0 ships
  without it and the limitation is named so users don't expect
  otherwise.

### 4. Image-data contract — Export-side, additive

`vision-calibration-core` gains a new `ImageManifest` type, optional
field on the relevant `*Export` types:

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

The contract is **viewer-side, not session-side**: the calibration
pipeline never reads the manifest; the runner (or a downstream tool that
shipped both calibration + images) populates it on the way out. ROI
supports tiled multi-camera formats (e.g. the puzzle 130×130 4320×540
6-camera strip) without a new tiled-image format — multiple `FrameRef`s
point at the same `path` with disjoint ROIs.

The field is `Option<…>` and serialized with
`#[serde(default, skip_serializing_if = "Option::is_none")]`, so existing
exports remain byte-identical when the field is absent.

For B0 only `PlanarIntrinsicsExport` carries the manifest. The other
seven `*Export` types extend in their own follow-up PRs once the viewer
needs them — single-type-end-to-end is the cheapest way to debug the
contract.

### 5. v0 fixture — synthesized + self-contained

A new example
(`crates/vision-calibration/examples/planar_synthetic_with_images.rs`)
generates the fixture deterministically: a 9×6 inner-corner checkerboard
at 5 hand-tuned poses, rendered to 640×480 PNGs via the closed-form
planar→image homography (no distortion in GT, so this is exact), with a
deterministic σ ≈ 0.3 px noise stream applied to the observations passed
into Zhang. Outputs land in
`target/fixtures/planar_synthetic_with_images/` (gitignored) and are
regenerated by `cargo run --example planar_synthetic_with_images`.

A regression test
(`crates/vision-calibration/tests/planar_synthetic_with_images.rs`)
re-uses the example's generation function and pins residual ceilings
(mean < 0.5 px, max per-feature < 1.5 px, all 270 expected records
present, manifest matches every PNG). Synthetic-with-known-noise is the
only fixture choice that lets CI assert numeric properties on the
output, not just file existence.

Real-data acceptance against the puzzle 130×130 rig is the
**B0.5 task** — once the synthetic fixture round-trips green, extend
`RigHandeyeExport` with the same `image_manifest` field, write a
populator that runs against the puzzle dataset, and point the viewer at
the resulting export. That is where ROI + tiled multi-camera frames
earn their keep.

### 6. App layout — `app/` as a sibling to `crates/`

```
app/
  package.json                  # vite + react + tauri (npm-managed)
  vite.config.ts
  tsconfig.json
  index.html
  README.md
  src/                          # React + TypeScript UI
    main.tsx
    App.tsx
    types.ts
    components/ResidualViewer.tsx
  src-tauri/                    # Rust backend (its own crate)
    Cargo.toml                  # depends on vision-calibration via path
    build.rs
    tauri.conf.json
    capabilities/default.json
    src/{main.rs,lib.rs,commands.rs}
    icons/icon.png
  dist/                         # Vite output; placeholder index.html committed
                                # so `cargo check` succeeds before
                                # `npm run build` has ever run
```

`app/` is **excluded from the root Cargo workspace**. The Tauri Rust
crate pulls in a substantial build-time dep tree (tauri-build,
tauri-macros, the entire webview FFI surface) that has no business
slowing down `cargo build --workspace` for the calibration library
itself. `Cargo.toml` adds `exclude = ["app"]` to make this explicit.

### 7. Tauri commands — two, exactly

- `load_export(path) -> {export: serde_json::Value, export_dir: String}` —
  read the export file, return its contents and the absolute parent
  directory (so the frontend can resolve manifest-relative paths).
  The export is returned as untyped JSON, not a typed Rust struct
  serialized over IPC; the frontend narrows it to a `PlanarExport`
  TypeScript interface that lists only the fields it consumes.
- `load_image(path) -> String` — read the PNG, return a `data:` URL
  the webview can drop straight into `Image()` for canvas rendering.

That is the entire IPC surface for B0. No events, no progress streams,
no long-lived state. Everything else lives in React `useState`.

## Considered alternatives

### A) Drive the original B0–B6 sequencing

Build the file loader, detection wrap, runner, and 3D viewer first;
land the diagnose UI as B5. **Rejected:** months of plumbing precede
the user-facing value, and a residuals-shape mistake found in month 5
forces re-work across the whole stack. The flip absorbs essentially
zero risk that the original ordering avoids — file load + runner are
trivial follow-ons once the viewer exists and we know what it needs.

### B) Render arrows on a bare coordinate grid (no image backdrop)

Considered to keep v0's contract single-file: just `export.json`, no
companion images. **Rejected by the user as not useful.** Diagnostic
patterns (radial distortion residuals, asymmetric Scheimpflug
residuals) read meaningfully against the underlying scene, not against
an empty axis. The `ImageManifest` contract is the cost of admission.

### C) Session-input-side image contract

A `CalibrationDataset` schema that the session ingests *and* re-emits,
carrying images + features + pose metadata as a unified bundle.
**Rejected for v0:** pulls dataset semantics (image-format
negotiation, ROI conventions, pose-source provenance, missing-camera
handling) into the viewer's compile-time contract before the viewer
has rendered a single arrow. Pre-B0 over-design. The Export-side
contract is additive and reversible; the Input-side one isn't.

### D) Borrow a public real-data fixture instead of synthesising

A small public chessboard dataset checked into the repo. **Rejected:**
buys nothing the synthesised fixture doesn't, and adds image-license
review. The synthesised fixture also lets CI assert numeric residual
properties (`mean < 0.5 px`, `max < 1.5 px`) instead of just
"the file exists".

### E) Multi-platform signed installers as part of B0

Originally B6 in the roadmap. **Out of v0 scope entirely.** Apple
Developer ID + notarization, Windows code signing, Linux AppImage /
Flatpak — each is its own engineering rabbit hole with operational
overhead (signing keys, CI integration, certificate renewal). Defer
indefinitely; revisit only if and when the tool earns the audience.
For now `npm run tauri dev` + `cargo install tauri-cli` + manual
local builds are the distribution model.

## Consequences

### Positive

- **Time-to-value drops from months to weeks.** B0 lands a usable
  diagnose surface in ~2–3 weeks (Rust contract + fixture +
  regression test + Tauri shell + ADR), versus 4–6 months under the
  original ordering.
- **The contract is exercised before downstream consumers commit.** The
  fixture + regression test catches manifest schema mistakes inside the
  same PR that introduces them, instead of cross-PR rework.
- **Single source of truth for `SensorMode` × `ImageManifest`.**
  `vision-calibration-core` owns the manifest; every downstream Export
  pulls the same definition. No risk of the desktop app re-defining its
  own image-reference type.
- **Workspace stays fast.** With `app/` excluded, `cargo build
  --workspace` stays at its current dep set; the Tauri build only fires
  when developers explicitly enter `app/`.

### Negative

- **Implicit invariant: `image_manifest` indexing matches
  `per_feature_residuals` indexing.** Both are pose-major over the
  same `(pose, camera)` slots; the type system does not enforce
  alignment. Wrong manifest content means wrong arrows over wrong
  images. Mitigated by: the fixture-emitter is the source of truth
  and is regression-tested; manifest entries always come from the
  same loop that emits the residuals.
- **Two-input loading surface (export.json + companion images).**
  Moving an export.json without its `images/` sibling silently breaks
  the viewer (frontend will surface the load error, but the failure
  mode is now part of the user's mental model). Documented in the ADR
  and surfaced as an inline UI error.
- **Pre-1.0 breakage on existing exports is absorbed silently.**
  Adding the optional field is byte-stable (`skip_serializing_if`),
  but if the manifest schema later grows (per-frame timestamp,
  exposure, gain), every fixture written between then and now is a
  migration. The `Option` wrapper contains the blast radius.
- **No init-failure diagnosis.** Failure mode (4) listed above is
  named-but-deferred. v0's value rests on (2) + (3) being common
  enough to justify the tool on their own. If real-world failures are
  dominated by (4), v0 is undersized and the next milestone shifts
  toward a runner integration earlier than planned.
- **Dist placeholder coupling.** `app/dist/index.html` is committed
  so `cargo check` for the Tauri backend succeeds before
  `npm run build` has ever run. The placeholder is overwritten by
  Vite on first build; the gitignore rules keep everything else
  out.

### Risk: Tauri 2 stability

Tauri 2 was stable by late 2024; we're picking it up in 2026, so the
plugin ecosystem and 2.x APIs are settled. Pinned to `tauri = "2"` in
the Cargo.toml, with `@tauri-apps/api ^2` and `@tauri-apps/plugin-dialog
^2` on the npm side. Breaking changes within 2.x are the standard
pre-1.0-of-the-app churn we already accept.

### Risk: webview rendering fidelity

A 640×480 PNG with ~50 arrow overlays in a `<canvas>` is well within
HTML5 canvas comfort zone. No performance concerns for v0. If a real
dataset (B0.5: puzzle 130×130 with 6 cameras × 20 poses × ~120
features per camera-view = ~14k arrows per export, with the user
viewing one (pose, camera) at a time) ever pushes past comfort, the
canvas can switch to WebGL via a tiny library — that swap is a v0.5
choice driven by measurement, not a B0 commitment.

## Out of scope

- Other `*Export` types' image manifests (B0.5+, one PR each).
- Calibration runner integration / live progress streaming (post-B0
  enrichment, priority TBD).
- Wrapping `chess-corners` / `calib-targets` for in-app detection
  (post-B0 enrichment).
- 3D viewer (frustums, target poses, laser planes — Three.js / R3F).
- Init-failure diagnosis (perturbed re-runs, cost-landscape
  signature) — v2 backend story.
- Multi-pose comparison / cross-camera residual matrix UI.
- Image format support beyond PNG (no TIFF, no RAW, no 16-bit).
- Multi-OS signed installers, auto-update, code signing.
- Python parity for the manifest field (deferred per the Track A
  re-plan, ADR 0013's `A5` row).

## Status of related ADRs

- **ADR 0011 (Manual init)** — unchanged.
- **ADR 0012 (Per-feature residuals)** — unchanged. The diagnose UI
  is the first downstream consumer the ADR named, so 0012's contract
  is now load-bearing for production code instead of just for tests.
- **ADR 0013 (rig_family refactor)** — unchanged. The `*Export`
  types it consolidates are the natural extension targets for the
  manifest field once B0 ships.

## Implementation map

- **B0 (this PR)** — `ImageManifest` in core; optional
  `image_manifest` on `PlanarIntrinsicsExport`; facade re-exports;
  `planar_synthetic_with_images` example + regression test;
  `app/` Tauri 2 + React + TS shell; this ADR; roadmap re-sequence
  note.
- **B0.5** — extend `RigHandeyeExport` (Scheimpflug variant) with
  the same field; populate against the puzzle 130×130 dataset; verify
  ROI + multi-camera tiled strips render correctly.
- **Post-B0 enrichments** — driven by what the user misses while
  using v0. Likely first pickups: a "re-run" button calling the
  facade in-process; multi-pose residual stats panel; cross-camera
  matrix view. None of these blocks B0 from landing.
