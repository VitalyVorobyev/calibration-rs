# ADR 0014: Desktop App

## Context

Calibration failures are diagnosed visually: a residual vector field over
the source images shows a wrong distortion model (radial pattern), a wrong
camera model (systematic asymmetry) or wrong target parameters. The library
alone gives numbers, not that view, and running a dataset end to end
(manifest, detection, every config knob, the result) from a terminal is
slow to iterate.

## Decision

### A diagnose-first desktop app on Tauri 2

The app (`app/`) runs a calibration and shows its result: workspaces for
Run, Diagnose (residuals over the frames), 3D, Epipolar and Depth, all
built on the export contract (ADR 0012).

- **Tauri 2 + React + TypeScript.** The webview only renders; all
  calibration logic runs in Rust through the `vision-calibration` facade.
- **Outside the Cargo workspace.** `app/src-tauri` is its own crate with
  its own lockfile, so the Tauri build tree never slows the library build.
- **Distribution is local builds.** No signed installers or auto-update.

### Forms and wire types are generated from Rust

Rust types are the single source of truth for everything the UI edits or
reads:

- Every user-editable config, the dataset manifest and every export derive
  `schemars::JsonSchema` behind the `schemars` feature.
- `cargo xtask emit-schemas` writes the config and manifest schemas into
  `app/src/schemas/`; they are committed so the frontend builds without a
  Rust toolchain and a config change shows its schema diff in review. CI
  fails when they drift (`--check`).
- One schema-driven component (`SchemaValueForm`, `@vitavision/forms`)
  renders every config form, so a new config field appears in the UI
  without frontend code.
- The TypeScript types of the export wire format are generated from the
  export schemas; there are no hand-written mirrors, and CI checks them.

How to build, test and extend the app is in `app/DEVELOPING.md`.

## Consequences

- The app cannot drift from the library's configs or exports silently;
  pre-1.0 config churn costs no UI work beyond hand-written presets.
- Configs stay free of foreign types (no `nalgebra` in a config tree), so
  their schemas need no shims; `*Input` types are internal and need no
  schema (ADR 0016).
- Moving an `export.json` away from its images breaks the image view; the
  app reports it as a load error.

## Alternatives rejected

- **rerun.io** — a developer-tool UI with a second (gRPC) transport.
- **egui** — less leverage for a polished production UI.
- **Hand-written forms and TypeScript mirrors** — drift on every config
  change.
- **Schema emission in `build.rs`** — slows clean builds and hides schema
  diffs from review.
