# ADR 0014: Desktop App

## Context

Calibration failures are diagnosed visually: a residual vector field over
the source images shows a wrong distortion model (radial pattern), a wrong
camera model (systematic asymmetry) or wrong target parameters. The library
alone gives numbers, not that view, and running a dataset end to end
(manifest, detection, every config knob, the result) from a terminal is
slow to iterate. The work also has to outlive a session: a dataset, the
manifest that describes it and the runs calibrated from it belong together
on disk, where they can be reopened, compared and shared.

## Decision

### A studio app on Tauri 2

The app (`app/`) runs a calibration and shows its result: workspaces for
Run, Diagnose (residuals over the frames), 3D, Epipolar and Depth, all
built on the export contract (ADR 0012).

- **Tauri 2 + React + TypeScript.** The webview only renders; all
  calibration logic runs in Rust through the `vision-calibration` facade.
- **Outside the Cargo workspace.** `app/src-tauri` is its own crate with
  its own lockfile, so the Tauri build tree never slows the library build.
- **Distribution is local builds.** No signed installers or auto-update.

### The workbench shell

The frame is `@vitavision/workbench`'s `AppShell`, the studio shell shared
with the other vitavision lab apps: a header (the project menu, Open
Export…, the theme), a `NavRail` of workspaces (`⌘1`…`⌘5`), a resizable
project navigator on the left while a project is open (its saved runs as a
`TreeView`), the active workspace, and a `StatusBar`. Components come from
`@vitavision/ui`, the 3D scene from `@vitavision/three(-react)`; the app
keeps no component kit of its own. Routing (a hash router, one route per
workspace) stays in the app, since the workbench is router-free.

App state is three stores: the loaded export and the selection (shared by
every viewer workspace), the Run setup (manifest, config, dataset folder),
and the project. The Run setup is app-wide rather than component state so
that opening a project fills it from any workspace, and a saved run records
exactly what ran.

### Project folders

A project is a folder:

```text
<project>/
  calibration-project.json   {"version": 1, "dataset": "<dir>"}
  dataset.json               the dataset manifest
  runs/<name>/               manifest.json, config.json, export.json, diagnostics.json
```

- **The marker file decides.** A folder without `calibration-project.json`
  is not a project: opening it is a `not_a_project` error shown to the
  user, and nothing is guessed from what else the folder holds (a
  `dataset.toml` alone does not make one). Every project failure is a typed
  `ProjectError { kind, message }` whose wire type is generated like the
  rest.
- **The dataset may live elsewhere.** `dataset` is the folder the
  manifest's relative paths resolve against: relative to the project when
  inside it (`"."` when the images sit beside the project file), absolute
  otherwise. A project can so describe a read-only dataset (a repository
  dataset, a share) without copying it.
- **The app writes JSON.** The manifest is `dataset.json`; a TOML manifest
  is imported once, when a project is made from it, and the hand-written
  file is never rewritten.
- **Runs are records.** A run that finishes while a project is open is
  saved as `runs/run-NNN/` (one past the highest number) and reopened from
  its `export.json`, so the viewers show what is on disk. Runs are never
  overwritten: a name that exists is a `run_exists` error. The saved
  export's image root is rebased onto the run folder, so it also opens on
  its own through Open Export….
- **Recent projects** are a per-viewer list in `localStorage`; a folder
  that no longer opens drops off it.

### Images over the asset protocol

Source images are not sent over IPC. The webview loads them from disk
through Tauri's asset protocol (`convertFileSrc`), whose scope starts empty
and grows only with the folders the user opens: a project's dataset, an
export's image root, the dataset of a live run. The CSP allows images from
the asset protocol, `blob:` and `data:`, and nothing remote. The protocol
answers with the page's origin, so the frame viewer reads pixels back from
a canvas (the cursor readout, the ROI histogram) without tainting it.
Computed images stay backend work: the undistorted frame is returned as
PNG bytes in a binary IPC response and shown through a blob URL; the depth
workspace's colourised results still come back as data URLs inside its
result.

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
- The TypeScript types of the export and command wire format (the project
  types included) are generated from the Rust types; there are no
  hand-written mirrors, and CI checks them.

How to build, test and extend the app is in `app/DEVELOPING.md`.

## Consequences

- The app cannot drift from the library's configs or exports silently;
  pre-1.0 config churn costs no UI work beyond hand-written presets.
- Configs stay free of foreign types (no `nalgebra` in a config tree), so
  their schemas need no shims; `*Input` types are internal and need no
  schema (ADR 0016).
- A project moved with its dataset inside it keeps working; one whose
  dataset lies outside it records that folder's absolute path and breaks
  when the dataset moves. Either way, an image the scope does not cover
  fails to load and is reported, rather than read from an unopened folder.
- Moving an `export.json` away from its images breaks the image view; the
  app reports it as a load error.

## Alternatives rejected

- **rerun.io** — a developer-tool UI with a second (gRPC) transport.
- **egui** — less leverage for a polished production UI.
- **Hand-written forms and TypeScript mirrors** — drift on every config
  change.
- **Schema emission in `build.rs`** — slows clean builds and hides schema
  diffs from review.
- **Base64 images over IPC** — every frame paid an encode, a JSON string
  of a third more bytes and a decode, and the backend read files on the
  webview's behalf with no scope at all.
- **A dataset folder as the project** (no marker file) — opening any folder
  would write `runs/` into it, and a folder holding both `dataset.toml` and
  `dataset.json` would be ambiguous.
- **An app-local shell** — the studio layout is shared with the other lab
  apps through `@vitavision/workbench`, so a second implementation would
  drift from it.
