# ADR 0014: Desktop App

## Context

Calibration failures are diagnosed visually. A residual vector field over
the source images shows a wrong distortion model (a radial pattern), a wrong
camera model (systematic asymmetry) or wrong target parameters. The library
alone gives numbers, not that view. Running a dataset end to end from a
terminal (manifest, detection, every config knob, the result) is also slow
to iterate.

The work has to outlive a session: a dataset, its manifest and the runs
calibrated from it belong together on disk, to be reopened, compared and
shared.

## Decision

**A studio app on Tauri 2 + React + TypeScript, outside the Cargo
workspace.**
- The webview only renders. All calibration logic runs in Rust through the
  `vision-calibration` facade, and every viewer is built on the export
  contract (ADR 0012).
- `app/src-tauri` has its own lockfile, so the Tauri build tree never slows
  the library build.
- Distribution is local builds: no signed installers, no auto-update.
- The frame is the shared `@vitavision/workbench` shell; components and the
  3D scene come from the `@vitavision/*` packages. The app keeps no component
  kit of its own.

**A project is a folder, and its marker file decides.**
- A project is a folder with `calibration-project.json`, its manifest
  (`dataset.json`) and its runs.
- A folder without the marker is not a project. Nothing is guessed from what
  else it holds, and every project failure is a typed error.
- The dataset may live elsewhere. The project records it relative to the
  project when inside it, absolute otherwise, so a read-only dataset is
  described without copying it.
- The app writes JSON. A TOML manifest is imported once, when a project is
  made from it; a hand-written file is never rewritten.
- **Runs are records.**
  - A run that finishes while a project is open is saved as the next
    `runs/run-NNN/`, then reopened from disk, so the viewers show what was
    saved.
  - Runs are never overwritten.
  - A saved export opens on its own as well.

**Images are read from disk, scoped to what the user opened.**
- Source images load through Tauri's asset protocol. Its scope starts empty
  and grows only with folders the user opens: a project's dataset, an
  export's image root, the dataset of a live run.
- The Content Security Policy allows nothing remote.
- Computed images (undistorted frames) come back as binary IPC responses,
  never as text.

**Forms and wire types are generated from Rust.**
- Every user-editable config, the manifest and every export derive a JSON
  Schema. `cargo xtask emit-schemas` writes them into the app, where one
  schema-driven form renders every config, so a new config field appears
  without frontend code.
- The TypeScript types of the export and command wire format are generated
  too, with no hand-written mirrors. CI fails when any of them drift.

How the shell, stores, commands and tests are built is in
`app/DEVELOPING.md`.

## Consequences

- The app cannot drift silently from the library's configs or exports, and
  pre-1.0 config churn costs no UI work beyond hand-written presets.
- Configs stay free of foreign types (no `nalgebra` in a config tree), so
  their schemas need no shims.
- A project moved together with its dataset keeps working. One whose dataset
  lies outside it breaks when that dataset moves, and the image load error
  says so. An image outside the opened folders fails to load; it is never
  read on the webview's behalf.

## Alternatives rejected

- **rerun.io.** A developer-tool UI with a second transport (gRPC).
- **egui.** Less leverage for a polished production UI.
- **Hand-written forms and TypeScript mirrors.** They drift on every config
  change.
- **Schema emission in `build.rs`.** It slows clean builds and hides schema
  diffs from review.
- **Base64 images over IPC.**
  - Every frame paid an encode, a third more bytes of JSON and a decode.
  - The backend read files for the webview with no scope at all.
- **A dataset folder as the project, with no marker file.**
  - Opening any folder would write `runs/` into it.
  - A folder holding two manifests would be ambiguous.
- **An app-local shell.** It would drift from the studio layout the other
  lab apps share.
