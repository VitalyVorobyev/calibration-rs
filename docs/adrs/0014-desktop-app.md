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
- `apps/studio` is its own cargo workspace with its own lockfile, so the
  Tauri build tree never slows the library build. What the app computes is
  the `studio-core` crate, with no Tauri in it; `src-tauri` holds the
  shells over it.
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

**One frontend, two back ends: the viewers also run in a browser.**
- A run is reviewed where it lies, without the desktop app:
  `calib-viewer <run-dir | export.json>` (a second binary of
  `apps/studio/src-tauri`) serves the built frontend and one run on localhost and
  opens a browser on it. A run is a `vcal run -o` folder
  (`calibration.json`), a run the app saved (`runs/<name>/export.json`) or
  one export file; anything else is an error that says what to pass.
- The back end is one command table in `studio-core`: each command a
  typed function, listed once, of three kinds (plain, job, resource), with
  one error envelope. Both shells dispatch through it: the Tauri shell over
  four IPC commands, `calib-viewer` over `POST /api/cmd/<name>` and
  `GET /api/r/<name>`. The frontend reaches it only through `src/api/`: a
  client generated from the table, over a transport (Tauri IPC or HTTP);
  the Tauri transport is the only module that imports Tauri. Capability
  flags hide what needs the desktop (dialogs, projects, calibration runs):
  the browser is a viewer, and `calib-viewer` answers only the table's
  read-only commands.
- The server binds 127.0.0.1 only, answers only its own `Host` (no DNS
  rebinding), refuses a foreign `Origin` and sends no CORS headers.
- Its file access is scoped like the asset protocol: files under the run
  folder and the export's image root, judged after canonicalising, so `..`
  and symbolic links cannot leave them. Images of a private rig stay on the
  machine.
- An export without an image list opens too: navigation comes from its
  residuals, and the image panes say there are no images.

**Forms and wire types are generated from Rust.**
- Every user-editable config, the manifest and every export derive a JSON
  Schema. `cargo xtask emit-schemas` writes them into the app, where one
  schema-driven form renders every config, so a new config field appears
  without frontend code.
- The TypeScript types of the export and command wire format are generated
  too, and CI fails when any of them drift. The target is no hand-written
  mirror of a Rust type in `apps/studio/src`; the ones left are backlog
  APP-WIRE-TYPES.

How the shell, stores, commands and tests are built is in
`apps/studio/DEVELOPING.md`.

## Consequences

- The app cannot drift silently from the library's configs or exports, and
  a config change costs no UI work beyond hand-written presets.
- Configs stay free of foreign types (no `nalgebra` in a config tree), so
  their schemas need no shims.
- A project moved together with its dataset keeps working. One whose dataset
  lies outside it breaks when that dataset moves, and the image load error
  says so. An image outside the opened folders fails to load; it is never
  read on the webview's behalf.
- Every viewer feature works on both back ends or is gated by a capability;
  a new command is one function and one line of the table, which both
  transports and the generated client pick up, and a lint rule keeps Tauri
  imports in the Tauri transport.
- `calib-viewer` needs the frontend built (`bun run build`); it serves
  `apps/studio/dist` from the source tree and is not distributed on its own.

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
- **A second, static report viewer** (a generated HTML page beside the run,
  like `report.html`). It would re-implement the residual overlay, the
  matrix, the 3D scene and the epipolar view, and drift from the app's; and
  undistortion, epipolar lines and projection uncertainty need the Rust
  camera models at view time.
- **Serving the run to the network or embedding images in a page.** A page
  that carries or serves a private rig's images can leave the machine;
  localhost with scoped files cannot.
- **An async HTTP stack (axum, hyper).** The viewer answers one local user;
  a small synchronous server (`tiny_http`) on a few threads is enough and
  adds no runtime.
