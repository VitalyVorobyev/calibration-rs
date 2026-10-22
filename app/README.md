# calibration-diagnose

A desktop app for `calibration-rs`: run calibrations end-to-end and diagnose
the results without leaving a GUI. It calls the same solvers as the
`vision-calibration` crate. For a hands-on tour, see the
[app chapters of the book](https://vitalyvorobyev.github.io/calibration-rs/app.html).

## Projects

A project keeps a dataset's manifest and its calibration runs together in
one folder, so the work can be closed and reopened:

```text
my-project/
  calibration-project.json   marks the folder as a project; names the dataset folder
  dataset.json               the dataset manifest
  runs/run-001/              manifest.json, config.json, export.json, diagnostics.json
```

The **Project** menu in the header creates one from the dataset the Run
workspace has loaded (a preset, a sniffed folder or a manifest file), opens
one, saves the edited manifest, and reopens recent projects. The images can
stay where they are: the dataset folder may be the project folder or any
other. While a project is open, every successful run is saved as the next
`runs/run-NNN/` (never over an existing run) and the project's runs are
listed on the left; choosing one loads its export. Opening a folder that is
not a project is an error, not a guess.

## Workspaces

Five workspaces, selectable from the left rail (`⌘1` to `⌘5`); they share
the loaded export and the selected pose and camera.

- **Run** — configure and launch a calibration end-to-end: pick a dataset
  folder (the app sniffs its layout), fill in the manifest and calibration
  config from schema-driven forms or a built-in preset for any of the eight
  problem types, and run it. Anything the folder sniffer cannot decide on its
  own is asked in a blocking dialog.
- **Diagnose** — load a calibration export and overlay the per-image
  reprojection residuals (and laser-feature residuals) on the source frames,
  with pose and camera steppers and error histograms. A **Stats** panel adds a
  sortable per-pose residual table and, for multi-camera exports, a cameras ×
  poses residual matrix. An **Uncertainty** panel lists each estimated
  parameter's value ± σ, their correlations, and the shown camera's
  projection σ across its image.
- **3D** — camera frustums, target board poses and laser planes for rig,
  hand-eye and laserline exports, with a panel of intrinsics and relative
  camera poses. A single-camera laserline export is shown as a one-camera rig.
- **Epipolar** — click-to-sample epipolar geometry between two camera views of
  a rig export, raw and undistorted.
- **Depth** — rectify a camera pair, compute disparity (block matching or
  SGM), and render disparity and depth overlays and a reprojected 3D point
  cloud.

## Running it

The app is built from a local checkout with [bun](https://bun.sh) (never
`npm`/`pnpm`/`yarn`); the Tauri CLI is a dev-dependency, so no global install
is needed.

```bash
bun install            # first-time setup / after package.json changes
bun run tauri dev      # launch the desktop app
bun run tauri build    # bundle the desktop app (installer/binary)
```

Use `bun run tauri dev`, **not** `bun run dev`: the latter only starts Vite at
<http://localhost:1420>, without the Tauri APIs the app needs. The app detects
this and shows a banner.

Signed installers and in-app detection outside the Run workspace are not
provided.

Contributing to the app: [DEVELOPING.md](DEVELOPING.md).
