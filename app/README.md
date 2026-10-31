# calibration-diagnose

A desktop app for `calibration-rs`: run calibrations end-to-end and diagnose
the results without leaving a GUI. It calls the same solvers as the
`vision-calibration` crate. For a hands-on tour, see the
[app chapters of the book](https://vitalyvorobyev.github.io/calibration-rs/app.html).

## What it does

Five workspaces (left rail, `⌘1` to `⌘5`) share the loaded export and the
selected pose and camera: **Run** (configure and launch any of the seven
problem types from a dataset folder or preset), **Diagnose** (residual
overlays, per-pose statistics, parameter uncertainty), **3D** (rig frustums,
target poses, laser planes), **Epipolar** and **Depth**. A **project** folder
keeps a dataset's manifest and every run calibrated from it, so the work can be
closed and reopened. The
[app chapters of the book](https://vitalyvorobyev.github.io/calibration-rs/app.html)
describe each workspace and the project layout.

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
