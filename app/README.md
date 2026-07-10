# calibration-diagnose

Tauri 2 + React 19 + TypeScript desktop app for `calibration-rs`: run
calibrations end-to-end and diagnose the results without leaving a GUI. The
Rust backend (`app/src-tauri`) depends on the workspace facade
(`vision-calibration`) via path — it does not bundle the calibration
algorithms separately, so its Tauri commands call straight into the same
solvers exercised by `cargo run --example`.

See `docs/adrs/0014-tauri-desktop-app.md` for the design rationale and
[`docs/tutorials/app-walkthrough.md`](../docs/tutorials/app-walkthrough.md)
for a hands-on walkthrough.

## Workspaces

The app is organised as five workspaces (routes), selectable from the left
rail — see `app/src/workspaces/`:

- **Run** (`RunWorkspace`) — configure and launch a calibration end-to-end:
  pick/sniff a dataset folder, fill in the manifest and calibration config
  (schema-driven forms, built-in presets for the eight problem types /
  topologies), and invoke the facade in-process. Surfaces `ask_user`
  ambiguities from folder-sniffing as a blocking modal.
- **Diagnose** (`DiagnoseWorkspace`) — the original v0 residual viewer: loads
  a calibration export JSON and overlays the per-image reprojection residual
  vector field (and laser-feature residuals) on the source frame, with
  pose/camera steppers and error histograms.
- **3D** (`Viewer3DWorkspace`) — a 3D rig scene: camera frustums, target board
  poses, and laser planes for rig/hand-eye/laserline exports, with a side
  panel breaking down intrinsics and relative camera poses.
- **Epipolar** (`EpipolarWorkspace`) — click-to-sample epipolar geometry
  between two camera views of a rig export (raw and undistorted), for
  sanity-checking extrinsics.
- **Depth** (`DepthWorkspace`) — dense stereo: rectifies a camera pair,
  computes disparity (block matching or SGM), and renders disparity/depth
  overlays and a reprojected 3D point cloud.

All workspaces share state (the loaded export, selected pose/camera, …)
through a single Zustand store (`app/src/store/`); routes carry no params.

## First-time setup

This app uses **bun** exclusively — the lockfile is `bun.lock`, and
`tauri.conf.json`'s `beforeDevCommand`/`beforeBuildCommand` invoke
`bun run …`. Never use `npm`/`pnpm`/`yarn`. If you don't have bun, install it
from <https://bun.sh>.

```bash
# from this directory
bun install
```

The Tauri CLI is pulled in as a dev-dep of `package.json`, so a global
install is not required.

## Commands

```bash
bun install            # first-time setup / after package.json changes
bun run tauri dev      # launch the desktop app
bun run build           # TS compile + Vite build (frontend only, no Tauri shell)
bun run tauri build    # bundle the desktop app (installer/binary)
bun run generate:types # regenerate TS wire types from the Rust source
```

> **Important.** Use `bun run tauri dev`, **not** `bun run dev`. The
> latter only starts Vite at <http://localhost:1420>; opening that URL
> in a regular browser bypasses the Tauri webview, so the
> `__TAURI_INTERNALS__` global is absent and any IPC call (file dialog,
> `load_export`, `run_calibration_cmd`, …) fails with `Cannot read
> properties of undefined (reading 'invoke')`. The app detects this and
> shows a banner, but the fix is to launch via `tauri dev`.

## Dev notes

- `app/src-tauri` is a **separate cargo project**, intentionally excluded
  from the root workspace (`/Cargo.toml`'s `exclude = ["app"]`) — Tauri's
  build-time dep tree is heavy and would otherwise pollute
  `cargo build --workspace` for the calibration library. It pins its own
  `Cargo.lock` and is built from inside `app/` (via `bun run tauri dev`/
  `build`), never from the repo root. `cargo test --workspace` at the repo
  root does **not** cover it — see `app/src-tauri/` for its own checks.
- Frontend code lives in `app/src/`: `workspaces/` (routed views),
  `components/`, `hooks/`, `layouts/`, `lib/`, `schemas/` (JSON Schemas for
  the Run workspace's config forms), `types/generated/` (generated wire
  types, see below), `store/` (Zustand + wire types).
- Backend (Tauri) code lives in `app/src-tauri/src/`: `commands.rs` (export
  loading, image loading/undistortion, epipolar overlay), `run.rs`
  (calibration runner + folder sniffing), `disparity.rs` (dense stereo).

### Generated wire types (`bun run generate:types`)

The TypeScript interfaces the app uses for calibration `*Export` payloads
and Tauri command responses are **generated from the Rust types**, not
hand-written (B-QUAL2). The single source of truth is the
`#[derive(schemars::JsonSchema)]` on the pipeline `*Export` types and the
`app/src-tauri` command structs; edit those and regenerate. Two stages:

```bash
bun run generate:types   # both stages (schema + TS); commit the results
# or run a stage on its own:
bun run generate:schemas   # stage 1 (cargo): Rust types → schemas-generated/diagnose_wire.json
bun run generate:types:ts  # stage 2 (bun):   schema → src/types/generated/*.ts (+ prettier)
```

Both outputs are committed. CI enforces they stay in sync: the
`app-src-tauri` job runs `generate:schemas:check` (Rust → schema drift) and
`app-frontend` regenerates the TS and `git diff --exit-code`s it (schema →
TS drift), so a Rust type change that isn't regenerated fails the build.
`src/types/generated/` is eslint-ignored and `schemas-generated/` is
prettier-ignored (they're machine-owned). The export discriminator lives in
`src/store/exportKind.ts` (`detectExportKind`), typed against these
generated shapes.

## Out of scope

Signed installers, in-app detection wrap beyond the Run workspace, and
init-failure diagnosis remain open backlog items — see `docs/backlog.md`
(`B-DIST-INSTALLERS` and related `B-*` entries) and ADR 0014's
deferred-features list.
