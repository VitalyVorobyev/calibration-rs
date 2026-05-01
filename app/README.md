# calibration-diagnose

Tauri 2 + React + TypeScript desktop viewer for `calibration-rs` exports.

**Scope (v0 / B0):** passive viewer of one `PlanarIntrinsicsExport` JSON.
Renders the per-image residual vector field overlaid on the source PNG —
the visualisation an experienced calibration engineer needs to identify
distortion-model and camera-model misspecification.

The Rust backend depends on the workspace facade
(`vision-calibration`) via path; it does not bundle the calibration
algorithms separately.

See `docs/adrs/0014-tauri-desktop-app.md` for the design rationale and
the deferred-features list.

## First-time setup

This app uses **bun** exclusively — the lockfile is `bun.lock`, the Tauri
`beforeDevCommand` runs `bun run dev`, and we don't carry alternative
lockfiles. If you don't have bun, install it from <https://bun.sh>.

```bash
# from this directory
bun install
```

The Tauri CLI is pulled in as a dev-dep of `package.json`, so a global
install is not required.

## Running

Generate the v0 fixture once from the workspace root:

```bash
cargo run -p vision-calibration --example planar_synthetic_with_images
```

Then launch the app:

```bash
bun run tauri dev
```

> **Important.** Use `bun run tauri dev`, **not** `bun run dev`. The
> latter only starts Vite at <http://localhost:1420>; opening that URL
> in a regular browser bypasses the Tauri webview, so the
> `__TAURI_INTERNALS__` global is absent and any IPC call (file dialog,
> `load_export`, `load_image`) fails with `Cannot read properties of
> undefined (reading 'invoke')`. The app detects this and shows a
> banner, but the fix is to launch via `tauri dev`.

In the running window: **Open Export…** → pick
`target/fixtures/planar_synthetic_with_images/export.json`.
You should see a pose/camera selector with five entries; selecting any
one renders the synthesised checkerboard PNG with short reprojection
arrows (≤ ~0.5 px) at every detected corner.

## Layout

```
app/
  package.json          # vite + react + tauri
  vite.config.ts
  tsconfig.json
  index.html
  src/                  # React + TypeScript UI
    main.tsx
    App.tsx
    types.ts
    components/
      ResidualViewer.tsx
  src-tauri/            # Rust backend (Tauri commands)
    Cargo.toml
    build.rs
    tauri.conf.json
    capabilities/default.json
    src/
      main.rs
      lib.rs
      commands.rs
```

## Out of scope for v0

Detection wrapping, calibration runner, 3D viewer, init-failure
diagnosis, multi-image format support, signed installers. See ADR 0014.
