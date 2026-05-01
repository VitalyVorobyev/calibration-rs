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

```bash
# from this directory
bun install      # or `npm install` / `pnpm install` / `yarn install`
```

The Tauri CLI is pulled in as a dev-dep of `package.json`, so a global
install is not required. The committed lockfile is `bun.lock`; the
other package managers will regenerate their own lockfile on first
install.

## Running

Generate the v0 fixture once from the workspace root:

```bash
cargo run -p vision-calibration --example planar_synthetic_with_images
```

Then launch the app:

```bash
bun run tauri dev      # or `npm run tauri dev`
```

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
