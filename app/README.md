# calibration-diagnose

Tauri 2 + React 19 + TypeScript desktop app for `calibration-rs`: run
calibrations end-to-end and diagnose the results without leaving a GUI. The
Rust backend (`app/src-tauri`) depends on the workspace facade
(`vision-calibration`) via path, so its Tauri commands call the same solvers
exercised by `cargo run --example`.

Design rationale: `docs/adrs/0014-tauri-desktop-app.md`. Hands-on walkthrough:
[`docs/tutorials/app-walkthrough.md`](../docs/tutorials/app-walkthrough.md).

## Workspaces

Five routes, selectable from the left rail (`app/src/workspaces/`):

- **Run** (`RunWorkspace`) — configure and launch a calibration end-to-end:
  pick/sniff a dataset folder, fill in the manifest and calibration config
  (schema-driven forms, built-in presets for the eight problem types), and
  invoke the facade in-process. Folder-sniffing ambiguities (`ask_user`) surface
  as a blocking modal.
- **Diagnose** (`DiagnoseWorkspace`) — loads a calibration export JSON and
  overlays the per-image reprojection residual vector field (and laser-feature
  residuals) on the source frame, with pose/camera steppers and error
  histograms. A **Stats** panel adds a sortable per-pose residual table and,
  for multi-camera exports, a cameras x poses residual matrix
  (`src/lib/residualStats.ts`).
- **3D** (`Viewer3DWorkspace`) — camera frustums, target board poses, and laser
  planes for rig/hand-eye/laserline exports, with a panel of intrinsics and
  relative camera poses. A single-camera `laserline_device` export is lifted
  into a one-camera rig at the origin (`src/lib/sceneExport.ts`). Drawn with the
  shared vitavision 3D primitives; see [3D scene primitives](#3d-scene-primitives).
- **Epipolar** (`EpipolarWorkspace`) — click-to-sample epipolar geometry
  between two camera views of a rig export (raw and undistorted).
- **Depth** (`DepthWorkspace`) — rectifies a camera pair, computes disparity
  (block matching or SGM), and renders disparity/depth overlays and a
  reprojected 3D point cloud.

All workspaces share state (loaded export, selected pose/camera, ...) through
one Zustand store (`src/store/`); routes carry no params.

## Setup and commands

This app uses **bun** exclusively: the lockfile is `bun.lock`, and
`tauri.conf.json`'s `beforeDevCommand`/`beforeBuildCommand` invoke `bun run …`.
Never use `npm`/`pnpm`/`yarn`. The Tauri CLI is a dev-dependency, so no global
install is needed.

```bash
bun install            # first-time setup / after package.json changes
bun run tauri dev      # launch the desktop app
bun run build          # TS compile + Vite build (frontend only, no Tauri shell)
bun run tauri build    # bundle the desktop app (installer/binary)
bun run generate:types # regenerate TS wire types from the Rust source
bun run test           # Vitest: pure-logic unit tests + jsdom component tests
bun run test:e2e       # Playwright smoke: boots the app, checks it doesn't fall over
bun run test:screens   # screenshots of every workspace vs. a local baseline
```

Use `bun run tauri dev`, **not** `bun run dev`: the latter only starts Vite at
<http://localhost:1420>, so `__TAURI_INTERNALS__` is absent and any IPC call
fails with `Cannot read properties of undefined (reading 'invoke')`. The app
detects this and shows a banner.

## Layout

- `app/src-tauri` is a **separate cargo project**, excluded from the root
  workspace (`exclude = ["app"]` in `/Cargo.toml`) because Tauri's build-time
  dependency tree is heavy. It pins its own `Cargo.lock` and is built from
  inside `app/`; `cargo test --workspace` at the repo root does not cover it.
- Frontend (`app/src/`): `workspaces/`, `components/`, `hooks/`, `layouts/`,
  `lib/`, `schemas/` (JSON Schemas for the Run config forms), `types/generated/`
  (generated wire types), `store/`.
- Backend (`app/src-tauri/src/`): `commands.rs` (export/image loading,
  undistortion, epipolar overlay), `run.rs` (calibration runner, folder
  sniffing), `disparity.rs` (dense stereo).

## 3D scene primitives

The 3D viewer's frustums, target boards, laser fans, rig axes and scene
colours come from [`@vitavision/three`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/three)
(framework-agnostic three.js objects plus the SE(3) wire helpers `lib/se3.ts`
delegates to) and [`@vitavision/three-react`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/three-react)
(R3F wrappers, `useSceneColors`). What stays in `Viewer3DWorkspace/` is this
app's own: the rig-frame `Canvas` and auto-fit, the camera label and apex
markers, board sizing from residuals, the laser-plane → fan placement
(`laserFanPose.ts`), and the laser/board cut lines.

- **Frustum rays.** The packages do no camera math: a frustum takes its
  image border's viewing rays. `useBorderRays` asks the backend's
  `undistort_points` to back-project the border through the full camera
  model (so a distorted field of view bows), and falls back to the pinhole
  `K⁻¹` outside Tauri (`src/lib/frustumRays.ts`).
- **Colours.** The packages read `@vitavision/ui` token names (`--signal`,
  `--fg-muted`, `--defect`, …), which `@vitavision/ui/styles.css` defines
  (see "Design system" below); `useSceneColors` re-reads them when `.dark`
  toggles.
- **Layers and picking.** Package gizmos live on `GIZMO_LAYER`, which this
  viewer's own `Canvas` enables on its camera and raycaster. Board and fan
  outlines are not pickable (three.js hits a `Line` within 1 world unit, a
  metre here); frustums pick through a hull padded by `pickPadding` 1.18
  plus a sphere at the optical centre.

## Design system

The UI is built on [`@vitavision/ui`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/ui),
the design system shared by the vitavision lab apps: one visual language,
no app-local component kit.

- **Components.** Use the package's primitives — `Button`, `ToggleChip`
  (on/off toolbar layers), `SegmentedControl`, `Select`, `Input`/`Textarea`,
  `Panel` (with `DensityProvider value="compact"` in side rails), `Table`,
  `Badge`, `Callout`/`ErrorBox`, `Empty`, `Dialog`, `Disclosure`, `Tooltip`,
  `ThemeToggle`. Read the props in the package (its `etc/ui.api.md` API
  report) rather than guessing. A shape the package lacks goes to lab-ui once
  a second app needs it, not into `src/components/`.
- **Tokens.** `src/index.css` imports `tailwindcss`, `@vitavision/ui/fonts.css`
  and `@vitavision/ui/styles.css` and adds only the root sizing. Colour comes
  from the semantic tokens: elevation `ground` / `surface` / `raised` /
  `overlay`, borders `line` / `line-strong`, text `fg` / `fg-muted` /
  `fg-subtle`, the one accent `signal` (focus, selection, the primary
  action), and the verdicts `normal` / `defect` / `warn`, reserved for
  verdicts. Utilities are `bg-surface`, `text-fg-muted`, `border-line`, …;
  in SVG or inline styles use the custom property itself
  (`fill="var(--signal)"`). The values are hex, so never wrap them in
  `hsl(...)`. Radii are `rounded-control` and `rounded-panel`.
- **Fonts.** IBM Plex Sans and IBM Plex Mono, served by `fonts.css` (no
  fontsource packages).
- **Theme.** Light / dark / system, stored in `localStorage` under
  `calib-theme` (`src/lib/theme.ts`). The inline script in `index.html`
  paints the `.dark` class before the first paint; `main.tsx` calls
  `initTheme("calib-theme")` so "system" keeps following the OS, and wraps
  the app in `TooltipProvider` (`ThemeToggle` and `Tooltip` throw without
  it). The header's `ThemeToggle` cycles system → light → dark.
- **Lint (gate G5.1).** `eslint.config.js` applies `tokensOnly(["src/**"])`
  from `@vitavision/config-eslint`: no raw Tailwind palette classes
  (`bg-slate-500`) and no hex literals in `src/`. The only exemptions are
  data-colour files listed there with their reason (the residual colour ramp
  in `lib/errorColors.ts`, the per-laser palette in `LaserTargetCuts.tsx`).

`components/ZoomControls.tsx` is the zoom/fit/1:1 cluster for `FrameCanvas`.

## Run progress and cancellation

`run_calibration_cmd` takes a per-run `runId` and a `tauri::ipc::Channel<RunProgress>`
(`src/lib/runCalibration.ts`). `run.rs` announces three coarse stages,
`detect` -> `solve` -> `export` (`RunStage`); per-camera and per-iteration
granularity are not emitted because the pipeline API exposes no hook.
Cancellation is a cooperative `AtomicBool` per `runId` in a Tauri-managed
`RunRegistry`: `cancel_run_cmd(runId)` sets it, and the runner checks it at each
stage boundary and returns `RunResponse::Cancelled` (the running stage finishes
first). The UI shows a distinct "Run cancelled" state; the checklist logic is
`src/workspaces/RunWorkspace/runStages.ts`.

## Repo-root resolution for built-in presets

Quick Start presets (`src/workspaces/RunWorkspace/presets.ts`) store a
**repo-root-relative** `manifestPath` (`data/…` or local-only `privatedata/…`).
`handleUsePreset` resolves the root through `lib/tauri.ts`'s `repoRoot()`, which
calls `repo_root_cmd` (`env!("CARGO_MANIFEST_DIR")` two levels up) and joins it
with `joinPath`. Because the app is always compiled from a local checkout, this
is the developer's own checkout path. Bundling public datasets in a release
installer would use Tauri's `resource_dir` instead.

## Testing

- **Vitest (`bun run test`)** — `src/**/*.test.ts` are pure-logic unit tests
  (Node); `src/**/*.test.tsx` are jsdom component tests (opt in with
  `// @vitest-environment jsdom`). Component tests mock IPC with
  `@tauri-apps/api/mocks`' `mockIPC`; `src/test/setupTests.ts` stubs
  `ResizeObserver` and `HTMLCanvasElement.getContext`.
- **Playwright (`bun run test:e2e`)** — `e2e/` smoke tests against the plain
  Vite dev server (no Tauri/Rust toolchain). `app.spec.ts` checks every
  workspace mounts with no console errors and no Tauri mock;
  `diagnose.spec.ts` injects a `window.__TAURI_INTERNALS__` shim
  (`e2e/support/tauriMock.ts`) via `page.addInitScript`. Real desktop behavior is
  only exercised manually via `bun run tauri dev`. Install the browser once with
  `bunx playwright install chromium --with-deps`.
- **Screenshots (`bun run test:screens`)** — `e2e/screens.spec.ts`
  (`playwright.screens.config.ts`, port 1421) captures every workspace; the
  baseline in `e2e/.screens/` is local and uncommitted (`--update-snapshots`
  before a change, compare after). Not run in CI.

## Generated wire types

TypeScript interfaces for `*Export` payloads and Tauri command responses are
generated from the Rust types (`#[derive(schemars::JsonSchema)]`); edit those
and regenerate:

```bash
bun run generate:types     # both stages; commit the results
bun run generate:schemas   # stage 1 (cargo): Rust types -> schemas-generated/diagnose_wire.json
bun run generate:types:ts  # stage 2 (bun): schema -> src/types/generated/*.ts
```

Both outputs are committed. CI enforces sync: `app-src-tauri` runs
`generate:schemas:check` and `app-frontend` regenerates the TS and
`git diff --exit-code`s it. `src/types/generated/` is eslint-ignored and
`schemas-generated/` prettier-ignored. The export discriminator lives in
`src/store/exportKind.ts` (`detectExportKind`).

## Out of scope

Signed installers, in-app detection beyond the Run workspace, and init-failure
diagnosis are not implemented (see ADR 0014).
