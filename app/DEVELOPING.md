# Developing the app

Tauri 2 + React 19 + TypeScript. The Rust backend (`app/src-tauri`) depends on
the workspace facade (`vision-calibration`) by path, so its Tauri commands
call the same solvers as `cargo run --example`. Design rationale:
`docs/adrs/0014-desktop-app.md`.

## Commands

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
- Frontend (`app/src/`): `shell/` (the workbench frame: header, project
  menu, workspace rail, project navigator, status line), `project/` (the
  project model: typed commands, store, run names and recent projects),
  `workspaces/` (one folder per workspace), `components/`, `hooks/`, `lib/`,
  `store/` (the loaded export and selection, `index.ts`; the Run setup,
  `runSetup.ts`), `schemas/` (JSON Schemas for the Run config forms, plus
  each config's `<name>.default.json`; both emitted by `cargo xtask
  emit-schemas`), `types/generated/` (generated wire types).
- Backend (`app/src-tauri/src/`): `project.rs` (project folders),
  `assets.rs` (the asset-protocol scope), `commands.rs` (export loading,
  undistortion, epipolar overlay, uncertainty), `run.rs` (calibration
  runner), `disparity.rs` (dense stereo).
- Module size: no `app/src` module over ~400 lines (generated files
  excepted). Split by responsibility: pure logic in a `.ts` module with its
  own unit test, views in `.tsx`, effects in a `use…` hook.

## Shell and projects

`shell/Shell.tsx` is `@vitavision/workbench`'s `AppShell`: `ShellHeader`
(project menu, Open Export…, theme), `WorkspaceRail` (a `NavRail` of router
links; `⌘1`…`⌘5` from `shell/workspaces.ts`), `ProjectPanel` in the left
slot while a project is open (its runs as a `TreeView`), the routed
workspace, and a `StatusBar`. The workbench has no router; the rail items
render react-router `Link`s through `asChild`.

A project folder holds `calibration-project.json` (`{"version": 1,
"dataset": "<dir>"}`), `dataset.json` and `runs/<name>/`
(`manifest.json`, `config.json`, `export.json`, `diagnostics.json`). The
backend commands are `create_project`, `open_project`,
`save_project_manifest` and `save_project_run`; each rejects with a
`ProjectError { kind, message }` (generated type). `project/projectStore.ts`
drives them: opening a project loads its manifest and dataset folder into
the Run setup (`store/runSetup.ts`), a successful run with a project open is
saved as the next `run-NNN` and reopened from its `export.json`, and a
failure is shown by the shell until dismissed. Rust tests for the project
rules are in `project.rs` (temporary folders); the store is tested against
`mockIPC` in `project/projectStore.test.tsx`.

## Images and the asset protocol

Source frames load straight from disk through Tauri's asset protocol:
`lib/tauri.ts`'s `assetUrl(path)` wraps `convertFileSrc`, and
`hooks/useImageData.ts` decodes the image (with `crossOrigin` set, so the
canvas the luminance is read from stays untainted). `tauri.conf.json`
enables the protocol with an empty scope; `assets.rs`'s `allow_dir` adds a
folder when the user opens it (`open_project`/`create_project`: the dataset
folder; `load_export`: the export's image root; `run_calibration_cmd`: the
dataset folder of the run). An image outside every opened folder fails to
load. The CSP allows images from `asset:`/`http://asset.localhost`, `blob:`
and `data:`, and IPC; it applies to the built app only (in `tauri dev` the
webview loads the Vite server directly, without it).
`dangerousDisableAssetCspModification` keeps Tauri from
adding nonces to `style-src`, which would disable the `'unsafe-inline'` that
React's `style` attributes need. `load_undistorted_image` returns PNG bytes
as a binary IPC response (`tauri::ipc::Response`), shown through a blob URL.

`lib/tauri.ts`'s `joinPath` resolves paths as the file system does: an
absolute segment (an absolute image root) replaces the base, and `.`/`..`
are folded, so the asset scope sees plain paths.

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
  a second app needs it, not into `src/components/`. Charts come from
  [`@vitavision/charts`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/charts)
  (Diagnose's ROI luminance histogram is its `Histogram`; `src/index.css`
  imports `@vitavision/charts/styles.css`). The frame — `AppShell`,
  `NavRail`, `TreeView`, `StatusBar` — comes from
  [`@vitavision/workbench`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/workbench)
  (its stylesheet is imported after ui's), and toasts from `@vitavision/ui`'s
  `toast()` with one `<Toaster />` in the shell. Icons are `lucide-react`,
  the icon set the packages use.
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
- **Lint.** `eslint.config.js` applies `tokensOnly(["src/**"])`
  from `@vitavision/config-eslint`: no raw Tailwind palette classes
  (`bg-slate-500`) and no hex literals in `src/`. The only exemptions are
  data-colour files listed there with their reason (the residual colour ramp
  in `lib/errorColors.ts`, the per-laser palette in `LaserTargetCuts.tsx`).

`components/FrameCanvas.tsx` is the 2D frame viewer: an `ImageStage` from
[`@vitavision/stage2d`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/stage2d)
with the ROI crop as its image and the residual arrows, laser overlay and
epipolar markers as batched SVG layers (`frameOverlayPaths.ts`,
`EpipolarWorkspace/markerPaths.ts`): one `<path>` per colour, never one
element per feature. Overlay coordinates are the detectors' own: the centre
of pixel *i* is at *i*. `components/ZoomControls.tsx` is the zoom/fit/1:1
cluster that drives it from outside the stage.

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
**repo-root-relative** `manifestPath` (`data/…`). The built-in list holds only
datasets a fresh checkout can run (`data/kuka_1`, and `data/DS8` once its public
archive is extracted there). Presets for local-only data go in an untracked
`presets.local.ts` next to it, exporting `LOCAL_PRESETS: Preset[]`; `PRESETS`
appends them when the file exists.
`setupActions.ts`'s `applyPreset` resolves the root through `lib/tauri.ts`'s `repoRoot()`, which
calls `repo_root_cmd` (`env!("CARGO_MANIFEST_DIR")` two levels up) and joins it
with `joinPath`. Because the app is always compiled from a local checkout, this
is the developer's own checkout path. Bundling public datasets in a release
installer would use Tauri's `resource_dir` instead.

## Testing

- **Vitest (`bun run test`)** — `src/**/*.test.ts` are pure-logic unit tests
  (Node); `src/**/*.test.tsx` are jsdom component tests (opt in with
  `// @vitest-environment jsdom`). Component tests mock IPC with
  `@tauri-apps/api/mocks`' `mockIPC` (and `mockConvertFileSrc` where a
  frame image is loaded); `src/test/setupTests.ts` stubs `ResizeObserver`.
- **Playwright (`bun run test:e2e`)** — `e2e/` smoke tests against the plain
  Vite dev server (no Tauri/Rust toolchain). `app.spec.ts` checks every
  workspace mounts with no console errors and no Tauri mock;
  `diagnose.spec.ts` injects a `window.__TAURI_INTERNALS__` shim
  (`e2e/support/tauriMock.ts`) via `page.addInitScript`; its `assetUrl`
  argument stands in for the asset protocol. Real desktop behavior is only
  exercised manually: `bun run tauri dev` loads the Vite server directly, so
  the CSP and the bundled `tauri://` origin are only in effect in a built app
  (`bun run tauri build --debug --no-bundle`, then run
  `src-tauri/target/debug/calibration-diagnose`). Install the browser once with
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
