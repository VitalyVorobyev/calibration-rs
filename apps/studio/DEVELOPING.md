# Developing Calibration Studio

Tauri 2 + React 19 + TypeScript. The Rust back end (`apps/studio/crates/studio-core`,
wrapped by the Tauri shell in `apps/studio/src-tauri`) depends on the workspace
facade (`vision-calibration`) by path, so its commands call the same solvers
as `cargo run --example`. Design rationale:
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
bun run generate:api   # regenerate the API schema, TS wire types and client
bun run lint           # oxlint, type-aware on TypeScript 7; no warnings allowed
bun run test           # Vitest: pure-logic unit tests + jsdom component tests
bun run test:e2e       # Playwright smoke: boots the app, checks it doesn't fall over
bun run test:screens   # screenshots of every workspace vs. a local baseline
bun run viewer -- <run-dir>  # the viewers in a browser over one run (calib-viewer)
bun run test:e2e:viewer      # Playwright against the real calib-viewer (needs cargo)
```

Use `bun run tauri dev`, **not** `bun run dev`: the latter only starts Vite at
<http://localhost:1420> with no back end, so nothing loads; the header says
"No back end" and the desktop-only features are hidden.

## Layout

- `apps/studio/` is a **separate cargo workspace** (`apps/studio/Cargo.toml`),
  excluded from the root workspace (`exclude = ["apps"]` in `/Cargo.toml`)
  because Tauri's build-time dependency tree is heavy. Its members are
  `crates/studio-core` (the back end: no Tauri, no HTTP) and `src-tauri`
  (the Tauri shell and the `calib-viewer` binary). It pins its own
  `apps/studio/Cargo.lock`, builds into `apps/studio/target/` and is built from
  inside `apps/studio/`; `cargo test --workspace` at the repo root does not
  cover it.
- Frontend (`apps/studio/src/`): `shell/` (the workbench frame: header, project
  menu, workspace rail, project navigator, status line), `project/` (the
  project model: typed commands, store, run names and recent projects),
  `workspaces/` (one folder per workspace), `components/`, `hooks/`, `lib/`,
  `store/` (the loaded export and selection, `index.ts`; the Run setup,
  `runSetup.ts`), `api/` (the back end: see "The back end and its
  client"). The Run workspace's forms carry no schema: `config_form`,
  `dataset_form` and `resolve_config` serve the library's schemas, its
  defaults and its merge rule (see "Forms" below).
- Back end (`apps/studio/crates/studio-core/src/`): `api/` (the command
  table, `Ctx`, the error envelope, jobs, events and the API schema; its
  `commands/` hold the command functions), `project.rs` (project folders),
  `ops.rs` (the viewer commands' bodies: export loading, undistortion,
  epipolar overlay, uncertainty, disparity), `run.rs` (calibration runner),
  `disparity.rs` (dense stereo), `epipolar.rs`, `uncertainty.rs`,
  `laser.rs`, `export_cache.rs`, `scope.rs` (the file scope: `RootScope`,
  `UserPath`), `run_source.rs` (the run `calib-viewer` serves),
  `bin/studio_schema.rs` (writes `schema/studio-api.json`). Shell
  (`apps/studio/src-tauri/src/`): `commands.rs` (the Tauri transport of the
  table), `assets.rs` (the asset-protocol scope), `viewer/` (the
  `calib-viewer` HTTP transport, see "Browser mode").
- Module size: no `apps/studio/src` module over ~400 lines (generated files
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
(`manifest.json`, `config.json`, `export.json`, `diagnostics.json`, and
`report.html` from the library's `analysis::html_report`). The
back-end commands are `create_project`, `open_project`,
`save_project_manifest` and `save_project_run`; each rejects with the
error envelope (`StudioError { kind, message }`). `project/projectStore.ts`
drives them: opening a project loads its manifest and dataset folder into
the Run setup (`store/runSetup.ts`), a successful run with a project open is
saved as the next `run-NNN` and reopened from its `export.json`, and a
failure is shown by the shell until dismissed. A run name that exists is a
`run_exists` error (runs are never overwritten), and the saved export's
image root is rebased onto the run folder so it also opens on its own
through Open Export…. Recent projects are a per-viewer list in
`localStorage`; a folder that no longer opens drops off it. The dataset path
in `calibration-project.json` is relative to the project when the dataset
is inside it (`"."` beside the project file), absolute otherwise. Rust
tests for the project rules are in studio-core's `project.rs` (temporary
folders); the store is tested against a mock transport in
`project/projectStore.test.tsx`.

## Images and the asset protocol

Source frames load straight from disk through Tauri's asset protocol
(in the browser viewer, through its scoped file route):
`api`'s `fileUrl(path)` wraps `convertFileSrc`, and
`hooks/useImageData.ts` decodes the image (with `crossOrigin` set, so the
canvas the luminance is read from stays untainted). `tauri.conf.json`
enables the protocol with an empty scope; `assets.rs`'s `allow_dir` adds
each folder the back end's `RootScope` grants, which is what the user opens
(`open_project`/`create_project`: the project and its dataset folder;
`load_export`: the export's folder and image root; `read_manifest`: the
manifest's folder; `sniff_folder` and `run_calibration`: the dataset
folder). An image outside every opened folder fails to load. The CSP allows images from `asset:`/`http://asset.localhost`, `blob:`
and `data:`, and IPC; it applies to the built app only (in `tauri dev` the
webview loads the Vite server directly, without it).
`dangerousDisableAssetCspModification` keeps Tauri from
adding nonces to `style-src`, which would disable the `'unsafe-inline'` that
React's `style` attributes need. The `undistorted_image` resource answers
PNG bytes, over a binary IPC response (`studio_resource`,
`tauri::ipc::Response`) in the app, shown through a blob URL.

`lib/paths.ts`'s `joinPath` resolves paths as the file system does: an
absolute segment (an absolute image root) replaces the base, and `.`/`..`
are folded, so the asset scope sees plain paths.

## Browser mode (`calib-viewer`)

The viewers (Diagnose, the 3D viewer, Epipolar, Depth) also run in a plain
browser over one calibration run, served from localhost:

```bash
bun run viewer -- <run-dir | export.json> [--port N] [--no-open]
```

builds the frontend when `dist/` is missing or older than `src/`, then runs
`src-tauri`'s second binary, `calib-viewer` (`src/bin/calib_viewer.rs`,
`src/viewer/`). A run is a `vcal run -o` folder (`calibration.json`), a run
the app saved (`runs/<name>/export.json`), or one export file. The server
binds `127.0.0.1` only (a free port unless `--port`), serves `apps/studio/dist`
(`--dist` overrides it) with `index.html` marked
`<meta name="calib-backend" content="http">`, prints the address and opens
the browser. On start the page asks `GET /api/session` for the export,
loads it and lands on Diagnose. An export without an image list
(`diagnostics.image_manifest`) opens too: its frames come from its
residuals and the image panes say there are no images.

**The seam.** The frontend reaches the back end only through `src/api/`
(see "The back end and its client"); `src/api/transport/tauri.ts` is the
only module that imports `@tauri-apps/*` (an oxlint rule enforces it
outside tests). The HTTP transport is `POST /api/cmd/<name>` with the
command's (snake_case) argument object, answered with its JSON reply or
with the `StudioError` envelope and a 4xx/5xx status; a resource is
`GET /api/r/<name>?q=<query as JSON>`. `calib-viewer` answers only the
table's read-only commands (`COMMANDS` and `RESOURCES` in
`viewer/http.rs`); every other name is `unknown_command`, 404. Without
either back end (plain `bun run dev`), `capabilities().kind` is `"none"`.
Desktop features (the project menu, Open Export…, the Run workspace) show
only when `capabilities().desktop`; there are no disabled stand-ins.

**File scope.** The server serves files only under the run folder and the
export's image root (resolved against the export's folder), judged after
canonicalising (studio-core's `scope.rs`): `..`, other absolute paths and symbolic
links out of them are 403. It answers only `Host: 127.0.0.1:<port>` or
`localhost:<port>` and refuses a `POST` with a foreign `Origin`.

**Developing in browser mode.** Run the server on a fixed port and Vite with
the HTTP back end; Vite proxies `/api` to it (`vite.config.ts`,
`CALIB_VIEWER_URL` overrides the target):

```bash
cargo run -p calibration-studio --bin calib-viewer -- <run> --port 8765 --no-open
VITE_CALIB_BACKEND=http bun run dev
```

**Sample runs.** `cargo test -p studio-core write_viewer_sample_runs --
--ignored --nocapture` calibrates `data/kuka_1`
and, when present locally, `data/stereo_charuco` (`CALIB_VIEWER_DATA`
points elsewhere) and saves each as an app project under `apps/studio/.tmp-runs/`
(git-ignored), ready for `bun run viewer -- .tmp-runs/kuka_1/runs/run-001`.

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
  `K⁻¹` without a back end (`src/lib/frustumRays.ts`).
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
- **Lint.** `.oxlintrc.json` turns on `vitavision/tokens-only` from
  `@vitavision/config-oxlint` for `src/`: no raw Tailwind palette classes
  (`bg-slate-500`) and no hex literals. The only exemptions are data-colour
  files listed there with their reason (the residual colour ramp in
  `lib/errorColors.ts`, the per-laser palette in `LaserTargetCuts.tsx`).

`components/FrameCanvas.tsx` is the 2D frame viewer: an `ImageStage` from
[`@vitavision/stage2d`](https://github.com/VitalyVorobyev/lab-ui/tree/main/packages/stage2d)
with the ROI crop as its image and the residual arrows, laser overlay and
epipolar markers as batched SVG layers (`frameOverlayPaths.ts`,
`EpipolarWorkspace/markerPaths.ts`): one `<path>` per colour, never one
element per feature. Overlay coordinates are the detectors' own: the centre
of pixel *i* is at *i*. `components/ZoomControls.tsx` is the zoom/fit/1:1
cluster that drives it from outside the stage.

## Run progress and cancellation

A calibration is the `run_calibration` job (`src/lib/runCalibration.ts`,
`api.jobs.runCalibration`). studio-core's `run.rs` announces three stages,
`detect` -> `solve` -> `export` (`RunStage`), and forwards the library's
reports within them as `RunProgress.report` (a `Progress`: each image
detected, each calibration step, each solver iteration; iterations are
throttled to one per 50 ms); each is one of the job's events. The job's
output is the runner's `RunResponse`. Cancelling the job (`cancel_job`, the
library `CancelToken` of the job) stops the library at its next image or
solver iteration, and the job ends `cancelled`. The UI shows the latest
report beside the active stage and a distinct "Run cancelled" state; the
checklist logic is `src/workspaces/RunWorkspace/runStages.ts`.

## Forms

The Run workspace's manifest and config forms (`SchemaValueForm` from
`@vitavision/forms`) render the schemas the back end serves:
`dataset_form` (the `DatasetSpec` schema and the manifest a fresh session
starts from) and `config_form { topology }` (the workflow's `*Config` schema
and the library's default), fetched by `RunWorkspace/useForms.ts`. A preset's
`configOverrides` become a config through `resolve_config { topology, patch
}`, the library's `ConfigPatch` rule (tables merge key by key, anything else
replaces; a field the workflow lacks is a `config` error). The Run setup
store (`store/runSetup.ts`) starts empty and fills from these, so the
frontend holds no copy of a Rust default and cannot lag the back end. A
studio-core test compares the served schemas and defaults with the ones
`cargo xtask emit-schemas` publishes in the repository's `schemas/`, which
the form round-trip tests (`configRoundTrip.test.tsx`,
`configForms.test.tsx`) render.

## Built-in presets

Quick Start presets (`src/workspaces/RunWorkspace/presets.ts`) name a sample of
the library's catalogue (`vision_calibration::dataset::samples`) in
`source: { sample: "<id>" }`. `setupActions.ts`'s `applyPreset` runs the
`fetch_sample` job, which downloads the sample into the platform cache
(`VISION_CALIBRATION_DATA` overrides it), verifies its SHA-256 and returns the
manifest path, which `read_manifest` reads; later calls reuse the cache. The installer bundles no data and
nothing resolves a source checkout at runtime (no `CARGO_MANIFEST_DIR` outside
tests). Presets for local-only data go in an untracked `presets.local.ts` next
to it, exporting `LOCAL_PRESETS: Preset[]` with
`source: { manifestPath: "<absolute path>" }`; `PRESETS` appends them when the
file exists.

## Testing

- **Vitest (`bun run test`)** — `src/**/*.test.ts` are pure-logic unit tests
  (Node); `src/**/*.test.tsx` are jsdom component tests (opt in with
  `// @vitest-environment jsdom`). Component tests replace the back end
  with `setTransport(createMockTransport({ commands, jobs }))`
  (`src/api/transport/mock.ts`): handlers typed by the command table, jobs
  that announce their states and events as the back end does, and every
  call recorded; `kind: "http"` makes it the browser viewer's. The
  transports have contract tests of their own (`src/api/transport/*.test.ts`,
  Tauri's against `mockIPC`). `src/test/setupTests.ts` stubs
  `ResizeObserver`.
- **Playwright (`bun run test:e2e`)** — `e2e/` smoke tests against the plain
  Vite dev server (no Tauri/Rust toolchain). `app.spec.ts` checks every
  workspace mounts with no console errors and no Tauri mock (the
  desktop-only Run workspace with an empty one);
  `diagnose.spec.ts` injects a `window.__TAURI_INTERNALS__` shim
  (`e2e/support/tauriMock.ts`, which answers `studio_invoke` by the
  command's name) via `page.addInitScript`; its `assetUrl`
  argument stands in for the asset protocol. Real desktop behavior is only
  exercised manually: `bun run tauri dev` loads the Vite server directly, so
  the CSP and the bundled `tauri://` origin are only in effect in a built app
  (`bun run tauri build --debug --no-bundle`, then run
  `target/debug/calibration-studio`). Install the browser once with
  `bunx playwright install chromium --with-deps`.
- **Browser viewer (`bun run test:e2e:viewer`)** — `e2e/viewer.spec.ts`
  (`playwright.viewer.config.ts`) builds the frontend and runs the real
  `calib-viewer` on the committed fixture runs in `e2e/fixtures/`: a
  synthetic two-camera `rig_extrinsics` run saved as the app saves one
  (`viewer-run/`, 160×120 rendered chessboard images) and the same export
  without an image list as `vcal run -o` writes it (`viewer-run-vcal/`).
  It needs the Rust toolchain and is not part of CI's smoke run; the server's
  routes, file scope and run detection are covered by `cargo test` in
  `src-tauri/src/viewer/` and studio-core on the same fixture.
- **Screenshots (`bun run test:screens`)** — `e2e/screens.spec.ts`
  (`playwright.screens.config.ts`, port 1421) captures every workspace; the
  baseline in `e2e/.screens/` is local and uncommitted (`--update-snapshots`
  before a change, compare after). Not run in CI.

## The back end and its client

Every command the back end offers is one line of studio-core's command
table (`crates/studio-core/src/api/table.rs`) over one typed function, of
three kinds: `plain` (argument object → reply), `job` (argument object →
events → output, on a worker lane, cancellable) and `resource` (query →
bytes). Both transports dispatch through it: the Tauri shell's
`studio_invoke`, `studio_subscribe`, `studio_unsubscribe` and
`studio_resource`, and `calib-viewer`'s HTTP routes. Every command runs
against a `Ctx` (the `RootScope`, the active export, the jobs, the event
bus), and every failure is one `StudioError { kind, message, path? }`.

- **Paths.** A path argument is a `UserPath`, which only the `RootScope`
  turns into a `Path`. The desktop app's scope grows with what the user
  opens (`open_dir`/`open_file`: a project, an export, a manifest, a
  dataset folder); `calib-viewer`'s is fixed (the run folder and its image
  root). Any other path is `forbidden_path`. A table test refuses a
  path-named argument (`path`, `dir`, `*_path`, `*_dir`, …) that is not a
  `UserPath`, an argument object that accepts unknown fields, and a name
  that is not snake_case.
- **Jobs.** Starting a job answers its `JobInfo` (`<counter>-<hex>` id) at
  once; its states (`queued` with a position, `running`, `done`, `failed`,
  `cancelled`) and events go out on the event stream. One worker per lane
  (`Heavy`: calibrations; `Light`: sample downloads), a FIFO queue of 8
  (`busy` beyond), and the newest 32 finished jobs kept for an hour
  (`list_jobs`, `job_status`, `cancel_job`). On the frontend,
  `src/api/jobs.ts` follows them over one subscription.
- **Adding a command.** Write a function `fn(&Ctx, Args) -> Result<Reply,
  StudioError>` in `crates/studio-core/src/api/commands/`, with an
  argument struct (`Deserialize`, `JsonSchema`, `deny_unknown_fields`; its
  rustdoc is the command's documentation in the client), add its line to
  `table()`, and run `bun run generate:api`. A command `calib-viewer`
  should answer goes into its `COMMANDS` list.

The client is generated:

```bash
bun run generate:api            # both stages; commit the results
bun run generate:api:schema     # stage 1 (cargo): the table -> schema/studio-api.json
bun run generate:api:ts         # stage 2 (bun): schema -> src/api/generated/{types,client}.ts
```

`schema/studio-api.json` holds every command with its argument, reply,
event and output types, the event and error types, and the wire types the
frontend reads that no command names (the exports, the viewer session), in
one `definitions` block. `generated/types.ts` is every type;
`generated/client.ts` is `createClient`, one typed function per command
(`api.openProject({ dir })`), job (`api.jobs.runCalibration(…)`) and
resource (`api.resources.undistortedImage(…)`), and the only file that
names commands. Both outputs are committed. CI enforces sync:
`app-src-tauri` runs `studio-schema --check` and `app-frontend`
regenerates the TS and `git diff --exit-code`s it. `src/api/generated/` is
lint-ignored and `schema/` prettier-ignored. The export discriminator
lives in `src/store/exportKind.ts` (`detectExportKind`).
