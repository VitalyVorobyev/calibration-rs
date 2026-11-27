# Developing Calibration Studio

Tauri 2 + React 19 + TypeScript. The Rust back end (`apps/studio/crates/studio-core`,
carried by the Tauri shell in `apps/studio/src-tauri` and by the HTTP server in
`apps/studio/crates/studio-server`) depends on the workspace
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
bun run test:e2e       # Playwright against the real server (needs cargo)
bun run test:e2e:smoke # Playwright on plain Vite, no back end (bun only)
bun run test:screens   # screenshots of every workspace vs. a local baseline
bun run serve -- --root <dir>  # the app in a browser (calibration-studio-server)
```

Use `bun run tauri dev` for the desktop app. Plain `bun run dev` starts Vite
at <http://localhost:1420> and proxies `/api` to a running server; without
one the page says no back end answers.

## Layout

- `apps/studio/` is a **separate cargo workspace** (`apps/studio/Cargo.toml`),
  excluded from the root workspace (`exclude = ["apps"]` in `/Cargo.toml`)
  because Tauri's build-time dependency tree is heavy. Its members are
  `crates/studio-core` (the back end: no Tauri, no HTTP),
  `crates/studio-server` (`calibration-studio-server`: axum, the frontend
  embedded) and `src-tauri` (the desktop shell). It pins its own
  `apps/studio/Cargo.lock`, builds into `apps/studio/target/` and is built from
  inside `apps/studio/`; `cargo test --workspace` at the repo root does not
  cover it.
- Frontend (`apps/studio/src/`): `shell/` (the workbench frame: header, project
  menu, workspace rail, project navigator with the chosen run's actions,
  status line; the session with the back end, `session.ts`; Settings),
  `project/` (the open project, `projectStore.ts`; its unsaved edits,
  `draftStore.ts`; the ways to a project, `projectActions.ts`; pure rules,
  `projectModel.ts`), `workspaces/` (one folder per workspace),
  `components/` (among them `PathPicker/`), `hooks/`, `lib/`, `store/` (the
  run the viewers show and the selection, `index.ts`), `api/` (the back
  end: see "The back end and its client"). The Run workspace's forms carry
  no schema: `config_form`, `dataset_form`, `resolve_config` and
  `config_patch` serve the library's schemas, its defaults and its merge
  rule (see "Forms" below).
- Back end (`apps/studio/crates/studio-core/src/`): `api/` (the command
  table, `Ctx`, the error envelope, jobs, events, resources and the API
  schema; its `commands/` hold the command functions), `runs.rs` (a run as
  commands name it, `RunRef`, and `RunCache`, the parsed exports),
  `disparity.rs`, `epipolar.rs`, `uncertainty.rs`, `laser.rs`, `images.rs`,
  `blobs.rs`, `planes.rs` (see "Images and resources"), `scope.rs` (the file
  scope: `RootScope`, `UserPath`, and the checks of what a project and its
  manifest name), `settings.rs` (the desktop's roots, the recent projects),
  `bin/studio_schema.rs` (writes `schema/studio-api.json`). Projects and
  runs are the library's (`vision_calibration::project`): the app has no
  project format, runner or run writer of its own.
  Desktop shell (`apps/studio/src-tauri/src/`): `commands.rs` (the Tauri
  transport of the table), `scheme.rs` (the `studio:` URI scheme). Server
  (`apps/studio/crates/studio-server/src/`): see "The server".
- Module size: no `apps/studio/src` module over ~400 lines (generated files
  excepted). Split by responsibility: pure logic in a `.ts` module with its
  own unit test, views in `.tsx`, effects in a `use…` hook.

## Shell and projects

`shell/Shell.tsx` is `@vitavision/workbench`'s `AppShell`: `ShellHeader`
(project menu, the run shown, Import Export…, Settings, theme),
`WorkspaceRail` (a `NavRail` of router links; `⌘1`…`⌘5` from
`shell/workspaces.ts`), `ProjectPanel` in the left slot while a project is
open (its runs as a `TreeView`, and `RunActions` for the chosen run: its
label and star, its report page, whether it is stale, its deletion), the
routed workspace, and a `StatusBar`. The workbench has no router; the rail
items render react-router `Link`s through `asChild`.

**Projects are the library's** (`vision_calibration::project`, the book's
Projects chapter): `calibration-project.toml` (or `.json`) with the
dataset's manifest in it or referenced, the configuration's changes,
exclusions, a device; `runs/run-NNN/` with `run.json`, `export.json`,
`quality.json`, `notes.json` and `report.html`; `.detections/`, the
detection cache. A project of the app's earlier format (version 1:
`calibration-project.json` + `dataset.json`, runs without `run.json`) opens
as it is, its runs listed as `unrecorded`, and becomes version 2 on its
first save. The back end keeps nothing open: every command names the
project by its folder (`dir`) and a run by its folder and id (`RunRef {
dir, id }`); there is no "active export".

- **The project commands**: `scan_library` (every project under the
  roots, with its runs), `open_project` (a `ProjectView`: the spec, the
  revision, the data folder, the dataset and configuration as they resolve
  or why not, whether the manifest is the file's, the digests of the
  inputs as they stand, the runs; opening again is how a client reloads),
  `create_project`, `save_project` (with the revision `open_project`
  answered: a file that changed since is `changed_on_disk`, 409),
  `project_views` (the views the images pair into, of the saved project or
  of a draft spec, with what a run would refuse as messages carrying their
  field), `list_runs`, `set_run_notes`, `delete_run`, `import_export`
  (an export file or a `vcal run -o` folder as the next run), and the
  `calibrate` job (`Project::calibrate`; its events are the library's
  `Progress`, its output the written run's `RunInfo`). A change publishes
  `project_changed { dir }` on the event stream.
- **The store** (`project/projectStore.ts`) holds the open `ProjectView`
  and replaces it from every answer; `draftStore.ts` holds the Run
  workspace's unsaved edits (the manifest when the file holds it, the
  configuration) and saves them through `save_project`, the configuration
  as its changes from the defaults (`config_patch`). A run is stale when one
  of its four input digests (`dataset`, `exclusions`, `config`, `quality`)
  differs from the project's `input_digest` (`projectModel.staleParts`).
- **New projects** (`projectActions.ts`): of a dataset folder (the sniffer
  drafts the manifest, written into the project; what it leaves open stays
  in `_unresolved` and blocks a run), of a manifest file (referenced, never
  written), or a Quick Start sample (`fetch_sample` downloads it into the
  samples folder and makes that folder a project referencing its manifest).
  A folder that already holds a project is opened instead.

Recent projects are the back end's list (`recent_projects`,
`forget_recent`; the desktop's, or a server's shared by its clients), put
first by `open_project` and `create_project`; a folder that no longer opens
drops off it. Rust tests for the commands are in studio-core
(`api/commands/project_tests.rs`, on a synthetic project whose detections
the cache holds); the store against a mock transport in
`project/projectStore.test.tsx`.

**Session, roots and the path picker.** `shell/session.ts` starts once:
it takes the start-up fragment (`#token=…&project=…`, captured by
`shell/captureStart.ts` before the hash router reads the address), logs in
with its token, asks `server_info`, and opens the fragment's project. The
shell then shows (`SessionNotice.tsx`) the token field when the server
answers `unauthorized` (instead of the workspace), the connection error when
no back end answers (above the workspace), and on the desktop's first run —
editable roots and none but the samples' folder — "Add a folder to start".
The roots are the only folders the back end reads and writes. The desktop's
start empty and live in `<config>/calibration-studio/settings.json`; the one
native dialog left, "Add folder…" (`AddRootButton.tsx`, the transport's
`pickNativeFolder`), grants one through `add_root`, and Settings lists and
removes them. Every other path question (`shell/dialogs.ts`: a folder, a
project, a manifest, an export) opens `components/PathPicker/`, a dialog
over `list_dir` on both transports: breadcrumb from the root, folders
first, projects marked, never outside a root.

## Images and resources

Commands never return bytes. Images and dense results are **resources**:
the frontend asks `api.urls.<resource>(query)` for a URL and loads it in an
`<img>`, a `fetch` or `fetchPlane` (`src/api/resources.ts`); nothing outside
`src/api/` builds a URL. Five resources:

- `image { path, max? }` — a source frame. JPEG, PNG (8 or 16-bit), WebP,
  GIF and BMP are the file itself; TIFF, PNM and the other formats the back
  end reads become an 8-bit PNG (16-bit reduced as the detectors reduce it,
  so the view is what was detected). With `max`, a JPEG thumbnail (quality
  80) whose longest side is the next of 160, 320, 640, 1280.
- `undistorted_image { run, path, camera, roi? }` — a frame rendered into a
  camera's undistorted pixel frame of the run's export (a PNG).
- `report { run }` — the run's `report.html`, served under its own
  `Content-Security-Policy: sandbox allow-scripts …`: its scripts run in an
  opaque origin with no access to the app, its cookie or its API.
- `blob { id }` — a derived blob a command made, by its id (a `BlobRef`).
  The `disparity` job answers its output as blobs: a `VAM1` value plane
  (disparity, metric depth and the rectified left image), the point cloud
  as raw little-endian `f32`, the rectified pair as a PNG. The Depth
  workspace colours the planes itself (`DepthWorkspace/depthView.ts`, the
  charts package's viridis through stage2d's `rasterizePlane`). A blob the
  store evicted answers `blob_expired`; the workspace runs the job again.

Behind them, in studio-core: `images.rs` (`ImageCache`: derived images on
disk under `<cache>/calibration-studio/images/<aa>/<key>.<ext>`, keyed by
SHA-256 of the canonical path, length, mtime, variant and encoder revision —
metadata, not content — capped at 1 GiB, least recently used pruned at start
and every 256 writes; decoding capped at 200 MP and 1 GiB), `blobs.rs`
(`BlobStore`: in memory, 256 MiB, ids the SHA-256 of command, canonical
arguments, calibration fingerprint and part, so the same request is also
memoized), `planes.rs` (`VAM1`: the magic, `u32` LE width, height, stride
and channels, 4 reserved bytes, then `f32` LE values, plane-major; a Rust
test pins `src/api/__fixtures__/golden.vam1`, which `resources.test.ts`
decodes with stage2d's `decodePlane`), and `api/resource.rs` (the HTTP
semantics, decided once: one `Range` → 206/416 on files, `ETag` +
`If-None-Match` → 304, `Cache-Control: private, no-cache` on images and
`immutable` on blobs). Every path goes through the `RootScope`.

**Transports.** In the browser a resource is `GET /api/r/<name>?q=<query
as JSON>`. In the desktop app it is the `studio:` URI scheme
(`src-tauri/src/scheme.rs`): `studio://localhost/r%2F<name>%3Fq%3D<json>`
(`http://studio.localhost/…` on Windows), built by Tauri's
`convertFileSrc(…, "studio")` and answered by the same `Api::resource` on a
blocking thread. The scheme is another origin, so its replies name the
app's origin in `Access-Control-Allow-Origin`, and `hooks/useImageData.ts`
loads images with `crossOrigin` set: the canvas the luminance is read from
stays untainted. The CSP (`tauri.conf.json`) allows `studio:` and
`http://studio.localhost` in `img-src` and `connect-src`; there is no asset
protocol. The CSP applies to the built app only (in `tauri dev` the webview
loads the Vite server directly). `dangerousDisableAssetCspModification`
keeps Tauri from adding nonces to `style-src`, which would disable the
`'unsafe-inline'` that React's `style` attributes need.

## The server (`calibration-studio-server`)

The same app runs in a browser, served by `crates/studio-server`: one axum
binary with the built frontend embedded (`rust-embed`, feature
`embed-dist`; a release build fails without `dist/assets`, a debug build
serves the placeholder and says so). It carries the same command table over
HTTP, so every workspace works in a browser, projects and runs included.

```bash
bun run serve -- --root ~/datasets          # builds dist/ when stale, runs the debug server on it
bun run serve -- path/to/project --open      # one project (and its dataset folder) as the only roots
cargo run -p studio-server -- --help         # every flag
```

```text
calibration-studio-server [PROJECT] [--root DIR]... [--bind ADDR:PORT] [--config FILE]
    [--token-file FILE] [--state-dir DIR] [--cache-dir DIR] [--dist DIR]
    [--allowed-host HOST]... [--public-origin URL] [--open]
calibration-studio-server token [--rotate] [--state-dir DIR]
```

- **Routes** (`src/lib.rs`, `src/routes.rs`): `POST /api/login` (the token
  → the session cookie; public), `POST /api/logout`, `GET /api/health`
  (public), `POST /api/cmd/{name}` (a command's JSON reply or the
  `StudioError` envelope with a status from its kind), `GET /api/events`
  (server-sent events, resumed from `Last-Event-ID` or `?after=`,
  `X-Accel-Buffering: no`), `GET /api/r/{name}?q=<query JSON>` (a resource,
  with ranges and revalidation), and the frontend for every other `GET`
  (unknown paths answer `index.html`; hashed `assets/*` are immutable).
  Plain commands run on blocking threads behind a semaphore of the CPU count
  with a 120 s limit; jobs run on studio-core's own workers.
- **Roots** are fixed while it runs: `--root` flags and `roots` in the TOML
  `--config` (flags win, lists add up), plus `<state-dir>/samples`. `PROJECT`
  makes a project folder and its dataset folder the only roots and opens the
  project in the page (`#project=` in the printed address).
  `server_info.roots_editable` is false, `add_root` answers `roots_fixed`.
  Recent projects are the device's, in `<state-dir>/state.json`.
- **Token** (`src/token.rs`, `src/auth.rs`): 32 random bytes, base64url, from
  `--token-file`, `$CALIBRATION_STUDIO_TOKEN`, `token_file` in the config, or
  generated once into `<state-dir>/token` (mode 0600); no flag takes the
  value. A loopback bind needs none unless one is configured; any other bind
  always has one. The start-up line prints `http://<host>:<port>/#token=…`;
  the page reads the fragment (`shell/fragment.ts`, before the hash router
  sees it), posts it to `/api/login` and replaces it with `#/`. The cookie is
  `studio_session=<hex SHA-256("studio-session-v1" ‖ token)>; HttpOnly;
  SameSite=Strict; Path=/` (`Secure` behind an https `--public-origin`): no
  session table, it survives a restart and dies with `token --rotate`.
  `Authorization: Bearer <token>` works too. Comparisons are constant-time;
  a failed login waits 250 ms. Without a session only the page, health and
  login answer; everything else is 401 with nothing about the server in it.
- **Requests** (`src/guard.rs`): the `Host` must be this machine (its bind
  address, `localhost`, an IP literal, its name, an `--allowed-host`), once;
  a non-GET must carry this page's `Origin` (or `--public-origin`) and a
  JSON `Content-Type`; `OPTIONS` is 405 and no `Access-Control-*` header is
  ever sent; bodies are at most 4 MiB, queries 8 KiB, event streams 32.
  Every response carries `nosniff`, `no-referrer`, `same-origin` resource
  policy and a CSP whose `script-src` names the hashes of `index.html`'s
  inline scripts.
- **TLS** is a reverse proxy's: bind `127.0.0.1`, pass `--public-origin
  https://calib.example.lan --allowed-host calib.example.lan`, and do not
  buffer `/api/events` (Caddy: `reverse_proxy 127.0.0.1:8757 {
  flush_interval -1 }`).
- **aarch64 Linux:** `cargo zigbuild --release -p studio-server --target
  aarch64-unknown-linux-gnu.2.28` after `bun run build` (any host with zig
  and cargo-zigbuild). `--no-default-features --features embed-dist` leaves
  the Quick Start samples (and their network access) out;
  `server_info.samples` then says so. The bundle workflow builds both Linux
  targets as artifacts.
- **Developing with hot reload:** run the server on its default address
  (`bun run serve -- --root DIR`) and `bun run dev`; Vite proxies `/api` to
  it (`CALIBRATION_STUDIO_URL` overrides the target), so log in once with
  the printed token.

**Sample projects.** The repository's datasets are projects
(`data/kuka_1`, `data/DS8`, `data/opencv_stereo`, `data/stereo/{left,right,rig}`):
`bun run serve -- ../../data/kuka_1 --open` serves one, and its Run
workspace calibrates it into `data/kuka_1/runs/` (git-ignored, as `vcal run
data/kuka_1` does). A studio-core test opens every one of them.

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

A calibration is the `calibrate { dir, notes?, config_patch?, redetect? }`
job (`api.jobs.calibrate`, driven by `RunWorkspace/useRunController.ts`):
the library's `Project::calibrate` on the project as it is on disk (the Run
workspace saves its edits first), with the HTML report. Its events are the
library's `Progress` reports (each image detected, each calibration step,
each solver iteration; iterations are throttled to one per 50 ms); the
checklist (`RunWorkspace/runStages.ts`) takes its stage, `detect` ->
`solve` -> `export`, from each report's kind. Its output is the written
run's `RunInfo`, which the workspace lists and shows before it hands off to
Diagnose. A failure is the job's `StudioError`: `needs_input` (a choice the
manifest leaves open, with its `field` and `suggestions`) opens the
ask-user dialog, which writes the choice into the draft manifest.
Cancelling the job (`cancel_job`, the job's `CancelToken`) stops the
library at its next image or solver iteration, nothing is written, and the
job ends `cancelled`.

## Forms

The Run workspace's manifest and config forms (`SchemaValueForm` from
`@vitavision/forms`) render the schemas the back end serves:
`dataset_form` (the `DatasetSpec` schema and a starting manifest) and
`config_form { topology }` (the workflow's `*Config` schema and the
library's default), fetched by `RunWorkspace/useForms.ts`. The values are the
open project's: its manifest and its configuration as `open_project`
resolves it (the defaults with the project's changes). A saved
configuration goes back as its changes from the defaults, `config_patch {
topology, config }` (the library's `ConfigPatch::from_config`), the inverse
of `resolve_config { topology, patch }` (tables merge key by key, anything
else replaces; a field the workflow lacks is a `config` error). The
frontend holds no copy of a Rust default and cannot lag the back end. A
studio-core test compares the served schemas and defaults with the ones
`cargo xtask emit-schemas` publishes in the repository's `schemas/`, which
the form round-trip tests (`configRoundTrip.test.tsx`,
`configForms.test.tsx`) render.

## Quick Start samples

The Quick Start cards are `server_info.samples`: the library's sample
catalogue (`vision_calibration::dataset::samples`; empty in a build without
the `samples` feature). Using one runs the `fetch_sample { id }` job, which
downloads the sample into the samples folder (`VISION_CALIBRATION_DATA`
overrides it on the desktop), verifies its SHA-256, makes that folder a
project referencing the sample's `dataset.toml` (once; it keeps its runs
afterwards) and answers the project's folder, which the app opens. The
installer bundles no data and nothing resolves a source checkout at runtime
(no `CARGO_MANIFEST_DIR` outside tests). A local dataset is a project too:
open it, or make one of its folder or manifest.

## Testing

- **Vitest (`bun run test`)** — `src/**/*.test.ts` are pure-logic unit tests
  (Node); `src/**/*.test.tsx` are jsdom component tests (opt in with
  `// @vitest-environment jsdom`). Component tests replace the back end
  with `setTransport(createMockTransport({ commands, jobs }))`
  (`src/api/transport/mock.ts`): handlers typed by the command table, jobs
  that announce their states and events as the back end does, and every
  call recorded; `nativeFolder` gives it the desktop's dialog. The
  transports have contract tests of their own (`src/api/transport/*.test.ts`,
  Tauri's against `mockIPC`), and `src/api/seam.test.ts` fails when a module
  outside `src/api/` imports `@tauri-apps/*`, calls `.invoke(` or
  `.resourceUrl(`, or writes an `/api/` URL (oxlint cannot express the
  calls). `src/test/setupTests.ts` stubs `ResizeObserver`.
- **Playwright (`bun run test:e2e`)** — `e2e/server/` against the real
  back end: `scripts/e2e-server.mjs` copies the committed fixture library
  (`e2e/fixtures/library/`: `planar-demo/`, a version 2 project of six
  320×240 rendered chessboard views, written by studio-core's ignored
  `write_planar_demo` and calibrated by a normal test; `rig-demo/`, a
  project of the app's earlier format with a two-camera `rig_extrinsics`
  dataset of 160×120 views and its run `runs/run-001`; `vcal-run/`, a bare
  export) to a temporary root and runs the debug
  `calibration-studio-server` on 127.0.0.1:1431 with a token, serving
  `dist/` (it builds both when missing). The specs: 401 without the token,
  the token field and the fragment login (no token left in the URL, an
  HttpOnly cookie), the path picker opening the earlier-format project and
  its run with its images, recent projects, refused paths, and
  `planar-demo` opened through the picker, run as the `calibrate` job with
  its progress over SSE, its run listed, labelled and deleted (and the
  earlier-format project's run failing as the library says). Install the
  browser once with
  `bunx playwright install chromium --with-deps`.
- **Smoke (`bun run test:e2e:smoke`)** — `e2e/smoke.spec.ts` on the plain
  Vite dev server with its `/api` proxy pointing nowhere (no Rust
  toolchain): every workspace mounts, says no back end answers, and logs no
  error of its own.
- **Rust.** studio-core's tests cover the commands (the project commands
  on a synthetic project: open, create, save and a stale revision, the
  views of a draft, a calibration job and its run read, noted and deleted,
  a cancelled one writing nothing, `needs_input`, the earlier format's
  migration and an import, the library scan, the repository's sample
  projects), the scope (including symlinks and projects and manifests that
  name paths outside the roots), the resources and the settings;
  studio-server's (`tests/http.rs`, in process) the routes: 401s, login
  and cookie flags, bearer, Host/Origin/content-type, no CORS, body and
  query limits, path escapes, the frontend's files, ranges and
  revalidation, SSE and its resume, a stale save as 409 and the report's
  sandbox policy; src-tauri's a mock-runtime test of the IPC commands (a
  project and a run's export among them) and the `studio:` scheme. Real desktop behaviour (the
  webview, its CSP and `tauri://` origin, the native dialog) is exercised
  manually: `bun run tauri build --debug --no-bundle`, then run
  `target/debug/calibration-studio`.
- **Screenshots (`bun run test:screens`)** — `e2e/screens.spec.ts`
  (`playwright.screens.config.ts`, the real server on port 1433) captures
  every workspace; the
  baseline in `e2e/.screens/` is local and uncommitted (`--update-snapshots`
  before a change, compare after). Not run in CI.

## The back end and its client

Every command the back end offers is one line of studio-core's command
table (`crates/studio-core/src/api/table.rs`) over one typed function, of
three kinds: `plain` (argument object → reply), `job` (argument object →
events → output, on a worker lane, cancellable) and `resource` (query → a
file or bytes behind a URL, see "Images and resources"). Both transports
dispatch through it: the Tauri shell's `studio_invoke`, `studio_subscribe`
and `studio_unsubscribe` and its `studio:` scheme, and the server's HTTP
routes. Every command runs against a `Ctx` (the `RootScope`, the settings,
the run cache, the image cache, the blob store, the jobs, the event bus),
and every failure is one `StudioError { kind, message, path?, field?,
suggestions? }`: each library error (`ProjectError`, the dataset runner's)
maps to its own kind, with the path and the field it names. A
shell sets the API up with a `Setup`: its folders (`Dirs`: cache, state,
samples) and, on the server, its fixed roots.

- **Paths.** A path argument is a `UserPath`, which only the `RootScope`
  turns into a `Path`: canonicalised and judged against the roots, so `..`,
  symbolic links out of a root, sibling-prefix names and relative paths are
  `forbidden_path`. Opening something never grants it: the roots change
  only through `add_root`/`remove_root` (the desktop). Paths a file names
  are judged too, before the library follows them: a project's data folder
  and referenced manifest on every open (`RootScope::check_project_paths`),
  and every file its manifest names — the glob bases lexically, then each
  file the library's `dataset_runner::input_files` lists, through the scope
  — before its dataset is shown, viewed, saved or run
  (`RootScope::check_dataset`, `check_project`); an imported export's
  image manifest is judged before the import reads its images. A table test
  refuses a path-named argument (`path`, `dir`, `*_path`, `*_dir`, …) that is
  not a `UserPath` (a `ProjectSpec` or `DatasetSpec` argument is a file's
  contents, judged as above), an argument object that accepts unknown
  fields, and a name that is not snake_case.
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
  `table()`, and run `bun run generate:api`. Both transports offer it at
  once.

The client is generated:

```bash
bun run generate:api            # both stages; commit the results
bun run generate:api:schema     # stage 1 (cargo): the table -> schema/studio-api.json
bun run generate:api:ts         # stage 2 (bun): schema -> src/api/generated/{types,client}.ts
```

`schema/studio-api.json` holds every command with its argument, reply,
event and output types, the event and error types, and the wire types the
frontend reads that no command names (the exports), in
one `definitions` block. `generated/types.ts` is every type;
`generated/client.ts` is `createClient`, one typed function per command
(`api.openProject({ dir })`), job (`api.jobs.calibrate(…)`) and
resource URL (`api.urls.undistortedImage(…)`), and the only file that
names commands. Both outputs are committed. CI enforces sync:
`app-src-tauri` runs `studio-schema --check` and `app-frontend`
regenerates the TS and `git diff --exit-code`s it. `src/api/generated/` is
lint-ignored and `schema/` prettier-ignored. The export discriminator
lives in `src/store/exportKind.ts` (`detectExportKind`).
