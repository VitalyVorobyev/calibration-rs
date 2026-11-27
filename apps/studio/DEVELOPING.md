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
- Frontend (`apps/studio/src/`): `shell/` (the workbench frame: the rail of
  seven screens, the context bar, the run list, the job tray, the status
  line, Settings, the keys; see "The shell"), `screens/` (one folder per
  screen: Library, Dataset, Detect, Calibrate, Inspect; the ones not built yet are a
  "coming" page), `session/` (the stores every screen shares: the session
  with the back end, the selection, the jobs and the connection, the query
  cache, the pure rules of a job row), `project/` (the open project, `projectStore.ts`; its unsaved edits,
  `draftStore.ts`; the ways to a project, `projectActions.ts`; pure rules,
  `projectModel.ts`), `workspaces/` (the viewers built before the screens,
  mounted by them),
  `components/` (among them `PathPicker/`), `hooks/`, `lib/`, `store/` (the
  run the viewers show and the selection, `index.ts`), `api/` (the back
  end: see "The back end and its client"). Calibrate's forms carry
  no schema: `config_form`, `dataset_form`, `resolve_config` and
  `config_patch` serve the library's schemas, its defaults and its merge
  rule (see "Forms" below).
- Back end (`apps/studio/crates/studio-core/src/`): `api/` (the command
  table, `Ctx`, the error envelope, jobs, events, resources and the API
  schema; its `commands/` hold the command functions), `runs.rs` (a run as
  commands name it, `RunRef`, and `RunCache`, the parsed exports),
  `disparity.rs`, `epipolar.rs`, `uncertainty.rs`, `laser.rs`, `images.rs`,
  `blobs.rs`, `planes.rs` (see "Images and resources"), `maps.rs` and
  `detections.rs` (see "Maps, residuals and detections"), `scope.rs` (the file
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

## The shell

`shell/Shell.tsx` is `@vitavision/workbench`'s `AppShell`. The **rail**
(`ScreenRail`) holds the seven screens in workflow order (Library, Dataset,
Detect, Calibrate, Inspect, Compare, Device), `⌘1`…`⌘7`; `shell/screens.ts`
is the one table of them. A screen that is not built, or needs a project
that is not open, is disabled and says why on hover; its address still opens
a "coming in a later release" page. The **context bar** (`ContextBar`) holds
`Breadcrumb` (the project's device and name; the project is the menu that
opens, creates and reopens projects), `RunSelector` (the current run, newest
first, with pills from `runPills.ts`: the state of its files, its verdict in
words, `Stale` with the inputs that changed), the job indicator, the
shortcut list, the theme and Settings. Under it, `ConnectionStrip`: "No back
end answers" with Retry, or "Connection lost" while the event stream is down
(what writes is disabled through `useReadOnly`; it clears itself). The
`RunList` is the left panel of Calibrate, Inspect and Compare: the runs as a
list of cards (`RunCard`, `RunDetails`; rules in `project/runRows.ts`) and, for
the current run, its label, notes and settings.
The `JobTray` is the bottom panel: a row per job (the running one, the queue,
the last three that ended, every failure until dismissed) with its stage, its
bar, Cancel, a plot of the solver's cost, and the run a calibration wrote.
`StatusLine` is the foot. While the server wants its token the page is
`LoginPanel` and nothing else. The workbench has no router; the rail items
render react-router `Link`s through `asChild`.

**The address** is `#/<screen>[/<tab>]?project=<dir>&run=<id>`
(`screens.ts`: `parsePlace`, `placePath`). `useUrlSync` keeps it and the
open project and run the same thing in both directions (a link opens the
project and run; choosing either writes the address; a project opened without
a run opens on its newest). Inspect's tabs are `Scene`, `Frames`, `Parameters`,
`Distortion`, `Residuals`, `Projection σ` (`sigma`), `Quality`, `Epipolar`
and `Depth` (see "Inspect" below); a tab a run cannot have is absent
(`screens/inspect/tabs.ts`). The old addresses (`#/diagnose`,
`#/viewer3d`, `#/epipolar`, `#/depth`, `#/run`) redirect. Calibrate is its own
screen (see "Calibrate" below).

**Keys** are pure rules in `keymap.ts` (a keystroke and whether the user is
typing become an action) carried out by `useShellKeys`: `⌘1`…`⌘7`, `⌘K` (the
palette, `CommandPalette`, searched by `lib/palette.ts`), `⌘S`, `[` `]`
views, `,` `.` cameras, `⇧[` `⇧]` runs, `Esc` (the selection, then the
tray), `?`. A bare key never fires while a text field has focus, and nothing
but `⌘K` while a dialog is open. The window's frame puts the context bar and
the rail before the screen in the tab order; `SkipLink` is the first stop.

**The stores** (`src/session/`): `session.ts` (the back end: token, roots,
what it serves), `selection.ts` (`{ camera, view, feature }` and a transient
hover, rules in `lib/selection.ts`; `startSelectionBridge` keeps it in step
with the export store's slots the older viewers still read), `jobs.ts` (the
jobs and the connection, from the one **standing** event stream opened for as
long as a session is ready; `list_jobs` fills in what ran before the page or
while the stream was down), `sync.ts` (starts the watch; `project_changed {
dir }` marks the cached answers stale and re-reads the open project, so a
change from another tab or program arrives without a reload; after a drop
everything is read again), `query.ts` (a cache of `(command, arguments)`
answers shared by panels, stale rather than dropped on a change), `ui.ts`
(which shell dialog is open). `lib/format.ts` writes every number: `value ± σ`
at the decimals of σ (`formatWithSigma`), `value ·` with the reason on hover
when there is none, units. The client's part is in `src/api/`:
`events.connect/listen/onConnection/onUnreachable`, and a transport's
`subscribe` reports the stream dropping and returning (the browser reopens one
it gave up on); a command that never reached the back end is a `StudioError`
with `network` set.

Component tests build the shell over placeholder screens and a mock back end
(`src/test/shellKit.tsx`); the Playwright specs on the real server are
`e2e/server/shell.spec.ts` (the rail, deep links, another client's change,
the tray and its Cancel, the keys, a dropped stream) with `auth.spec.ts`
(sign-in).

## Calibrate

`screens/calibrate/` is the editor (`ParameterEditor`), "This run" (`RunPanel`,
`RunProgress`, `FailureView`, `RerunNotice`) and the pure rules beside them;
the run list is the shell's (`shell/RunList.tsx`).

- **The working configuration** is `project/runEditor.ts`: `edits` is `null`
  while the editor holds the project's configuration, else the whole config
  value the form edits. A run takes `patchOver(edits, projectConfig)` as its
  one-off `config_patch` (`project/configEdit.ts`: the fewest changes that
  merge over the project's patch, a list as one value, a removed optional
  field as `null`); the run records it. `saveAsDefault` is the only write of
  the project (`config_patch` then `save_project`, with the revision, so
  `changed_on_disk` is the shell's callout). The baseline (library default,
  project, or a run's recorded patch resolved by `resolve_config`) only
  decides what the form marks as changed.
- **Validation** (`useEditorCheck.ts`): after 300 ms of rest, `config_patch`
  (does it read) then `resolve_config` (does it check). The library's message
  starts with its field (`solver.max_iters must be positive`), which
  `configCheck.ts` maps to the form's `messages`; any other message is the
  form's. Run is disabled with the reason while it fails.
- **Curation** (`curate.ts`) reads the schema's structure, never a topology:
  a `camera` with a `model`, `camera_overrides`, any `…_loss`, `solver`,
  `…_init`, and the `refine_…`/`fix…` switches get groups; whatever no rule
  names is "All other fields", folded. A test renders it for all seven
  workflows' schemas and checks every field of the default configuration is
  in a group. `cameraRenderer.tsx` takes over two things the schema's
  controls do badly: a camera's `fix` object (chips; the names per model are
  `cameraParams.ts`, the library's `CameraFix` names, and a model it does not
  know keeps the form's own control) and `camera_overrides` (a row per
  camera). Switching a robust loss to a kind with a scale seeds 1 (the schema
  would seed 0, which no loss accepts).
- **History**: `runRows.ts` reads a run's record into a row (pooled and
  per-camera RMS, views, features, changes against the project's patch).
  `staleSince` does not call a run stale for settings of its own: the
  library's digest differs by them, and the changes say it. The lowest-RMS
  mark is only a mark; the star is the person's acceptance. Ticks feed
  `session/compareSet.ts`, which the Compare screen will read.
- **Seed row**: "not available yet"; no command serves a seed.

## Inspect

`screens/inspect/` holds the per-camera tabs; `Scene`, `Frames`, `Epipolar`
and `Depth` are the viewers of `workspaces/` (the Frames viewer lost its Stats
and Uncertainty panels to the tabs). Every per-camera tab is a `CameraFrame`
(`CameraFrame.tsx`, `target.ts`: the run in the export store and the camera of
the selection, else the run's first; plain words while no run is shown and
while a project's run is opening) over queries of the typed client.

- **Maps.** `maps/useMaps.ts` makes each map command a query and joins its
  reply (grid, channels, the planes' blob ids) to the decoded plane
  (`fetchPlane`, one cache entry per blob id, so a plane is decoded once;
  `blob_expired` runs the command again and reads the fresh blob).
  `maps/MapViewer.tsx` is the one viewer of the distortion, residual and
  projection σ maps: an `ImageStage` over the camera's image extent with
  `HeatmapLayer` (smooth, over the plane's block rectangle, which a partial
  last block makes run past the image: the layers sit in a clipping wrapper),
  `ContourLayer`, `VectorField`, a vertical `ColormapLegend` (unit, clipping,
  a named limit, the pointer's value), `MapReadout` and the toolbar
  (`components/ViewerToolbar.tsx`, also the Frames viewer's). The tab passes
  its own controls (`controls`) and what the map says (`build` in
  `DistortionTab`, `SigmaTab`, `ResidualMapPanel`); `MapPanel` keeps the last
  plane of the camera while the next loads, so the viewer keeps its choices.
  The pure rules, each tested, are `mapModel.ts` (domain: 99.5th percentile
  rounded up to 1/2/5·10ⁿ; node geometry; readout; arrows; the significance
  mask; peaks), `viewerModel.ts` (colour map: viridis, inferno over an image,
  blue-orange for a signed quantity; iso-line step; legend markers) and
  `mapText.ts` (the statements). A significance test the plane lacks (too few
  effective views) disables the fade instead of counting nodes as "not
  significant"; 3.03 is the library's p < 0.01 cut.
- **Residuals tab.** `ResidualsTab` owns the views scope (`residualModel.ts`:
  all, the selected view, a chosen few, which become `residual_map`'s
  `views`), then the map, `ResidualRecords` (`ScatterChart` and a windowed
  `Table` of `residual_records`, worst first; a point or row selects camera,
  view and feature in `session/selection.ts`) and `ResidualStatsTable`
  (`residual_stats`). `components/FrameCanvas` rings the selected feature;
  the arrows are anchored at the observed corner and run along
  `observed − projected`, which the legend says.
- **Parameters and Quality.** `parameterModel.ts` groups
  `camera_parameter_values`, writes value ± σ with `lib/format.ts` (radians
  as degrees), the reason there is no σ, and the JSON for the clipboard; pose
  parameters are the covariance's labels no camera owns, by σ only
  (`lib/uncertainty.ts`). `qualityModel.ts` holds the words of each check and
  the measured value and limit; verdict colour is on the `Badge` there only.
- **Tests.** Vitest per pure module, `maps/MapViewer.test.tsx` (legend,
  controls, readout), `maps/useMaps.test.tsx` (decode once, `blob_expired`),
  `InspectPanels.test.tsx` and `MapPanels.test.tsx` (each tab over a mock back
  end, `test/inspectKit.tsx`; value planes from `test/planeFixtures.ts`, which
  writes `VAM1` as `planes.rs` does). The
  Playwright spec `e2e/server/inspect.spec.ts` makes a run of `planar-demo`
  through Calibrate and walks the tabs, then opens the committed rig run.
  The stage's own drawing needs a layout engine, so it is seen only there.

## Projects

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
  and replaces it from every answer; `draftStore.ts` holds the
  unsaved edits (the manifest when the file holds it, the
  configuration) and saves them through `save_project`, the configuration
  as its changes from the defaults (`config_patch`). A run is stale when one
  of its four input digests (`dataset`, `exclusions`, `config`, `quality`)
  differs from the project's `input_digest` (`projectModel.staleParts`).
- **New projects** (`projectActions.ts`): of a dataset folder (the sniffer
  drafts the manifest, written into the project; what it leaves open stays
  in `_unresolved` and blocks a run), of a manifest file (referenced, never
  written), or a Quick Start sample (`fetch_sample` downloads it into the
  samples folder and makes that folder a project referencing its manifest).
  A folder that already holds a project is opened instead. The Run
  workspace still uses these; the Library and the project menu use the
  dialogs below.

**The Library and the project dialogs.** `screens/library/` is the Library:
`libraryModel.ts` (pure: a scan as rows, grouped `product → device →
project`, the search and filters, what a node's table lists), `libraryUrl.ts`
(the search, filters and selected node in the address, beside `project` and
`run`), `LibraryTree`, `ProjectTable`, `LibraryToolbar`. `scan_library` is
read through `project/useLibraryScan.ts` (any `project_changed` marks it
stale; the Library also rescans when the window regains focus). The dialogs
are mounted by the shell (`shell/ProjectDialogs.tsx`) and opened through
`project/dialogs.ts` from the Library, the project menu and the context bar:
New project (`screens/library/newProject/`: five steps over the draft
manifest of `useNewProject`, with the rules of each step in
`newProjectModel.ts`), Import export (`ImportDialog.tsx`) and Set device
(`shell/SetDeviceDialog.tsx`). The breadcrumb's device type and device are
`SiblingPopover`s over the same scan. New project writes nothing until its
last step: the layout preview asks `project_views` for the draft spec in the
data folder itself (a folder that is not a project yet; the back end judges
it as a project in a scratch folder), and what the sniffer left open
(`_unresolved`) is asked as one question per field, its control taken from the
dataset form's schema. The camera × view matrix is `components/ViewMatrix/`
(`SequenceMatrix` with thumbnails from the `image` resource; at most six
loads at a time, `lib/thumbQueue.ts`), shared with the Dataset screen.

**The Dataset screen** (`screens/dataset/`) edits the open project's
manifest as an explicit draft. The draft is `project/draftStore.ts`
(`manifest`, its saved baseline, `manifestDirty`); a change of the project
file under unsaved edits keeps them and sets `disk` (`rebase` puts the edits
back on the disk's version, a field changed in both keeps mine and is marked
changed; `keepEditing` overwrites; `revert` drops them). `useProjectViews`
asks `project_views` for the draft spec 400 ms after the last edit; the
library's messages are placed on the form's fields (`datasetModel.ts`:
`placeMessages`, which reads the field out of a message that names none; the
app's own warning for a pattern that matched nothing; counts), the rail's
badge reads the error count (`project/problems.ts`) and `⌘S` is refused while
there are errors. `DatasetForm` is `SchemaValueForm` with `baseline`,
`messages` and the groups of `datasetUi`; the cameras are `CamerasEditor`
(`renderField`, since the schema's control for a list of structures is a JSON
box), each pattern an `ImagePatternField` with what it matches. The matrix is
`components/ViewMatrix` in `MatrixPane` (filters), the selection the app's
(`session/selection.ts`), the selected image `ViewDetail` on stage2d. The
per-view robot pose and laser frame the spec shows as matrix columns are not
in `project_views` (the library pairs only the cameras' images), so the screen
lists what the manifest declares instead.

**The Detect screen** (`screens/detect/`) reads the detection cache and
writes the project's `[exclude]`. `useDetectData.ts` holds its reads:
`detection_overview` and `detection_detail` through the query cache (a saved
project marks them stale, so a changed detector or an exclusion refetches
them), `detect_preview` as a screen-owned, never cached answer, and the
`detect_all` job (`describeJob` reads its `{ done, total }` report; a
finished job invalidates the project's queries). An image with no cached
detection is detected only when the user asks (`detection_detail` would
detect and cache it). The pure rules are `detectModel.ts` (a cell's status
and tone, the sentences for an outcome, a run's `left_out` and a
`PointRejection`, the filters), `overlayModel.ts` (a `DetectionDetail` into
`@vitavision/overlays`' `TargetDetection` per target kind, the table rows, the
rejected items, the diff of a preview) and `exclusions.ts` (toggle a view, an
image or features by their `world_xyz` within the library's micrometre; counts;
notes). Overlay ids: a feature is its index in `detail.features`, a marker
`m<n>`, an edge bit `e<n>`, a rejected candidate `r<n>`, a rejected marker
`x<n>`. The stage is `StagePane` (`DetectionLayers` draws the accepted layer
`acc-*`, the dimmed rejected layer `rej-*`, the excluded crosses and the ring
covariance ellipses); `PickSurface` is the one press target (`useStageHitTest`:
a press on an accepted feature picks it, `⇧`-drag boxes, anything else pans).
Picking is `usePicking.ts` (state keyed by image and shown detection, so
another image starts clean without an effect). Every exclusion is one
`save_project` through `useExclusions.ts` (`changeExclusions`, `undoExclusion`
with a history of the `[exclude]` before each write; on `changed_on_disk` it
reloads and retries once); the keys are `useDetectKeys.ts` (`X`, `⇧X`, `⌘Z`;
views and cameras are the Dataset screen's `useViewKeys`). The Detector tab
cuts the manifest schema (`dataset_form`) down to its `detector` section
(`detectorModel.ts`) and applies by writing the saved manifest with the new
`detector`; it refuses while the Dataset draft has edits and for a manifest
that is a file of its own. The matrix is `components/ViewMatrix` with
`renderCellContent` (`CountCell`). Tests: `detect/*.test.ts(x)` (the pure
rules, each tab on every target kind, the screen over a mock back end) and
`e2e/server/detect.spec.ts` on the committed fixtures.

Recent projects are the back end's list (`recent_projects`,
`forget_recent`; the desktop's, or a server's shared by its clients), put
first by `open_project` and `create_project`; a folder that no longer opens
drops off it. Rust tests for the commands are in studio-core
(`api/commands/project_tests.rs`, on a synthetic project whose detections
the cache holds); the store against a mock transport in
`project/projectStore.test.tsx`.

**Session, roots and the path picker.** `session/session.ts` starts once:
it takes the start-up fragment (`#token=…&project=…`, captured by
`shell/captureStart.ts` before the hash router reads the address), logs in
with its token, asks `server_info`, and opens the fragment's project. The
shell then shows the sign-in panel when the server answers `unauthorized`
(instead of everything), the connection strip when no back end answers, and
on the desktop's first run — editable roots and none but the samples' folder
— the Library's one prompt, "Add a folder that holds your projects".
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

## Maps, residuals and detections

What the Inspect, Detect and Compare screens read. Every displacement is
`actual − reference` in pixels; a residual is **`observed − projected`**
(the library's sign; the Frames overlay draws its arrows that way too).

- **Maps of a run's camera** (`api/commands/maps.rs`, `maps.rs`):
  `distortion_map`, `residual_map` and `projection_sigma_map` take
  `{ run, camera, columns? }` and share one grid (`MapGrid`: the camera's
  image size, its region of interest for a tiled camera, cut into
  `stride × stride` blocks; `columns` defaults to 64, at most 1024). Each
  answers JSON with `MapPlane`s: a `BlobRef` to a `VAM1` plane and its
  channels, each with a name, a unit (`px`, `chi`, `samples`, `views`) and
  the `[min, max]` of its finite values (`null` when none). The distortion
  map has three planes (`distortion`, `sensor`, `total`) of `dx`, `dy`,
  `magnitude`; the residual map one plane of `mean_dx`, `mean_dy`,
  `mean_magnitude`, `rms`, `std_error`, `significance`,
  `effective_samples`, `effective_views`, plus the bandwidth, how it was
  chosen, the options as used (`options.min_effective_samples` is the
  support boundary; `options.views` maps a subset of views), `chain_level`
  (rig and hand-eye runs: the residuals include the pose chain's error),
  the counts and the outliers' record indices; the projection σ map one
  plane `sigma`, or `{ status: "unavailable", reason }`. `NaN` in a plane is
  "no value", never a colour.
- **Memo and expiry.** A reply is kept in the blob store under a content
  address of the command, its resolved options, the image size and the
  export's SHA-256, so asking again answers at once with the same blob ids.
  The store holds 256 MiB; a plane it dropped answers `blob_expired`, and
  the command run again makes the same ids. A different run, camera,
  `columns` or option is a different address.
- **Parameters and residuals.** `camera_parameter_values { run }` lists
  every parameter of every camera, whatever the model (label in the
  export's uncertainty, name, group, value, unit, σ or why there is none:
  `no_covariance` with the library's reason, `fixed`, `not_listed`,
  `unsupported_model`; `at_bound`). The bounds' values are not in the
  export. `residual_records { run, camera, view? }` answers the records as
  columns (`index` into `per_feature_residuals.target`, `view`, `feature`,
  `target_xyz_m`, `observed_px`, `projected_px`, `residual_px`);
  `residual_stats { run }` the library's mean, RMS, max and count per
  camera, per view and per camera and view, with the diverged records.
- **Detections** (`api/commands/detection.rs`, `detections.rs`), each over
  a project `dir`; an image is named as `project_views` lists it (relative
  to the data folder). `detection_overview { dir, run? }` reads the
  project's detection cache for every view and camera (features, the
  detector's outcome, or `not_detected`; the expected count of a complete
  board; why the latest calibrated run, or `run`, left an image out). It
  decodes nothing, but reads every image to find its cache entry.
  `detection_detail { dir, camera, image }` is the library's
  `DetectionDetail` from the cache, detected and cached when missing (in
  the camera's region-of-interest pixels; exclude a feature by its
  `world_xyz`). `detect_preview { dir, camera, image, detector }` detects
  one image with edited detector settings and writes nothing. The
  `detect_all { dir, redetect? }` job fills the cache (several images at a
  time), reports `{ done, total, camera, image }` and ends with the
  overview; it then announces `project_changed`, so cached overviews go
  stale.
- **Exclusions** need no command: edit `spec.exclude` and `save_project`.
  A stale entry is the library's `exclusion` error with its entry as
  `field` (`project_views` lists it in `messages`).

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
job (`api.jobs.calibrate`, driven by `project/runEditor.ts`):
the library's `Project::calibrate` on the project as it is on disk, with the
HTML report and the editor's one-off `config_patch`. Its events are the
library's `Progress` reports (each image detected, each calibration step,
each solver iteration; iterations are throttled to one per 50 ms); the
checklist (`screens/calibrate/stages.ts`) takes its stage, `detect` ->
`solve` -> `export`, from each report's kind. Its output is the written
run's `RunInfo`, which the run list shows and selects. A failure is the job's
`StudioError`, shown with the stage it stopped in and what to try
(`project/configCheck.ts`).
Cancelling the job (`cancel_job`, the job's `CancelToken`) stops the
library at its next image or solver iteration, nothing is written, and the
job ends `cancelled`.

## Forms

Calibrate's config form (`SchemaValueForm` from `@vitavision/forms`)
renders the schema the back end serves, `config_form { topology }` (the
workflow's `*Config` schema and the library's default), fetched by
`project/runEditor.ts`; `dataset_form` serves the manifest's. The values are the
open project's: its manifest and its configuration as `open_project`
resolves it (the defaults with the project's changes). A saved
configuration goes back as its changes from the defaults, `config_patch {
topology, config }` (the library's `ConfigPatch::from_config`), the inverse
of `resolve_config { topology, patch }` (tables merge key by key, anything
else replaces; a field the workflow lacks is a `config` error). The
frontend holds no copy of a Rust default and cannot lag the back end. A
studio-core test compares the served schemas and defaults with the ones
`cargo xtask emit-schemas` publishes in the repository's `schemas/`, which
the curation test (`screens/calibrate/curate.test.ts`) renders for all seven.

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
  call recorded; `nativeFolder` gives it the desktop's dialog, `emit`
  sends it an event, `setConnected` and `setReachable` drop the stream or the
  back end. The
  transports have contract tests of their own (`src/api/transport/*.test.ts`,
  Tauri's against `mockIPC`), and `src/api/seam.test.ts` fails when a module
  outside `src/api/` imports `@tauri-apps/*`, calls `.invoke(` or
  `.resourceUrl(`, or writes an `/api/` URL (oxlint cannot express the
  calls). `src/test/setupTests.ts` stubs `ResizeObserver` and
  `scrollIntoView`.
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
  earlier-format project's run failing as the library says), and the
  shell's (`shell.spec.ts`, see "The shell"). Install the
  browser once with
  `bunx playwright install chromium --with-deps`.
- **Smoke (`bun run test:e2e:smoke`)** — `e2e/smoke.spec.ts` on the plain
  Vite dev server with its `/api` proxy pointing nowhere (no Rust
  toolchain): every screen's address mounts (the not-built and the
  need-a-project ones say so), says no back end answers, and logs no error of
  its own.
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
  every screen's address, empty, and Inspect with the fixture run; the
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
