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
  pose/camera steppers and error histograms. A **Stats** side panel adds a
  sortable per-pose residual table (mean/median/max px, click-to-jump) and,
  for multi-camera exports, a cameras × poses residual matrix — both computed
  by the pure `src/lib/residualStats.ts`.
- **3D** (`Viewer3DWorkspace`) — a 3D rig scene: camera frustums, target board
  poses, and laser planes for rig/hand-eye/laserline exports, with a side
  panel breaking down intrinsics and relative camera poses. A single-camera
  `laserline_device` export renders too — it is lifted into a one-camera rig
  at the origin so its camera-frame laser plane + `camera_se3_target` poses
  show up (`src/lib/sceneExport.ts`).
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
bun run test           # Vitest: pure-logic unit tests + jsdom component tests
bun run test:e2e       # Playwright smoke: boots the app, checks it doesn't fall over
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

## Design system

B-UX1-DESIGN-SYSTEM consolidated the hand-rolled chrome that had
accumulated across the five workspaces (buttons, panels, section headers,
toolbar selects, tables, status banners, empty states — each workspace had
grown its own copy) into one small component set, and closed the token
gaps that copy-pasting had been quietly working around.

**Component set** — `app/src/components/ui/` (barrel: `components/ui/index.ts`):

- `Button` — every plain, toggle, and primary-CTA button in the app.
  `variant` is `default` (bordered, neutral — the global `button` rule in
  `index.css`) or `primary` (brand-filled CTA: Run, Compute disparity,
  Apply). `size` is `sm` (h-7, toolbar default) · `md` (h-9, page-level
  CTA) · `icon` (h-7 square). `pressed` drives the on/off toolbar-toggle
  look (Compare, Linked, SGM, …) and wires `aria-pressed` — before this
  pass these toggles had no ARIA state at all, just a color change.
- `Panel` — the bordered card surface for side rails (Diagnose's stats
  panel, the 3D viewer's info rail).
- `SectionHeader` — small uppercase subsection heading with an optional
  right-aligned subtitle or action (replaces two byte-identical
  `PanelHeading` copies).
- `Select` — compact toolbar "label + native `<select>`" control (camera/
  pose pickers); replaces three near-identical hand-rolled copies.
- `Table` / `Th` / `Td` — shared table shell (border-collapse, font-mono,
  tabular-nums, alignment, sticky-column support) for the residual stats
  table and the cross-camera matrix. Sort/hover/severity logic stays with
  each table — this is chrome, not a generic sortable-DataTable engine.
- `Banner` — left-accent status banner (`error` / `success` / `warning` /
  `neutral`). Also closes a divergence: RunWorkspace had grown a second,
  separate full-border banner shape with its own inline-styled colors;
  everything now uses the one left-accent shape the rest of the app
  already agreed on.
- `Badge` — small pill label (unresolved-field counts, preset topology/
  milestone tags).
- `EmptyState` — whole-workspace "no data loaded" placeholder (title +
  centered dashed box). Diagnose's smaller inline canvas-area placeholder
  is a different idiom (a note inside an already-visible canvas, not a
  whole-workspace state) and intentionally isn't folded into this.

`components/ZoomControls.tsx` (not under `ui/` — it's specific to the
`FrameCanvas` zoom/fit/1:1 interaction, not a generic primitive)
consolidates Diagnose's and Epipolar's identical zoom-button clusters.

**Tokens** (`app/src/index.css`, Tailwind 4 CSS-first `@theme`) — HSL
triplets in `:root` / `.dark`, exposed as `--color-*` custom properties so
both `hsl(var(--name))` and Tailwind's `bg-brand`/`text-destructive`/…
utilities work off the same source. This pass:

- Added `--success` / `--warning` (light + dark) — RunWorkspace had been
  reaching for `var(--color-success, #22c55e)` / `var(--color-warning,
  #f59e0b)` inline-style fallbacks because no such tokens existed; those
  hex fallbacks were also identical in both themes, so `#22c55e` on the
  light theme's near-white background measured well under WCAG AA.
- Fixed `--brand`'s light-mode lightness (55% → 34% L, same hue/chroma).
  `text-brand` sits directly on `--background`/`--surface` throughout the
  app (active nav item, active table row, active preset, …); at 55% L it
  measured ~2.1:1 contrast on white — dark mode's 55% L was left alone
  (already ~8.6:1 against the dark background).
- Fixed several `style={{ color: "var(--brand)" }}` / `backgroundColor:
  "var(--brand)"` call sites — invalid CSS, since `--brand` is a bare `H
  S% L%` triplet, not a `hsl(...)` color; the declarations were silently
  dropped, leaving the "Run" button, an ActivePresetBar checkmark, and the
  AskUserModal's Apply button and icon without their intended brand fill.
  Convert to `hsl(var(--brand))` (SVG/canvas contexts) or the `bg-brand`/
  `text-brand` Tailwind utilities (DOM), never a bare `var(--brand)`.
- Fixed `bg-bg`, used in ~15 places (CollapsibleSection, PresetCard,
  RunWorkspace, and every bare `<input>`/`<select>`/`<textarea>` in
  `lib/configForm.tsx`): no `--color-bg` token was ever declared, so it
  silently rendered no background at all. Config-form fields in
  particular sit inside a `bg-bg-soft` fieldset, so this wasn't always
  visible — but it meant every field lost its distinct surface. Fixed to
  `bg-surface` (elevated content) or removed outright on `<button>`/
  `<select>` elements, which already fall back to the global base rule's
  `background: var(--color-surface)`.

**Rules for adding UI** — use the `ui/` set; don't hand-roll a button,
panel, banner, badge, or table shell in a workspace file. If a genuinely
new shape recurs three-plus times, add it to `ui/` following the existing
pattern (typed props, Tailwind classes built on the tokens above, no new
runtime dependency). Don't add a variant "just in case" — this set is
deliberately small (YAGNI): it covers exactly what the five workspaces use
today, not a speculative general-purpose design system.

### Run progress & cancellation

Long solves stream **stage progress** and are **cancellable at stage
boundaries** (B-UX2). The wiring, end to end:

- The frontend mints a per-run `runId` (`crypto.randomUUID`) and a
  `tauri::ipc::Channel<RunProgress>`, and passes both to
  `run_calibration_cmd` (`src/lib/runCalibration.ts`). A request-scoped
  `Channel` is the right fit over a global `emit`/`listen` bus: progress is
  inherently one-per-invocation, so there's no cross-run fan-out to filter
  and no listener to tear down.
- `run.rs` announces exactly three coarse stages over the channel —
  `detect` → `solve` → `export` (`RunStage`). These are the only boundaries
  the runner honestly owns: `dataset_runner::build_*_input` fuses image
  decode + feature detection (and loops cameras internally with no
  callback), and each per-topology `run_calibration` wrapper fuses linear
  init + bundle adjustment. **Per-camera and per-LM-iteration granularity
  are deliberately not emitted** — the pipeline API exposes no hook, and
  faking one would misrepresent progress. `RunProgress` is schema-generated
  (`emit_schemas` → `src/types/generated/diagnose-wire.ts`).
- Cancellation is a cooperative `AtomicBool` per `runId`, held in a
  Tauri-managed `RunRegistry` (`run.rs`). `cancel_run_cmd(runId)` flips the
  flag; the runner checks it at each stage boundary and returns
  `RunResponse::Cancelled`. The **currently executing stage runs to
  completion** — there is no solver/detector interrupt — so cancel takes
  effect at the next boundary. The Run workspace surfaces this as a distinct
  "Run cancelled" state, never an error banner. The stage-checklist logic
  lives in the pure, tested `src/workspaces/RunWorkspace/runStages.ts`.

### Repo-root resolution for built-in presets

`RunWorkspace`'s Quick Start presets (`src/workspaces/RunWorkspace/presets.ts`)
point at datasets committed to the repo (`data/…`) or private local-only ones
(`privatedata/…`); `manifestPath` on every preset is **repo-root-relative**,
never a hard-coded personal absolute path. At load time
(`RunWorkspace`'s `handleUsePreset`) the frontend resolves the repo root via
`lib/tauri.ts`'s `repoRoot()`, which calls the `repo_root_cmd` Tauri command
(`src-tauri/src/commands.rs`) and joins it onto the preset's relative path
with `joinPath`.

`repo_root_cmd` reports `env!("CARGO_MANIFEST_DIR")` two levels up
(`app/src-tauri` → `app` → repo root) — a compile-time constant, but since
`app/src-tauri` is never shipped as a prebuilt binary (every `bun run tauri
dev`/`build` recompiles it locally), it always reflects *the developer
running the command's own checkout path*, not a value baked in by whoever
last edited the source. That's the simplest fix that removes the personal
path from source control while keeping presets working out of the box on
any machine: no extra dev-root configuration, no bundled-resource plumbing
(which wouldn't apply to `privatedata/` presets anyway — private datasets
are never meant to ship inside an installer). True asset bundling via
Tauri's `resource_dir` API remains a distinct, separate concern for
`B-DIST` (shipping *public* bundled datasets inside a release installer),
not the dev-preset path.

### Testing (`bun run test` / `bun run test:e2e`)

Two layers, run by different tools:

- **Vitest (`bun run test`)** — `src/**/*.test.ts` are pure-logic unit tests
  (Node environment, no DOM); `src/**/*.test.tsx` are component tests
  (jsdom, via `@testing-library/react`) selected by
  `vitest.config.ts`'s `environmentMatchGlobs`. Component tests mock the
  Tauri IPC layer with `@tauri-apps/api/mocks`' `mockIPC` (one seam:
  `window.__TAURI_INTERNALS__.invoke`) — see
  `src/workspaces/DiagnoseWorkspace/DiagnoseWorkspace.test.tsx` and
  `src/workspaces/RunWorkspace/RunWorkspace.test.tsx` for the pattern.
  `src/test/setupTests.ts` stubs `ResizeObserver` and
  `HTMLCanvasElement.getContext` (jsdom has neither) so `FrameCanvas` can
  mount without pulling in `node-canvas`.
- **Playwright (`bun run test:e2e`)** — smoke tests in `app/e2e/` that boot
  the real app in a real Chromium tab via the plain Vite dev server
  (`bun run dev`, **not** `bun run tauri dev` — no Tauri/Rust toolchain
  needed, keeping the CI job toolchain-pure). Two flavours:
  - `app/e2e/app.spec.ts` — the app boots and each of the five workspaces
    mounts via left-rail navigation with zero console errors, running
    *without* any Tauri mock (exactly like a developer opening
    localhost:1420 in a plain browser tab — `isTauriContext()` is false
    and every workspace's empty state must render on its own).
  - `app/e2e/diagnose.spec.ts` — exercises the mocked-IPC path: Playwright
    can't reach into the page's own module graph the way Vitest can, so
    `@tauri-apps/api/mocks` doesn't apply; `e2e/support/tauriMock.ts`
    instead injects a `window.__TAURI_INTERNALS__` shim via
    `page.addInitScript` (the same seam `@tauri-apps/api/core`'s
    `invoke()` and `@tauri-apps/plugin-dialog`'s `open()` call into),
    before any of the app's own scripts run. That's the
    Tauri-native-vs-mocked-IPC boundary: real desktop behavior is only
    ever exercised manually via `bun run tauri dev`; both automated test
    layers stop at the IPC seam.

`bunx playwright install chromium --with-deps` installs the browser once
(cached locally / in CI); CI runs `test:e2e` as an extra step in the
`app-frontend` job.

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
