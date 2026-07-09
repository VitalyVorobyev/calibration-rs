# Desktop app walkthrough

> Practical guide for running calibrations and inspecting results through
> the Tauri desktop app — aimed at an engineer *using* the app, not at
> app development. Design records: [ADR 0014](../adrs/0014-tauri-desktop-app.md)
> (diagnose viewer), [ADR 0018](../adrs/0018-schema-driven-ui.md)
> (schema-driven config forms), [ADR 0019](../adrs/0019-fail-fast-on-ambiguity.md)
> (ask-user-on-ambiguity).

## Why

Everything the app does is also reachable from Rust (the tutorials in
this directory are the canonical reference for that). The app exists so
you don't have to write Rust to run a calibration or look at its
residuals: point it at a dataset folder, click Run, and get an
interactive view of the reprojection errors, laser residuals, and 3D rig
geometry.

## Mental model

- **Five workspaces**, reachable from the left rail (hotkeys `⌘1`–`⌘5`):
  Diagnose, 3D viewer, Epipolar, Depth, Run calibration.
- **One shared store** holds the currently loaded `*Export` JSON plus
  derived pose/camera indices. Every workspace except Run reads from it;
  they all react to the same loaded export.
- **Two ways to populate the store**: the top bar's **Open Export…**
  button loads a `*Export.json` file directly; running a calibration from
  the **Run** workspace hands off to **Diagnose** automatically on
  success.
- **The export needs an `image_manifest` field to render anything
  visual.** v0 requires it — an export JSON with residual numbers but no
  manifest produces an error banner ("This export has no image_manifest
  field") instead of a blank viewer. Runs launched from the Run workspace
  populate it for you; a hand-written or older export may not have one.

## Launching

```bash
cd app
bun install            # first-time setup
bun run tauri dev       # launches the desktop app
```

**Always `bun`, never `npm`/`pnpm`/`yarn`.** And always `bun run tauri dev`
— plain `bun run dev` only starts the Vite dev server; the file dialog and
every Tauri IPC command (running a calibration, computing dense stereo,
sniffing a folder) are absent, and the app tells you so as soon as you try
to open a file.

## Workspace tour

### Diagnose (`⌘1`, default)

The residual viewer. Step through `(pose, camera)` frames with the pose /
camera stepper; the source image renders with detected corners color-coded
by reprojection error (a heatmap, brightest = worst). A histogram panel
summarizes the pixel-value or error distribution under the cursor's
region of interest. Toggle **compare mode** to view two frames
side-by-side with a linked viewport (pan/zoom stays in sync); toggle the
**laser frame** view to switch a target frame for its paired laser frame
and see point-to-plane residuals instead of corner reprojection error.
Works for any of the 8 export kinds (single-camera or rig, with or
without a laser stage) — the header shows which one is loaded.

### 3D viewer (`⌘2`)

A Three.js scene of the calibrated rig: one frustum per camera, the
target board at the selected pose (or all poses at once), and laser
planes when the export carries them. Click a frustum to select a camera
and see its intrinsics plus its pose relative to a chosen reference
camera; click a board to select a pose. Requires a rig export (needs
`cameras[]` + `cam_se3_rig[]`) — single-camera exports show an empty-state
message instead.

### Epipolar (`⌘3`)

Click a point in one camera's image and the corresponding epipolar line
is drawn in a second camera's image, computed from the calibrated
relative pose — a quick visual sanity check that two cameras in a rig
are calibrated consistently with each other. Requires a rig export with
at least two cameras.

### Depth / dense stereo (`⌘4`)

Pick two cameras and a pose from a loaded rig export and click compute:
the app rectifies the pair, runs block or semi-global dense matching
(`vision_mvg::dense`), and gives you rectified / disparity / overlay /
depth / 3D point-cloud views. Toggle semi-global aggregation to fill in
low-texture regions the plain block matcher leaves blank. Requires a
stereo rig export (two cameras with extrinsics) — same gate as Epipolar.

### Run calibration (`⌘5`)

Drives a calibration end-to-end without leaving the app:

1. **Quick-start presets** — a card grid of committed datasets
   (`data/stereo`, `data/stereo_charuco`, `data/kuka_1`, plus a few
   `privatedata/rtv3d` cards for local development) spanning most
   problem types. Clicking a card loads its manifest and applies any
   `configOverrides` the preset carries (e.g. the rtv3d presets need
   Scheimpflug sensors and EyeToHand — see
   [`presets.ts`](../../app/src/workspaces/RunWorkspace/presets.ts)).
2. **Sniff folder** — pick an arbitrary dataset folder and the app
   heuristically infers a manifest (`sniff_folder`). Fields it can't
   determine are left `_unresolved` and block Run until you fill them in
   the manifest form (a red badge marks each one).
3. **Manifest** and **calibration config** sections — collapsible,
   schema-driven forms (ADR 0018) generated from the same JSON Schema the
   Rust config types derive, so every field the pipeline accepts is
   editable without hand-writing JSON. An **advanced JSON editor** is
   available for anything the form doesn't expose yet.
4. **Run** — on success, the app flashes a status banner and navigates to
   Diagnose after a short delay. On an unresolvable ambiguity (e.g. a
   hand-eye dataset with no `pose_pairing` set — see the
   [single-camera hand-eye tutorial](./single-cam-handeye.md)) a modal
   asks you to pick between the suggested options (ADR 0019) instead of
   guessing. Validation failures and IPC errors surface inline.

## Walkthrough: load an export into Diagnose

1. Click **Open Export…** in the top bar.
2. Pick any `*Export.json` produced by a calibration session (from a Run,
   or written by `session.export()` in your own Rust code with
   `image_manifest` populated).
3. Diagnose infers the export kind from its JSON shape (planar, rig
   hand-eye, rig + laserline, …) and shows the right header label; step
   through frames with the arrow keys or the pose/camera stepper.

## Walkthrough: run a calibration from Run

1. Pick a preset card, or click **Sniff folder** and point at your own
   dataset directory.
2. Review the manifest and config sections — the schema-driven forms
   surface every field with its default and description.
3. Click **Run**. Resolve any `AskUserModal` prompt if one appears.
4. On success you land in Diagnose with the fresh export already loaded.

## Common variations

- **Preset paths are hard-coded to one dev checkout.** `presets.ts`
  documents this: `REPO_ROOT` is an absolute path constant, not yet
  resolved through Tauri's bundle-asset API. If your checkout lives
  elsewhere, use **Sniff folder** or the manifest file picker instead of
  a preset card.
- **Config overrides**: a preset can deep-merge a config patch over the
  topology's schema defaults (`configOverrides`) and a manifest patch
  over the loaded TOML (`manifestOverrides`) — useful for datasets whose
  schema defaults don't match the physical rig (Scheimpflug sensors,
  EyeToHand mode, laser residual type). Inspect a preset's overrides in
  [`presets.ts`](../../app/src/workspaces/RunWorkspace/presets.ts) as a
  worked example of a non-default config.
- **Re-running with different settings**: edit the config form (or the
  advanced JSON editor) after loading a preset and click Run again — the
  preset selection clears the moment you touch the form, so nothing
  silently reverts your edits.
- **Laser datasets**: see [Laser topologies from a dataset
  manifest](./laser-dataset-manifest.md) for the two-stage
  (`rig_handeye` → `rig_laserline_device`) manifest shape the rtv3d
  presets exercise.

## What to read next

- [ADR 0014](../adrs/0014-tauri-desktop-app.md) — why a Tauri desktop app
  at all, and the v0 diagnose-viewer scope.
- [ADR 0018](../adrs/0018-schema-driven-ui.md) — how the Run workspace's
  config forms stay in sync with the Rust config types.
- [ADR 0019](../adrs/0019-fail-fast-on-ambiguity.md) — the ask-user
  contract behind the Run workspace's modal.
- [Laser dataset manifest](./laser-dataset-manifest.md) — the
  `dataset.toml` shape the Run workspace's manifest form edits.
- [Puzzle 130×130 walkthrough](./puzzle-130x130-walkthrough.md) — the
  full-scale calibration one of these workspaces would diagnose.
