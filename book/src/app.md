# The Desktop App

Everything the library does is reachable from Rust and Python. The desktop app exists so that you do not have to write code to run a calibration or look at its result: point it at a dataset folder, click **Run**, and get an interactive view of the reprojection errors, the laser residuals, the parameter uncertainty and the 3D rig geometry. It calls the same solvers as the `vision-calibration` crate.

## Launching

The app is built from a local checkout with [bun](https://bun.sh). It uses bun exclusively (never `npm`, `pnpm` or `yarn`); the Tauri CLI is a development dependency, so no global install is needed.

```bash
cd app
bun install            # first-time setup, and after package.json changes
bun run tauri dev      # launches the desktop app
bun run tauri build    # bundles the app (installer or binary)
```

**Always use `bun run tauri dev`.** Plain `bun run dev` starts only the Vite development server in a browser: the file dialog and every command that reaches the Rust backend (running a calibration, computing dense stereo, sniffing a folder) are absent, and the app shows a banner saying so as soon as you try to open a file. Signed installers are not provided.

## Mental Model

- **Five workspaces**, reachable from the left rail (hotkeys `⌘1` to `⌘5`): Diagnose, 3D viewer, Epipolar, Depth and Run calibration.
- **One shared store** holds the currently loaded `*Export` JSON plus derived pose and camera indices. Every workspace except Run reads from it; they all react to the same loaded export, the selected pose and the selected camera.
- **Two ways to fill the store.** The top bar's **Open Export…** button loads a `*Export.json` file directly. Running a calibration from the **Run** workspace hands off to **Diagnose** automatically on success.
- **The export needs an `image_manifest` field to render anything visual.** An export with residual numbers but no manifest produces an error banner ("This export has no image_manifest field") instead of a blank viewer. Runs launched from the Run workspace fill it in; a hand-written export may not have one.

## What Is Where

| Workspace | Hotkey | Use it to |
|---|---|---|
| **Run** | `⌘5` | configure and launch a calibration from a dataset folder ([Running Calibrations](app_run.md)) |
| **Diagnose** | `⌘1` | look at per-corner residuals on the source images, per-pose statistics and the parameter uncertainty ([Inspecting Results](app_inspect.md)) |
| **3D viewer** | `⌘2` | see the calibrated rig: camera frustums, target boards, laser planes |
| **Epipolar** | `⌘3` | click a point in one image, see its epipolar line in another |
| **Depth** | `⌘4` | rectify a stereo pair and compute disparity, depth and a point cloud |

The Run workspace is the entry point for a new dataset; the other four work on any export, whether produced by the app or by `session.export()` in your own code.

## Relation to the Library

The app's forms are generated from the same JSON Schemas as the Rust configuration and manifest types (`cargo xtask emit-schemas`), so every field the pipeline accepts is editable without hand-writing JSON. The manifest it edits is the [dataset manifest](manifest.md); the results it displays are the [exports](per_feature_residuals.md) and [uncertainty](uncertainty.md) described in this book.

To work on the app itself, see `app/DEVELOPING.md` in the repository.
