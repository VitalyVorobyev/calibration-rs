# The Desktop App

Everything the library does is reachable from Rust and Python. The desktop app, Calibration Studio, exists so that you do not have to write code to run a calibration or look at its result: point it at a dataset folder, click **Run**, and get an interactive view of the reprojection errors, the laser residuals, the parameter uncertainty and the 3D rig geometry. It calls the same solvers as the `vision-calibration` crate.

## Launching

The app is built from a local checkout with [bun](https://bun.sh). It uses bun exclusively (never `npm`, `pnpm` or `yarn`); the Tauri CLI is a development dependency, so no global install is needed.

```bash
cd apps/studio
bun install            # first-time setup, and after package.json changes
bun run tauri dev      # launches the desktop app
bun run tauri build    # bundles the app (installer or binary)
```

**Use `bun run tauri dev` for the desktop app.** Plain `bun run dev` starts only the Vite development server, which talks to a running Calibration Studio server (below); without one the app says that no back end answers. Signed installers are not provided.

On its first start the desktop app reads no folder at all: it asks you to **Add folder…**, and every dataset and project you open must lie inside a folder you added (**Settings** lists them, and removes one).

## In a Browser

The same app, every workspace included, runs in a browser, served by one program with the app inside it: `calibration-studio-server`. It suits a calibration station or a small device that several people reach over the network.

```bash
cd apps/studio
bun run serve -- --root ~/datasets           # from a checkout: builds the page, starts the server
calibration-studio-server --root /data/calib --bind 0.0.0.0:8757   # an installed binary
```

It serves only the folders given with `--root` (or a project folder given alone, with its dataset). On the machine itself (the default address `127.0.0.1:8757`) it needs nothing more. On any other address it always needs its token: it prints an address with the token in it once it starts; open that address, or paste the token when the page asks for it (`calibration-studio-server token` prints it again, `--rotate` replaces it). The page then keeps a session cookie, and the token never stays in the address bar. There is no upload: datasets reach the device by copying them into a root folder. For HTTPS, put it behind a reverse proxy and pass `--public-origin` and `--allowed-host`; `--help` lists every option.

## Mental Model

- **Five workspaces**, reachable from the left rail (hotkeys `⌘1` to `⌘5`): Diagnose, 3D viewer, Epipolar, Depth and Run calibration.
- **One run at a time.** The viewers show one run of the open [project](projects.md): its export plus derived pose and camera indices. Every workspace except Run reads it; they all react to the same run, the selected pose and the selected camera.
- **Three ways to a run.** Choose one of the project's runs in the panel on the left. Running a calibration from the **Run** workspace writes the project's next run and hands off to **Diagnose** with it. **Import Export…** in the top bar adds an export calibrated elsewhere (a `*Export.json`, or a `vcal run -o` folder) to the open project as its next run.
- **The export needs `diagnostics.image_manifest` to render anything visual.** An export with residual numbers but no manifest produces an error banner ("This export has no image_manifest field") instead of a blank viewer. Runs launched from the Calibrate screen fill it in, as do `vcal run` and the library's `calibrate_dataset`; a hand-written export may not have one.

## What Is Where

| Workspace | Hotkey | Use it to |
|---|---|---|
| **Run** | `⌘5` | set up and run a calibration of the open project ([Running Calibrations](app_run.md)) |
| **Diagnose** | `⌘1` | look at per-corner residuals on the source images, per-pose statistics and the parameter uncertainty ([Inspecting Results](app_inspect.md)) |
| **3D viewer** | `⌘2` | see the calibrated rig: camera frustums, target boards, laser planes |
| **Epipolar** | `⌘3` | click a point in one image, see its epipolar line in another |
| **Depth** | `⌘4` | rectify a stereo pair and compute disparity, depth and a point cloud |

The Calibrate screen is where a run is set up and started; the other screens work on any run of a project, calibrated by the app, by `vcal run` or imported.

## Projects

The app works on [projects](projects.md), the same folders `vcal` and the Python package read and write: a dataset (its manifest in the project file, or a manifest file of its own), the changes to its workflow's configuration, and the runs calibrated from it in `runs/run-NNN/`. A project of the app's earlier format (`calibration-project.json` with a `dataset.json`) opens as it is and is written in the current format when it is first saved; its runs are listed as they are.

The **Project** menu in the top bar:

- **Open project…** opens a folder browser over the folders the app may read (the same one on the desktop and in a browser) and lists the project's runs in a panel on the left; choosing a run loads it into the viewers. A folder without a project file is reported as "not a project folder"; the app never guesses one from a folder's contents.
- **New project from a dataset folder…** makes the folder a project whose manifest the [folder sniffer](manifest.md#the-folder-sniffer) drafts; **New project from a manifest…** makes the manifest's folder a project referencing that file, which the app shows but never rewrites.
- **Save project** writes the project's unsaved edits to the project file. If the file changed on disk since it was opened (another program, another browser tab), nothing is written and the app asks you to reload it.
- **Import an export as a run…** is the menu's twin of the top bar's **Import Export…**; **Reload project** reads it again; **Recent** reopens one of the last projects opened (in a browser: the server's list, shared by everyone using it).

The panel on the left lists the runs with their views and verdict, a ★ on starred ones and "stale" on a run made from other inputs (dataset, exclusions, configuration or quality thresholds) than the project states now. Below it, the chosen run takes a label and a star, opens its `report.html` ([The Calibration Report](report.md)), and can be deleted.

The app reads and writes only inside its root folders: a project whose dataset, or a manifest whose images or pose file, lie elsewhere is refused with an error instead of being read silently.

## Relation to the Library

The app's forms are generated from the JSON Schemas of the Rust configuration and manifest types, which the app's back end serves with their defaults, so every field the linked library accepts is editable without hand-writing JSON. The manifest it edits is the [dataset manifest](manifest.md); the results it displays are the [exports](per_feature_residuals.md) and [uncertainty](uncertainty.md) described in this book.

To work on the app itself, see `apps/studio/DEVELOPING.md` in the repository.
