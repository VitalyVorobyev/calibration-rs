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

- **Screens** in the left rail, in workflow order (hotkeys `⌘1` to `⌘7`): Library, Dataset, Detect, Calibrate, Inspect, Compare and Device. Inspect is a row of tabs (Scene, Frames, Parameters, Distortion, Residuals, Projection σ, Quality, Epipolar, Depth).
- **One run at a time.** The viewers show one run of the open [project](projects.md): its export plus derived pose and camera indices. Every workspace except Run reads it; they all react to the same run, the selected pose and the selected camera.
- **Three ways to a run.** Choose one of the project's runs in the panel on the left. Running a calibration from the **Run** workspace writes the project's next run and hands off to **Diagnose** with it. **Import Export…** in the top bar adds an export calibrated elsewhere (a `*Export.json`, or a `vcal run -o` folder) to the open project as its next run.
- **The export needs `diagnostics.image_manifest` to render anything visual.** An export with residual numbers but no manifest produces an error banner ("This export has no image_manifest field") instead of a blank viewer. Runs launched from the Calibrate screen fill it in, as do `vcal run` and the library's `calibrate_dataset`; a hand-written export may not have one.

## What Is Where

| Screen | Hotkey | Use it to |
|---|---|---|
| **Calibrate** | `⌘4` | set up and run a calibration of the open project ([Running Calibrations](app_run.md)) |
| **Inspect** | `⌘5` | judge a run: parameters with σ, distortion, residual and projection-σ maps, per-feature residuals, quality checks, the frames, the rig in 3D, epipolar lines and depth, each a tab ([Inspecting Results](app_inspect.md)) |

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

## The Detect Screen

The **Detect** screen shows what the target detector found in each image of the open project, lets you try other detector settings on one image, and writes what the project leaves out of its calibrations.

- **The matrix** (left) has a row per view and a column per camera. Each cell holds the number of features found in that image, with a bar scaled to the number a complete board has (a PuzzleBoard is made to be seen in part, so it has none). A warning mark means fewer than expected, an ambiguous board, or an image the last calibrated run left out (the reason is in the cell's tooltip: too few features, only part of a chessboard, a turned board); a red cell means no board was found or the image failed; `·` means nothing is cached for the image with the project's current detector settings; a struck-through number is an excluded image. **Only problems**, **Only excluded**, a count limit and a view-token search find a bad image among hundreds. **Detect all** detects every image that has no cached detection, several at a time, with progress and **Cancel** (what was detected stays); **Detect again** ignores the cache.
- **The image** (middle) is the selected image in a zoomable stage with the detection drawn over it. The features are drawn for every target: chessboard corners with their grid indices and the lattice between them (and, from the layers menu, each corner's two edge directions); ChArUco corners with the decoded marker quads and their ids; PuzzleBoard corners with their cell ids and the code bits read on the edges; ring-grid markers with their outer and inner ellipse, code or lattice cell, and, from the layers menu, the one-sigma ellipse of a photometric centre (drawn magnified; the stage prints the gain). Candidates the detector left out are drawn dimmed. The layers menu switches each layer; hovering a feature reads its position, score, centre estimator and confidence under the stage. `,` and `.` step the camera and `[` and `]` the view; the zoom stays across images of one size.
- **Features** (right) lists the image's features (hover and press are mirrored with the stage) with the detector's verdict, the board's cell size and the noise level it measured. **Rejected** groups the candidates the detector left out by reason, in words ("Corner candidate not taken by any labelled grid", "Marker decodes as 17, board expects 23"), above the detector's verdict on the whole image and what the last run did with it; pressing one shows it on the image. A PuzzleBoard whose bits mostly disagree with the master pattern, or a ring grid whose view is not anchored on the board, says so: its labels are likely wrong or only relative.
- **Detector** shows the detector settings of the project's manifest as a form, with every change from the saved settings marked. **Preview on this image** runs the detector on the open image with the edited settings and writes nothing: the stage shows the preview (switch to the saved detection with **Saved**), and a line says what differs ("+3 features, −1 feature, 2 labels changed against the saved settings"). **Apply to project** saves the settings into the manifest; the other images' detections were made with the old settings, so their cells show `·` until **Detect all with the new settings**. **Discard** drops the edit and the preview. A manifest that is a file of its own is never written: the preview works, Apply is disabled, and the Dataset screen can copy the manifest into the project file first.

**Exclusions.** Three things can be left out of every run, each with one gesture, and each is one write of the [project file](projects.md)'s `[exclude]` section:

| What | How | Written as |
|---|---|---|
| Some features of one image | click a feature (`⇧`-click adds), or `⇧`-drag a box on the stage, then **Exclude** (or `X`) | `exclude.features`: the image and the features' board points |
| One camera's image of a view | `X` with no feature picked, or **Exclude image** | `exclude.images` |
| A whole view, every camera | `⇧X`, or **Exclude view** | `exclude.views` |

The same gesture restores. An excluded feature is named by its board point, so it stays excluded when the image is detected again; a board point that matches no feature of an image's detection leaves nothing out, and the Features tab says so. Excluded things stay visible: struck-through in the matrix, crossed on the stage, an "excluded" status in the table, and a banner on an excluded image. `⌘Z` (or **Undo**) takes the last exclusion back; **Exclusions** at the foot of the stage lists them all with an editable note each, **Restore** per row and **Restore all**. A run made before an exclusion shows "stale" in the panel of runs. With the connection to the back end lost, every write is disabled. If the project file changed on disk under a write, the app tries once more on the file as it is now.

## Relation to the Library

The app's forms are generated from the JSON Schemas of the Rust configuration and manifest types, which the app's back end serves with their defaults, so every field the linked library accepts is editable without hand-writing JSON. The manifest it edits is the [dataset manifest](manifest.md); the results it displays are the [exports](per_feature_residuals.md) and [uncertainty](uncertainty.md) described in this book.

To work on the app itself, see `apps/studio/DEVELOPING.md` in the repository.
