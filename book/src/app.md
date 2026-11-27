# The Desktop App

Everything the library does is reachable from Rust and Python. The desktop app, Calibration Studio, exists so that you do not have to write code to run a calibration or look at its result: point it at a folder of images, say what the target and the cameras are, click **Run**, and get an interactive view of the reprojection errors, the parameter uncertainty, the residual and distortion maps and the 3D rig geometry. It calls the same solvers as the `vision-calibration` crate and works on the same [projects](projects.md) as `vcal` and the Python package.

## Launching

The app is built from a local checkout with [bun](https://bun.sh). It uses bun exclusively (never `npm`, `pnpm` or `yarn`); the Tauri CLI is a development dependency, so no global install is needed.

```bash
cd apps/studio
bun install            # first-time setup, and after package.json changes
bun run tauri dev      # launches the desktop app
bun run tauri build    # bundles the app (installer or binary)
```

**Use `bun run tauri dev` for the desktop app.** Plain `bun run dev` starts only the Vite development server, which talks to a running Calibration Studio server (below); without one the app says that no back end answers. Signed installers are not provided.

On its first start the desktop app reads no folder at all: the Library asks you to **Add folder…**, and every dataset and project you open must lie inside a folder you added. **Settings** (the gear in the top bar) lists those folders and removes one. The app reads and writes only inside them: a project whose dataset, or a manifest whose images or pose file, lie elsewhere is refused with an error instead of being read silently.

## In a Browser

The same app, every screen included, runs in a browser, served by one program with the app inside it: `calibration-studio-server`. It suits a calibration station or a small device that several people reach over the network.

```bash
cd apps/studio
bun run serve -- --root ~/datasets           # from a checkout: builds the page, starts the server
calibration-studio-server --root /data/calib --bind 0.0.0.0:8757   # an installed binary
```

It serves only the folders given with `--root` (or a project folder given alone, with its dataset). On the machine itself (the default address `127.0.0.1:8757`) it needs nothing more. On any other address it always needs its token: it prints an address with the token in it once it starts; open that address, or paste the token into the sign-in page when it asks (`calibration-studio-server token` prints it again, `--rotate` replaces it). The page then keeps a session cookie, and the token never stays in the address bar. There is no upload: datasets reach the device by copying them into a root folder. For HTTPS, put it behind a reverse proxy and pass `--public-origin` and `--allowed-host`; `--help` lists every option.

A change made to a project by another browser tab, another person or a program (`vcal run`, say) arrives in the open page without a reload. When the connection to the back end drops, the page says so, everything that writes is disabled, and it recovers by itself when the connection returns.

## The Window

A rail on the left lists the screens in workflow order. A bar across the top says where you are and what is open; the screen fills the rest, a tray of running jobs and a status line sit at the foot.

| Screen | Key | Use it to |
|---|---|---|
| **Library** | `⌘1` | find, open and start projects |
| **Dataset** | `⌘2` | describe how the images lay out: cameras, views, target, poses |
| **Detect** | `⌘3` | see what the detector found in each image and leave out what should not count |
| **Calibrate** | `⌘4` | set up and run a calibration ([Running Calibrations](app_run.md)) |
| **Inspect** | `⌘5` | judge a run ([Inspecting Results](app_inspect.md)) |
| **Compare** | `⌘6` | not built yet |
| **Device** | `⌘7` | not built yet |

(`Ctrl` takes the place of `⌘` off macOS.) A screen that is not built yet, or that needs a project when none is open, is dimmed in the rail and says why when you point at it; its address still opens a page that says so. The Library is the one to start from: Dataset, Detect, Calibrate and Inspect work on the open project.

**The top bar** holds, from the left, the project's device type, device and name, the run, then the job indicator, the shortcut list, the theme (light or dark) and Settings. The device type and the device are lists of their siblings; choosing one shows it in the Library. The project's name is its menu (below). The **run** list chooses the run every screen that reads a run shows, newest first, with what that run is: whether it was calibrated here (`Recorded`) or imported from an export with no record of its settings (`Imported`), its verdict in words, and `Stale` when the project's dataset, exclusions, configuration or quality thresholds are no longer those it was made from.

**The address** is part of the app: `#/<screen>`, then for Inspect its tab, then the open project and run. A link opens the same screen with the same project and run, and the browser's back button steps between screens.

**The run list** is the left panel of Calibrate and Inspect: every run of the open project as a card, with its verdict, its RMS (and per camera for a rig), the views and features it used and when it was made; a star is how you mark the run you accept, a tick collects runs for the Compare screen to come. The panel is described in [Running Calibrations](app_run.md#the-history).

**The job tray** at the foot lists what is running or has just ended: a row per job (detection, calibration) with its stage, a progress bar, **Cancel**, and for a calibration the solver's cost curve and the run it wrote. A failure stays until you dismiss it.

## Projects

The app works on [projects](projects.md): a folder with a project file (a dataset's manifest, in the file or referenced; the changes to its workflow's configuration; exclusions; the device it belongs to) and the runs calibrated from it in `runs/run-NNN/`. A folder without a project file is reported as "not a project folder"; the app never guesses one from a folder's contents.

The **project menu** (the project's name in the top bar) offers:

- **Open project…**, a folder browser over the folders the app may read (the same one on the desktop and in a browser);
- **New project…** and **New project from a manifest…**, below;
- **Save project**, which writes the project's unsaved edits (an unsaved edit is a dot beside the name). If the file changed on disk since it was opened, nothing is written and the app says so;
- **Reload project**, **Import an export as a run…**, **Set device…** and **Close project**;
- the project's device's other projects, one click to switch.

A new project is made in five steps, and nothing is written until the last: **Data** (a folder of images, whose layout the [folder sniffer](manifest.md#the-folder-sniffer) drafts; a manifest file, which the project then references and the app never rewrites), **Layout** (the camera by view matrix the draft finds, with what it could not decide asked as a question per field), **Target**, **Device** (the device type and device it belongs to, both optional) and **Create**.

Importing an export (a `*Export.json` made elsewhere, or a `vcal run -o` folder) adds it to the open project as its next run, so it can be inspected beside the project's own; such a run has no record of its configuration and cannot be rerun.

## The Library

The Library lists every project under the app's folders, grouped by device type and device, with how its latest calibration stands: its run, verdict and RMS, its views and when it was last changed. Search by project, device or label, filter by verdict or workflow, and choose a node of the tree to list its projects; the search, filters and node are in the address, so a link reopens them. **Open…** takes a folder, **Import export** and **New project** start one; **Quick start samples** at the foot lists the datasets a back end can download and open as projects. The folders are scanned again when the window regains focus and whenever a project changes.

## The Dataset Screen

The Dataset screen edits the open project's [manifest](manifest.md) as an explicit draft. The form on the left is generated from the manifest's JSON Schema: the workflow, the target, and one block per camera with its images (a pattern with its matches counted, or a list), an optional crop for a tiled strip, robot poses and, for a laser setup, its frames. The middle is the camera by view matrix of what the description finds, with thumbnails, and a search by view token; the right shows the selected image with the robot pose and laser frame the view pairs with. Every edit asks the library again, shortly after the last keystroke, what the layout is and what it would refuse to run, and the problems appear on the fields that cause them; the rail shows their count. **Save** writes the manifest into the project file (`⌘S`), **Revert** drops the draft. A manifest that is a file of its own is shown and never written: the Dataset screen can copy it into the project file first. If the project file changes on disk under unsaved edits, they are kept and the screen offers to put them on the new version or to overwrite it.

## The Detect Screen

The **Detect** screen shows what the target detector found in each image of the open project, lets you try other detector settings on one image, and writes what the project leaves out of its calibrations.

- **The matrix** (left) has a row per view and a column per camera. Each cell holds the number of features found in that image, with a bar scaled to the number a complete board has (a PuzzleBoard is made to be seen in part, so it has none). A warning mark means fewer than expected, an ambiguous board, or an image the last calibrated run left out (the reason is in the cell's tooltip: too few features, only part of a chessboard, a turned board); a red cell means no board was found or the image failed; `·` means nothing is cached for the image with the project's current detector settings; a struck-through number is an excluded image. **Only problems**, **Only excluded**, a count limit and a view-token search find a bad image among hundreds. **Detect all** detects every image that has no cached detection, several at a time, with progress and **Cancel** (what was detected stays); **Detect again** ignores the cache.
- **The image** (middle) is the selected image in a zoomable stage with the detection drawn over it. The features are drawn for every target: chessboard corners with their grid indices and the lattice between them (and, from the layers menu, each corner's two edge directions); ChArUco corners with the decoded marker quads and their ids; PuzzleBoard corners with their cell ids and the code bits read on the edges; ring-grid markers with their outer and inner ellipse, code or lattice cell, and, from the layers menu, the one-sigma ellipse of a photometric centre (drawn magnified; the stage prints the gain). Candidates the detector left out are drawn dimmed. A PuzzleBoard corner at the end of an edge whose bit disagrees with the master pattern has a thin ring around it. The layers menu switches each layer; hovering a feature reads its position, score, centre estimator and confidence under the stage. `,` and `.` step the camera and `[` and `]` the view; the zoom stays across images of one size.
- **Features** (right) lists the image's features (hover and press are mirrored with the stage) with the detector's verdict, the board's cell size and the noise level it measured. **Rejected** groups the candidates the detector left out by reason, in words ("Corner candidate not taken by any labelled grid", "Marker decodes as 17, board expects 23"), above the detector's verdict on the whole image and what the last run did with it; pressing one shows it on the image. A PuzzleBoard whose bits mostly disagree with the master pattern, or a ring grid whose view is not anchored on the board, says so: its labels are likely wrong or only relative.
- **Detector** shows the detector settings of the project's manifest as a form, with every change from the saved settings marked. **Preview on this image** runs the detector on the open image with the edited settings and writes nothing: the stage shows the preview (switch to the saved detection with **Saved**), and a line says what differs ("+3 features, −1 feature, 2 labels changed against the saved settings"). **Apply to project** saves the settings into the manifest; the other images' detections were made with the old settings, so their cells show `·` until **Detect all with the new settings**. **Discard** drops the edit and the preview. A manifest that is a file of its own is never written: the preview works, Apply is disabled, and the Dataset screen can copy the manifest into the project file first.

**Exclusions.** Three things can be left out of every run, each with one gesture, and each is one write of the [project file](projects.md)'s `[exclude]` section:

| What | How | Written as |
|---|---|---|
| Some features of one image | click a feature (`⇧`-click adds), or hold `⇧` and drag a box on the stage (`⌘` as well adds to the picked, `Esc` cancels), then **Exclude** (or `X`) | `exclude.features`: the image and the features' board points |
| One camera's image of a view | `X` with no feature picked, or **Exclude image** | `exclude.images` |
| A whole view, every camera | `⇧X`, or **Exclude view** | `exclude.views` |

The same gesture restores. An excluded feature is named by its board point, so it stays excluded when the image is detected again; a board point that matches no feature of an image's detection leaves nothing out, and the Features tab says so. Excluded things stay visible: struck-through in the matrix, dimmed and crossed on the stage, an "excluded" status in the table, and a banner on an excluded image. `⌘Z` (or **Undo**) takes the last exclusion back; **Exclusions** at the foot of the stage lists them all with an editable note each, **Restore** per row and **Restore all**. A run made before an exclusion shows `Stale`. With the connection to the back end lost, every write is disabled. If the project file changed on disk under a write, the app tries once more on the file as it is now.

## Keyboard Shortcuts

The list is under the keyboard button in the top bar (or `?`). A bare key never acts while a text field has focus, and nothing but `⌘K` acts while a dialog is open.

| Keys | Does |
|---|---|
| `⌘1` … `⌘7` | go to Library … Device |
| `⌘K` | the command palette: jump to a screen, an Inspect tab, a recent project, a run or a camera by typing a few letters |
| `[` `]` | previous and next view |
| `,` `.` | previous and next camera |
| `⇧[` `⇧]` | previous and next run |
| `Esc` | clear the selection, then close panels |
| `⌘S` | save the project's unsaved edits |
| `⌘↵` | Calibrate: run with the editor's parameters |
| `X`, `⇧X`, `⌘Z` | Detect: exclude the picked features or the image, the whole view; undo the last exclusion |
| `←` `→`, `↑` `↓` | Inspect, Frames: previous and next pose, camera |
| `f`, `1`, `+` `-` | Inspect, Frames: fit, one-to-one, zoom |
| `?` | this list |

## Relation to the Library Crates

The app's forms are generated from the JSON Schemas of the Rust configuration and manifest types, which the app's back end serves with their defaults, so every field the linked library accepts is editable without hand-writing JSON. The manifest it edits is the [dataset manifest](manifest.md); the results it displays are the [exports](per_feature_residuals.md) and [uncertainty](uncertainty.md) described in this book.

To work on the app itself, see `apps/studio/DEVELOPING.md` in the repository.
