# Running Calibrations

The **Run calibration** workspace (`⌘5`) drives a calibration of the open [project](projects.md) end to end without leaving the app: review the manifest and the configuration, click **Run**, and the run lands in the project's `runs/` beside every other. It runs any of the seven problem types and all four target detectors; the manifest it edits is the [dataset manifest](manifest.md).

## Choosing a Dataset

A calibration runs on a project. With none open, the workspace offers the ways to one.

### Quick Start

A card grid of the library's public sample datasets, among them:

- `kuka_1`: single-camera hand-eye, a KUKA arm with a 17×28 chessboard;
- `ds8`: a three-camera rig hand-eye dataset, published as open data ([doi:10.15482/USDA.ADC/1340592](https://doi.org/10.15482/USDA.ADC/1340592)).

The app carries no dataset. The first click on a card downloads the sample from its publisher into the app's samples folder (`VISION_CALIBRATION_DATA` overrides the location on the desktop), checks every file against its SHA-256 and makes that folder a project referencing the sample's manifest; later clicks reopen it with its runs. The first click needs a network connection, and a failed download is reported. A server built without the samples shows no cards.

### A Dataset Folder or a Manifest

**New project from a dataset folder…** makes the folder a project whose manifest the [folder sniffer](manifest.md#the-folder-sniffer) drafts from file names and structure only. Fields it cannot determine are left unresolved and **block Run** until you fill them in; a red badge marks each one. This is the unresolved-fields rule of the manifest in action. **New project from a manifest…** makes the manifest's folder a project referencing the file; the app shows that manifest but never rewrites it, so edit it where it is and reload the project. A folder that already holds a project is opened instead.

## Reviewing the Setup

The **Manifest** and **Calibration config** sections are collapsible, schema-driven forms generated from the JSON Schema the Rust types derive, served by the app's back end together with the library's defaults. They show the project's manifest and its configuration (the workflow's defaults with the project's changes); every field shows its description. An **Advanced JSON editor** is available for anything the forms do not expose.

Edits are the project's until saved: **Save** writes them to the project file, the configuration as its changes from the defaults, and **Run** saves them first. Changing the workflow (the manifest's topology) starts the configuration from that workflow's defaults.

## Run

While the solve runs, a **stage checklist** (detecting features, solving, exporting) with a live elapsed clock replaces a plain spinner; beside the active stage it shows the images detected so far, then the calibration step and its solver iteration with the cost. A **Cancel** button stops the run at its next image or solver iteration. A cancelled run shows a distinct "Run cancelled" banner, not an error, and writes nothing.

On success the run is the project's next `runs/run-NNN/` (its record, export, quality report and `report.html`); the banner names it, and the app navigates to Diagnose with it after a short delay. Detection goes through the project's [detection cache](detection.md#the-detection-cache), so a second run on the same images skips it.

When the runner meets an ambiguity it refuses to guess (a hand-eye dataset with no `pose_pairing`, for example), a dialog asks you to pick one of the suggested options, which goes into the manifest, instead of continuing. Other errors appear inline, with the field they are about.

## Walkthrough: Calibrate the KUKA Dataset

1. Open the Run workspace and click **Use** on the **kuka_1** card.
2. Review the manifest (a chessboard of 17x28 corners with 20 mm squares and `by_index` pairing) and the config (eye-in-hand is the default).
3. Click **Run**.
4. On success you land in Diagnose with the run loaded: the intrinsics, the hand-eye transform and the reprojection error (well under a pixel on this dataset) are in the numbers, and the residual heatmap is on the images.

## Laser Datasets

Laser datasets run in two stages (`rig_handeye`, then `rig_laserline_device`); the manifest shape is described in [Laser Datasets](manifest_laser.md). Such setups usually need non-default configuration (Scheimpflug sensors, eye-to-hand mounting, point-to-plane laser residuals); set it in the config form and save it with the project.
