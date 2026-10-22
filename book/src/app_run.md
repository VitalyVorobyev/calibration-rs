# Running Calibrations

The **Run calibration** workspace (`⌘5`) drives a calibration end to end without leaving the app: choose a dataset, review the manifest and the configuration, click **Run**. It runs any of the eight problem types and all four target detectors; the manifest it edits is the [dataset manifest](manifest.md).

## Choosing a Dataset

### Presets

A card grid of public datasets. Two ship with the repository:

- `data/kuka_1`: single-camera hand-eye, committed to the repository;
- `data/DS8`: a three-camera rig hand-eye dataset, published as open data (extract its archive, [doi:10.15482/USDA.ADC/1340592](https://doi.org/10.15482/USDA.ADC/1340592), into that folder).

Clicking a card loads its manifest and applies the configuration overrides the preset carries. Some datasets need non-default settings that the topology's schema defaults do not give, for example Scheimpflug sensors or an eye-to-hand rig; the preset states them. Preset paths are resolved against the repository root, so the cards work on any checkout. The registry lives in `app/src/workspaces/RunWorkspace/presets.ts`.

### Sniff Folder

Pick an arbitrary dataset folder and the app infers a draft manifest with the [folder sniffer](manifest.md#the-folder-sniffer). It reads file names and structure only. Fields it cannot determine are left unresolved and **block Run** until you fill them in; a red badge marks each one. This is the unresolved-fields rule of the manifest in action.

## Reviewing the Setup

The **Manifest** and **Calibration config** sections are collapsible, schema-driven forms generated from the same JSON Schema the Rust types derive. Every field shows its default and description. An **Advanced JSON editor** is available for anything the forms do not expose.

Editing the form clears the preset selection, so nothing silently reverts your edits. To re-run with different settings, edit the config and click **Run** again.

## Run

While the solve runs, a **stage checklist** (detecting features, solving, exporting) with a live elapsed clock replaces a plain spinner. A **Cancel** button stops the run at the next stage boundary; the stage in flight always finishes, since there is no solver interrupt. A cancelled run shows a distinct "Run cancelled" banner, not an error.

On success the app flashes a status banner and navigates to Diagnose after a short delay, with the fresh export loaded. Detection goes through the [detection cache](detection.md#the-detection-cache), so a second run on the same images skips it.

When the runner meets an ambiguity it refuses to guess (a hand-eye dataset with no `pose_pairing`, for example), a modal dialog asks you to pick one of the suggested options instead of continuing. Validation failures and other errors appear inline.

## Walkthrough: Calibrate the KUKA Dataset

1. Open the Run workspace and click the **KUKA handeye** preset card.
2. Review the manifest (a chessboard of 17x28 corners with 20 mm squares and `by_index` pairing) and the config (eye-in-hand is the default).
3. Click **Run**.
4. On success you land in Diagnose with the export loaded: the intrinsics, the hand-eye transform and the reprojection error (well under a pixel on this dataset) are in the numbers, and the residual heatmap is on the images.

## Laser Datasets

Laser datasets run in two stages (`rig_handeye`, then `rig_laserline_device`); the manifest shape is described in [Laser Datasets](manifest_laser.md). Such setups usually need non-default configuration (Scheimpflug sensors, eye-to-hand mounting, point-to-plane laser residuals); set it in the config form, or put it in a preset.
