# Running Calibrations

The **Calibrate** screen (`⌘4`) sets up what to estimate, runs it and keeps the history of attempts of the open [project](projects.md). Its left panel is the project's run list, the centre is the parameter editor and the right is **This run**: what pressing **Run** will do, and its progress. It runs any of the seven problem types and all four target detectors; the dataset is the project's [manifest](manifest.md).

## The Parameter Editor

The editor shows the configuration the next run takes. It is generated from the JSON Schema of the workflow's configuration, served by the app's back end with the library's defaults, so it works for every problem type with no form of its own. The fields that matter most come first, in groups found by the configuration's structure:

- **Camera**: the camera model (projection, distortion, sensor) and, for a rig, **per-camera overrides**, a row per camera whose model differs from the rig's;
- **Fix and free**: which parameters are held fixed in the solve (chips for the intrinsics, the distortion terms and the sensor tilt; a list left out means the model's default, which the chips show) and the switches that free or fix an estimate, among them **fix poses**, the views whose target pose stays at its initial value, typed as view numbers counted from 0 (`0, 3, 5-7`; empty holds none);
- **Robust loss**: the loss of each stage that has one;
- **Solver and iterations**; **Initialization**;
- **All other fields**, folded: the rest of the schema, so the curated view never hides a real option.

**Compare with** chooses what changes are measured from: the library default (the default), the project's configuration, or the configuration a run recorded. Every field that differs is marked with what it was, with a reset per field and a **Reset all**. Each edit is checked by the library after a short pause; a value it rejects shows the library's message on its field and disables **Run**. **Resulting patch** shows the changes from the library defaults that the run will record, as TOML, with a copy button.

## Running

**Run** (`⌘↵`) calibrates with the editor's configuration as a one-off change over the project's: the run records it, and the project file is not touched. **Save as project default** is the separate, explicit way to write the editor's configuration into the project file. An optional label names the run; **Re-detect images** detects every image again instead of reading the project's [detection cache](detection.md#the-detection-cache).

The **Seed** row of the right panel shows what the project's runs start from: nothing (each camera starts from what its own images give), or the project's seed with its cameras. Choose one of the project's runs to make it the seed, or **Clear** it; the link goes to the Device screen for a seed from the device or its type. A seed is a starting value, not a prior: the run still estimates its values from its own images, and the one thing it limits is the tilt of a tilted (Scheimpflug) sensor. A run's details say which seed it started from and how many of its values the solve took.

While the solve runs, the right panel shows a stage checklist (detecting, solving, writing the run) with the images detected so far, the solver iteration and its cost, the cost curve, the elapsed time and **Cancel**, which stops the run at its next image or solver iteration and writes nothing. The same progress is in the job indicator and the job tray. A run that fails says what failed in the library's words, where it stopped and what to try; the editor keeps the settings that failed.

## The History

Every run of the project is a card in the left panel: its verdict in words, its reprojection RMS (and per camera for a rig), the views and features it used, how many settings it changed from the project's, when it was made and whether the project's dataset, exclusions or thresholds changed since. **Lowest RMS** marks the run with the lowest error among those that did not fail; a lower error is not a better model by itself, so a star is how you mark the run you accept. The tick on a card collects runs for comparison.

Choosing a card opens its details: its label and notes (saved when the field is left), the settings it changed, and how they differ from the editor's now. The card's menu opens the run in Inspect, opens its report, copies its id, deletes it after a confirmation, and **Rerun from this run**, which loads the run's recorded configuration into the editor and compares with it. If the project's exclusions are no longer those of that run, the editor offers to restore them or to keep the current ones. A run of an earlier format or an imported export has no record of its configuration and cannot be rerun.

## Walkthrough: Calibrate the KUKA Dataset

1. Open the `kuka_1` project from the Library and go to Calibrate.
2. Change **Max iters** to 30 and release `k3` in **Fix and free**; the editor marks both as changed.
3. Click **Run**. The new run appears at the top of the list with "2 changes".
4. Choose **Rerun from this run** in its menu, change the robust loss of the hand-eye stage to Huber and run again; the two runs sit side by side with their RMS.
5. Star the one you accept.

## Laser Datasets

Laser datasets run in two stages (`rig_handeye`, then `rig_laserline_device`); the manifest shape is described in [Laser Datasets](manifest_laser.md). Such setups usually need non-default configuration (Scheimpflug sensors, eye-to-hand mounting, point-to-plane laser residuals); set it in the editor and save it as the project default.
