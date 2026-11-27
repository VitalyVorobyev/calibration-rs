# The Command Line

`vcal` calibrates a folder of images without a line of code: it drafts the
dataset manifest, checks it, detects the target, calibrates, and writes the
result and its quality report as JSON. It runs the same library calls as
[Calibrate Your Own Images](getting_started.md#calibrate-your-own-images), so
the numbers are the same.

```bash
cargo install vision-calibration-cli --version 1.0.0-rc.1   # or: pip install --pre vision-calibration
```

The Rust crate and the Python package install the same program.

| Command | Does |
|---|---|
| `vcal init <folder>` | drafts `<folder>/dataset.toml` from the file names |
| `vcal check <manifest>` | checks the manifest and its images, without detecting |
| `vcal run <manifest>` | detects, calibrates, writes `calibration.json` and `quality.json` (`--report` adds `report.html`; `--seed` starts from a previous calibration) |
| `vcal run <project>` | the same, into the project's next `runs/run-NNN/` ([Projects](#projects)) |
| `vcal project init <folder>` | makes a folder a calibration project |
| `vcal project runs <project>` | lists a project's runs |
| `vcal project import <project> <export>` | adds an export calibrated elsewhere as a run |
| `vcal project note <project> <run>` | labels, annotates or stars a run |
| `vcal project scan <folder>...` | lists the projects under folders, each device's runs together |
| `vcal seed from-runs <run>...` | writes the seed of a calibrated run, or of a device type from several units' runs |
| `vcal seed from-spec <spec.json>` | writes the datasheet seed of a device spec |
| `vcal sample list` | lists the sample datasets |
| `vcal sample fetch <id>` | downloads a sample dataset and prints its manifest's path |

`vcal <command> --help` lists each command's options.

## A folder of images

```bash
vcal init data/my_camera
```

```text
wrote data/my_camera/dataset.toml
fill in, then delete `_unresolved`:
  target
```

The draft lists what the file names cannot tell, the board above all, under
`_unresolved`. Describe the board ([The Dataset Manifest](manifest.md)) and
delete the list:

```toml
[target]
kind = "chessboard"
rows = 6            # inner corners down
cols = 9            # inner corners across
square_size_m = 0.025
```

`vcal check` then reads the manifest and each image's header. It reports a
manifest that is incomplete or invalid, a camera whose images it cannot find,
and a camera whose images differ in size, all before any detection:

```text
$ vcal check data/my_camera/dataset.toml
data/my_camera/dataset.toml: planar_intrinsics
  camera cam0: 640 × 480 px
ok
```

`vcal run` calibrates. On the 13 views of the `opencv_left` sample, fetched
to `~/.cache/vision-calibration`, it prints:

```text
$ vcal run ~/.cache/vision-calibration/opencv_left/dataset.toml
planar_intrinsics: 13 of 13 views used
camera 0 (left), 640 × 480 px:
  fx = 534.108 ± 0.930 px
  fy = 534.155 ± 0.966 px
  cx = 341.029 ± 1.075 px
  cy = 233.586 ± 1.144 px
verdict: warn
  not_applicable variance_factor: no expected feature noise was stated (expected_feature_noise_px)
  warn projection_sigma camera 0: projection σ up to 5.490 px over the 640×480 image (centre 1.146 px, corners 5.490 px)
wrote ~/.cache/vision-calibration/opencv_left/calibration.json and ~/.cache/vision-calibration/opencv_left/quality.json
```

Each parameter comes with its σ, and the verdict with the checks that did not
pass ([Judging a Result](quality.md)). Here the 13 views leave the image
corners uncertain by 5 px; [Capturing Calibration Images](capture.md) says
which views would fix that. A view left out of the calibration is listed with
its reason, such as a chessboard whose corners do not reach every edge of the
board.

`calibration.json` is the workflow's export, the same JSON the library and
the desktop app write, and `quality.json` is the quality report. `--report`
adds `report.html`, all of it as one page with the residual and
projection-σ figures ([The Calibration Report](report.md)). `-o <dir>`
writes them elsewhere. Detections are cached in `.detections` next to the
manifest, so a second run skips detection; `--redetect` detects again.

The export names the image of every view and camera it used, in
`diagnostics.image_manifest`: each frame's path is relative to the
manifest's folder, and the `root` leads there from the folder
`calibration.json` is in (`..` for `-o out`, the absolute folder for an
output folder outside the dataset). A viewer that overlays the residuals on
the images, such as the desktop app, opens it directly, and a later run
that builds on it, such as a rig laser calibration, finds its views by
their images.

## Configuration

Without a configuration file the workflow runs with its defaults: a pinhole
camera with Brown-Conrady distortion. `-c <file>` changes the defaults the
file names and keeps the rest. The file is TOML or JSON and holds the
workflow's `*Config` fields: the library's `ConfigPatch`
([Changes to the Defaults](getting_started.md#changes-to-the-defaults)). A
fisheye camera needs a fisheye model:

```toml
# fisheye.toml
camera.model = { projection = "equidistant", distortion = "kannala_brandt4", sensor = "identity" }
```

```bash
vcal run "$(vcal sample fetch opencv_fisheye_stereo)" -c fisheye.toml
```

On that sample the default model leaves each focal length uncertain by
14 px; the fisheye model brings σ(fx) to 1.1 px. A field the workflow does
not have is an error, so a misspelt name never passes silently. The fields
are those of [Configuring a Run](getting_started.md#configuring-a-run) and
of each workflow's chapter.

## Starting from a previous calibration

`--seed <file>` starts every stage of the calibration from a seed instead
of the closed-form initialization: the lenses, mounts, hand-eye and laser
planes of a previous calibration of the device, or of another unit of its
type. The file is a `DeviceSeed`, JSON (TOML by its extension), which the
library and the Python package make from an export
([Starting from a previous calibration](manual_init.md#starting-from-a-previous-calibration)).
`--device-spec <spec.json>` starts from a datasheet instead: each camera's
focal length and mount tilt and the drawing's mounts
([Device Specifications](manifest.md#device-specifications)).

```bash
vcal run unit_0008/dataset.toml --seed unit_0007/seed.json
```

```text
planar_intrinsics: 13 of 13 views used
started from the seed unit_0007/seed.json
...
```

The seed is a starting point, not a constraint: the calibration still
estimates everything, and a good seed ends where the unseeded run ends. Its
cameras must be the manifest's, each for images of the same size and, for a
calibrated lens, of the configured camera model; anything else fails before
any detection with a message that says which. The two options exclude each
other and are for a manifest's run; a project names its seed in its
`[seed]` ([Starting from a Seed](projects.md#starting-from-a-seed)).

`vcal seed` writes seed files, JSON (TOML by a `.toml` extension):

```bash
vcal seed from-runs unit_0007/runs/run-003 -o unit_0007/seed.json
vcal seed from-runs unit_0001/runs/run-002 unit_0002/runs/run-001 unit_0003/runs/run-004 -o type.json
vcal seed from-spec spec.json -o datasheet.json
```

`from-runs` takes project run folders, whose records name the cameras and
their image sizes. One run gives its seed. Several runs, one per unit of a
device type, give the type's seed: each value the median across the units,
printed with its spread (median, MAD σ, min and max); quantities only some
units have are left out and listed; `--json` prints the whole aggregate
instead. The type's seed is a starting point for the next unit, not a
calibration of any unit
([A seed for a device type](manual_init.md#a-seed-for-a-device-type)). An
export file outside a project (or a `vcal run -o` folder) needs its cameras'
ids and image sizes: `--camera-ids cam0,cam1 --image-sizes
1920x1200,1920x1200`.

## Projects

A [project](projects.md) keeps a dataset with its configuration changes,
what to leave out and every run. `vcal check` and `vcal run` take one when
their path is a project folder or its `calibration-project.toml`; any other
path is a manifest, read as above. The project file decides: a folder
without one is not a project.

```bash
vcal project init my-camera --manifest my-camera/dataset.toml --device-id SN-0042
vcal check my-camera
vcal run my-camera --label "after refocus" --report
```

```text
single_cam_handeye: 30 of 30 views used
camera 0 (cam0), 1928 × 1208 px:
  fx = 2058.456 ± 0.490 px
  fy = 2058.993 ± 0.486 px
  cx = 961.973 ± 0.391 px
  cy = 610.085 ± 0.331 px
verdict: pass
  not_applicable variance_factor: no expected feature noise was stated (expected_feature_noise_px)
wrote my-camera/runs/run-001 (run-001)
```

A project's run writes `export.json`, `quality.json` and `run.json`, the
run's record, into the next `runs/run-NNN/` (`report.html` with `--report`);
`-c` changes the configuration for this run only, over the project's
`[config]`, and the record keeps it; `--label` names the run. `-o` is a
manifest's option and `--label` a project's: the other combination is a
command-line error (status 2). Detections are cached in the project's
`.detections`, shared with `vcal run` on a manifest in the same folder.

`vcal project init <folder>` writes `calibration-project.toml`:

- `--manifest <file>` references a manifest file, which stays the one
  source of the dataset; `--embed` copies it into the project file instead.
- Without `--manifest`, a manifest is drafted from the file names in
  `--data <folder>` (default: the project folder) and written into the
  project file; fill in what it lists under `_unresolved`. A folder that
  already holds `dataset.toml` or `dataset.json` is refused, so the project
  never forks a manifest you keep by hand.
- `--name`, and `--device-id` with `--device-product` and `--device-label`,
  fill in the project's name and the device it calibrates.
- `--seed-run <run>`, `--seed-file <file>` or `--seed-spec <spec.json>`
  (one of them) writes the project's `[seed]`: every run starts from that
  run's calibration, that seed file or that datasheet. The path must exist;
  `vcal check` checks that it makes a seed that fits.

The other commands:

```text
$ vcal project runs my-camera
run-001  2026-11-26 11:21  pass  30/30 views  cam0 0.152 px  * after refocus
$ vcal project note my-camera run-001 --star --notes "reference"
$ vcal project import my-camera out/kuka --label "from the lab PC"
$ vcal project scan ~/calibrations
```

`runs` lists each run's id, time (UTC), verdict, views, each camera's RMS
reprojection error, the seed it started from (`seed: run ../unit/runs/run-002`),
star and label (`--json` prints the records); `note`
sets a run's label (`--label`), text (`--notes`) and star (`--star`,
`--unstar`), an empty text clearing a field; `import` adds an export file
or a `vcal run -o` folder whose workflow and camera count are the
project's; `scan` lists the projects under the folders, each device's runs
together (`--json`, `--max-depth`), naming on standard error the folders it
could not read and exiting 0 once the walk is done.

## Sample datasets

```bash
vcal sample list
vcal run "$(vcal sample fetch kuka_1)" -o out/kuka
```

`fetch` downloads a sample from its publisher, verifies every file's SHA-256
and prints the path of its manifest, so it composes with `run`. A second
fetch reuses the cache: `VISION_CALIBRATION_DATA`, else `vision-calibration`
in the platform's cache directory, or `--cache <dir>`.

## Exit status

| Status | Meaning |
|---:|---|
| 0 | done; for `run`, the verdict is pass or warn |
| 1 | an error, printed on standard error |
| 2 | a wrong command line (`-o`, `--seed` or `--device-spec` with a project, `--label` with a manifest, `--seed` with `--device-spec`, included) |
| 3 | `run` wrote its files, but the verdict is fail |
| 130 | interrupted with Ctrl-C: the run stopped and wrote nothing |

A script can stop on a failed calibration with `vcal run ... || exit`.

## Progress and Ctrl-C

On a terminal, `vcal run` keeps one line of standard error up to date: the
images detected (`detecting 12/40`), then each step and its solver
iterations (`optimize: iteration 7/50, cost 1.2345e2`). Piped or redirected,
it prints no progress. Ctrl-C stops the run at the next image or iteration,
prints `interrupted` and exits with status 130; a second Ctrl-C exits at
once.

## Limits

The laser workflows (`laserline_device`, `rig_laserline_device`,
`rig_handeye_laserline`) need a laser-line extractor, which the library takes
from code ([Laser Datasets](manifest_laser.md)); `vcal run`
refuses them before any detection. Calibrate them from Rust or Python.
