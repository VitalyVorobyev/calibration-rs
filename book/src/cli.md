# The Command Line

`vcal` calibrates a folder of images without a line of code: it drafts the
dataset manifest, checks it, detects the target, calibrates, and writes the
result and its quality report as JSON. It runs the same library calls as
[Calibrate Your Own Images](getting_started.md#calibrate-your-own-images), so
the numbers are the same.

```bash
cargo install vision-calibration-cli   # or: pip install vision-calibration
```

The Rust crate and the Python package install the same program.

| Command | Does |
|---|---|
| `vcal init <folder>` | drafts `<folder>/dataset.toml` from the file names |
| `vcal check <manifest>` | checks the manifest and its images, without detecting |
| `vcal run <manifest>` | detects, calibrates, writes `calibration.json` and `quality.json` |
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
the desktop app write, and `quality.json` is the quality report. `-o <dir>`
writes them elsewhere. Detections are cached in `.detections` next to the
manifest, so a second run skips detection; `--redetect` detects again.

## Configuration

Without a configuration file the workflow runs with its defaults: a pinhole
camera with Brown-Conrady distortion. `-c <file>` changes the defaults the
file names and keeps the rest. The file is TOML or JSON and holds the
workflow's `*Config` fields. A fisheye camera needs a fisheye model:

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
| 2 | a wrong command line |
| 3 | `run` wrote its files, but the verdict is fail |

A script can stop on a failed calibration with `vcal run ... || exit`.

## Limits

The laser workflows (`laserline_device`, `rig_laserline_device`,
`rig_handeye_laserline`) need a laser-line extractor, which the library takes
from code ([Laser Datasets](manifest_laser.md)); `vcal run`
refuses them before any detection. Calibrate them from Rust or Python.
