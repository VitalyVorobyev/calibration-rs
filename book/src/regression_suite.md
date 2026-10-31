# Regression Suite

The repository carries a versioned regression suite in `regression/`. Each
entry pairs a set of **observations** — the 2D–3D target correspondences, robot
poses and image sizes a calibration consumes, extracted once from the images —
with the **reference result** calibration-rs produces from them. Every change to
the library is checked against these results, and because the observations are
plain JSON, other calibration tools can run on exactly the same data.

## Entries

| Id | Problem | Data | Licence |
|----|---------|------|---------|
| `kuka_1` | `single_cam_handeye` | 30 eye-in-hand views of a 17×28 chessboard (20 mm) from a KUKA arm | None stated |
| `stereo_left` | `planar_intrinsics` | 20 views of a 7×11 chessboard (30 mm), left camera | None stated |
| `stereo_right` | `planar_intrinsics` | The same 20 views, right camera | None stated |
| `stereo_rig` | `rig_extrinsics` | Both cameras of the stereo pair together | None stated |
| `ds8` | `rig_handeye` | Three cameras on a Denso arm, 33 of 42 poses with a complete 10×14 chessboard (52 mm) | U.S. public domain |
| `synth_planar` | `planar_intrinsics` | Synthetic: 20 views of a 13×9 board, 0.5 px noise | CC0 |
| `synth_scheimpflug` | `planar_intrinsics` | Synthetic: a tilted (Scheimpflug) sensor, 20 views, 0.5 px noise | CC0 |
| `synth_handeye` | `single_cam_handeye` | Synthetic: eye-in-hand, 20 robot stations, 0.5 px noise | CC0 |
| `synth_rig` | `rig_extrinsics` | Synthetic: two cameras, intrinsics refined with the rig, 20 views | CC0 |
| `synth_laserline` | `laserline_device` | Synthetic: one camera and a laser plane, 20 views | CC0 |
| `synth_rig_handeye_laserline` | `rig_handeye_laserline` | Synthetic: two cameras on a robot, each with a laser plane, solved jointly | CC0 |

The real-image entries come from their publishers:

- `kuka_1` is dataset `kuka_1` of RWHE-Calib, from I. Ali, O. Suominen,
  A. Gotchev and E. Ruiz Morales, *Methods for Simultaneous
  Robot-World-Hand–Eye Calibration: A Comparative Study*, Sensors 19(12),
  2837 (2019), <https://doi.org/10.3390/s19122837>. The images and robot
  poses are published at <https://github.com/ihtishamaliktk/RWHE-Calib> as
  open-access data; the authors state no licence and ask users to cite the
  paper.
- The stereo entries use *Chessboard Pictures for Stereocamera Calibration*
  by Daniel (Kaggle user `danielwe14`), version 3,
  <https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures>,
  which states no licence.
- `ds8` is dataset 8 of A. Tabb and K. M. Ahmad Yousef, *Solving the
  robot-world hand-eye(s) calibration problem with iterative methods*,
  Machine Vision and Applications 28 (2017); the images are published at
  <https://doi.org/10.15482/USDA.ADC/1340592> in the U.S. public domain.

`regression/suite.toml` records each entry's licence (an SPDX identifier,
`NOASSERTION` where the publisher states none), citation and source.

The synthetic entries come with their ground truth, so they check accuracy
as well as stability, and the laser entries are the suite's laser coverage.
Their data was generated once with fixed seeds and is frozen like the rest:
a change to the generator cannot change it.

## Layout

```text
regression/
├── suite.toml                          entries: description, licence, config
├── SHA256SUMS                          checksums of the observation and truth files
├── observations/<id>.json              the entry's observations
├── truth/<id>.json                     a synthetic entry's true parameters
├── results/<id>/calibration-rs.json    the entry's reference result
├── results/<id>/<tool>.json            another library's result, for comparison
├── fetch/<id>.sha256                   the files `suite fetch` downloads, with checksums
└── schema/                             JSON Schemas of the three file formats
```

`SHA256SUMS` is in the format of `sha256sum`, so the data can be verified
without any of our tools:

```bash
cd regression && sha256sum -c SHA256SUMS      # shasum -a 256 -c on macOS
```

## Running it

```bash
cargo run --release -p vision-calibration-bench --bin calib-bench -- suite check
```

`suite check` verifies each observation file against `SHA256SUMS` (a missing
or mismatched checksum fails the entry; it is never skipped), calibrates the
entry with its config and compares every number in the result with the
reference. Each entry reports `PASS` or `FAIL` with the values that moved, and
the largest change of an estimated parameter in units of the reference's
standard deviation (see [Parameter Uncertainty](uncertainty.md)); the command
exits non-zero when any entry fails. `--only <id>` selects entries.

`--sigma <k>` compares only the estimated parameters, each within `k`
standard deviations of the reference, for results that may legitimately
differ in their last digits. `--backend factrs` (or `tiny_solver`) runs every
solver stage on that backend:

```bash
calib-bench suite check --backend factrs --sigma 1
```

Two more subcommands maintain the suite:

- `suite bless` recalibrates the selected entries and overwrites their
  reference results. Run it when a change is meant to move the results, and
  review the diff of `results/` like any other code change.
- `suite freeze` re-extracts an entry's observations from the images its
  dataset manifest names (built with `--features tier-b`), or regenerates a
  synthetic entry's observations and truth from its scene, and rewrites
  `SHA256SUMS`. Observations change only when the detector does. The images
  are read from the manifest's folder, or with `--images <dir>` from the
  cache `suite fetch` fills (below).

## Fetching the images

The observations are all `suite check` needs. To re-extract them, or to run
another tool on the images themselves, `suite fetch` downloads each entry's
images from its publisher and checks every file against the entry's
`fetch/<id>.sha256`:

```bash
cargo run --release -p vision-calibration-bench --features tier-b \
    --bin calib-bench -- suite fetch
cargo run --release -p vision-calibration-bench --features tier-b \
    --bin calib-bench -- suite freeze --images target/suite-images
```

The files land in `target/suite-images/<id>/` (`--images <dir>` picks
another folder), laid out as the dataset manifest expects. A file already
there that matches its checksum is not downloaded again, so the command is
cheap to repeat; a downloaded file that does not match fails the entry, as
it means the publisher's files changed. `suite freeze --images <dir>`
verifies the cached files again, freezes the entries `suite fetch`
downloads (or those named with `--only`) and rewrites `SHA256SUMS`. On
Apple-silicon macOS, where the observations are frozen, a clean checkout
reproduces them byte for byte. Corner detection is not bit-identical across
platforms: on x86-64 Linux the same images give corners up to about
0.001 px apart, so the files differ there while every view and corner is
the same.

| Entry | Downloaded from |
|-------|-----------------|
| `kuka_1` | The 30 images and the robot poses of `Datasets/kuka_1` in the RWHE-Calib repository, at a pinned commit |
| `ds8` | `DS8.zip` of the Ag Data Commons record; the images as published (RGB TIFF for the two JAI cameras, colour JPEG for the third) |
| `stereo_left`, `stereo_right`, `stereo_rig` | Not fetched: Kaggle needs an account. `suite fetch` prints the dataset page; unpack the download into `data/stereo/` |

`suite.toml` gives each fetchable entry a `fetch` source: `files`, each file
downloaded from a `base_url` followed by its path, or `zip`, one archive
whose `root` folder is the dataset's. A scheduled CI job fetches the images
from the publishers every week, re-freezes these entries and fails when the
observations no longer match `SHA256SUMS`.

The image entries are also the public datasets of the benchmark commands
that run detection: `calib-bench run --dataset <id>` and `calib-bench
feature-cov` read each entry's dataset manifest, and `--images <dir>` reads
the fetchable entries from the `suite fetch` cache.

## The observation format

```json
{
  "version": 1,
  "cameras": [{ "id": "cam0", "image_size": [1928, 1208] }],
  "problem": "single_cam_handeye",
  "input": {
    "views": [
      {
        "obs": {
          "points_3d": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
          "points_2d": [[109.811, 92.798], [140.950, 91.702]]
        },
        "meta": {
          "base_se3_gripper": {
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "translation": [1.205, -0.808, 0.524]
          }
        }
      }
    ]
  }
}
```

- `problem` names the problem type and `input` is its input, the same payload
  `CalibrationSession::set_input` takes in Rust and the `run_*` functions take
  in Python (see [Data Types Quick Reference](types_reference.md)). Rig inputs
  hold `num_cameras` and, per view, one observation per camera in `cameras`
  order, `null` where that camera did not see the target.
- `cameras` lists the cameras in input order with their image size
  `[width, height]` in pixels.
- `points_3d` are target points in metres in the target frame; planar targets
  lie in `z = 0`. `points_2d` are the matching image points in pixels, with
  integer coordinates at pixel centres (the OpenCV convention).
- A pose is `{"rotation": [qx, qy, qz, qw], "translation": [x, y, z]}`: a unit
  quaternion, scalar last, and a translation in metres. `a_se3_b` maps
  coordinates in frame `b` to frame `a`, so `base_se3_gripper` takes gripper
  coordinates to robot-base coordinates — OpenCV's `R_gripper2base`,
  `t_gripper2base`.
- `version` is the format version; it changes only with an incompatible
  format change.

`schema/observation_set.json` is the format's JSON Schema, generated from the
types the library reads, with the input of every problem type; `truth.json`
and `external_result.json` describe the truth files and the results of other
libraries below.

## Ground truth

`truth/<id>.json` names each true value by the label the result's
covariance uses (see [Parameter Uncertainty](uncertainty.md)):

```json
{
  "parameters": { "fx": 1000.0, "fy": 990.0, "k1": -0.12, "plane.nx": 0.287 },
  "poses": {
    "gripper_se3_camera": { "rotation": [0.05, -0.02, 0.01, 0.998], "translation": [0.05, -0.03, 0.08] }
  }
}
```

Scalars sit under `parameters`: intrinsics, distortion coefficients, the
sensor tilt and laser-plane components, prefixed `cam<i>.` in a rig with more
than one camera. Poses sit under `poses` by their export names. `suite check`
compares every estimated parameter the truth names with its true value, in
units of the result's own standard deviation (a pose by the right
perturbation `T_true⁻¹ · T`), and fails the entry when one is 5σ or more
off. Each entry reports its largest error against the truth.

## Comparing with other libraries

The reference results are the problem's export, minus the per-feature residual
lists and the projected laser lines (they follow from the parameters and the
observations; the residual histograms and the reprojection statistics stay). When comparing them with
another tool's output:

- **Intrinsics** `k` holds `fx`, `fy`, `cx` and `cy` in pixels, in the
  same pixel convention as the observations.
- **Distortion** `dist` holds the Brown-Conrady coefficients by name; OpenCV's
  five-element `distCoeffs` is `(k1, k2, p1, p2, k3)` of the same model. Most
  configs fix `k3 = 0`.
- **Reprojection error**: `mean_reproj_error` and `per_cam_reproj_errors` are
  the mean Euclidean distance in pixels between observed and reprojected
  points. OpenCV's `calibrateCamera` returns the root-mean-square distance,
  which is larger for the same fit.
- **Poses** follow the `a_se3_b` convention above.
- **Scheimpflug tilt**: `tilt_x`, `tilt_y` are OpenCV's `τx`, `τy` of
  `CALIB_TILTED_MODEL`, with the same sign and the same place in the
  projection (after distortion, before `K`).

### OpenCV

`tools/compare/opencv/compare.py` calibrates every entry OpenCV has a method
for and writes `results/<id>/opencv.json`: the parameters under the labels of
our results, the poses under our export names, OpenCV's standard deviations
and its mean reprojection error. The OpenCV version is pinned in the
harness's `uv` project:

```bash
uv run --project tools/compare/opencv tools/compare/opencv/compare.py
cargo run --release -p vision-calibration-bench --bin calib-bench -- suite compare
```

| Problem | OpenCV |
|---|---|
| `planar_intrinsics` | `calibrateCameraExtended` with `CALIB_FIX_K3` |
| `planar_intrinsics` with a Scheimpflug sensor | the same with `CALIB_TILTED_MODEL` and `CALIB_ZERO_TANGENT_DIST`, the parameters our Scheimpflug model estimates (higher terms fixed) |
| `single_cam_handeye` | `calibrateCameraExtended`, then `calibrateRobotWorldHandEye` (Shah) |
| `rig_extrinsics` | `calibrateCameraExtended` per camera, then `stereoCalibrate` with `CALIB_FIX_INTRINSIC`, or refining the intrinsics when the entry does |
| `rig_handeye` | per camera as for `single_cam_handeye`; the rig frame is camera 0's |

OpenCV has no laser-plane calibration, so the laser entries have no
comparison. `suite compare` sets each result against our reference:
differences in units of our standard deviation, OpenCV's standard deviations
against ours, the mean reprojection error by the same definition, and for a
synthetic entry each library's largest error against the truth, in units of
our σ:

| Entry | Library | Parameters | Largest difference | Median | σ ratio (ours / theirs) | Mean reprojection (ours / theirs) | Truth (ours / theirs) |
|---|---|---|---|---|---|---|---|
| `kuka_1` | opencv 4.12.0 | 20 | 12.21σ (`base_se3_target.rz`) | 2.25σ | 0.878–0.993 | 0.1347 / 2.2545 px | — |
| `stereo_left` | opencv 4.12.0 | 8 | 0.02σ (`cy`) | 0.01σ | 1.000–1.000 | 0.2339 / 0.2339 px | — |
| `stereo_right` | opencv 4.12.0 | 8 | 0.01σ (`cy`) | 0.01σ | 1.000–1.000 | 0.2442 / 0.2442 px | — |
| `stereo_rig` | opencv 4.12.0 | 22 | 7.22σ (`cam1.cam_se3_rig.tz`) | 1.15σ | — | — | — |
| `ds8` | opencv 4.12.0 | 36 | 18.72σ (`cam2.p2`) | 1.98σ | 0.344–1.327 | — | — |
| `synth_planar` | opencv 4.12.0 | 8 | 0.01σ (`cx`) | 0.00σ | 1.000–1.000 | 0.6570 / 0.6570 px | 1.87σ / 1.86σ |
| `synth_scheimpflug` | opencv 4.12.0 | 8 | 0.01σ (`cx`) | 0.01σ | 1.000–1.000 | 0.6568 / 0.6568 px | 1.52σ / 1.53σ |
| `synth_handeye` | opencv 4.12.0 | 20 | 10.09σ (`base_se3_target.tx`) | 1.82σ | 0.303–0.631 | 0.6645 / 0.7521 px | 1.20σ / 10.46σ |
| `synth_rig` | opencv 4.12.0 | 22 | 0.09σ (`cam0.cy`) | 0.02σ | — | — | 2.07σ / 2.06σ |

Reading it:

- On plain and Scheimpflug intrinsics the two libraries agree within
  0.02σ, and OpenCV's standard deviations match ours to 0.1 %: both report
  the classical covariance for these fits. The Scheimpflug fit needs the same
  parameter set on both sides: with the tangential terms `p1`, `p2` free, a
  tilt and a decentring distortion explain the same images, and OpenCV's fit
  wanders along that valley (its own σ of `cy` is 40 px).
- Hand-eye results differ by design. Our hand-eye workflows refine the robot
  poses within their priors, so on `kuka_1` the reprojection error is the
  corner noise (0.13 px) rather than the robot's pose error (2.3 px with the
  measured poses). OpenCV takes the robot poses as exact for the hand-eye
  pose, but calibrates the intrinsics with a free pose per view, as the
  refined robot poses effectively are here: the intrinsics standard
  deviations of the two agree within 12 %. The synthetic hand-eye entry's
  robot poses are exact, and our solve estimates a robot error near zero
  from the corrections: it lands 1.2σ from the truth with standard
  deviations a third to two thirds of OpenCV's, whose hand-eye, from the
  same poses, is 10σ from the truth in our units.
- Rig results refine each camera's intrinsics jointly with the rig, from
  every view of every camera. On `stereo_rig` OpenCV's `stereoCalibrate`
  refines them too, but from the views both cameras see; the two estimates
  differ by up to 7σ, along the coupling between camera 0's focal length and
  camera 1's depth offset. On the synthetic rig the joint estimate lands
  within 2.1σ of the truth.
- `ds8` is a three-camera rig; OpenCV calibrates each camera's intrinsics
  alone and camera 0's hand-eye alone, while ours refines all three cameras
  with the rig and the hand-eye, so the difference includes what the joint
  solve moves.

## Drift tolerance

A number passes when it is within `abs + rel · max(|reference|, |result|)` of
its reference, with `rel = 1e-6` and `abs = 1e-9` unless the entry sets its own
`tolerance`. That flags any change in the solution while allowing the last-bit
differences between platforms and compilers. Strings, flags and the shape of
the result must match exactly.

With `--sigma k`, a parameter passes when its change is within `k` of the
reference's standard deviations. A pose's change is measured as the right
perturbation `T_ref⁻¹ · T` in the coordinates of its covariance. The two
solver backends agree within 0.02σ on every entry.

## Adding an entry

A synthetic entry names a scene of the solver benchmark's full matrix under
`synthetic`, by its id `<problem>/<sensor>/<scale>/n<σ>/<clean|huber|cauchy>`
(as `calib-bench solver run --preset full` reports them), with the config the
scene needs, and carries the CC0 licence. Run `suite freeze --only <id>` and
`suite bless --only <id>` and commit the observation, truth and result files
with `SHA256SUMS`. For an entry with images:

1. Check that the data's licence allows publishing observations derived from
   it, and record the licence (and citation, if the authors ask for one) in
   the entry.
2. Write a dataset manifest (`dataset.toml`, the file the desktop app's Run
   workspace reads; `data/kuka_1/dataset.toml` is an example) and an
   `[[entry]]` in `suite.toml` that points `manifest` at it.
3. Run `suite freeze --only <id>`, then `suite bless --only <id>`, and commit
   the observation file, `SHA256SUMS` and the reference result.
4. If the publisher serves the files without a login, give the entry a
   `fetch` source and list every file the manifest reads, with its SHA-256,
   in `fetch/<id>.sha256` (the `sha256sum` format, paths relative to the
   manifest's folder). `suite fetch --only <id>` and `suite freeze --only
   <id> --images target/suite-images` must then reproduce the observation
   file.
