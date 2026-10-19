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
| `kuka_1` | `single_cam_handeye` | 30 eye-in-hand views of a 17×28 chessboard (20 mm) from a KUKA arm | CC BY 4.0 |
| `stereo_left` | `planar_intrinsics` | 20 views of a 7×11 chessboard (30 mm), left camera | CC BY 4.0 |
| `stereo_right` | `planar_intrinsics` | The same 20 views, right camera | CC BY 4.0 |
| `stereo_rig` | `rig_extrinsics` | Both cameras of the stereo pair together | CC BY 4.0 |
| `ds8` | `rig_handeye` | Three cameras on a Denso arm, 33 of 42 poses with a complete 10×14 chessboard (52 mm) | U.S. public domain |

`ds8` is dataset 8 of Tabb and Ahmad Yousef, *Solving the robot-world
hand-eye(s) calibration problem with iterative methods*, Machine Vision and
Applications 28 (2017); the images are published at
<https://doi.org/10.15482/USDA.ADC/1340592>.

## Layout

```text
regression/
├── suite.toml                          entries: description, licence, config
├── SHA256SUMS                          checksums of the observation files
├── observations/<id>.json              the entry's observations
└── results/<id>/calibration-rs.json    the entry's reference result
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
reference. Each entry reports `PASS` or `FAIL` with the values that moved; the
command exits non-zero when any entry fails. `--only <id>` selects entries.

Two more subcommands maintain the suite:

- `suite bless` recalibrates the selected entries and overwrites their
  reference results. Run it when a change is meant to move the results, and
  review the diff of `results/` like any other code change.
- `suite freeze` (built with `--features tier-b`) re-extracts an entry's
  observations from the images its dataset manifest names and rewrites
  `SHA256SUMS`. Observations change only when the detector does.

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

## Comparing with other libraries

The reference results are the problem's export, minus the per-feature residual
lists (they follow from the parameters and the observations; the residual
histograms and the reprojection statistics stay). When comparing them with
another tool's output:

- **Intrinsics** `k` holds `fx`, `fy`, `cx`, `cy` and `skew` in pixels, in the
  same pixel convention as the observations.
- **Distortion** `dist` holds the Brown-Conrady coefficients by name; OpenCV's
  five-element `distCoeffs` is `(k1, k2, p1, p2, k3)` of the same model. Most
  configs fix `k3 = 0`.
- **Reprojection error**: `mean_reproj_error` and `per_cam_reproj_errors` are
  the mean Euclidean distance in pixels between observed and reprojected
  points. OpenCV's `calibrateCamera` returns the root-mean-square distance,
  which is larger for the same fit.
- **Poses** follow the `a_se3_b` convention above.

## Drift tolerance

A number passes when it is within `abs + rel · max(|reference|, |result|)` of
its reference, with `rel = 1e-6` and `abs = 1e-9` unless the entry sets its own
`tolerance`. That flags any change in the solution while allowing the last-bit
differences between platforms and compilers. Strings, flags and the shape of
the result must match exactly.

## Adding an entry

1. Check that the data's licence allows publishing observations derived from
   it, and record the licence (and citation, if the authors ask for one) in
   the entry.
2. Write a dataset manifest (`dataset.toml`, the file the desktop app's Run
   workspace reads; `data/kuka_1/dataset.toml` is an example) and an
   `[[entry]]` in `suite.toml` that points `manifest` at it.
3. Run `suite freeze --only <id>`, then `suite bless --only <id>`, and commit
   the observation file, `SHA256SUMS` and the reference result.
