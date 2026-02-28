# Stereo Rig with Real ChArUco Data

This chapter walks through the `stereo_charuco_session` example, which calibrates a stereo camera rig from synchronized ChArUco board images.

## Dataset

The example uses `data/stereo_charuco/`:

- `cam1/Cam1_*.png` for camera 0
- `cam2/Cam2_*.png` for camera 1
- Pairing by shared filename suffix (deterministic sorted intersection)

The loader ignores non-PNG files (for example `Thumbs.db`).

## ChArUco Target Parameters

- Board squares: `22 x 22`
- Cell size: `0.45 mm` (`0.00045 m`)
- Marker size scale: `0.75`
- Dictionary: `DICT_4X4_1000`
- Layout: OpenCV ChArUco (`OpenCvCharuco`)

## Detector Tuning

For this `2048x1536` dataset, the example increases ChArUco graph spacing:

- `params.graph.max_spacing_pix = 120.0`

Without this, the default graph spacing may be too strict for this board scale in image space.

## Workflow

The example follows the standard rig extrinsics 4-step session pipeline:

1. `step_intrinsics_init_all`
2. `step_intrinsics_optimize_all`
3. `step_rig_init`
4. `step_rig_optimize`

It also runs `run_calibration` as a convenience-path comparison.

## Running the Example

```bash
cargo run -p vision-calibration --example stereo_charuco_session
cargo run -p vision-calibration --example stereo_charuco_session -- --max-views 8
cargo run -p vision-calibration --example stereo_charuco_session -- --max-views=15
```

- Default: uses all detected stereo pairs
- Optional `--max-views` caps the number of pairs used (deterministic prefix of sorted pairs)

## Output Interpretation

The example prints:

- Dataset summary (`total_pairs`, `used_views`, `skipped_views`, `usable_left`, `usable_right`)
- Per-camera intrinsics and per-camera reprojection errors
- Rig BA mean reprojection error and per-camera BA errors
- Baseline magnitude in meters and millimeters

As with any real-data calibration, verify that:

- Both cameras have enough usable views (at least 3 each)
- Reprojection errors are within your application tolerance
- Estimated baseline is physically plausible for the hardware setup
