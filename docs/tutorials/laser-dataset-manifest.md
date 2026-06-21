# Laser topologies from a dataset manifest

How to describe a laser-triangulation dataset in `dataset.toml` so the
Run workspace (or the `dataset_runner` API) can calibrate
`LaserlineDevice` / `RigLaserlineDevice` end-to-end. Design record:
[ADR 0021](../adrs/0021-laser-frame-manifest.md).

## Why

A laser-triangulation device captures **two frames per view**: a target
frame (board, projector off) and a laser frame (projector on). The
manifest describes both; the runner detects target features, extracts
laser-line pixels (cached like detections), and assembles the problem
input — including, for the rig topology, the frozen upstream rig
hand-eye calibration.

## Mental model

- `laser_images` per camera mirrors `images` and pairs with target
  views through the same `pose_pairing` (`by_index`: i-th laser ↔ i-th
  target; `shared_filename_token`: same regex on both filename sets).
  A target view without a laser frame is a gap; a laser frame without
  a target view is an error.
- `[laser]` tunes extraction (`scan_axis`, `sigma`, thresholds,
  `min_points`). Omit it for defaults — extraction tuning is not a
  fail-fast ambiguity.
- `RigLaserlineDevice` is a *second stage*: `upstream_calibration`
  points at a frozen `RigHandeyeExport` JSON. Per-view board poses are
  recomputed from the laser dataset's own robot poses through the
  frozen hand-eye chain, so the laser views do not need to match the
  upstream run's views.

## Walkthrough — rig laserline (rtv3d shape)

```toml
version = 1
topology = "rig_laserline_device"
upstream_calibration = "rig_handeye_export.json"

[target]
kind = "charuco"
rows = 22
cols = 22
square_size_m = 0.0052
marker_size_m = 0.0039
dictionary = "DICT_4X4_1000"

[laser]
min_points = 120          # views below the bar become gap slots

[robot_poses]
path = "poses.json"
format = "json"
matrix_field = "tcp2base" # nested 4x4 per row (vendor JSON)

[pose_convention]
transform = "t_base_tcp"
rotation_format = "matrix4x4_row_major"
translation_units = "mm"

[pose_pairing]
kind = "by_index"

[[cameras]]
id = "cam0"
roi_xywh = [0, 0, 720, 540]   # tile 0 of the 6-camera strip
[cameras.images]
kind = "glob"
pattern = "target_*.png"
[cameras.laser_images]
kind = "glob"
pattern = "laser_*.png"
# … one [[cameras]] entry per tile …
```

Run the hand-eye stage first (`topology = "rig_handeye"`, same cameras
and poses, no laser fields) and save its export as
`rig_handeye_export.json`. In the app, both stages are one Run click
each; the rtv3d presets carry the right config overrides
(Scheimpflug sensors, EyeToHand, `PointToPlane` laser residuals).

Single-camera `LaserlineDevice` is the same minus `robot_poses` /
`upstream_calibration` — it calibrates intrinsics and the laser plane
jointly from the target + laser frames alone.

## Library consumers: inject an extractor

The published crates do not ship a laser extractor (the reference
implementation, `vision-metrology`, is not on crates.io). Implement
the open trait and hand it to the converter:

```rust
use vision_calibration_pipeline::dataset_runner::{
    LaserPixelExtractor, build_rig_laserline_device_input,
};

struct MyExtractor;
impl LaserPixelExtractor for MyExtractor {
    fn name(&self) -> &str { "my-extractor-v1" } // cache-key namespace
    fn extract(
        &self,
        image: &image::DynamicImage,
        spec: &vision_calibration_dataset::LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        // subpixel laser-line points in the (ROI-cropped) image
        todo!()
    }
}

let run = build_rig_laserline_device_input(
    &spec, base_dir, &cache, &MyExtractor, false,
)?;
```

The runner owns caching (keyed `laser:<name>` + extraction-spec hash),
ROI cropping, and coordinate lifting — the extractor only extracts.
The Tauri app's implementation is
[`app/src-tauri/src/laser.rs`](../../app/src-tauri/src/laser.rs).

## Common variations

- **Sparse laser coverage (rig):** cameras may miss the laser in some
  views (`None` slots), but every camera needs ≥1 usable laser view —
  otherwise its plane is unconstrained and the runner fails fast.
- **Vertical laser lines:** `scan_axis = "rows"` (one point per row).
- **Re-extraction:** edit `[laser]` and the cache key changes —
  nothing stale is served. Same contract as detector configs
  (ADR 0017).

## What to read next

- [ADR 0021](../adrs/0021-laser-frame-manifest.md) — design decisions
  and the publishing constraint behind the injected extractor.
- [ADR 0016](../adrs/0016-dataset-manifest.md) — the manifest itself.
- `app/src-tauri/src/run.rs` (`rtv3d_laser_end_to_end`) — the
  two-stage acceptance test over the rtv3d dataset.
