# Laser Datasets

A laser-triangulation device captures **two frames per view**: a target frame (board visible, projector off) and a laser frame (projector on). The manifest describes both. The runner detects the target features, extracts the laser-line pixels (cached like detections) and assembles the input of the laser problem types: `LaserlineDevice`, `RigLaserlineDevice` and `RigHandeyeLaserline`. For the rig laser topology that includes the frozen upstream rig hand-eye calibration.

This chapter extends [The Dataset Manifest](manifest.md) with the laser-specific fields. The calibration workflows themselves are in [Laserline Device Calibration](laserline.md) and [Scheimpflug and Laser Rigs](scheimpflug_laser_rigs.md).

## How the Manifest Describes Laser Data

- `laser_images` per camera mirrors `images`: a glob or an explicit list. It pairs with the target views through the same `pose_pairing`: with `by_index` the i-th laser image pairs with the i-th target image, with `shared_filename_token` the same regex runs on both filename sets. A target view without a laser frame is a gap; a laser frame without a target view is an error.
- `[laser]` tunes the extraction: `scan_axis` (`cols` walks image columns, one point per column, for lines running mostly left to right; `rows` walks rows, for vertical lines), `sigma` (Gaussian derivative scale in pixels), `pos_thresh` and `neg_thresh` (edge-response thresholds) and `min_points` (the smallest usable extraction, at least 2). Omit the table for the defaults: extraction tuning has safe defaults and is not an ambiguity the runner refuses to guess.
- `RigLaserlineDevice` is a *second stage*: `upstream_calibration` points at a frozen `RigHandeyeExport` JSON. The per-view board poses are recomputed from the laser dataset's own robot poses through the frozen hand-eye chain, so the laser views need not match the upstream run's views.
- `RigHandeyeLaserline` is the joint solve: it computes its own hand-eye warm start and takes no `upstream_calibration`.

## Walkthrough: A Scheimpflug Rig, Eye-to-Hand

```toml
version = 1
topology = "rig_laserline_device"
upstream_calibration = "rig_handeye_export.json"

[target]
kind = "charuco"
rows = 12
cols = 9
square_size_m = 0.01
marker_size_m = 0.0075
dictionary = "DICT_4X4_1000"

[laser]
min_points = 120          # views below the bar become gap slots

[robot_poses]
path = "poses.json"
format = "json"
matrix_field = "tcp2base" # a nested 4x4 per row (vendor JSON)

[pose_convention]
transform = "t_base_tcp"
rotation_format = "matrix4x4_row_major"
translation_units = "mm"

[pose_pairing]
kind = "by_index"

[[cameras]]
id = "cam0"
roi_xywh = [0, 0, 720, 540]   # one tile of a multi-camera strip image
[cameras.images]
kind = "glob"
pattern = "target_*.png"
[cameras.laser_images]
kind = "glob"
pattern = "laser_*.png"
# ... one [[cameras]] entry per camera
```

Run the hand-eye stage first (`topology = "rig_handeye"`, the same cameras and poses, no laser fields) and save its export as `rig_handeye_export.json`. In the desktop app each stage is one Run click; such a setup needs non-default configuration (Scheimpflug sensors, eye-to-hand, `PointToPlane` laser residuals), set in the config form or in a preset.

A single-camera `LaserlineDevice` is the same minus `robot_poses` and `upstream_calibration`: it calibrates the intrinsics and the laser plane jointly from the target and laser frames alone.

## Library Consumers: Inject an Extractor

The published crates do not ship a laser-line extractor, and extraction is the one step of the runner that you provide. Implement the open `LaserPixelExtractor` trait and hand it to the converter:

```rust
use vision_calibration::dataset::LaserExtractionSpec;
use vision_calibration::dataset_runner::{LaserPixelExtractor, build_rig_laserline_device_input};

struct MyExtractor;

impl LaserPixelExtractor for MyExtractor {
    fn name(&self) -> &str { "my-extractor-v1" } // cache-key namespace
    fn extract(
        &self,
        image: &image::DynamicImage,
        spec: &LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        // subpixel laser-line points in the (ROI-cropped) image
        todo!()
    }
}

let run = build_rig_laserline_device_input(&spec, base_dir, Detection::Cached(&cache), &MyExtractor)?;
```

The runner owns the caching (keyed `laser:<name>` plus a hash of the extraction spec), the ROI cropping and the coordinate lifting; the extractor only extracts. Unlike the target detectors, whose trait is sealed, this one is open. `name()` is part of the cache key, so switching implementations re-extracts instead of serving another extractor's pixels.

### In Python

Any object with a `name` and an `extract(image, spec)` method is an extractor (the `LaserPixelExtractor` protocol of `vision_calibration.dataset_runner`). `extract` receives the cropped laser image as a `(H, W)` `uint8` grayscale array and the manifest's `[laser]` section as a dict, and returns the line's points as an `(N, 2)` array of `(x, y)` pixels:

```python
import numpy as np
import vision_calibration as vc


class ColumnPeak:
    name = "column-peak-v1"

    def extract(self, image, spec):
        """Per column, the row centroid of the pixels above half the peak."""
        values = image.astype(float)
        points = []
        for col in range(image.shape[1]):
            column = values[:, col]
            if column.max() < 50:
                continue
            weights = np.where(column > column.max() / 2, column, 0.0)
            points.append((col, weights @ np.arange(len(column)) / weights.sum()))
        return np.array(points).reshape(-1, 2)


result = vc.calibrate_dataset("data/my_laser_rig/dataset.toml", laser_extractor=ColumnPeak())
```

An exception the extractor raises fails the run with a `RunError` of `kind == "laser_extraction"`.

## Variations

- **Sparse laser coverage (rig)**: cameras may miss the laser in some views (`None` slots), but every camera needs at least one usable laser view; otherwise its plane is unconstrained and the runner fails fast.
- **Vertical laser lines**: `scan_axis = "rows"`.
- **Re-extraction**: edit `[laser]` and the cache key changes, so nothing stale is served. The same contract holds for detector configurations.
