# calib-cli

Command-line interface for `calibration-rs` pipelines.

Currently supports planar intrinsics calibration with JSON input/output.

## Installation

```bash
cargo install --path crates/calib-cli
# or run directly:
cargo run -p calib-cli -- --help
```

## Usage

```bash
calib-cli --input views.json [--config config.json] > report.json
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to JSON file containing calibration views |
| `--config` | No | Path to JSON configuration file (uses defaults if omitted) |

Output is written to stdout as JSON.

## Input Format

The input JSON file contains calibration views with 3D-2D point correspondences:

```json
{
  "views": [
    {
      "points_3d": [
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.08, 0.0, 0.0]
      ],
      "points_2d": [
        [320.5, 240.2],
        [380.1, 239.8],
        [440.3, 239.5]
      ],
      "weights": null
    }
  ]
}
```

Each view represents one image of the calibration pattern:
- `points_3d`: 3D coordinates on the calibration pattern (in meters)
- `points_2d`: Corresponding 2D pixel coordinates in the image
- `weights`: Optional per-point weights (null for uniform weighting)

## Config Format

Optional configuration for optimization:

```json
{
  "solve_opts": {
    "robust_loss": { "Huber": { "scale": 2.0 } },
    "fix_intrinsics": { "fx": false, "fy": false, "cx": false, "cy": false },
    "fix_distortion": { "k1": false, "k2": false, "k3": true, "p1": false, "p2": false },
    "fix_poses": []
  },
  "backend_opts": {
    "max_iters": 100,
    "function_tol": 1e-8,
    "gradient_tol": 1e-10,
    "param_tol": 1e-12,
    "verbosity": 0
  }
}
```

### Robust Loss Options

- `"None"` - No robust loss (least squares)
- `{ "Huber": { "scale": 2.0 } }` - Huber loss
- `{ "Cauchy": { "scale": 3.0 } }` - Cauchy loss
- `{ "Arctan": { "scale": 1.0 } }` - Arctan loss

## Output Format

The report contains estimated camera parameters:

```json
{
  "camera": {
    "projection": "Pinhole",
    "distortion": {
      "BrownConrady5": {
        "params": { "k1": -0.1, "k2": 0.01, "k3": 0.0, "p1": 0.001, "p2": -0.001, "iters": 8 }
      }
    },
    "sensor": "Identity",
    "intrinsics": {
      "FxFyCxCySkew": {
        "params": { "fx": 800.0, "fy": 795.0, "cx": 640.0, "cy": 360.0, "skew": 0.0 }
      }
    }
  },
  "final_cost": 0.00012345
}
```

## Example Workflow

1. **Detect corners** in calibration images using your preferred method
2. **Create input JSON** with 3D pattern coordinates and detected 2D points
3. **Run calibration**:
   ```bash
   calib-cli --input my_views.json > calibration_result.json
   ```
4. **Use results** in your application

## Error Handling

Errors are printed to stderr with a non-zero exit code:

```bash
$ calib-cli --input missing.json
error: No such file or directory (os error 2)
$ echo $?
1
```

## See Also

- [calib-pipeline](../calib-pipeline): Underlying pipeline implementation
- [Book: CLI Usage](../../book/src/cli.md): Detailed tutorial
