# Laserline with Industrial Data

> **[COLLAB]** This chapter requires user collaboration for industrial application context, data collection protocols, and accuracy requirements.

This chapter discusses practical considerations for laser triangulation calibration in industrial settings.

## Industrial Context

<!-- [COLLAB]: Describe typical industrial applications -->

Laser triangulation is widely used in industrial inspection:

- **Weld seam inspection**: Verifying weld bead geometry (width, height, penetration)
- **3D surface profiling**: Measuring surface topology for quality control
- **Gap and flush measurement**: Checking assembly tolerances in automotive manufacturing
- **Tire and rubber inspection**: Measuring tread depth and sidewall profiles

In each case, a laser line is projected onto the workpiece and a camera observes the deformed line. The calibrated laser plane, combined with camera parameters, allows converting pixel positions to metric 3D coordinates.

## Accuracy Requirements

<!-- [COLLAB]: Provide typical accuracy requirements for different applications -->

| Application | Typical accuracy needed |
|-------------|----------------------|
| Weld inspection | 0.1-0.5 mm |
| Surface profiling | 0.01-0.1 mm |
| Gap/flush | 0.05-0.2 mm |
| Coarse 3D scanning | 0.5-2 mm |

## Data Collection for Laser Calibration

<!-- [COLLAB]: Describe the data collection procedure in detail -->

### Calibration Board Setup

The calibration board serves dual purpose:
1. Provides chessboard corners for camera intrinsics calibration
2. Provides a known plane to constrain the laser line observations

### Capturing Views

For each view:
1. Position the calibration board at a known angle to the laser plane
2. Capture an image with both the chessboard pattern and the laser line visible
3. Record chessboard corners and laser line pixel positions separately

### View Diversity

- **Board angles**: Vary the board tilt so the laser-board intersection line covers different positions in the image
- **Board distances**: Vary the distance to cover the working range
- **Minimum views**: 5-10 views with diverse orientations

## The `laserline_device_session` Example

```bash
cargo run -p vision-calibration --example laserline_device_session
```

This example uses synthetic data to demonstrate the workflow. For real data, the laser pixel extraction must be performed by an external algorithm (typically a peak detector applied to each image column).

## Interpreting Results

<!-- [COLLAB]: Discuss how to interpret and validate laser calibration results -->

Key outputs:

- **Laser plane normal**: The direction the laser projects in. Should align with the physical mounting.
- **Laser plane distance**: Distance from camera center to the laser plane. Should match the physical geometry.
- **Mean laser error**: The average residual of laser observations. In pixel units for `LineDistNormalized`, in meters for `PointToPlane`.

## DS8 Dataset

<!-- [COLLAB]: Describe the DS8 dataset if available for examples -->

The `data/DS8/` directory contains a real industrial laser triangulation dataset:

- `ExperimentDetails.txt`: Experiment metadata
- `calibration_object.txt`: Calibration pattern specification
- `images/`: Captured images
- `robot_cali.txt`: Robot calibration data (if applicable)

<!-- [COLLAB]: Walk through this dataset and provide example calibration results -->

## Troubleshooting

<!-- [COLLAB]: Add troubleshooting based on industrial experience -->

- **Poor laser plane estimate**: Usually caused by insufficient view diversity or laser pixels too close to collinear (all from similar board angles)
- **High laser residuals**: May indicate laser pixel detection errors, or that the laser is not well-described by a single plane (diverging laser, curved projection)
- **Scheimpflug convergence**: If sensor tilt parameters diverge, try initializing with `fix_sensor: true` for the first few iterations, then release
