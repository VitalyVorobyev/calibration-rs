# Roadmap

Planned evolution of the project. We will update this as features land.

## Phase 1: solid linear core (in progress)
- Stabilize homography, PnP, epipolar, rig extrinsics, and hand–eye modules with robust tests.
- Add visualization-friendly report structures and numeric sanity checks.
- Publish the Rust book quickstart with sample datasets.

## Phase 2: richer optimisation backends
- Extend `calib-optim` with additional solvers (Dogleg, trust region) and damping strategies.
- Bundle-adjustment style problems for multi-view and multi-camera rigs.
- Improve parameter handling (manifolds, gauge fixing) and Jacobian benchmarks.

## Phase 3: pipelines and IO
- CLI subcommands for each pipeline; consistent JSON schemas and versioning.
- Dataset ingestion crate (`calib-io`) for common formats and detector outputs.
- Calibration report generation (metrics, plots, HTML/PDF-friendly data).

## Phase 4: advanced models
- Laserline calibration models, rolling shutter support, and timing calibration.
- Cross-sensor calibration: stereo, LiDAR-camera, IMU-camera, projector-camera.
- Robustness: outlier rejection, temporal filtering, and uncertainty estimates.

## Phase 5: ecosystem & distribution
- Benchmarks and regression tests on public datasets.
- `no_std`/embedded-friendly math where feasible.
- Optional C/FFI bindings and Python wheels for integration.

> Feedback and contributions are welcome—open an issue with proposals or experiment reports.
