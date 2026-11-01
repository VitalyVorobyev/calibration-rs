# Summary

[Introduction](introduction.md)
[Architecture Overview](architecture.md)

---

# Getting Started

- [Getting Started](getting_started.md)
- [Stereo Rig with Real Data](stereo_real_data.md)

---

# Calibration Workflows

- [Planar Intrinsics Calibration](planar_intrinsics.md)
- [Planar Intrinsics with Real Data](planar_real_data.md)
- [Choosing a Distortion Model](distortion_selection.md)
- [Single-Camera Hand-Eye](handeye_workflow.md)
- [Hand-Eye with KUKA Robot](handeye_real_data.md)
- [Multi-Camera Rig Extrinsics](rig_extrinsics.md)
- [Multi-Camera Rig Hand-Eye](rig_handeye.md)
- [Laserline Device Calibration](laserline.md)
- [Scheimpflug and Laser Rigs](scheimpflug_laser_rigs.md)
- [Seeding with Prior Knowledge](manual_init.md)
- [Per-Feature Residuals](per_feature_residuals.md)

---

# The Session Framework

- [CalibrationSession](session.md)
- [ProblemType Trait](problem_type.md)
- [Step Functions vs Pipeline Functions](step_functions.md)

---

# Data and Detection

- [Calibration Targets and Detection](detection.md)
- [The Dataset Manifest](manifest.md)
- [Laser Datasets](manifest_laser.md)
- [Manifest Field Reference](manifest_reference.md)

---

# Uncertainty

- [Parameter Uncertainty](uncertainty.md)
- [Reading the Covariance](uncertainty_reading.md)
- [Per-Point Pixel Covariance](uncertainty_point_covariance.md)
- [Projection Uncertainty](uncertainty_projection.md)
- [Planning the Next View](uncertainty_next_view.md)
- [Limits and Unavailable Results](uncertainty_limits.md)
- [Judging a Result](quality.md)

---

# Multiple-View Geometry

- [Multiple-View Geometry](mvg.md)
- [Pose Recovery and Triangulation](mvg_pose.md)
- [Bundle Adjustment](mvg_bundle_adjust.md)
- [Stereo Rectification](mvg_rectification.md)
- [Dense Stereo](mvg_dense.md)

---

# The Desktop App

- [The Desktop App](app.md)
- [Running Calibrations](app_run.md)
- [Inspecting Results](app_inspect.md)

---

# Theory: Camera Model

- [Composable Camera Pipeline](camera_pipeline.md)
- [Pinhole Projection](pinhole.md)
- [Brown-Conrady Distortion](distortion.md)
- [Fisheye Projection](fisheye.md)
- [Intrinsics Matrix](intrinsics.md)
- [Sensor Models and Scheimpflug Tilt](sensor.md)
- [Serialization and Runtime-Dynamic Types](params.md)

---

# Theory: Geometric Primitives and Robust Estimation

- [Rigid Transformations and SE(3)](rigid_transforms.md)
- [RANSAC](ransac.md)
- [Robust Loss Functions](robust_loss.md)

---

# Theory: Linear Initialization

- [Why Linear Initialization Matters](linear_overview.md)
- [Hartley Normalization](hartley.md)
- [Homography Estimation (DLT)](homography.md)
- [Zhang's Intrinsics from Homographies](zhang.md)
- [Distortion Estimation from Homography Residuals](distortion_fit.md)
- [Iterative Intrinsics + Distortion](iterative_intrinsics.md)
- [Pose from Homography](planar_pose.md)
- [Fisheye Intrinsics from Bearings](fisheye_init.md)
- [Perspective-n-Point Solvers](pnp.md)
- [Epipolar Geometry](epipolar.md)
- [Camera Matrix and RQ Decomposition](camera_matrix.md)
- [Linear Triangulation](triangulation.md)
- [Hand-Eye Calibration (Tsai-Lenz)](handeye_linear.md)
- [Multi-Camera Rig Initialization](rig_init.md)
- [Laser Plane Initialization](laser_init.md)
- [Polynomial Solvers](polynomial.md)

---

# Theory: Non-Linear Optimization

- [Non-Linear Least Squares Overview](nlls_overview.md)
- [Manifold Optimization](manifolds.md)
- [Backend-Agnostic IR Architecture](ir_architecture.md)
- [Autodiff and Generic Residual Functions](autodiff.md)
- [Solver Backends](solver_backends.md)
- [Factor Catalog Reference](factor_catalog.md)

---

# Reference and Development

- [Regression Suite](regression_suite.md)
- [Adding a New Optimization Problem](new_problem.md)
- [Adding a New Pipeline Problem Type](new_pipeline.md)
- [Synthetic Data Generation](synthetic.md)
- [Data Types Quick Reference](types_reference.md)
- [References and Further Reading](references.md)
