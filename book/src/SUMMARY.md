# Summary

[Introduction](introduction.md)
[Quickstart](quickstart.md)
[Architecture Overview](architecture.md)

---

# Camera Model

- [Composable Camera Pipeline](camera_pipeline.md)
- [Pinhole Projection](pinhole.md)
- [Brown-Conrady Distortion](distortion.md)
- [Intrinsics Matrix](intrinsics.md)
- [Sensor Models and Scheimpflug Tilt](sensor.md)
- [Serialization and Runtime-Dynamic Types](params.md)

---

# Geometric Primitives and Robust Estimation

- [Rigid Transformations and SE(3)](rigid_transforms.md)
- [RANSAC](ransac.md)
- [Robust Loss Functions](robust_loss.md)

---

# Linear Initialization Algorithms

- [Why Linear Initialization Matters](linear_overview.md)
- [Hartley Normalization](hartley.md)
- [Homography Estimation (DLT)](homography.md)
- [Zhang's Intrinsics from Homographies](zhang.md)
- [Distortion Estimation from Homography Residuals](distortion_fit.md)
- [Iterative Intrinsics + Distortion](iterative_intrinsics.md)
- [Pose from Homography](planar_pose.md)
- [Perspective-n-Point Solvers](pnp.md)
- [Epipolar Geometry](epipolar.md)
- [Camera Matrix and RQ Decomposition](camera_matrix.md)
- [Linear Triangulation](triangulation.md)
- [Hand-Eye Calibration (Tsai-Lenz)](handeye_linear.md)
- [Multi-Camera Rig Initialization](rig_init.md)
- [Laser Plane Initialization](laser_init.md)
- [Polynomial Solvers](polynomial.md)

---

# Non-Linear Optimization

- [Non-Linear Least Squares Overview](nlls_overview.md)
- [Manifold Optimization](manifolds.md)
- [Backend-Agnostic IR Architecture](ir_architecture.md)
- [Autodiff and Generic Residual Functions](autodiff.md)
- [Levenberg-Marquardt Backend](lm_backend.md)
- [Factor Catalog Reference](factor_catalog.md)

---

# Calibration Workflows

- [Planar Intrinsics Calibration](planar_intrinsics.md)
- [Planar Intrinsics with Real Data](planar_real_data.md)
- [Single-Camera Hand-Eye](handeye_workflow.md)
- [Hand-Eye with KUKA Robot](handeye_real_data.md)
- [Multi-Camera Rig Extrinsics](rig_extrinsics.md)
- [Stereo Rig with Real Data](stereo_real_data.md)
- [Multi-Camera Rig Hand-Eye](rig_handeye.md)
- [Laserline Device Calibration](laserline.md)
- [Laserline with Industrial Data](laserline_real_data.md)

---

# Session Framework

- [CalibrationSession](session.md)
- [ProblemType Trait](problem_type.md)
- [Step Functions vs Pipeline Functions](step_functions.md)

---

# Extending the Library

- [Adding a New Optimization Problem](new_problem.md)
- [Adding a New Pipeline Problem Type](new_pipeline.md)
- [Adding a New Solver Backend](new_backend.md)

---

# Appendices

- [Synthetic Data Generation](synthetic.md)
- [Data Types Quick Reference](types_reference.md)
- [References and Further Reading](references.md)
