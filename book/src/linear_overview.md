# Why Linear Initialization Matters

Non-linear optimization is the workhorse of camera calibration: it achieves sub-pixel reprojection error by jointly optimizing all parameters. But non-linear optimizers — Levenberg-Marquardt, Gauss-Newton, and their variants — are local methods. They converge to the nearest local minimum, which is only the global minimum if the starting point is sufficiently close.

**The role of linear initialization is to provide that starting point.**

## The Init-Then-Refine Paradigm

Every calibration workflow in calibration-rs follows a two-phase pattern:

1. **Linear initialization**: Closed-form solvers estimate approximate parameters. These are fast, deterministic, and require no initial guess — but they achieve only ~5-40% accuracy on camera parameters.

2. **Non-linear refinement**: Levenberg-Marquardt bundle adjustment minimizes reprojection error starting from the linear estimate. This converges to <2% parameter accuracy and <1 px reprojection error.

The linear estimate does not need to be highly accurate. It needs to be in the **basin of convergence** of the non-linear optimizer — close enough that gradient-based iteration converges to the correct solution rather than a local minimum.

## Why Not Just Optimize Directly?

Without initialization, you would need to guess intrinsics ($f_x, f_y, c_x, c_y$), distortion ($k_1, k_2, p_1, p_2$), and all camera poses ($R, \mathbf{t}$ per view). For a typical 20-view calibration:

- 4 intrinsics + 5 distortion + 20 × 6 pose = **129 parameters**
- The cost landscape has many local minima
- A random starting point will almost certainly diverge or converge to a wrong solution

Linear initialization eliminates this problem by computing a reasonable estimate from the data structure alone.

## Linear Solvers in calibration-rs

| Solver | Estimates | Method | Chapter |
|--------|-----------|--------|---------|
| Homography DLT | $3\times3$ homography | SVD nullspace | [Homography](homography.md) |
| Zhang's method | Intrinsics $K$ | IAC constraints | [Zhang](zhang.md) |
| Distortion fit | $k_1, k_2, k_3, p_1, p_2$ | Linear LS on residuals | [Distortion Fit](distortion_fit.md) |
| Iterative intrinsics | $K$ + distortion | Alternating refinement | [Iterative](iterative_intrinsics.md) |
| Planar pose | $R, \mathbf{t}$ from $H$ | Decomposition + SVD | [Planar Pose](planar_pose.md) |
| P3P (Kneip) | $R, \mathbf{t}$ from 3 points | Quartic polynomial | [PnP](pnp.md) |
| DLT PnP | $R, \mathbf{t}$ from $\geq 6$ points | SVD + SO(3) projection | [PnP](pnp.md) |
| 8-point fundamental | $F$ matrix | SVD + rank constraint | [Epipolar](epipolar.md) |
| 7-point fundamental | $F$ matrix | Cubic polynomial | [Epipolar](epipolar.md) |
| 5-point essential | $E$ matrix | Action matrix eigenvalues | [Epipolar](epipolar.md) |
| Camera matrix DLT | $P = K[R\|t]$ | SVD + RQ decomposition | [Camera Matrix](camera_matrix.md) |
| Triangulation | 3D point | SVD nullspace | [Triangulation](triangulation.md) |
| Tsai-Lenz | Hand-eye $T_{G,C}$ | Quaternion + linear LS | [Hand-Eye](handeye_linear.md) |
| Rig extrinsics | $T_{R,C}$ per camera | SE(3) averaging | [Rig Init](rig_init.md) |
| Laser plane | Plane normal + distance | SVD on covariance | [Laser Init](laser_init.md) |

## Common Mathematical Tools

Most linear solvers share a common toolkit:

- **SVD (Singular Value Decomposition)**: Solves homogeneous systems $A\mathbf{x} = 0$ via the right singular vector corresponding to the smallest singular value. This is the core of DLT methods.
- **Hartley normalization**: Conditions the DLT system for numerical stability by centering and scaling the input points.
- **Polynomial root finding**: Minimal solvers (P3P, 7-point F, 5-point E) reduce to polynomial equations.

The following chapters derive each algorithm in detail.
