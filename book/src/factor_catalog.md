# Factor Catalog Reference

This chapter is a complete reference for all factor types (residual computations) supported by the optimization IR. Each factor defines a residual function connecting specific parameter blocks.

## Reprojection Factors

All reprojection factors compute the weighted pixel residual:

$$\mathbf{r} = w \cdot \left( \pi(\boldsymbol{\theta}, \mathbf{P}_w) - \mathbf{p}_{\text{obs}} \right) \in \mathbb{R}^2$$

They differ in the transform chain from world point to pixel.

### ReprojPointPinhole4

**Parameters**: `[intrinsics(4), pose(7)]`

**Transform chain**: $\mathbf{P}_c = T \cdot \mathbf{P}_w$, then pinhole projection with intrinsics only (no distortion).

**Use case**: Distortion-free cameras or when distortion is handled externally.

---

### ReprojPointPinhole4Dist5

**Parameters**: `[intrinsics(4), distortion(5), pose(7)]`

**Transform chain**: $\mathbf{P}_c = T \cdot \mathbf{P}_w$, then pinhole + Brown-Conrady distortion + intrinsics.

**Use case**: Standard single-camera calibration (planar intrinsics).

---

### ReprojPointPinhole4Dist5Scheimpflug2

**Parameters**: `[intrinsics(4), distortion(5), sensor(2), pose(7)]`

**Transform chain**: $\mathbf{P}_c = T \cdot \mathbf{P}_w$, then pinhole + distortion + Scheimpflug tilt + intrinsics.

**Use case**: Laser profilers with tilted sensors.

---

### ReprojPointPinhole4Dist5TwoSE3

**Parameters**: `[intrinsics(4), distortion(5), extrinsics(7), rig_pose(7)]`

**Transform chain**:

$$\mathbf{P}_c = T_{\text{extrinsics}} \cdot T_{\text{rig\_pose}} \cdot \mathbf{P}_w$$

The camera pose is composed from two SE(3) transforms: the camera-to-rig extrinsics and the rig-to-target pose.

**Use case**: Multi-camera rig calibration.

---

### ReprojPointPinhole4Dist5HandEye

**Parameters**: `[intrinsics(4), distortion(5), extrinsics(7), handeye(7), target_pose(7)]`

**Per-residual data**: `robot_pose` ($T_{B,G}$, known from robot kinematics)

**Transform chain** (eye-in-hand):

$$T_{C,T} = T_{\text{extrinsics}} \cdot T_{\text{handeye}}^{-1} \cdot T_{\text{robot}}^{-1} \cdot T_{\text{target}}$$

Then project $\mathbf{P}_c = T_{C,T} \cdot \mathbf{P}_w$.

For single-camera hand-eye, $T_{\text{extrinsics}} = I$.

**Use case**: Hand-eye calibration (camera on robot arm).

---

### ReprojPointPinhole4Dist5HandEyeRobotDelta

**Parameters**: `[intrinsics(4), distortion(5), extrinsics(7), handeye(7), target_pose(7), robot_delta(7)]`

**Per-residual data**: `robot_pose`

**Transform chain**: Same as HandEye, but with a per-view SE(3) correction $\Delta T$ applied to the robot pose:

$$T_{\text{robot}}' = T_{\text{robot}} \cdot \exp(\boldsymbol{\xi}_{\text{delta}})$$

The correction is regularized by an `Se3TangentPrior` factor.

**Use case**: Hand-eye calibration with imprecise robot kinematics (accounts for robot pose uncertainty).

---

## Laser Factors

Both laser factors have residual dimension 1 and connect `[intrinsics(4), distortion(5), pose(7), plane(3+1)]`.

### LaserPlanePixel

**Computation**:
1. Undistort laser pixel to normalized coordinates
2. Back-project to a ray in camera frame
3. Intersect ray with target plane (known from pose)
4. Compute 3D point in camera frame
5. Measure signed distance from 3D point to laser plane

**Residual**: $r = \sqrt{w} \cdot (\hat{\mathbf{n}}^T \mathbf{P}_c - d)$ — distance in meters.

---

### LaserLineDist2D

**Computation**:
1. Compute 3D intersection line of laser plane and target plane (in camera frame)
2. Project this line onto the $Z = 1$ normalized camera plane
3. Undistort laser pixel to normalized coordinates
4. Measure perpendicular distance from pixel to projected line (2D geometry)
5. Scale by $\sqrt{f_x \cdot f_y}$ for pixel-comparable units

**Residual**: $r = \sqrt{w} \cdot d_{\perp} \cdot \sqrt{f_x f_y}$ — distance in effective pixels.

**Advantages over LaserPlanePixel**: Undistortion done once per pixel (not per-iteration), residuals in pixel units (directly comparable to reprojection error), simpler 2D geometry.

**Default**: This is the recommended laser residual type.

---

## Prior Factors

### Se3TangentPrior

**Parameters**: `[se3_param(7)]`

**Computation**: Maps the SE(3) parameter to its tangent vector (via logarithm map) and divides by the prior standard deviations:

$$\mathbf{r} = \begin{bmatrix} \boldsymbol{\omega} / \sigma_{\text{rot}} \\ \mathbf{v} / \sigma_{\text{trans}} \end{bmatrix} \in \mathbb{R}^6$$

where $[\boldsymbol{\omega}, \mathbf{v}] = \log(T)$ is the 6D tangent vector.

**Use case**: Zero-mean Gaussian prior on SE(3) parameters. Applied to robot pose corrections (`robot_delta`) to penalize deviations from the nominal robot kinematics.

**Effect**: Adds a regularization term $\frac{1}{2} \left( \frac{\|\boldsymbol{\omega}\|^2}{\sigma_r^2} + \frac{\|\mathbf{v}\|^2}{\sigma_t^2} \right)$ to the cost function.

---

## Summary Table

| Factor | Params | Res. dim | Units | Domain |
|--------|--------|---------|-------|--------|
| ReprojPointPinhole4 | 2 blocks | 2 | pixels | Simple pinhole |
| ReprojPointPinhole4Dist5 | 3 blocks | 2 | pixels | Standard calibration |
| ReprojPointPinhole4Dist5Scheimpflug2 | 4 blocks | 2 | pixels | Tilted sensor |
| ReprojPointPinhole4Dist5TwoSE3 | 4 blocks | 2 | pixels | Multi-camera rig |
| ReprojPointPinhole4Dist5HandEye | 5 blocks | 2 | pixels | Hand-eye |
| ReprojPointPinhole4Dist5HandEyeRobotDelta | 6 blocks | 2 | pixels | Hand-eye + robot refinement |
| LaserPlanePixel | 4 blocks | 1 | meters | Laser triangulation |
| LaserLineDist2D | 4 blocks | 1 | pixels | Laser triangulation |
| Se3TangentPrior | 1 block | 6 | normalized | Regularization |
