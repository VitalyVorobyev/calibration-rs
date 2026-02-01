# Brown-Conrady Distortion

Real lenses introduce geometric distortion: straight lines in the world project as curves in the image. The Brown-Conrady model captures the two dominant distortion effects — radial and tangential — as polynomial functions of the distance from the optical axis.

## Distortion Model

Given undistorted normalized coordinates $\mathbf{n}_u = [x, y]^T$, the distorted coordinates $\mathbf{n}_d = [x_d, y_d]^T$ are:

$$r^2 = x^2 + y^2$$

**Radial distortion** (barrel/pincushion):

$$\alpha = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6$$

**Tangential distortion** (decentering):

$$\delta_x = 2 p_1 x y + p_2 (r^2 + 2x^2)$$
$$\delta_y = p_1 (r^2 + 2y^2) + 2 p_2 x y$$

**Combined**:

$$x_d = x \cdot \alpha + \delta_x$$
$$y_d = y \cdot \alpha + \delta_y$$

The model has five parameters: radial coefficients $(k_1, k_2, k_3)$ and tangential coefficients $(p_1, p_2)$.

> **OpenCV equivalence**: This is identical to OpenCV's `distortPoints` with the 5-parameter model `(k1, k2, p1, p2, k3)`. Note OpenCV's parameter ordering differs from the mathematical ordering.

## Physical Interpretation

- **$k_1 > 0$**: barrel distortion (lines bow outward from center)
- **$k_1 < 0$**: pincushion distortion (lines bow inward)
- **$k_2, k_3$**: higher-order radial corrections
- **$p_1, p_2$**: tangential distortion from imperfect lens-sensor alignment (lens elements not perfectly centered)

Typical values for industrial cameras: $|k_1| \sim 0.01\text{-}0.3$, $|k_2| \sim 0.001\text{-}0.1$, $|p_1|, |p_2| \sim 0.0001\text{-}0.01$.

## When to Fix $k_3$

The sixth-order radial term $k_3 r^6$ is only significant for wide-angle lenses where $r$ reaches large values. For typical industrial cameras with moderate field of view:

- $k_3$ is poorly constrained by the data and often absorbs noise
- Estimating $k_3$ can cause **overfitting**, leading to worse extrapolation outside the calibration region
- The library defaults to `fix_k3: true` in most problem configurations

**Recommendation**: Only estimate $k_3$ with high-quality data covering the full image area, or for lenses with field of view > 90°.

## Undistortion (Inverse)

Given distorted coordinates $\mathbf{n}_d$, recovering undistorted coordinates $\mathbf{n}_u$ requires inverting the distortion model. Since the forward model is a polynomial without a closed-form inverse, calibration-rs uses **iterative fixed-point refinement**:

$$\mathbf{n}_u^{(0)} = \mathbf{n}_d$$

$$\mathbf{n}_u^{(k+1)} = \mathbf{n}_d - \left( D(\mathbf{n}_u^{(k)}) - \mathbf{n}_u^{(k)} \right)$$

This rearranges the distortion equation to isolate $\mathbf{n}_u$ and iterates to convergence. The default is 8 iterations, which provides accuracy well below sensor noise for typical distortion magnitudes.

### Convergence

The fixed-point iteration converges when the Jacobian of the distortion residual has spectral radius less than 1, which holds for physically realistic distortion values (small $k_i$, $p_i$ relative to $r$). For extreme distortion, more iterations may be needed via the `iters` parameter.

> **OpenCV equivalence**: `cv::undistortPoints` performs the same iterative inversion.

## The `DistortionModel` Trait

```rust
pub trait DistortionModel<S: RealField> {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S>;
    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S>;
}
```

The `BrownConrady5<S>` struct:

```rust
pub struct BrownConrady5<S: RealField> {
    pub k1: S, pub k2: S, pub k3: S,
    pub p1: S, pub p2: S,
    pub iters: u32,   // undistortion iterations (default: 8)
}
```

`NoDistortion` is the identity implementation for distortion-free cameras.

## Distortion in the Projection Pipeline

Distortion operates in **normalized image coordinates** (after pinhole projection, before intrinsics):

$$\text{pixel} = K(\underbrace{D(\Pi(\mathbf{P}_c))}_{\text{distorted normalized}})$$

This is the standard convention: distortion is defined on the $Z = 1$ plane, not in pixel space. The advantage is that distortion parameters are independent of image resolution and focal length.
