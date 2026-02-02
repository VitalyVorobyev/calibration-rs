# Distortion Estimation from Homography Residuals

After estimating intrinsics $K$ via Zhang's method (which assumes distortion-free observations), the residuals between observed and predicted pixel positions encode the lens distortion. This chapter derives a linear method to estimate Brown-Conrady distortion coefficients from these residuals.

## Problem Statement

**Given**:
- Camera intrinsics $K$ (from Zhang's method)
- $M$ homographies $\{H_k\}$ from the calibration board to the image
- Point correspondences: board points $\{\mathbf{P}_j\}$ and observed pixels $\{\mathbf{p}_{kj}\}$

**Find**: Distortion coefficients $(k_1, k_2, k_3, p_1, p_2)$.

**Assumptions**:
- $K$ is a reasonable estimate (possibly biased by distortion)
- Distortion is moderate (the linear approximation holds)
- The homographies were computed from distorted pixel observations

## Derivation

### Ideal vs. Observed Coordinates

For each correspondence in view $k$, the **ideal** (undistorted) pixel position predicted by the homography is:

$$\tilde{\mathbf{p}} = \text{dehomogenize}(H_k \cdot [\mathbf{P}_j, 1]^T)$$

Convert both observed and ideal pixels to **normalized coordinates**:

$$\mathbf{n}_{\text{obs}} = K^{-1} \begin{bmatrix} \mathbf{p}_{\text{obs}} \\ 1 \end{bmatrix}, \quad \mathbf{n}_{\text{ideal}} = K^{-1} \begin{bmatrix} \tilde{\mathbf{p}} \\ 1 \end{bmatrix}$$

The residual in normalized coordinates encodes the distortion effect:

$$\Delta \mathbf{n} = \mathbf{n}_{\text{obs}} - \mathbf{n}_{\text{ideal}} \approx D(\mathbf{n}_{\text{ideal}}) - \mathbf{n}_{\text{ideal}}$$

### Linearized Distortion Model

The Brown-Conrady distortion applied to point $\mathbf{n} = [x, y]^T$ with $r^2 = x^2 + y^2$ produces a displacement:

$$\Delta x = x(k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2x^2)$$
$$\Delta y = y(k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2y^2) + 2 p_2 x y$$

This is **linear in the distortion coefficients** $(k_1, k_2, k_3, p_1, p_2)$.

### Building the Linear System

For each correspondence, write the $x$ and $y$ components:

$$\begin{bmatrix} x r^2 & x r^4 & x r^6 & 2xy & r^2 + 2x^2 \\ y r^2 & y r^4 & y r^6 & r^2 + 2y^2 & 2xy \end{bmatrix} \begin{bmatrix} k_1 \\ k_2 \\ k_3 \\ p_1 \\ p_2 \end{bmatrix} = \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$

where $x, y$ are the ideal normalized coordinates and $\Delta x, \Delta y$ are the observed residuals.

Stacking all correspondences from all views gives an overdetermined system:

$$A \mathbf{c} = \mathbf{b}$$

where $A$ is $2NM \times 5$ (or smaller if some coefficients are fixed), and $\mathbf{c} = [k_1, k_2, k_3, p_1, p_2]^T$.

### Solving

The least-squares solution is:

$$\mathbf{c} = (A^T A)^{-1} A^T \mathbf{b}$$

In practice, this is solved via SVD of $A$ for numerical stability.

## Options

```rust
pub struct DistortionFitOptions {
    pub fix_tangential: bool,  // Fix p1 = p2 = 0
    pub fix_k3: bool,          // Fix k3 = 0 (default: true)
    pub iters: u32,            // Undistortion iterations
}
```

- **`fix_k3`** (default `true`): Removes $k_3$ from the system, reducing to a 4-parameter or 2-parameter model. Recommended unless the lens has strong higher-order radial distortion.
- **`fix_tangential`** (default `false`): Removes $p_1, p_2$, estimating only radial distortion. Useful when tangential distortion is known to be negligible.

When parameters are fixed, the corresponding columns are removed from $A$.

## Accuracy

This linear estimate is typically within 10-50% of the true distortion values. The accuracy depends on:

- **Quality of $K$**: If intrinsics are biased (from distorted observations), the estimated distortion absorbs some of the bias
- **Number of points**: More correspondences improve the overdetermined system
- **Point distribution**: Points across the full image area constrain distortion better than points clustered near the center (where distortion is small)

This accuracy is sufficient for initializing non-linear refinement.

## API

```rust
let dist = estimate_distortion_from_homographies(
    &k_matrix, &views, opts
)?;
```

Returns `BrownConrady5` with the estimated coefficients.

> **OpenCV equivalence**: OpenCV estimates distortion jointly with intrinsics inside `cv::calibrateCamera`. The separate distortion estimation step is specific to calibration-rs's initialization approach.
