# Iterative Intrinsics + Distortion

Zhang's method assumes distortion-free observations. When applied to images from a camera with significant distortion, the estimated intrinsics are biased because the distorted pixels violate the linear homography model. The iterative intrinsics solver addresses this by alternating between intrinsics estimation and distortion correction.

## Problem Statement

**Given**: $M$ views of a planar calibration board, with observed (distorted) pixel coordinates.

**Find**: Camera intrinsics $K$ and distortion coefficients $(k_1, k_2, k_3, p_1, p_2)$ jointly.

**Assumptions**:
- The observations are distorted (raw detector output)
- Distortion is moderate enough that Zhang's method on distorted pixels gives a usable initial $K$
- 1-3 iterations of alternating refinement suffice

## Why Alternation Works

The joint estimation of $K$ and distortion is a non-convex problem. However, it decomposes naturally:

- **Given $K$**: Distortion estimation is a linear problem (see [Distortion Fit](distortion_fit.md))
- **Given distortion**: Undistorting the pixels and re-estimating $K$ is a linear problem (Zhang's method)

Each step solves a convex subproblem. The alternation converges because:

1. The initial Zhang estimate (ignoring distortion) is typically within the basin where the alternation contracts
2. Each step reduces the residual between the model and observations
3. The coupling between $K$ and distortion is relatively weak for moderate distortion

## Algorithm

$$\boxed{\text{Iterative Intrinsics Estimation}}$$

**Input**: Views $\{(\mathbf{P}_j, \mathbf{p}_{kj})\}$ with distorted observations $\mathbf{p}_{kj}$

**Output**: Intrinsics $K$, distortion $\mathbf{d}$

1. Compute homographies $\{H_k\}$ from distorted pixels via DLT
2. Estimate initial $K^{(0)}$ from $\{H_k\}$ via Zhang's method

3. **For** $i = 1, \ldots, n_{\text{iter}}$:

   a. Estimate distortion $\mathbf{d}^{(i)}$ from homography residuals using $K^{(i-1)}$

   b. Undistort all observed pixels: $\hat{\mathbf{p}}_{kj} = \text{undistort}(\mathbf{p}_{kj}, K^{(i-1)}, \mathbf{d}^{(i)})$

   c. Recompute homographies $\{H_k^{(i)}\}$ from undistorted pixels

   d. Re-estimate $K^{(i)}$ from $\{H_k^{(i)}\}$ via Zhang's method

   e. (Optional) Enforce zero skew: $K^{(i)}_{12} = 0$

4. **Return** $K^{(n_{\text{iter}})}, \mathbf{d}^{(n_{\text{iter}})}$

## Convergence

Typically **1-2 iterations** suffice:

- **Iteration 0** (Zhang on distorted pixels): 10-40% intrinsics error
- **Iteration 1**: Distortion estimate corrects the dominant radial effect; intrinsics error drops to 5-20%
- **Iteration 2**: Further refinement; diminishing returns

More iterations are safe but rarely improve the estimate significantly. The default is 2 iterations.

## Configuration

```rust
pub struct IterativeIntrinsicsOptions {
    pub iterations: usize,                    // 1-3 typical (default: 2)
    pub distortion_opts: DistortionFitOptions, // Controls fix_k3, fix_tangential, iters
    pub zero_skew: bool,                      // Force skew = 0 (default: true)
}
```

## The Undistortion Step

Step 3b converts distorted pixels back to "ideal" pixels by inverting the distortion model:

1. Convert distorted pixel to normalized coordinates: $\mathbf{n}_d = K^{-1} [\mathbf{p}, 1]^T$
2. Undistort using the current distortion estimate: $\mathbf{n}_u = D^{-1}(\mathbf{n}_d)$ (iterative fixed-point, see [Brown-Conrady Distortion](distortion.md))
3. Convert back to pixel: $\hat{\mathbf{p}} = K \cdot [\mathbf{n}_u, 1]^T$

Note that this uses $K$ both for normalization and for converting back — the undistorted pixels are in the same coordinate system as the original distorted pixels, just without the distortion effect.

## Accuracy Expectations

| Stage | Typical intrinsics error |
|-------|------------------------|
| Zhang on distorted pixels | 10-40% |
| After 1 iteration | 5-20% |
| After 2 iterations | 5-15% |
| After non-linear refinement | <2% |

The iterative linear estimate is not meant to be highly accurate. Its purpose is to provide a starting point good enough for the non-linear optimizer to converge.

## API

```rust
use vision_calibration::linear::iterative_intrinsics::*;
use vision_calibration::linear::DistortionFitOptions;

let opts = IterativeIntrinsicsOptions {
    iterations: 2,
    distortion_opts: DistortionFitOptions {
        fix_k3: true,
        fix_tangential: false,
        iters: 8,
    },
    zero_skew: true,
};

let camera = estimate_intrinsics_iterative(&dataset, opts)?;
// camera is PinholeCamera = Camera<f64, Pinhole, BrownConrady5, IdentitySensor, FxFyCxCySkew>
// camera.k: FxFyCxCySkew (intrinsics)
// camera.dist: BrownConrady5 (distortion)
```
