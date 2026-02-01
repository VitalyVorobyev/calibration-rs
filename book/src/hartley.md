# Hartley Normalization

DLT algorithms build large linear systems from point coordinates and solve them via SVD. When the coordinates span vastly different scales (e.g., pixel values in the hundreds vs. homogeneous scale factor of 1), the resulting system is **ill-conditioned**: small perturbations in the data cause large changes in the solution. Hartley normalization is a simple preprocessing step that dramatically improves numerical stability.

## The Problem

Consider the homography DLT: we build a $2N \times 9$ matrix $A$ from pixel coordinates $(u, v)$ and solve $A\mathbf{h} = 0$. If $(u, v) \sim (640, 480)$, the entries of $A$ range from $\sim 1$ to $\sim 640^2 \approx 4 \times 10^5$. The condition number of $A$ becomes large, and the SVD solution is sensitive to floating-point errors.

## The Solution

**Normalize the input points** so they are centered at the origin with a controlled scale, solve the system, then **denormalize** the result.

### 2D Normalization

Given $N$ points $\{\mathbf{p}_i = (x_i, y_i)\}$:

1. **Centroid**: $\bar{x} = \frac{1}{N}\sum x_i$, $\bar{y} = \frac{1}{N}\sum y_i$

2. **Mean distance** from centroid: $\bar{d} = \frac{1}{N}\sum \sqrt{(x_i - \bar{x})^2 + (y_i - \bar{y})^2}$

3. **Scale factor**: $s = \sqrt{2} \,/\, \bar{d}$

4. **Normalization matrix**:

$$T = \begin{bmatrix} s & 0 & -s\bar{x} \\ 0 & s & -s\bar{y} \\ 0 & 0 & 1 \end{bmatrix}$$

5. **Normalized points**: $\tilde{\mathbf{p}}_i = T \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$

After normalization, the points have centroid at the origin and mean distance $\sqrt{2}$ from the origin.

### 3D Normalization

For 3D points $\{\mathbf{P}_i = (X_i, Y_i, Z_i)\}$, the analogous normalization uses:

- Mean distance target: $\sqrt{3}$ (instead of $\sqrt{2}$)
- A $4 \times 4$ normalization matrix $T_{\text{3D}}$

## Denormalization

After solving the normalized system, the result must be denormalized. For a homography estimated from normalized points:

$$H = T_{\text{image}}^{-1} \cdot H_{\text{norm}} \cdot T_{\text{world}}$$

For a fundamental matrix:

$$F = T_2^T \cdot F_{\text{norm}} \cdot T_1$$

The denormalization formula depends on the specific problem. Each chapter states the appropriate denormalization.

## When It Is Used

Hartley normalization is applied in every DLT-based solver in calibration-rs:

| Solver | Normalized spaces |
|--------|------------------|
| Homography DLT | World 2D + image 2D |
| Fundamental 8-point | Image 1 2D + image 2 2D |
| Essential 5-point | Image 1 2D + image 2 2D |
| Camera matrix DLT | World 3D + image 2D |
| PnP DLT | World 3D (image points use normalized coordinates via $K^{-1}$) |

## Reference

Hartley, R.I. (1997). "In Defense of the Eight-Point Algorithm." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(6), 580-593.

The normalization technique is described in Algorithm 4.2 of Hartley & Zisserman (2004).
