# Zhang's Intrinsics from Homographies

Zhang's method is the most widely used technique for estimating camera intrinsics from a planar calibration board. Given homographies from multiple views of the board, it recovers the intrinsics matrix $K$ using constraints from the **image of the absolute conic** (IAC).

## Problem Statement

**Given**: $M \geq 3$ homographies $\{H_k\}_{k=1}^M$ from a planar calibration board to the image, computed from different viewpoints.

**Find**: Camera intrinsics $K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$.

**Assumptions**:
- The calibration board is planar
- At least 3 homographies from distinct viewpoints
- The camera intrinsics are constant across all views
- Distortion is negligible (or has been accounted for — see [Iterative Intrinsics](iterative_intrinsics.md))

## Derivation

### Homography and Intrinsics

From the previous chapter, the homography from a $Z = 0$ board plane is:

$$H = \lambda K [\mathbf{r}_1 \; \mathbf{r}_2 \; \mathbf{t}]$$

where $\lambda$ is an arbitrary scale and $\mathbf{r}_1, \mathbf{r}_2$ are columns of the rotation matrix. Let $H = [\mathbf{h}_1 \; \mathbf{h}_2 \; \mathbf{h}_3]$. Then:

$$\mathbf{h}_1 = \lambda K \mathbf{r}_1, \quad \mathbf{h}_2 = \lambda K \mathbf{r}_2$$

### Orthogonality Constraints

Since $\mathbf{r}_1$ and $\mathbf{r}_2$ are columns of a rotation matrix, they satisfy:

$$\mathbf{r}_1^T \mathbf{r}_2 = 0 \quad \text{(orthogonality)}$$
$$\mathbf{r}_1^T \mathbf{r}_1 = \mathbf{r}_2^T \mathbf{r}_2 \quad \text{(equal norm)}$$

Substituting $\mathbf{r}_i = \frac{1}{\lambda} K^{-1} \mathbf{h}_i$:

$$\mathbf{h}_1^T K^{-T} K^{-1} \mathbf{h}_2 = 0$$
$$\mathbf{h}_1^T K^{-T} K^{-1} \mathbf{h}_1 = \mathbf{h}_2^T K^{-T} K^{-1} \mathbf{h}_2$$

### The Image of the Absolute Conic (IAC)

Define the symmetric matrix:

$$B = K^{-T} K^{-1} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}$$

Since $B$ is symmetric, it has **6 independent entries**. The constraints become:

$$\mathbf{h}_1^T B \, \mathbf{h}_2 = 0$$
$$\mathbf{h}_1^T B \, \mathbf{h}_1 - \mathbf{h}_2^T B \, \mathbf{h}_2 = 0$$

Each homography gives **2 linear equations** in the 6 unknowns of $B$.

### The $\mathbf{v}_{ij}$ Vectors

To write the constraints as a linear system, define the 6-vector:

$$\mathbf{v}_{ij} = \begin{bmatrix} h_{1i} h_{1j} \\ h_{1i} h_{2j} + h_{2i} h_{1j} \\ h_{2i} h_{2j} \\ h_{3i} h_{1j} + h_{1i} h_{3j} \\ h_{3i} h_{2j} + h_{2i} h_{3j} \\ h_{3i} h_{3j} \end{bmatrix}$$

where $h_{ki}$ is the $(k,i)$-th element of $H$ (1-indexed, column $i$, row $k$). Then:

$$\mathbf{h}_i^T B \, \mathbf{h}_j = \mathbf{v}_{ij}^T \mathbf{b}$$

where $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^T$.

### The Linear System

For each homography $H_k$, the two constraints become:

$$\begin{bmatrix} \mathbf{v}_{12}^T \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^T \end{bmatrix} \mathbf{b} = \mathbf{0}$$

Stacking $M$ homographies gives a $2M \times 6$ system:

$$V \mathbf{b} = \mathbf{0}$$

With $M \geq 3$ (giving $\geq 6$ equations), this is solved via SVD: $\mathbf{b}$ is the right singular vector of $V$ corresponding to the smallest singular value.

### Extracting Intrinsics from $B$

Given $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]$, the intrinsics are recovered by a Cholesky-like factorization of $B = K^{-T} K^{-1}$.

The closed-form extraction:

$$v_0 = \frac{B_{12} B_{13} - B_{11} B_{23}}{B_{11} B_{22} - B_{12}^2}$$

$$\lambda = B_{33} - \frac{B_{13}^2 + v_0 (B_{12} B_{13} - B_{11} B_{23})}{B_{11}}$$

$$f_x = \sqrt{\frac{\lambda}{B_{11}}}$$

$$f_y = \sqrt{\frac{\lambda B_{11}}{B_{11} B_{22} - B_{12}^2}}$$

$$s = -\frac{B_{12} f_x^2 f_y}{\lambda}$$

$$c_x = \frac{s \cdot v_0}{f_y} - \frac{B_{13} f_x^2}{\lambda}$$

$$c_y = v_0$$

### Validity Checks

The extraction can fail when:

- $B_{11} B_{22} - B_{12}^2 \approx 0$ (degenerate: insufficient view diversity)
- $|B_{11}| \approx 0$
- $\lambda$ and $B_{11}$ have different signs (negative focal length squared)

These cases indicate that the input homographies do not sufficiently constrain the intrinsics, typically because the views are too similar (e.g., all near-frontal).

## Minimum Number of Views

- **$M = 3$** homographies: 6 equations for 6 unknowns — the minimum for a unique solution (with skew)
- **$M = 2$**: Only 4 equations; requires the additional assumption that $s = 0$ (zero skew) to reduce unknowns to 5
- **$M > 3$**: Overdetermined system; SVD gives the least-squares solution

## Practical Considerations

**View diversity**: The views should include significant rotation around both axes. Pure translations or rotations around a single axis lead to degenerate configurations where some intrinsic parameters are undetermined.

**Distortion**: Zhang's method assumes distortion-free observations. When applied to distorted pixels, the estimated $K$ is biased. The [Iterative Intrinsics](iterative_intrinsics.md) chapter addresses this with an alternating refinement scheme.

> **OpenCV equivalence**: Zhang's method is the internal initialization step of `cv::calibrateCamera`. OpenCV does not expose it as a separate API.

## API

```rust
let K = estimate_intrinsics_from_homographies(&homographies)?;
```

Returns `FxFyCxCySkew` with the estimated intrinsics. Typically followed by distortion estimation and non-linear refinement.

## Reference

Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.
