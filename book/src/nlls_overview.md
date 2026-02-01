# Non-Linear Least Squares Overview

Non-linear least squares (NLLS) is the mathematical framework underlying all refinement in calibration-rs. After linear initialization provides approximate parameter estimates, NLLS minimizes the reprojection error to achieve sub-pixel accuracy.

## Problem Formulation

**Objective**: Minimize the sum of squared residuals:

$$\min_{\boldsymbol{\theta}} \quad F(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{N} \| \mathbf{r}_i(\boldsymbol{\theta}) \|^2 = \frac{1}{2} \mathbf{r}(\boldsymbol{\theta})^T \mathbf{r}(\boldsymbol{\theta})$$

where $\boldsymbol{\theta} \in \mathbb{R}^n$ is the parameter vector and $\mathbf{r}(\boldsymbol{\theta}) \in \mathbb{R}^m$ is the stacked residual vector.

In camera calibration, a typical residual is the **reprojection error**: the difference between an observed pixel and the predicted projection:

$$\mathbf{r}_i = \pi(K, \mathbf{d}, T_v, \mathbf{P}_j) - \mathbf{p}_{vj}$$

where $\pi$ is the camera projection function, and the parameters $\boldsymbol{\theta}$ include intrinsics $K$, distortion $\mathbf{d}$, and poses $\{T_v\}$.

## Gauss-Newton Method

The Gauss-Newton method exploits the least-squares structure. At the current estimate $\boldsymbol{\theta}$, linearize the residuals:

$$\mathbf{r}(\boldsymbol{\theta} + \boldsymbol{\delta}) \approx \mathbf{r}(\boldsymbol{\theta}) + J \boldsymbol{\delta}$$

where $J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{m \times n}$ is the **Jacobian**.

Substituting into the objective and minimizing with respect to $\boldsymbol{\delta}$:

$$\frac{\partial}{\partial \boldsymbol{\delta}} \frac{1}{2} \| \mathbf{r} + J\boldsymbol{\delta} \|^2 = 0$$

gives the **normal equations**:

$$J^T J \, \boldsymbol{\delta} = -J^T \mathbf{r}$$

The matrix $H_{\text{GN}} = J^T J$ is the Gauss-Newton approximation to the Hessian ($H_{\text{GN}} \approx \nabla^2 F$ when residuals are small).

**Update**: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{\delta}$

## Levenberg-Marquardt Method

Gauss-Newton can diverge when the linearization is poor (far from the minimum) or when $J^T J$ is singular. Levenberg-Marquardt (LM) adds a damping term:

$$(J^T J + \lambda \, \text{diag}(J^T J)) \, \boldsymbol{\delta} = -J^T \mathbf{r}$$

The damping parameter $\lambda > 0$ interpolates between:

- **$\lambda \to 0$**: Pure Gauss-Newton (fast convergence near minimum)
- **$\lambda \to \infty$**: Gradient descent with small step (safe far from minimum)

### Trust Region Interpretation

LM is a **trust region method**: $\lambda$ controls the size of the region where the linear approximation is trusted.

- If the update reduces the cost: accept the step, decrease $\lambda$ (expand trust region)
- If the update increases the cost: reject the step, increase $\lambda$ (shrink trust region)

### Convergence Criteria

LM terminates when any of:

- **Cost threshold**: $F(\boldsymbol{\theta}) < \epsilon_{\text{min}}$
- **Absolute decrease**: $|F_k - F_{k+1}| < \epsilon_{\text{abs}}$
- **Relative decrease**: $|F_k - F_{k+1}| / F_k < \epsilon_{\text{rel}}$
- **Maximum iterations**: $k > k_{\max}$
- **Parameter change**: $\|\boldsymbol{\delta}\| < \epsilon_{\text{param}}$

## Sparsity

In bundle adjustment problems, the Jacobian $J$ is **sparse**: each residual depends on only a few parameter blocks (one camera intrinsics, one distortion, one pose). The $J^T J$ matrix has a block-arrow structure that can be exploited by sparse linear solvers:

- **Sparse Cholesky**: Efficient for well-structured problems
- **Sparse QR**: More robust when the normal equations are ill-conditioned

calibration-rs uses sparse linear solvers through the tiny-solver backend.

## Cost Function vs. Reprojection Error

The optimizer minimizes the **cost** $F = \frac{1}{2} \sum r_i^2$. The commonly reported **mean reprojection error** is:

$$\bar{e} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{r}_i\|$$

These are related but not identical: the cost weights large errors quadratically, while the mean error weighs them linearly. calibration-rs reports both: the cost from the solver and the mean reprojection error computed post-optimization.

## With Robust Losses

When a robust loss $\rho$ is used, the objective becomes:

$$\min_{\boldsymbol{\theta}} \sum_{i=1}^{N} \rho(\|\mathbf{r}_i\|)$$

The normal equations are modified to incorporate the loss function's weight:

$$J^T W J \, \boldsymbol{\delta} = -J^T W \mathbf{r}$$

where $W = \text{diag}(\rho'(\|\mathbf{r}_i\|) / \|\mathbf{r}_i\|)$ is the iteratively reweighted diagonal matrix. See [Robust Loss Functions](robust_loss.md) for details.
