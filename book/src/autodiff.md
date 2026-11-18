# Autodiff and Generic Residual Functions

The Levenberg-Marquardt algorithm requires the Jacobian $J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\theta}}$ of the residual vector. Computing this analytically for complex camera models (distortion, Scheimpflug tilt, SE(3) transforms) is error-prone and tedious. calibration-rs uses **automatic differentiation** (autodiff) via dual numbers to compute exact Jacobians from the same code that evaluates residuals.

## Dual Numbers

A dual number extends a real number with an infinitesimal part:

$$\tilde{x} = x + \epsilon \dot{x}, \quad \epsilon^2 = 0$$

Arithmetic preserves the chain rule automatically:

$$f(\tilde{x}) = f(x) + \epsilon \, f'(x) \dot{x}$$

By setting $\dot{x} = 1$ for the parameter of interest and $\dot{x} = 0$ for others, evaluating $f(\tilde{x})$ simultaneously computes both $f(x)$ and $\frac{\partial f}{\partial x}$.

For multi-variable functions, **hyper-dual numbers** or **multi-dual numbers** generalize this to compute full Jacobians. The `num-dual` crate provides these types.

## The `RealField` Pattern

All residual functions in calibration-rs are written **generic over the scalar type**:

A simplified pinhole + Brown-Conrady reprojection residual looks like this (the shipped residuals additionally take the camera model and pose chain as data):

```rust,ignore
fn reproj_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,  // [fx, fy, cx, cy]
    dist: DVectorView<'_, T>,  // [k1, k2, k3, p1, p2]
    pose: DVectorView<'_, T>,  // [qx, qy, qz, qw, tx, ty, tz]
    pw: [f64; 3],              // 3D world point (constant)
    uv: [f64; 2],              // observed pixel (constant)
    sqrt_info: [f64; 3],       // [l00, l01, l11] of L = [[l00, l01], [0, l11]] (constant)
) -> SVector<T, 2>
```

When called with `T = f64`, this computes the residual value. When called with a dual-number type `T`, it simultaneously computes the residual and its derivatives with respect to all parameters.

### Design Rules for Autodiff-Compatible Code

1. **Use `.clone()` liberally**: `RealField` is not `Copy`, so a value used twice is cloned. Dual numbers are small structs, so cloning is cheap.

2. **Convert constants**: Use `T::from_f64(1.0).unwrap()` to convert floating-point literals. Do not use `1.0` directly where a `T` is expected.

3. **Constant data as `f64`**: World points, observed pixels, and the square-root information are constant data — not optimized. Keep them as `f64` and convert to `T` inside the function.

## Walkthrough: Reprojection Residual

The core reprojection residual computes:

$$\mathbf{r} = L \left( \mathbf{p}_{\text{obs}} - \pi(K, \mathbf{d}, T, \mathbf{P}) \right)$$

where $L$ is the upper-triangular square-root information of the observation: $\sqrt{w}\,I$ for a point without a stated covariance, or $L^T L = \Sigma^{-1}$ for a point stating its covariance $\Sigma$ (see the [Factor Catalog](factor_catalog.md)).

Step by step:

### 1. Unpack Parameters

```rust,ignore
let fx = intr[0].clone();
let fy = intr[1].clone();
let cx = intr[2].clone();
let cy = intr[3].clone();

let k1 = dist[0].clone();
let k2 = dist[1].clone();
// ...
```

### 2. SE(3) Transform: World to Camera

Apply the pose block's rotation and translation to the 3D world point:

$$\mathbf{P}_c = R(\mathbf{q}) \cdot \mathbf{P}_w + \mathbf{t}$$

The implementation unpacks the quaternion $[q_x, q_y, q_z, q_w]$ and applies the rotation formula.

### 3. Pinhole Projection

$$n_x = P_{cx} / P_{cz}, \quad n_y = P_{cy} / P_{cz}$$

### 4. Apply Distortion

$$r^2 = n_x^2 + n_y^2$$
$$\alpha = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6$$
$$x_d = n_x \cdot \alpha + 2 p_1 n_x n_y + p_2(r^2 + 2 n_x^2)$$
$$y_d = n_y \cdot \alpha + p_1(r^2 + 2 n_y^2) + 2 p_2 n_x n_y$$

### 5. Apply Intrinsics

$$u = f_x \cdot x_d + c_x$$
$$v = f_y \cdot y_d + c_y$$

### 6. Compute Residual

$$\mathbf{e} = (u_{\text{obs}} - u,\ v_{\text{obs}} - v)$$
$$r_x = l_{00}\, e_x + l_{01}\, e_y, \qquad r_y = l_{11}\, e_y$$

Every arithmetic operation on `T` values propagates derivatives automatically through the dual number mechanism.

## Backend Integration

Each backend wraps the same generic residual function in its own residual type. The factor carries its camera model as data, and the residual evaluates it per slot through the core camera-model kinds:

```text
FactorKind::ReprojPoint { model: pinhole + Brown-Conrady, chain: SinglePose, pw, uv, sqrt_info }
    ↓ (compile step)
residual(params) = reproj_residual(model, chain, params, pw, uv, sqrt_info)
```

Both evaluate it with dual numbers to get residuals and Jacobians in one pass. tiny-solver's duals have a dynamic size; factrs' are sized at compile time to the factor's tangent dimension (for example 15 for a Brown-Conrady camera and one SE(3) pose), so they live on the stack.

## SE(3) Exponential Map

A pose block is stored as a quaternion and a translation, so the residual above applies it directly. A residual that reads a 6D se(3) tangent vector instead, such as the per-view robot-pose correction `robot_delta` applied as $T_{B,G} \cdot \exp(\boldsymbol{\xi})$, maps it to a rotation and a translation inside the residual:

```rust,ignore
fn se3_exp<T: RealField>(xi: DVectorView<'_, T>) -> (UnitQuaternion<T>, Vector3<T>)
```

For $\boldsymbol{\xi} = [\boldsymbol{\omega}, \mathbf{v}]$ it returns $\exp([\boldsymbol{\omega}]_\times)$ and $V(\boldsymbol{\omega})\,\mathbf{v}$, the exponential map of [Manifold Optimization](manifolds.md). Below a rotation angle of $10^{-4}$ rad the rotation and the coefficients of $V$ are their Taylor series, exact to working precision there. The closed forms are not differentiable at $\boldsymbol{\omega} = 0$ (an axis-angle conversion returns the identity for a zero vector, so its derivative there is zero), and a correction that starts at zero, as every robot-pose correction does, would otherwise see no rotation Jacobian.

## Performance Considerations

Autodiff with dual numbers is typically 2-10x slower than hand-coded Jacobians, but:

- **Correctness**: Eliminates the risk of Jacobian implementation bugs
- **Maintainability**: Adding a new factor only requires writing the forward evaluation
- **Flexibility**: The same code works for any parameter configuration

For calibration problems (hundreds to thousands of residuals, tens to hundreds of parameters), the autodiff overhead is negligible compared to the linear solve in each LM iteration.
