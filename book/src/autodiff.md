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

```rust
fn reproj_residual_pinhole4_dist5_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,  // [fx, fy, cx, cy]
    dist: DVectorView<'_, T>,  // [k1, k2, k3, p1, p2]
    pose: DVectorView<'_, T>,  // [qx, qy, qz, qw, tx, ty, tz]
    pw: [f64; 3],              // 3D world point (constant)
    uv: [f64; 2],              // observed pixel (constant)
    w: f64,                    // weight (constant)
) -> SVector<T, 2>
```

When called with `T = f64`, this computes the residual value. When called with `T = DualNumber`, it simultaneously computes the residual and its derivatives with respect to all parameters.

### Design Rules for Autodiff-Compatible Code

1. **Use `.clone()` liberally**: Dual numbers are small structs; cloning is cheap and avoids borrow checker issues with operator overloading.

2. **Avoid in-place operations**: Operations like `x += y` or `v[i] = expr` can cause issues with dual number tracking. Prefer functional style.

3. **Convert constants**: Use `T::from_f64(1.0).unwrap()` to convert floating-point literals. Do not use `1.0` directly where a `T` is expected.

4. **Constant data as `f64`**: World points, observed pixels, and weights are constant data — not optimized. Keep them as `f64` and convert to `T` inside the function.

## Walkthrough: Reprojection Residual

The core reprojection residual computes:

$$\mathbf{r} = w \cdot \left( \pi(K, \mathbf{d}, T, \mathbf{P}) - \mathbf{p}_{\text{obs}} \right)$$

Step by step:

### 1. Unpack Parameters

```rust
let fx = intr[0].clone();
let fy = intr[1].clone();
let cx = intr[2].clone();
let cy = intr[3].clone();

let k1 = dist[0].clone();
let k2 = dist[1].clone();
// ...
```

### 2. SE(3) Transform: World to Camera

Apply the pose to the 3D world point using the exponential map or direct quaternion rotation:

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

$$r_x = w \cdot (u - u_{\text{obs}})$$
$$r_y = w \cdot (v - v_{\text{obs}})$$

Every arithmetic operation on `T` values propagates derivatives automatically through the dual number mechanism.

## Backend Integration

The backend wraps each generic residual function in a solver-specific factor struct that implements the solver's cost function interface:

```
FactorKind::ReprojPointPinhole4Dist5 { pw, uv, w }
    ↓ (compile step)
TinySolverFactor {
    evaluate: |params| {
        reproj_residual_pinhole4_dist5_se3_generic(
            params["cam"], params["dist"], params["pose/k"],
            pw, uv, w
        )
    }
}
```

The tiny-solver backend then calls `evaluate` with dual numbers to compute both residuals and Jacobians in a single pass.

## SE(3) Exponential Map

A critical autodiff-compatible function is `se3_exp<T>()`, which maps a 6D tangent vector to an SE(3) transform:

```rust
fn se3_exp<T: RealField>(xi: &[T; 6]) -> [[T; 4]; 4]
```

This implements Rodrigues' formula with a special case for small angles ($\|\boldsymbol{\omega}\| < \epsilon$) using Taylor expansions to avoid numerical issues with the dual number's infinitesimal part.

## Performance Considerations

Autodiff with dual numbers is typically 2-10x slower than hand-coded Jacobians, but:

- **Correctness**: Eliminates the risk of Jacobian implementation bugs
- **Maintainability**: Adding a new factor only requires writing the forward evaluation
- **Flexibility**: The same code works for any parameter configuration

For calibration problems (hundreds to thousands of residuals, tens to hundreds of parameters), the autodiff overhead is negligible compared to the linear solve in each LM iteration.
