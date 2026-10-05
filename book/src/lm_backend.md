# Levenberg-Marquardt Backend

The `TinySolverBackend` is the optimization backend in calibration-rs. It wraps the `tiny-solver` crate, providing Levenberg-Marquardt optimization with sparse linear solvers, manifold support, and robust loss functions.

The IR and backend types described in this chapter are internal to `vision-calibration-optim`: callers reach them through the `optimize_*` entry points and the pipeline `step_optimize` functions. Of the backend types, only `BackendSolveOptions` and `SolveReport` are public.

## Backend Trait

All backends implement the `OptimBackend` trait:

```rust
pub trait OptimBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error>;
}

pub enum BackendKind {
    TinySolver,
}

pub fn solve_with_backend(
    backend: BackendKind,
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution, Error>;
```

The backend receives the problem IR, initial parameter values, and solver options, and returns the optimized parameters with a solve report. Problems call `solve_with_backend`, which dispatches on `BackendKind`.

## Compilation: IR to Solver

Inside `OptimBackend::solve`, a compile step translates the abstract IR into tiny-solver's concrete types:

### Parameters

For each `ParamBlock` in the IR:

1. **Create parameter** with the correct dimension
2. **Set manifold** based on `ManifoldKind`:
   - `Euclidean` → no manifold (standard addition)
   - `SE3` → `SE3Manifold` (7D ambient, 6D tangent)
   - `SO3` → `QuaternionManifold` (4D ambient, 3D tangent)
   - `S2` → `UnitVector3Manifold` (3D ambient, 2D tangent)
3. **Fix parameters** according to `FixedMask`:
   - Euclidean: fix individual indices
   - Manifolds: fix entire block (all-or-nothing)
4. **Set bounds** if present (box constraints on parameter values)

### Residuals

For each `ResidualBlock`:

1. **Compile the factor**: Create a closure that calls the appropriate generic residual function
2. **Apply robust loss**: Wrap in Huber/Cauchy/Arctan if specified
3. **Connect parameters**: Reference the correct parameter blocks by their compiled IDs

### Factor Compilation

`compile_factor` matches the factor's `CameraModelDesc` against the
`dispatch_camera_model!` table once and monomorphizes a generic factor
struct over the selected kernel types; the chain is evaluated as data inside
the residual:

```
FactorKind::ReprojPoint { model: PINHOLE4_DIST5, chain, pw, uv, w }
    → TinyReprojFactor::<PinholeKernel, BrownConrady5Kernel, IdentitySensorKernel>
      (calls reproj_residual_model_generic::<P, D, S, T>(chain, params, pw, uv, w))

FactorKind::LaserLineDistance { model, chain, laser_pixel, w }
    → TinyLaserLineFactor::<BrownConrady5Kernel, Scheimpflug2Kernel>
      (calls laser_line_distance_model_generic::<D, S, T>(chain, params, laser_pixel, w))

FactorKind::Se3TangentPrior { sqrt_info }
    → TinySe3TangentPriorFactor (element-wise scaled tangent residual)
```

## Solver Options

```rust
pub struct BackendSolveOptions {
    pub max_iters: usize,                       // Maximum LM iterations (default: 100)
    pub verbosity: usize,                       // 0 = silent
    pub linear_solver: Option<LinearSolverKind>, // default: Some(SparseCholesky)
    pub min_abs_decrease: Option<f64>,          // default: Some(1e-5)
    pub min_rel_decrease: Option<f64>,          // default: Some(1e-5)
    pub min_error: Option<f64>,                 // default: Some(1e-10)
}

pub enum LinearSolverKind {
    SparseCholesky,  // Default: fast for well-conditioned problems
    SparseQR,        // More robust for ill-conditioned problems
}
```

### Choosing the Linear Solver

- **SparseCholesky** (default): Factors the normal equations $J^T J + \lambda D = -J^T \mathbf{r}$ directly. Fast but can fail if $J^T J$ is poorly conditioned.
- **SparseQR**: Factors $J$ directly (QR decomposition). More robust but slower. Use when Cholesky fails or when the problem has near-singular directions.

## Solution

```rust
pub struct BackendSolution {
    pub params: HashMap<String, DVector<f64>>,  // Optimized values by name
    pub solve_report: SolveReport,
}

pub struct SolveReport {
    pub final_cost: f64,
    pub num_iters: usize,
}
```

`final_cost` is the robust objective at the solution, $F = \frac{1}{2} \sum_i \rho_i(\lVert \mathbf{r}_i \rVert^2)$, where $\rho_i$ is the residual block's loss ($\rho(s) = s$ without one, so $F = \frac{1}{2} \sum \lVert \mathbf{r}_i \rVert^2$ for plain least squares). The Levenberg–Marquardt loop accepts or rejects each step on the change of this same objective. Problem-specific code extracts domain types (cameras, poses, planes) from the raw parameter vectors.

## Typical Convergence

For a well-initialized planar intrinsics problem:

- **Final cost**: $\sim 10^{-2}$ - $10^0$ (sub-pixel residuals)
- **Iterations**: 10-50 (depends on problem size and initial quality)
- **Termination**: Usually relative decrease below `min_rel_decrease`; the loop also stops once a step can no longer change the parameters ($\lVert \Delta x \rVert \le 10^{-8} (\lVert x \rVert + 10^{-8})$)

## Error Handling

The backend propagates errors for:

- Missing initial values for a parameter block
- Manifold dimension mismatch
- Linear solver failure (singular system)
- NaN/Inf in residual evaluation
