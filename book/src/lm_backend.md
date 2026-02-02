# Levenberg-Marquardt Backend

The `TinySolverBackend` is the current optimization backend in calibration-rs. It wraps the `tiny-solver` crate, providing Levenberg-Marquardt optimization with sparse linear solvers, manifold support, and robust loss functions.

## Backend Trait

All backends implement the `OptimBackend` trait:

```rust
pub trait OptimBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial_params: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution>;
}
```

The backend receives the problem IR, initial parameter values, and solver options, and returns the optimized parameters with a solve report.

## Compilation: IR to Solver

The `compile()` step translates the abstract IR into tiny-solver's concrete types:

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

Each `FactorKind` maps to a specific generic function call:

```
FactorKind::ReprojPointPinhole4Dist5 { pw, uv, w }
    → reproj_residual_pinhole4_dist5_se3_generic(intr, dist, pose, pw, uv, w)

FactorKind::LaserLineDist2D { laser_pixel, w }
    → laser_line_dist_normalized_generic(intr, dist, pose, plane_normal, plane_distance, laser_pixel, w)

FactorKind::Se3TangentPrior { sqrt_info }
    → se3_tangent_prior_generic(pose, sqrt_info)
```

## Solver Options

```rust
pub struct BackendSolveOptions {
    pub max_iters: usize,          // Maximum LM iterations (default: 100)
    pub verbosity: u32,            // 0=silent, 1=summary, 2=per-iteration
    pub linear_solver: LinearSolverType,  // SparseCholesky or SparseQR
    pub min_abs_decrease: f64,     // Absolute cost decrease threshold
    pub min_rel_decrease: f64,     // Relative cost decrease threshold
    pub min_error: f64,            // Minimum cost to stop early
    pub initial_lambda: Option<f64>, // Initial damping (None = auto)
}

pub enum LinearSolverType {
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
    pub report: SolveReport,
}

pub struct SolveReport {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub termination: TerminationReason,
}
```

The cost is $F = \frac{1}{2} \sum r_i^2$ (half sum of squared residuals). Problem-specific code extracts domain types (cameras, poses, planes) from the raw parameter vectors.

## Typical Convergence

For a well-initialized planar intrinsics problem:

- **Initial cost**: $\sim 10^2$ - $10^4$ (from linear initialization)
- **Final cost**: $\sim 10^{-2}$ - $10^0$ (sub-pixel residuals)
- **Iterations**: 10-50 (depends on problem size and initial quality)
- **Termination**: Usually relative decrease below threshold

## Error Handling

The backend propagates errors for:

- Missing initial values for a parameter block
- Manifold dimension mismatch
- Linear solver failure (singular system)
- NaN/Inf in residual evaluation
