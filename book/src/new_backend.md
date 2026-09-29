# Adding a New Solver Backend

The backend-agnostic IR design allows adding new optimization backends without modifying problem definitions. This chapter is a guide for contributors working inside the `vision-calibration-optim` crate; the backend types are not part of its public API.

## The `OptimBackend` Trait

```rust
pub trait OptimBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error>;
}
```

A backend receives:
- **`ir`**: The problem structure (parameter blocks, residual blocks, factor kinds)
- **`initial`**: Initial values for all parameter blocks (keyed by name)
- **`opts`**: Solver options (max iterations, tolerances, verbosity)

And returns:
- **`BackendSolution`**: Optimized parameter values (`params`, keyed by parameter name) and a `solve_report` (`final_cost`, `num_iters`)

## What a Backend Must Handle

### 1. Parameter Blocks

For each `ParamBlock`, the backend must:

- Allocate storage for a parameter vector of the given dimension
- Initialize from the provided initial values
- Apply the manifold (if not Euclidean)
- Respect the fixed mask (hold specified indices constant)
- Apply box constraints (if bounds are specified)

### 2. Manifold Constraints

The backend must implement the plus ($\oplus$) and minus ($\ominus$) operations for each `ManifoldKind`:

| Manifold | Ambient dim | Tangent dim | Plus operation |
|----------|-------------|-------------|----------------|
| `Euclidean` | $n$ | $n$ | $\mathbf{x} + \boldsymbol{\delta}$ |
| `SE3` | 7 | 6 | $\exp(\boldsymbol{\xi}) \cdot T$ |
| `SO3` | 4 | 3 | $\exp([\boldsymbol{\omega}]_\times) \cdot R$ |
| `S2` | 3 | 2 | Retract via tangent plane basis |

### 3. Residual Evaluation

For each `ResidualBlock`, the backend must:

- Call the appropriate residual function based on `FactorKind`
- Pass the correct parameter block values (referenced by `ParamId`)
- Include per-residual constant data (3D points, observed pixels, weights)
- Compute Jacobians (via autodiff or finite differences)

### 4. Robust Loss Functions

The backend must apply the `RobustLoss` to each residual:

- `None` → standard squared loss
- `Huber { scale }` → Huber loss with the given scale
- `Cauchy { scale }` → Cauchy loss
- `Arctan { scale }` → Arctan loss

### 5. Solution Extraction

Return optimized values as a `HashMap<String, DVector<f64>>` keyed by parameter block **name** (not ID).

## Implementation Pattern

`TinySolverBackend` (in `backend/tiny_solver_backend.rs`) is the reference implementation. It has two phases inside `solve`:

1. **Compile** — validate the IR, then create one solver parameter per `ParamBlock` (manifold, fixed indices, bounds) and one cost function per `ResidualBlock` (from its `FactorKind`, with the robust loss applied).
2. **Solve** — run the optimizer with the convergence criteria from `BackendSolveOptions`, then extract the final parameter values into a `BackendSolution`.

## Registering the Backend

Add a variant to `BackendKind` and a match arm in `solve_with_backend`, which is the single dispatch point used by every problem:

```rust
pub enum BackendKind {
    TinySolver,
    // MyBackend,
}
```

## Testing

A new backend should pass the same convergence tests as the existing backend: solve the same synthetic problems from the same initial values and check that `solution.solve_report.final_cost` reaches an equivalent level. Compare the results of the existing `vision-calibration-optim` problem tests between the two backends.
