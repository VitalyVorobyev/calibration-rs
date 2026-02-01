# Adding a New Solver Backend

The backend-agnostic IR design allows adding new optimization backends without modifying problem definitions. This chapter describes what a backend must implement and how to integrate it.

## The `OptimBackend` Trait

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

A backend receives:
- **`ir`**: The problem structure (parameter blocks, residual blocks, factor kinds)
- **`initial_params`**: Initial values for all parameter blocks (keyed by name)
- **`opts`**: Solver options (max iterations, tolerances, verbosity)

And returns:
- **`BackendSolution`**: Optimized parameter values (keyed by parameter name) and a solve report

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

A typical backend has two phases:

### Compile Phase

Translate the IR into solver-specific data structures:

```rust
fn compile(&self, ir: &ProblemIR) -> SolverProblem {
    for param in &ir.params {
        // Create solver parameter with manifold and fixing
    }
    for residual in &ir.residuals {
        // Create solver cost function from FactorKind
    }
}
```

### Solve Phase

Run the optimizer and extract results:

```rust
fn solve(&self, problem: SolverProblem, opts: &BackendSolveOptions)
    -> BackendSolution
{
    // Set convergence criteria from opts
    // Run optimization loop
    // Extract final parameter values
    // Build SolveReport
}
```

## Potential Backends

| Backend | Description | Advantages |
|---------|-------------|------------|
| tiny-solver | Current. Rust-native LM. | Pure Rust, no external deps |
| Ceres-RS | Rust bindings to Google Ceres | Battle-tested, many features |
| Custom GN | Hand-written Gauss-Newton | Full control, educational |
| L-BFGS | Quasi-Newton for large problems | Memory-efficient |

## Testing

A new backend should pass the same convergence tests as the existing backend:

```rust
#[test]
fn new_backend_planar_converges() {
    // Same synthetic data and initial values as tiny-solver tests
    let (ir, init) = build_planar_test_problem();

    let solution = MyNewBackend.solve(&ir, &init, &opts)?;

    // Verify same convergence quality
    assert!(solution.report.final_cost < 1e-4);
}
```

Run the full test suite with both backends to ensure equivalent results.
