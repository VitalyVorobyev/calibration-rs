# Adding a New Optimization Problem

This chapter walks through adding a new optimization problem to `vision-calibration-optim`, following the pattern established by the planar intrinsics implementation.

## Overview

Adding a new problem requires these steps:

1. Define the generic residual function in `factors/`
2. Add a `FactorKind` variant in `ir/types.rs`
3. Create a problem builder in `problems/`
4. Integrate with the backend in `backend/tiny_solver_backend.rs`
5. Write tests with synthetic ground truth

## Step 1: Generic Residual Function

Create a new file or add to an existing file in `crates/vision-calibration-optim/src/factors/`:

```rust
pub fn my_residual_generic<T: RealField>(
    param_a: DVectorView<'_, T>,  // optimizable parameters
    param_b: DVectorView<'_, T>,
    constant_data: [f64; 3],      // per-residual constants (f64, not T)
    w: f64,                        // weight
) -> SVector<T, 2> {              // residual dimension
    // Convert constants to T
    let cx = T::from_f64(constant_data[0]).unwrap();

    // Compute residual using param_a, param_b, cx
    // Use .clone() liberally, avoid in-place ops
    let r_x = /* ... */;
    let r_y = /* ... */;

    let wt = T::from_f64(w).unwrap();
    SVector::<T, 2>::new(r_x * wt.clone(), r_y * wt)
}
```

**Key rules**:
- Generic over `T: RealField` for autodiff
- Optimizable parameters as `DVectorView<'_, T>`
- Constants as `f64` (converted inside)
- Use `.clone()`, no in-place mutation
- Return `SVector<T, N>` for fixed residual dimension

## Step 2: FactorKind Variant

Add a new variant to `FactorKind` in `crates/vision-calibration-optim/src/ir/types.rs`:

```rust
pub enum FactorKind {
    // ... existing variants ...
    MyNewFactor {
        constant_data: [f64; 3],
        w: f64,
    },
}
```

Add validation in the `validate` method:

```rust
FactorKind::MyNewFactor { .. } => {
    ensure!(params.len() == 2, "MyNewFactor requires 2 param blocks");
    ensure!(params[0].dim == 4, "param_a must be dim 4");
    ensure!(params[1].dim == 7, "param_b must be dim 7 (SE3)");
}
```

Update the `residual_dim()` method:

```rust
FactorKind::MyNewFactor { .. } => 2,
```

## Step 3: Problem Builder

Create `crates/vision-calibration-optim/src/problems/my_problem.rs`:

```rust
pub struct MyProblemParams {
    pub param_a: DVector<f64>,
    pub param_b: Iso3,
}

pub struct MySolveOptions {
    pub fix_param_a: FixedMask,
    pub robust_loss: RobustLoss,
}

pub fn build_my_problem_ir(
    data: &MyDataset,
    initial: &MyProblemParams,
    opts: &MySolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    let mut ir = ProblemIR::new();
    let mut initial_values = HashMap::new();

    // Add parameter blocks via the builder API
    let param_a_id = ir.add_param_block(
        "param_a", 4, ManifoldKind::Euclidean,
        opts.fix_param_a.clone(), None,
    );
    initial_values.insert("param_a".to_string(), initial.param_a.clone());

    let param_b_id = ir.add_param_block(
        "param_b", 7, ManifoldKind::SE3,
        FixedMask::all_free(), None,
    );
    initial_values.insert(
        "param_b".to_string(), iso3_to_se3_dvec(&initial.param_b),
    );

    // Add residual blocks
    for obs in &data.observations {
        ir.add_residual_block(ResidualBlock {
            params: vec![param_a_id, param_b_id],
            loss: opts.robust_loss,
            factor: FactorKind::MyNewFactor {
                constant_data: obs.constant_data,
                w: obs.weight,
            },
            residual_dim: 2,
        });
    }

    Ok((ir, initial_values))
}
```

## Step 4: Backend Integration

In `crates/vision-calibration-optim/src/backend/tiny_solver_backend.rs`, add a match arm in `compile_factor()`:

```rust
FactorKind::MyNewFactor { constant_data, w } => {
    let constant_data = *constant_data;
    let w = *w;
    Box::new(move |params: &[DVectorView<'_, T>]| {
        my_residual_generic(
            params[0], params[1],
            constant_data, w,
        ).as_slice().to_vec()
    })
}
```

## Step 5: Tests

Write a test with synthetic ground truth in `crates/vision-calibration-optim/tests/`:

```rust
#[test]
fn my_problem_converges() {
    // 1. Define ground truth parameters
    let gt_param_a = /* ... */;
    let gt_param_b = /* ... */;

    // 2. Generate synthetic observations
    let data = generate_synthetic_data(&gt_param_a, &gt_param_b);

    // 3. Create perturbed initial values
    let initial = perturb(&gt_param_a, &gt_param_b);

    // 4. Build and solve
    let (ir, init_vals) = build_my_problem_ir(&data, &initial, &opts)?;
    let solution = solve_with_backend(
        BackendKind::TinySolver, &ir, &init_vals, &backend_opts,
    )?;

    // 5. Verify convergence
    let solved_a = &solution.params["param_a"];
    assert!((solved_a[0] - gt_param_a[0]).abs() < tolerance);
}
```

## Checklist

- [ ] Generic residual function with `T: RealField`
- [ ] `FactorKind` variant with validation
- [ ] Problem builder producing `ProblemIR` + initial values
- [ ] Backend compilation for the new factor
- [ ] Synthetic ground truth test verifying convergence
- [ ] (Optional) Pipeline integration with session framework
