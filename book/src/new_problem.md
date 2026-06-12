# Adding a New Optimization Problem

This chapter walks through adding a new optimization problem to `vision-calibration-optim`, following the pattern established by the planar intrinsics implementation.

## Overview

Most new problems compose the **existing** factor families (`ReprojPoint`,
`LaserPointToPlane`, `LaserLineDistance`, `Se3TangentPrior`) with a
`CameraModelDesc` and a chain — in that case skip straight to Step 3 and emit
the existing factors from your builder. The steps below cover the rarer case
of a genuinely **new residual family**:

1. Define the generic residual function in `factors/`
2. Add a `FactorKind` variant in `ir/types.rs`
3. Create a problem builder in `problems/`
4. Integrate with the backend in `backend/tiny_solver_backend.rs`
5. Write tests with synthetic ground truth

(Adding a new **camera model** is a different, smaller recipe: one descriptor
enum variant + one kernel type + one dispatch-table row; see
[ADR 0020](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/adrs/0020-camera-model-as-data-factor-ir.md)
and the [Factor Catalog](factor_catalog.md).)

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

Describe its parameter layout in `param_layout()` — validation is derived
from the layout, so there is no separate validation arm to write:

```rust
FactorKind::MyNewFactor { .. } => vec![
    ParamSlotSpec { dim: 4, manifold: ManifoldKind::Euclidean, role: "param_a" },
    ParamSlotSpec { dim: 7, manifold: ManifoldKind::SE3, role: "param_b" },
],
```

Update the `residual_dim()` and `name()` methods:

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

In `crates/vision-calibration-optim/src/backend/tiny_solver_backend.rs`, add
a factor struct implementing tiny-solver's `Factor<T>` plus a match arm in
`compile_factor()`:

```rust
#[derive(Debug, Clone)]
struct TinyMyNewFactor {
    constant_data: [f64; 3],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyMyNewFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let r = my_residual_generic(
            params[0].as_view(),
            params[1].as_view(),
            self.constant_data,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

// in compile_factor():
FactorKind::MyNewFactor { constant_data, w } => Ok((
    Box::new(TinyMyNewFactor { constant_data: *constant_data, w: *w }),
    loss,
)),
```

If the new family is camera-model-aware, make the struct generic over the
kernel types and construct it through the `dispatch_camera_model!` table the
way `TinyReprojFactor<P, D, S>` is.

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
