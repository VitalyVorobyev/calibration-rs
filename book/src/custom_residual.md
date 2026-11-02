# Your Own Residual

The calibration workflows solve the problems the library knows. When you
have a measurement they do not use, such as a distance taken with a laser
rangefinder, a constraint from a CAD drawing or a reading from another
sensor, write it as a residual and solve it together with the library's own.
The solver is the one the workflows run, and the result comes with the
covariance of everything it estimates.

This is Rust only: a residual is generic over its scalar type, which a
Python function cannot be.

## The pieces

[`optim::problem`](https://docs.rs/vision-calibration/latest/vision_calibration/optim/problem/)
holds them:

- a `Problem` of **parameter blocks**: a Euclidean vector, an SE(3) pose or
  a unit 3-vector, each with a name;
- **residuals** that read blocks: ones you write (the `Residual` trait), and
  the library's reprojection residual through a camera you add;
- `solve`, which returns a `Solution`: each block's value, the solve report,
  the noise level of each residual family and the parameter covariance.

## Writing a residual

A residual states its length, the dimension of each block it reads, and
its value:

```rust
use vision_calibration::nalgebra::{DVector, RealField};
use vision_calibration::optim::problem::{Residual, se3};

/// A measured distance from the camera centre to the board's origin: the
/// length of the translation of `camera_se3_target`.
struct Range {
    measured: f64,
}

impl Residual for Range {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![7] // one SE(3) pose
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let range = se3(&blocks[0]).translation.vector.norm();
        DVector::from_element(1, range - T::from_f64(self.measured).unwrap())
    }
}
```

`eval` is generic over `T: RealField`. The solver evaluates it at `f64`
for the cost and at dual numbers for the Jacobian, so the derivatives are
exact and you write none. Write it with `RealField` operations and nalgebra
types over `T`, and lift constants with `T::from_f64(c).unwrap()`.

Each block arrives as its stored values: a Euclidean block's components, a
pose's `[qx, qy, qz, qw, tx, ty, tz]` (`se3` turns it into an
`Isometry3<T>`), a unit vector's `[x, y, z]`. `eval` must return `dim`
values and must not panic. A value that is not finite makes the solver
reject that step, as for any residual.

## Building the problem

Here the board's distance in the first view joins a planar calibration.
`camera`, `poses`, `board` and `pixels` are the starting values and the
observations, from the closed-form initializers in
[`init`](https://docs.rs/vision-calibration/latest/vision_calibration/init/)
or a workflow's result:

```rust
use vision_calibration::core::CameraFix;
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::{Noise, Problem, SolveOptions, solve};

let mut problem = Problem::new();
let cam = problem.add_camera(&camera, &CameraFix::default())?;
let mut views = Vec::new();
for (v, (pose, pixels)) in poses.iter().zip(&pixels).enumerate() {
    let pose = problem.add_pose(format!("view{v}"), pose)?;
    for (point, pixel) in board.iter().zip(pixels) {
        problem.add_reprojection(cam, pose, *point, *pixel, RobustLoss::None)?;
    }
    views.push(pose);
}
problem.add_residual(
    Range { measured: 0.4524 },
    &[views[0]],
    RobustLoss::None,
    Noise::Known(2e-4), // the rangefinder's σ: 0.2 mm
)?;

let solution = solve(&problem, &SolveOptions::default())?;
let refined = solution.camera(cam).unwrap();
let board_pose = solution.pose(views[0]).unwrap();
```

`add_camera` lays out the camera's blocks from its model, holding what the
`CameraFix` names, just as a workflow's config does (by default `k3` for
Brown–Conrady). `add_reprojection` takes a point in the target frame, the
`camera_se3_target` pose block that moves it into the camera frame, and its
observed pixel.

## Noise: what the covariance needs to know

The covariance needs the standard deviation of every residual. A residual
gets it in one of two ways:

- **`Noise::Known(σ)`** for a measurement with a stated σ, or a prior. The
  solve divides the residual by σ, so it weighs as much as its precision
  says, and the covariance counts it as known information.
- **`Noise::Estimated(name)`** for residuals that share an unknown noise
  level. They are pooled into a family of that name, and its σ is
  estimated from the family's scatter at the solution, as the reprojection
  residuals' σ is. A family needs enough residuals to estimate a spread.
  It is never pooled with the library's own residuals.

The reprojection residuals form the reprojection family, in pixels.

An estimated family enters the solve in its own unit, unweighted. The
covariance accounts for whatever weights the solve used, so it stays right.
But a family in metres next to residuals in pixels barely moves the
solution. To have it count, divide its residual by its expected σ, as
`Noise::Known` does.

`solution.noise` lists each estimated family's level. A family of yours
appears as `ResidualKind::Custom(name)`.

## Reading the result

`solution.uncertainty` is the same `Uncertainty` the workflows' results
carry, labelled by coordinate:

- **a camera's parameters** are labelled as in a workflow's result: `fx`,
  `k1`, … with one camera, or `cam0.fx`, `cam1.fx`, … with several. So
  [`projection_uncertainty`](uncertainty_projection.md) reads a camera
  solved here;
- **a block of yours** is labelled `<name>.<i>` by default, or
  `<name>.<label>` after `set_labels`;
- **a pose** is labelled `<name>.rx … <name>.tz`. These are its
  right-perturbation tangent `T·Exp(δ)`: `tx … tz` step along the axes of
  the frame the pose is of (the board's), not the camera's. To get the
  variance of the board's depth in the camera frame, rotate the translation
  block: `r₃ᵀ Σ_t r₃`, with `r₃` the third row of the pose's rotation.

The range does more than place one board. A planar calibration determines
the focal length and the boards' depths only together: a longer focal
length with the boards farther away fits the pixels almost as well. One
measured distance fixes that scale. Over 120 repetitions of 8 views of a
9×6 board with 0.3 px pixel noise, the 0.2 mm range on view 0 gives:

| | pixels only | with the range |
|---|---|---|
| σ(fx) | 3.66 px | 0.84 px |
| σ(depth), view 0 (frontal) | 2.05 mm | 0.49 mm |
| σ(depth), view 3 (tilted) | 2.14 mm | 0.66 mm |

The stated σ matches the scatter of the estimates. With the range, the
ratio of actual scatter to stated σ was 0.93 for `fx`, 1.05 for `cx`, and
1.00 and 1.01 for the two depths.

## What is checked, and when

Every input is checked as it is added, so a problem that was built is one
`solve` runs:

- a block's values are finite. A pose has a unit quaternion and a unit
  vector has unit norm;
- a residual's blocks match its `block_dims`, and its value at the initial
  blocks has `dim` finite entries. An error names the residual by its index;
- a reprojected point lies in front of the camera at the initial pose;
- bounds hold the initial value, and a known σ is positive.

At the solution:

- a block no residual reads has no information, and the uncertainty says
  so (`RankDeficient`, naming its coordinates);
- a free parameter that ends on a bound has no Gaussian covariance either
  (`AtBound`).

## Limits

- **Backend.** A problem with a residual of yours runs on the default
  tiny-solver backend. Asking for factrs is an error before any work,
  because factrs fixes each variable's size at compile time. The library's
  residuals alone run on both.
- **Independent residuals.** The covariance treats every residual as
  independent. The workflows also measure how much the estimate varies
  between views beyond that (`view_scatter`). A problem of yours does not.
- **The library's residuals.** Reprojection through a single pose is the
  only built-in residual a problem can use so far.
