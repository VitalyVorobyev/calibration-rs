# ADR 0020: Camera Model as Data in the Factor IR

- Status: Accepted
- Date: 2026-06-12

## Context

The optimization IR (ADR 0008) described every residual with a `FactorKind`
variant whose *name* encoded the full camera model and pose chain:
`ReprojPointPinhole4`, `ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta`,
`LaserPlanePixelRigHandEye`, … — 18 variants total. Each variant carried its
own residual kernel, its own ~40-line parameter-layout validation arm
(`validate_inner` had grown to ~550 lines), and its own boxed factor struct in
the tiny-solver backend.

The variant set is the product of three independent axes:

1. **Camera model** — projection × distortion × sensor
   (pinhole · {none, Brown-Conrady-5} · {identity, Scheimpflug-2} today).
2. **Pose chain** — how the target-frame point reaches the camera frame
   (single pose / two-SE3 rig / hand-eye / hand-eye + robot-delta, plus the
   laser chains).
3. **Residual family** — point reprojection, laser point-to-plane, laser
   line-distance.

Four new camera models are planned (rational k4–k6 distortion, thin-prism
s1–s4, division model, Kannala-Brandt fisheye projection). Added naively, each
would multiply variants across every chain and family — dozens of new enum
arms, kernels, validation blocks, and backend structs per model. That growth
pattern made new camera models effectively unaffordable, which is the gate
this ADR removes.

## Decision

**One factor variant per residual family; camera model and pose chain carried
as data.**

```rust
pub enum FactorKind {
    ReprojPoint       { model: CameraModelDesc, chain: ReprojChain, pw: [f64; 3], uv: [f64; 2], w: f64 },
    LaserPointToPlane { model: CameraModelDesc, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    LaserLineDistance { model: CameraModelDesc, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    Se3TangentPrior   { sqrt_info: [f64; 6] },
}

pub struct CameraModelDesc {
    pub projection: ProjectionKind, // Pinhole (intrinsics dim 4)
    pub distortion: DistortionKind, // None (dim 0) | BrownConrady5 (dim 5)
    pub sensor: SensorKind,         // None (dim 0) | Scheimpflug2 (dim 2)
}
```

`ReprojChain` (`SinglePose` / `TwoSe3` / `HandEye` / `HandEyeRobotDelta`) and
`LaserChain` (`SinglePose` / `RigHandEye` / `RigHandEyeRobotDelta`) carry the
per-view measured robot pose (`base_se3_gripper: [f64; 7]`, named per ADR
0009) and `HandEyeMode` as data.

### Layout-derived validation

A factor's expected parameter blocks are *derived*, never hand-listed:
`FactorKind::param_layout()` returns `Vec<ParamSlotSpec { dim, manifold,
role }>` — intrinsics, then a distortion block iff `distortion.dim() > 0`,
then a sensor block iff `sensor.dim() > 0`, then the chain blocks (laser
chains end with `plane_normal` (S²) and `plane_distance`; a robot-delta block
always comes last). `ProblemIR::validate` zips the layout against the
referenced blocks. Variable-size distortion blocks (rational 8, thin-prism 4,
division 1–2) need no new validation code — only a `dim()` value. Per-index
fix masks (e.g. `fix_k3`) are untouched: fixing remains a property of the
parameter block, not the factor.

### Kernel monomorphization (no dyn dispatch in autodiff)

Residual kernels stay generic over `T: RealField` for dual-number autodiff,
so the camera model cannot be a `dyn` object (the dyn-safe `CameraProject`
trait in core is `f64`-only). Instead, each descriptor slot maps to a
zero-sized kernel type with static generic methods
(`optim/src/factors/camera_kernels.rs`):

- `ProjectionKernel::normalize<T>` — camera-frame point → z=1 plane;
- `DistortionKernel::{distort, undistort}<T>`;
- `SensorKernel::{to_sensor, to_normalized}<T>`.

One shared residual per family (`reproj_residual_model_generic<P, D, S, T>`,
`laser_*_model_generic<D, S, T>`) composes the kernels; the chain is matched
as data inside the residual (a small enum branch, negligible next to
dual-number quaternion algebra). The backend matches the descriptor **once
per factor** in `compile_factor` via a single `dispatch_camera_model!` table
and monomorphizes a generic factor struct over the kernel types. The
camera-model axis — the axis that grows — is therefore fully compile-time.

Combinatorics: dispatch rows = |projection·distortion·sensor| (4 today, 20
with all planned models), shared by all families and all chains. Chains do
not multiply anything.

### Extension recipe (one new camera model)

1. Add a descriptor enum variant with its `dim()` (IR).
2. Add a kernel type implementing the slot trait (autodiff math).
3. Add one row to `dispatch_camera_model!` (backend).
4. Add the corresponding parameter pack/fix-mask plumbing in `core`/problem
   configs as needed.

No new factor variants, no validation arms, no per-chain duplication.

### Numerics policy

Pinhole normalization is unified on the `max(z, 1e-12)` clamp that 14 of the
15 enumerated reprojection/laser kernels already used. The single divergent
path (the no-distortion `ReprojPointPinhole4`, which normalized via
`z + 1e-9`) was used only by a validation unit test, never by a production
problem builder; the drift is ~1e-9 relative in normalized coordinates
(≈4e-7 px at f≈800), far below every test tolerance. The now-unused
`math::projection` module was removed.

Equivalence was pinned before deletion: every enumerated factor was asserted
**bit-identical** to its descriptor replacement at the kernel level and
through the backend (the no-distortion case bounded < 1e-5 px). The pins
survive as golden-value tests captured from the enumerated kernels, plus
invariant tests (zero Scheimpflug tilt ≡ identity sensor; zero robot delta ≡
plain hand-eye chain; rig chain ≡ explicit isometry composition).

## Consequences

- `ir/types.rs` validation collapsed from ~550 lines of per-variant arms to a
  ~30-line layout zip; the backend lost 15 concrete factor structs (net
  ≈ −1,900 LoC across the crate).
- Pinhole rig laserline is unblocked: laser factors always carry a sensor
  block and the rig laser bundle freezes it, so a pinhole upstream is exactly
  zero tilt. `RigHandeyeExport::to_upstream_calibration` and
  `pixel_to_gripper_point` now accept pinhole exports.
- A future second backend (the planned apex-solver integration) consumes the
  same descriptors and builds its own dispatch — nothing tiny-solver-specific
  leaks into the IR.
- Public (pre-1.0) breaking change: the enumerated `FactorKind` variants are
  gone; emitters construct descriptor factors. The laser factor names changed
  to the more accurate `LaserPointToPlane` (1D, meters) and
  `LaserLineDistance` (1D, pixels); the config-level
  `LaserlineResidualType` enum is unchanged.

## Alternatives considered

- **`dyn CameraProject` inside factors** — rejected: the trait is `f64`-only;
  dyn dispatch cannot be generic over the autodiff scalar `T`.
- **Matching the model enum per residual evaluation** — rejected: puts a
  growing enum match (and its branch mispredictions) inside the hot autodiff
  loop; monomorphizing once per factor keeps evaluation straight-line.
- **Const-generic dimensions** (`Factor<const D: usize>`) — rejected: fights
  variable-size distortion blocks and complicates the backend object graph
  without removing the descriptor enum.
- **Keeping enumerated variants and generating them with macros** — rejected:
  hides, rather than removes, the product-growth; validation and backend
  arms still multiply.
