//! factrs residuals over the shared factor kernels.
//!
//! Each residual unpacks its factrs variables into the IR's parameter blocks
//! (`&[DVector<T>]`, in the factor's slot order) and evaluates the same
//! [`Kernel`] the tiny-solver backend evaluates. factrs differentiates it in
//! forward mode with a static-size dual of width `DIN`, the residual's total
//! tangent dimension, so each variable layout is its own residual type,
//! instantiated per fused camera size.

use std::fmt;

use factrs::containers::{Factor, FactorBuilder, Key, Values};
use factrs::linalg::{
    AllocatorBuffer, Const, DefaultAllocator, DiffResult, DualAllocator, DualVector, ForwardProp,
    MatrixX, Numeric, VectorX,
};
use factrs::residuals::{
    Residual, Residual1, Residual2, Residual3, Residual4, Residual5, Residual6,
};
use factrs::variables::{SE3, VectorVar};
use nalgebra::DVector;

use super::robust::IrRobustCost;
use super::variables::{PlaneVar, se3_to_ir};
use crate::backend::kernels::Kernel;
use crate::factors::camera_blocks::CameraBlocks;
use crate::ir::{CameraModelSpec, FactorKind, LaserChain, ReprojChain, RobustLoss};

/// Size of the fused camera variable of a camera model.
fn camera_dim(model: CameraModelSpec) -> usize {
    4 + model.projection.dim() + model.distortion.dim() + model.sensor.dim()
}

/// Appends a variable's IR parameter blocks to a kernel's argument list.
trait IrBlocks<T> {
    fn push_blocks(&self, out: &mut Vec<DVector<T>>);
}

impl<T: Numeric> IrBlocks<T> for SE3<T> {
    fn push_blocks(&self, out: &mut Vec<DVector<T>>) {
        out.push(se3_to_ir(self));
    }
}

impl<T: Numeric> IrBlocks<T> for PlaneVar<T> {
    fn push_blocks(&self, out: &mut Vec<DVector<T>>) {
        out.push(DVector::from_column_slice(self.normal.as_slice()));
        out.push(DVector::from_element(1, self.distance));
    }
}

impl<const D: usize, T: Numeric> IrBlocks<T> for VectorVar<D, T> {
    fn push_blocks(&self, out: &mut Vec<DVector<T>>) {
        out.push(DVector::from_column_slice(self.0.as_slice()));
    }
}

/// Splits a fused camera variable into its intrinsics, projection,
/// distortion and sensor blocks (each present only when non-empty).
fn push_camera<const N: usize, T: Numeric>(
    model: CameraModelSpec,
    camera: &VectorVar<N, T>,
    out: &mut Vec<DVector<T>>,
) {
    let (camera, _) = CameraBlocks::from_flat(model, camera.0.as_slice());
    out.extend(camera.blocks().map(DVector::from_column_slice));
}

/// Defines a residual over a fused camera variable followed by chain
/// variables, for one factrs arity.
macro_rules! camera_residual {
    (
        $(#[$doc:meta])*
        $name:ident: $trait:ident { $method:ident, $values:ident, $jacobian:ident },
        out = $out:literal,
        chain = [$($V:ident $v:ident: $VT:ty),*]
    ) => {
        $(#[$doc])*
        struct $name<const N: usize, const DIN: usize> {
            /// The kernel's camera, read once.
            model: CameraModelSpec,
            kernel: Kernel,
        }

        impl<const N: usize, const DIN: usize> $name<N, DIN> {
            fn new(model: CameraModelSpec, kernel: Kernel) -> Self {
                Self { model, kernel }
            }
        }

        // By hand: a derive would bound the const parameters needlessly.
        impl<const N: usize, const DIN: usize> Clone for $name<N, DIN> {
            fn clone(&self) -> Self {
                Self::new(self.model, self.kernel.clone())
            }
        }

        impl<const N: usize, const DIN: usize> fmt::Debug for $name<N, DIN> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("kernel", &self.kernel)
                    .finish()
            }
        }

        impl<const N: usize, const DIN: usize> Residual for $name<N, DIN>
        where
            AllocatorBuffer<Const<DIN>>: Sync + Send,
            DefaultAllocator: DualAllocator<Const<DIN>>,
            DualVector<Const<DIN>>: Copy,
        {
            fn dim_in(&self) -> usize {
                DIN
            }

            fn dim_out(&self) -> usize {
                $out
            }

            fn residual(&self, values: &Values, keys: &[Key]) -> VectorX {
                $trait::$values(self, values, keys)
            }

            fn residual_jacobian(
                &self,
                values: &Values,
                keys: &[Key],
            ) -> DiffResult<VectorX, MatrixX> {
                $trait::$jacobian(self, values, keys)
            }
        }

        impl<const N: usize, const DIN: usize> $trait for $name<N, DIN>
        where
            AllocatorBuffer<Const<DIN>>: Sync + Send,
            DefaultAllocator: DualAllocator<Const<DIN>>,
            DualVector<Const<DIN>>: Copy,
        {
            type V1 = VectorVar<N>;
            $(type $V = $VT;)*
            type DimIn = Const<DIN>;
            type DimOut = Const<$out>;
            type Differ = ForwardProp<Const<DIN>>;

            fn $method<T: Numeric>(
                &self,
                camera: VectorVar<N, T>,
                $($v: <$VT as factrs::variables::Variable>::Alias<T>,)*
            ) -> VectorX<T> {
                let mut params = Vec::with_capacity(9);
                push_camera(self.model, &camera, &mut params);
                $($v.push_blocks(&mut params);)*
                self.kernel.eval(&params)
            }
        }
    };
}

camera_residual! {
    /// Reprojection, `[camera, cam_se3_target]`.
    ReprojSinglePose: Residual2 { residual2, residual2_values, residual2_jacobian },
    out = 2,
    chain = [V2 pose: SE3]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, pose]`.
    ReprojTwoSe3: Residual3 { residual3, residual3_values, residual3_jacobian },
    out = 2,
    chain = [V2 extrinsics: SE3, V3 pose: SE3]
}

camera_residual! {
    /// Reprojection of an estimated point, `[camera, cam_se3_target,
    /// point]`.
    ReprojFreePose: Residual3 { residual3, residual3_values, residual3_jacobian },
    out = 2,
    chain = [V2 pose: SE3, V3 point: VectorVar<3>]
}

camera_residual! {
    /// Reprojection of an estimated point, `[camera, cam_se3_rig,
    /// rig_se3_target, point]`.
    ReprojFreeRig: Residual4 { residual4, residual4_values, residual4_jacobian },
    out = 2,
    chain = [V2 rig: SE3, V3 pose: SE3, V4 point: VectorVar<3>]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, handeye, target]`.
    ReprojHandeye: Residual4 { residual4, residual4_values, residual4_jacobian },
    out = 2,
    chain = [V2 extrinsics: SE3, V3 handeye: SE3, V4 target: SE3]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, handeye, target, robot_delta]`.
    ReprojHandeyeRobotDelta: Residual5 { residual5, residual5_values, residual5_jacobian },
    out = 2,
    chain = [V2 extrinsics: SE3, V3 handeye: SE3, V4 target: SE3, V5 delta: VectorVar<6>]
}

camera_residual! {
    /// Laser, `[camera, cam_se3_target, plane]`.
    LaserSinglePose: Residual3 { residual3, residual3_values, residual3_jacobian },
    out = 1,
    chain = [V2 pose: SE3, V3 plane: PlaneVar]
}

camera_residual! {
    /// Laser, `[camera, cam_se3_rig, handeye, target_ref, plane]`.
    LaserRigHandeye: Residual5 { residual5, residual5_values, residual5_jacobian },
    out = 1,
    chain = [V2 rig: SE3, V3 handeye: SE3, V4 target: SE3, V5 plane: PlaneVar]
}

camera_residual! {
    /// Laser, `[camera, cam_se3_rig, handeye, target_ref, plane, robot_delta]`.
    LaserRigHandeyeRobotDelta: Residual6 { residual6, residual6_values, residual6_jacobian },
    out = 1,
    chain = [
        V2 rig: SE3,
        V3 handeye: SE3,
        V4 target: SE3,
        V5 plane: PlaneVar,
        V6 delta: VectorVar<6>
    ]
}

/// Zero-mean prior on a 6-D se(3) tangent correction, `[robot_delta]`.
#[derive(Clone, Debug)]
struct TangentPrior(Kernel);

impl Residual for TangentPrior {
    fn dim_in(&self) -> usize {
        6
    }

    fn dim_out(&self) -> usize {
        6
    }

    fn residual(&self, values: &Values, keys: &[Key]) -> VectorX {
        Residual1::residual1_values(self, values, keys)
    }

    fn residual_jacobian(&self, values: &Values, keys: &[Key]) -> DiffResult<VectorX, MatrixX> {
        Residual1::residual1_jacobian(self, values, keys)
    }
}

impl Residual1 for TangentPrior {
    type V1 = VectorVar<6>;
    type DimIn = Const<6>;
    type DimOut = Const<6>;
    type Differ = ForwardProp<Const<6>>;

    fn residual1<T: Numeric>(&self, delta: VectorVar<6, T>) -> VectorX<T> {
        let mut params = Vec::with_capacity(1);
        delta.push_blocks(&mut params);
        self.0.eval(&params)
    }
}

/// Expands `$mk!(N)` for the fused camera size `$n`: the sizes that the
/// camera models produce (intrinsics plus projection, distortion and sensor
/// parameters). factrs needs the variable size at compile time, so this is
/// a table over sizes, not over models.
macro_rules! dispatch_camera_dim {
    ($n:expr, $mk:ident) => {
        match $n {
            4 => $mk!(4),
            5 => $mk!(5),
            6 => $mk!(6),
            7 => $mk!(7),
            8 => $mk!(8),
            9 => $mk!(9),
            10 => $mk!(10),
            11 => $mk!(11),
            12 => $mk!(12),
            13 => $mk!(13),
            14 => $mk!(14),
            15 => $mk!(15),
            n => unreachable!("compile rejects a fused camera of {n} parameters"),
        }
    };
}

/// The factrs factor of `kernel` whose variables are `keys`: the fused
/// camera variable first (absent for the tangent prior), then one key per
/// chain variable in slot order.
pub(super) fn build_factor(kernel: Kernel, keys: &[Key], loss: RobustLoss) -> Factor {
    let robust = IrRobustCost(loss);
    let Some(model) = kernel.camera() else {
        debug_assert!(matches!(
            kernel.factor(),
            FactorKind::Se3TangentPrior { .. }
        ));
        return FactorBuilder::new1_unchecked(TangentPrior(kernel), keys[0])
            .robust(robust)
            .build();
    };
    macro_rules! mk {
        ($n:literal) => {{
            const N: usize = $n;
            match kernel.factor() {
                FactorKind::ReprojPoint { chain, .. } => match chain {
                    ReprojChain::SinglePose => FactorBuilder::new2_unchecked(
                        ReprojSinglePose::<N, { N + 6 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                    )
                    .robust(robust)
                    .build(),
                    ReprojChain::TwoSe3 | ReprojChain::Rig => FactorBuilder::new3_unchecked(
                        ReprojTwoSe3::<N, { N + 12 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                    )
                    .robust(robust)
                    .build(),
                    ReprojChain::Handeye { .. } => FactorBuilder::new4_unchecked(
                        ReprojHandeye::<N, { N + 18 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                        keys[3],
                    )
                    .robust(robust)
                    .build(),
                    ReprojChain::HandeyeRobotDelta { .. } => FactorBuilder::new5_unchecked(
                        ReprojHandeyeRobotDelta::<N, { N + 24 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                        keys[3],
                        keys[4],
                    )
                    .robust(robust)
                    .build(),
                },
                FactorKind::ReprojFreePoint { chain, .. } => match chain {
                    ReprojChain::Rig => FactorBuilder::new4_unchecked(
                        ReprojFreeRig::<N, { N + 15 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                        keys[3],
                    )
                    .robust(robust)
                    .build(),
                    // The IR admits a free point through a single pose or a
                    // rig only.
                    _ => FactorBuilder::new3_unchecked(
                        ReprojFreePose::<N, { N + 9 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                    )
                    .robust(robust)
                    .build(),
                },
                FactorKind::LaserPointToPlane { chain, .. }
                | FactorKind::LaserLineDistance { chain, .. } => match chain {
                    LaserChain::SinglePose => FactorBuilder::new3_unchecked(
                        LaserSinglePose::<N, { N + 9 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                    )
                    .robust(robust)
                    .build(),
                    LaserChain::RigHandeye { .. } => FactorBuilder::new5_unchecked(
                        LaserRigHandeye::<N, { N + 21 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                        keys[3],
                        keys[4],
                    )
                    .robust(robust)
                    .build(),
                    LaserChain::RigHandeyeRobotDelta { .. } => FactorBuilder::new6_unchecked(
                        LaserRigHandeyeRobotDelta::<N, { N + 27 }>::new(model, kernel),
                        keys[0],
                        keys[1],
                        keys[2],
                        keys[3],
                        keys[4],
                        keys[5],
                    )
                    .robust(robust)
                    .build(),
                },
                FactorKind::Se3TangentPrior { .. } | FactorKind::Custom(_) => {
                    unreachable!("a factor with a camera")
                }
            }
        }};
    }
    dispatch_camera_dim!(camera_dim(model), mk)
}
