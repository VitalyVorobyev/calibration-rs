//! factrs residuals over the IR factor kernels.
//!
//! Each residual unpacks its factrs variables into the IR's parameter blocks
//! (`&[DVector<T>]`, in the factor's slot order) and calls the same generic
//! kernel the tiny-solver backend calls. factrs differentiates it in forward
//! mode with a static-size dual of width `DIN`, the residual's total tangent
//! dimension.

use std::fmt;
use std::marker::PhantomData;

use factrs::containers::{Factor, FactorBuilder, Key, Values};
use factrs::linalg::{
    AllocatorBuffer, Const, DefaultAllocator, DiffResult, DualAllocator, DualVector, ForwardProp,
    MatrixX, Numeric, VectorX,
};
use factrs::residuals::{
    Residual, Residual1, Residual2, Residual3, Residual4, Residual5, Residual6,
};
use factrs::variables::{SE3, VectorVar};
use nalgebra::{DVector, RealField};

use super::robust::IrRobustCost;
use super::variables::{PlaneVar, se3_to_ir};
use crate::factors::camera_kernels::{DistortionKernel, ProjectionKernel, SensorKernel};
use crate::factors::laserline::{
    laser_line_distance_model_generic, laser_point_to_plane_model_generic,
};
use crate::factors::reprojection_model::reproj_residual_model_generic;
use crate::ir::{FactorKind, LaserChain, ReprojChain, RobustLoss};

/// Size of the fused camera variable for distortion `D` and sensor `S`.
const fn camera_dim<D: DistortionKernel, S: SensorKernel>() -> usize {
    4 + D::DIM + S::DIM
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

impl<T: Numeric> IrBlocks<T> for VectorVar<6, T> {
    fn push_blocks(&self, out: &mut Vec<DVector<T>>) {
        out.push(DVector::from_column_slice(self.0.as_slice()));
    }
}

/// Splits a fused camera variable into its intrinsics, distortion (if any)
/// and sensor (if any) blocks.
fn push_camera<D: DistortionKernel, S: SensorKernel, const N: usize, T: Numeric>(
    camera: &VectorVar<N, T>,
    out: &mut Vec<DVector<T>>,
) {
    let v = camera.0.as_slice();
    let intrinsics = N - D::DIM - S::DIM;
    out.push(DVector::from_column_slice(&v[..intrinsics]));
    if D::DIM > 0 {
        out.push(DVector::from_column_slice(
            &v[intrinsics..intrinsics + D::DIM],
        ));
    }
    if S::DIM > 0 {
        out.push(DVector::from_column_slice(&v[N - S::DIM..]));
    }
}

/// A reprojection factor's data.
#[derive(Clone, Debug)]
struct Reproj {
    chain: ReprojChain,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl Reproj {
    fn eval<P, D, S, T>(&self, params: &[DVector<T>]) -> VectorX<T>
    where
        P: ProjectionKernel,
        D: DistortionKernel,
        S: SensorKernel,
        T: RealField,
    {
        let r = reproj_residual_model_generic::<P, D, S, T>(
            &self.chain,
            params,
            self.pw,
            self.uv,
            self.w,
        );
        DVector::from_column_slice(r.as_slice())
    }
}

/// Which laser residual a [`Laser`] factor evaluates.
#[derive(Clone, Copy, Debug)]
enum LaserMetric {
    PointToPlane,
    LineDistance,
}

/// A laser factor's data.
#[derive(Clone, Debug)]
struct Laser {
    metric: LaserMetric,
    chain: LaserChain,
    laser_pixel: [f64; 2],
    w: f64,
}

impl Laser {
    fn eval<P, D, S, T>(&self, params: &[DVector<T>]) -> VectorX<T>
    where
        P: ProjectionKernel,
        D: DistortionKernel,
        S: SensorKernel,
        T: RealField,
    {
        let r = match self.metric {
            LaserMetric::PointToPlane => laser_point_to_plane_model_generic::<D, S, T>(
                &self.chain,
                params,
                self.laser_pixel,
                self.w,
            ),
            LaserMetric::LineDistance => laser_line_distance_model_generic::<D, S, T>(
                &self.chain,
                params,
                self.laser_pixel,
                self.w,
            ),
        };
        DVector::from_column_slice(r.as_slice())
    }
}

/// Defines a residual over a fused camera variable followed by chain
/// variables, for one factrs arity.
macro_rules! camera_residual {
    (
        $(#[$doc:meta])*
        $name:ident: $trait:ident { $method:ident, $values:ident, $jacobian:ident },
        $payload:ty, out = $out:literal,
        chain = [$($V:ident $v:ident: $VT:ty),*]
    ) => {
        $(#[$doc])*
        struct $name<P, D, S, const N: usize, const DIN: usize> {
            payload: $payload,
            _kernels: PhantomData<fn() -> (P, D, S)>,
        }

        impl<P, D, S, const N: usize, const DIN: usize> $name<P, D, S, N, DIN> {
            fn new(payload: $payload) -> Self {
                Self {
                    payload,
                    _kernels: PhantomData,
                }
            }
        }

        // By hand: a derive would require the kernel markers to be
        // `Clone + Debug`.
        impl<P, D, S, const N: usize, const DIN: usize> Clone for $name<P, D, S, N, DIN> {
            fn clone(&self) -> Self {
                Self::new(self.payload.clone())
            }
        }

        impl<P, D, S, const N: usize, const DIN: usize> fmt::Debug for $name<P, D, S, N, DIN> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("payload", &self.payload)
                    .finish()
            }
        }

        impl<P, D, S, const N: usize, const DIN: usize> Residual for $name<P, D, S, N, DIN>
        where
            P: ProjectionKernel + 'static,
            D: DistortionKernel + 'static,
            S: SensorKernel + 'static,
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

        impl<P, D, S, const N: usize, const DIN: usize> $trait for $name<P, D, S, N, DIN>
        where
            P: ProjectionKernel + 'static,
            D: DistortionKernel + 'static,
            S: SensorKernel + 'static,
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
                push_camera::<D, S, N, T>(&camera, &mut params);
                $($v.push_blocks(&mut params);)*
                self.payload.eval::<P, D, S, T>(&params)
            }
        }
    };
}

camera_residual! {
    /// Reprojection, `[camera, camera_se3_target]`.
    ReprojSinglePose: Residual2 { residual2, residual2_values, residual2_jacobian },
    Reproj, out = 2,
    chain = [V2 pose: SE3]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, pose]`.
    ReprojTwoSe3: Residual3 { residual3, residual3_values, residual3_jacobian },
    Reproj, out = 2,
    chain = [V2 extrinsics: SE3, V3 pose: SE3]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, handeye, target]`.
    ReprojHandEye: Residual4 { residual4, residual4_values, residual4_jacobian },
    Reproj, out = 2,
    chain = [V2 extrinsics: SE3, V3 handeye: SE3, V4 target: SE3]
}

camera_residual! {
    /// Reprojection, `[camera, extrinsics, handeye, target, robot_delta]`.
    ReprojHandEyeRobotDelta: Residual5 { residual5, residual5_values, residual5_jacobian },
    Reproj, out = 2,
    chain = [V2 extrinsics: SE3, V3 handeye: SE3, V4 target: SE3, V5 delta: VectorVar<6>]
}

camera_residual! {
    /// Laser, `[camera, camera_se3_target, plane]`.
    LaserSinglePose: Residual3 { residual3, residual3_values, residual3_jacobian },
    Laser, out = 1,
    chain = [V2 pose: SE3, V3 plane: PlaneVar]
}

camera_residual! {
    /// Laser, `[camera, cam_se3_rig, handeye, target_ref, plane]`.
    LaserRigHandEye: Residual5 { residual5, residual5_values, residual5_jacobian },
    Laser, out = 1,
    chain = [V2 rig: SE3, V3 handeye: SE3, V4 target: SE3, V5 plane: PlaneVar]
}

camera_residual! {
    /// Laser, `[camera, cam_se3_rig, handeye, target_ref, plane, robot_delta]`.
    LaserRigHandEyeRobotDelta: Residual6 { residual6, residual6_values, residual6_jacobian },
    Laser, out = 1,
    chain = [
        V2 rig: SE3,
        V3 handeye: SE3,
        V4 target: SE3,
        V5 plane: PlaneVar,
        V6 delta: VectorVar<6>
    ]
}

/// Zero-mean prior on a 6-D se(3) tangent correction, scaled element-wise by
/// the diagonal square-root information.
#[derive(Clone, Debug)]
struct TangentPrior {
    sqrt_info: [f64; 6],
}

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
        DVector::from_fn(6, |i, _| delta.0[i] * T::from(self.sqrt_info[i]))
    }
}

/// The factrs factor for an IR factor whose variables are `keys`: the fused
/// camera variable first (absent for the tangent prior), then one key per
/// chain variable in slot order.
pub(super) fn build_factor(factor: &FactorKind, keys: &[Key], loss: RobustLoss) -> Factor {
    let robust = IrRobustCost(loss);
    match factor {
        FactorKind::Se3TangentPrior { sqrt_info } => FactorBuilder::new1_unchecked(
            TangentPrior {
                sqrt_info: *sqrt_info,
            },
            keys[0],
        )
        .robust(robust)
        .build(),
        FactorKind::ReprojPoint {
            model,
            chain,
            pw,
            uv,
            w,
        } => {
            let payload = Reproj {
                chain: *chain,
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            macro_rules! mk {
                ($P:ty, $D:ty, $S:ty) => {{
                    const N: usize = camera_dim::<$D, $S>();
                    match chain {
                        ReprojChain::SinglePose => FactorBuilder::new2_unchecked(
                            ReprojSinglePose::<$P, $D, $S, N, { N + 6 }>::new(payload),
                            keys[0],
                            keys[1],
                        )
                        .robust(robust)
                        .build(),
                        ReprojChain::TwoSe3 => FactorBuilder::new3_unchecked(
                            ReprojTwoSe3::<$P, $D, $S, N, { N + 12 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                        )
                        .robust(robust)
                        .build(),
                        ReprojChain::HandEye { .. } => FactorBuilder::new4_unchecked(
                            ReprojHandEye::<$P, $D, $S, N, { N + 18 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                            keys[3],
                        )
                        .robust(robust)
                        .build(),
                        ReprojChain::HandEyeRobotDelta { .. } => FactorBuilder::new5_unchecked(
                            ReprojHandEyeRobotDelta::<$P, $D, $S, N, { N + 24 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                            keys[3],
                            keys[4],
                        )
                        .robust(robust)
                        .build(),
                    }
                }};
            }
            crate::backend::dispatch_camera_model!(model, mk)
        }
        FactorKind::LaserPointToPlane {
            model,
            chain,
            laser_pixel,
            w,
        }
        | FactorKind::LaserLineDistance {
            model,
            chain,
            laser_pixel,
            w,
        } => {
            let metric = if matches!(factor, FactorKind::LaserPointToPlane { .. }) {
                LaserMetric::PointToPlane
            } else {
                LaserMetric::LineDistance
            };
            let payload = Laser {
                metric,
                chain: *chain,
                laser_pixel: *laser_pixel,
                w: *w,
            };
            macro_rules! mk {
                ($P:ty, $D:ty, $S:ty) => {{
                    const N: usize = camera_dim::<$D, $S>();
                    match chain {
                        LaserChain::SinglePose => FactorBuilder::new3_unchecked(
                            LaserSinglePose::<$P, $D, $S, N, { N + 9 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                        )
                        .robust(robust)
                        .build(),
                        LaserChain::RigHandEye { .. } => FactorBuilder::new5_unchecked(
                            LaserRigHandEye::<$P, $D, $S, N, { N + 21 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                            keys[3],
                            keys[4],
                        )
                        .robust(robust)
                        .build(),
                        LaserChain::RigHandEyeRobotDelta { .. } => FactorBuilder::new6_unchecked(
                            LaserRigHandEyeRobotDelta::<$P, $D, $S, N, { N + 27 }>::new(payload),
                            keys[0],
                            keys[1],
                            keys[2],
                            keys[3],
                            keys[4],
                            keys[5],
                        )
                        .robust(robust)
                        .build(),
                    }
                }};
            }
            crate::backend::dispatch_camera_model!(model, mk)
        }
    }
}
