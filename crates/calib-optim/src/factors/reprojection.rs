//! Reprojection residual factors.

use crate::math::projection::project_pinhole;
use nalgebra::{DVector, Point3, RealField, Vector2, Vector3};
use tiny_solver::factors::Factor;
use tiny_solver::manifold::se3::SE3;

/// Single-point reprojection residual for planar calibration.
#[derive(Debug, Clone)]
pub struct ReprojPointFactor {
    pub pw: Point3<f64>,
    pub uv: Vector2<f64>,
    pub w: f64,
}

impl ReprojPointFactor {
    fn residual_generic<T: RealField>(&self, cam: &DVector<T>, pose: &DVector<T>) -> DVector<T> {
        debug_assert!(cam.len() >= 4, "intrinsics must have 4 params");
        debug_assert!(pose.len() == 7, "pose must have 7 params");

        let fx = cam[0].clone();
        let fy = cam[1].clone();
        let cx = cam[2].clone();
        let cy = cam[3].clone();

        let se3 = SE3::<T>::from_vec(pose.as_view());
        let pw_t = Vector3::new(
            T::from_f64(self.pw.x).unwrap(),
            T::from_f64(self.pw.y).unwrap(),
            T::from_f64(self.pw.z).unwrap(),
        );
        let pc = se3 * pw_t.as_view();

        let proj = project_pinhole(fx, fy, cx, cy, pc);
        let sqrt_w = T::from_f64(self.w.sqrt()).unwrap();
        let u_meas = T::from_f64(self.uv.x).unwrap();
        let v_meas = T::from_f64(self.uv.y).unwrap();
        let ru = (u_meas - proj.x.clone()) * sqrt_w.clone();
        let rv = (v_meas - proj.y.clone()) * sqrt_w;
        nalgebra::dvector![ru, rv]
    }
}

impl<T: RealField> Factor<T> for ReprojPointFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(params.len(), 2, "expected [cam, pose] parameter blocks");
        self.residual_generic(&params[0], &params[1])
    }
}
