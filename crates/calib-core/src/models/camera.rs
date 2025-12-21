use nalgebra::{Point3, RealField, Vector2, Vector3};

use super::{DistortionModel, IntrinsicsModel, ProjectionModel, SensorModel};

#[derive(Clone, Copy, Debug)]
pub struct Ray<S: RealField + Copy> {
    pub dir: Vector3<S>,
}

#[derive(Clone, Debug)]
pub struct Camera<S, P, D, Sm, K>
where
    S: RealField + Copy,
    P: ProjectionModel<S>,
    D: DistortionModel<S>,
    Sm: SensorModel<S>,
    K: IntrinsicsModel<S>,
{
    pub proj: P,
    pub dist: D,
    pub sensor: Sm,
    pub k: K,
    _phantom: core::marker::PhantomData<S>,
}

impl<S, P, D, Sm, K> Camera<S, P, D, Sm, K>
where
    S: RealField + Copy,
    P: ProjectionModel<S>,
    D: DistortionModel<S>,
    Sm: SensorModel<S>,
    K: IntrinsicsModel<S>,
{
    pub fn new(proj: P, dist: D, sensor: Sm, k: K) -> Self {
        Self {
            proj,
            dist,
            sensor,
            k,
            _phantom: core::marker::PhantomData,
        }
    }

    pub fn project_point_c(&self, p_c: &Vector3<S>) -> Option<Vector2<S>> {
        if p_c.z <= S::zero() {
            return None;
        }
        let dir = *p_c;
        let n_u = self.proj.project_dir(&dir)?;
        let n_d = self.dist.distort(&n_u);
        let s = self.sensor.to_sensor(&n_d);
        Some(self.k.to_pixel(&s))
    }

    pub fn project_point(&self, p_c: &Point3<S>) -> Option<Vector2<S>> {
        self.project_point_c(&p_c.coords)
    }

    pub fn backproject_pixel(&self, px: &Vector2<S>) -> Ray<S> {
        let s = self.k.from_pixel(px);
        let n_d = self.sensor.from_sensor(&s);
        let n_u = self.dist.undistort(&n_d);
        let dir = self.proj.unproject_dir(&n_u);
        let dir = dir / dir.norm();
        Ray { dir }
    }
}
