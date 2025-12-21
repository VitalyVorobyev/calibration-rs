use nalgebra::{RealField, Vector2};
use serde::{Deserialize, Serialize};

pub trait DistortionModel<S: RealField + Copy> {
    fn distort(&self, n_undist: &Vector2<S>) -> Vector2<S>;
    fn undistort(&self, n_dist: &Vector2<S>) -> Vector2<S>;
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct NoDistortion;

impl<S: RealField + Copy> DistortionModel<S> for NoDistortion {
    fn distort(&self, n_undist: &Vector2<S>) -> Vector2<S> {
        *n_undist
    }

    fn undistort(&self, n_dist: &Vector2<S>) -> Vector2<S> {
        *n_dist
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct BrownConrady5<S: RealField> {
    pub k1: S,
    pub k2: S,
    pub k3: S,
    pub p1: S,
    pub p2: S,
    pub iters: u32,
}

impl<S: RealField + Copy> BrownConrady5<S> {
    fn distort_impl(&self, x: S, y: S) -> (S, S) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let two = S::one() + S::one();
        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;

        let x_tan = two * self.p1 * xy + self.p2 * (r2 + two * x2);
        let y_tan = self.p1 * (r2 + two * y2) + two * self.p2 * xy;

        (x * radial + x_tan, y * radial + y_tan)
    }
}

impl<S: RealField + Copy> DistortionModel<S> for BrownConrady5<S> {
    fn distort(&self, n_undist: &Vector2<S>) -> Vector2<S> {
        let (xd, yd) = self.distort_impl(n_undist.x, n_undist.y);
        Vector2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Vector2<S>) -> Vector2<S> {
        let mut x = n_dist.x;
        let mut y = n_dist.y;

        let iters = if self.iters == 0 { 8 } else { self.iters };
        for _ in 0..iters {
            let (xd, yd) = self.distort_impl(x, y);
            let ex = xd - n_dist.x;
            let ey = yd - n_dist.y;
            x = x - ex;
            y = y - ey;
        }
        Vector2::new(x, y)
    }
}
