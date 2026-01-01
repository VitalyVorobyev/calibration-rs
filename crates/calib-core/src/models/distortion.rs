use nalgebra::{RealField, Vector2};
use serde::{Deserialize, Serialize};

/// Distortion model mapping between ideal and distorted normalized coordinates.
pub trait DistortionModel<S: RealField + Copy> {
    /// Apply distortion to undistorted normalized coordinates.
    fn distort(&self, n_undist: &Vector2<S>) -> Vector2<S>;
    /// Remove distortion from distorted normalized coordinates.
    fn undistort(&self, n_dist: &Vector2<S>) -> Vector2<S>;
}

/// No distortion (identity mapping).
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

/// Brown-Conrady 5-parameter radial-tangential distortion model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct BrownConrady5<S: RealField> {
    /// Radial coefficient k1.
    pub k1: S,
    /// Radial coefficient k2.
    pub k2: S,
    /// Radial coefficient k3.
    pub k3: S,
    /// Tangential coefficient p1.
    pub p1: S,
    /// Tangential coefficient p2.
    pub p2: S,
    /// Iterations for undistortion.
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
            x -= ex;
            y -= ey;
        }
        Vector2::new(x, y)
    }
}
