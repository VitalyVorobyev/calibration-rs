use nalgebra::{Matrix3, RealField, Vector2};
use serde::{Deserialize, Serialize};

pub trait IntrinsicsModel<S: RealField + Copy> {
    fn to_pixel(&self, s: &Vector2<S>) -> Vector2<S>;
    fn from_pixel(&self, px: &Vector2<S>) -> Vector2<S>;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FxFyCxCySkew<S: RealField + Copy> {
    pub fx: S,
    pub fy: S,
    pub cx: S,
    pub cy: S,
    pub skew: S,
}

impl<S: RealField + Copy> FxFyCxCySkew<S> {
    pub fn k_matrix(&self) -> Matrix3<S> {
        Matrix3::new(
            self.fx,
            self.skew,
            self.cx,
            S::zero(),
            self.fy,
            self.cy,
            S::zero(),
            S::zero(),
            S::one(),
        )
    }
}

impl<S: RealField + Copy> IntrinsicsModel<S> for FxFyCxCySkew<S> {
    fn to_pixel(&self, s: &Vector2<S>) -> Vector2<S> {
        let u = self.fx * s.x + self.skew * s.y + self.cx;
        let v = self.fy * s.y + self.cy;
        Vector2::new(u, v)
    }

    fn from_pixel(&self, px: &Vector2<S>) -> Vector2<S> {
        let sy = (px.y - self.cy) / self.fy;
        let sx = (px.x - self.cx - self.skew * sy) / self.fx;
        Vector2::new(sx, sy)
    }
}
