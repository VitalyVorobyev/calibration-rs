use serde::{Deserialize, Serialize};

use crate::{Mat3, Pt3, Vec2};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
}

impl CameraIntrinsics {
    pub fn matrix(&self) -> Mat3 {
        Mat3::new(
            self.fx, self.skew, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RadialTangential {
    BrownConrady {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    },
    // extend later
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinholeCamera {
    pub intrinsics: CameraIntrinsics,
    pub distortion: Option<RadialTangential>,
}

impl PinholeCamera {
    /// Project a 3D point in camera coords to distorted image coords (pixels).
    pub fn project(&self, p_c: &Pt3) -> Vec2 {
        let x = p_c.x / p_c.z;
        let y = p_c.y / p_c.z;

        let (x_d, y_d) = if let Some(dist) = self.distortion {
            distort_normalised(dist, x, y)
        } else {
            (x, y)
        };

        let k = &self.intrinsics;
        let u = k.fx * x_d + k.skew * y_d + k.cx;
        let v = k.fy * y_d + k.cy;
        Vec2::new(u, v)
    }
}

fn distort_normalised(model: RadialTangential, x: f64, y: f64) -> (f64, f64) {
    match model {
        RadialTangential::BrownConrady { k1, k2, k3, p1, p2 } => {
            let r2 = x * x + y * y;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            let x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

            (x * radial + x_t, y * radial + y_t)
        }
    }
}
