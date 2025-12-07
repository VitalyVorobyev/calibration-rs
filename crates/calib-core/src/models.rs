use serde::{Deserialize, Serialize};

use crate::{Mat3, Pt3, Vec2, Vec3, Real};

/// Brown–Conrady-style camera intrinsics for a pinhole model.
///
/// The corresponding calibration matrix `K` has the form:
///
/// ```text
/// [ fx  skew  cx ]
/// [  0   fy   cy ]
/// [  0    0    1 ]
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
}

impl CameraIntrinsics {
    /// Build the 3×3 calibration matrix `K`.
    pub fn k_matrix(&self) -> Mat3 {
        Mat3::new(
            self.fx, self.skew, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }

    /// Attempt to construct intrinsics from a 3×3 calibration matrix `K`.
    ///
    /// The matrix is first normalised so that `K[2, 2] == 1`. Then it is
    /// checked against the standard form:
    ///
    /// ```text
    /// [ fx  skew  cx ]
    /// [  0   fy   cy ]
    /// [  0    0    1 ]
    /// ```
    ///
    /// If the structure does not match within a small tolerance, `None` is
    /// returned.
    pub fn try_from_k_matrix(k: &Mat3) -> Option<Self> {
        let mut k_norm = *k;
        let eps = 1e-9;

        let k33 = k_norm[(2, 2)];
        if k33.abs() < eps {
            return None;
        }
        k_norm /= k33;

        // Enforce standard lower row / zero structure.
        if k_norm[(1, 0)].abs() > eps
            || k_norm[(2, 0)].abs() > eps
            || k_norm[(2, 1)].abs() > eps
            || (k_norm[(2, 2)] - 1.0).abs() > eps
        {
            return None;
        }

        Some(Self {
            fx: k_norm[(0, 0)],
            skew: k_norm[(0, 1)],
            cx: k_norm[(0, 2)],
            fy: k_norm[(1, 1)],
            cy: k_norm[(1, 2)],
        })
    }
}

/// Radial–tangential distortion models supported by [`PinholeCamera`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RadialTangential {
    /// Classic Brown–Conrady 5-parameter model.
    BrownConrady {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    },
    // extend later
}

/// Simple pinhole camera model with optional radial–tangential distortion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinholeCamera {
    pub intrinsics: CameraIntrinsics,
    pub distortion: Option<RadialTangential>,
}

impl PinholeCamera {
    /// Project a 3D point in camera coordinates to distorted image coordinates (pixels).
    ///
    /// The input point `p_c` is expressed in the camera frame. Its `z`
    /// component must be non-zero; behaviour is undefined if `z == 0`.
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

    /// Undistort a pixel coordinate using the current distortion model.
    ///
    /// This maps a measured, **distorted** pixel `(u_d, v_d)` to an
    /// approximately ideal pixel `(u, v)` that would be observed with zero
    /// lens distortion but the same calibration matrix `K`.
    ///
    /// Returns `None` if the intrinsic matrix is singular.
    pub fn undistort(&self, uv_distorted: &Vec2) -> Option<Vec2> {
        let k = self.intrinsics.k_matrix();
        let k_inv = k.try_inverse()?;

        // Back-project distorted pixel into (approximate) normalised coordinates.
        let p_d = Vec3::new(uv_distorted.x, uv_distorted.y, 1.0);
        let p_n = k_inv * p_d;
        let x_d = p_n.x / p_n.z;
        let y_d = p_n.y / p_n.z;

        let (x_u, y_u) = match self.distortion {
            Some(model) => undistort_normalised(model, x_d, y_d),
            None => (x_d, y_d),
        };

        // Re-apply intrinsics to get undistorted pixel coordinates.
        let p_u = k * Vec3::new(x_u, y_u, 1.0);
        let u = p_u.x / p_u.z;
        let v = p_u.y / p_u.z;
        Some(Vec2::new(u, v))
    }

    /// Unproject a pixel and depth into a 3D point in camera coordinates.
    ///
    /// - `uv` is a **distorted** pixel coordinate,
    /// - `depth` is the distance along the optical axis (i.e. `z` in the
    ///   camera frame, in the same units as the 3D world you work in).
    ///
    /// Returns `None` if the intrinsic matrix is singular or if `depth <= 0`.
    pub fn unproject(&self, uv: &Vec2, depth: Real) -> Option<Pt3> {
        if depth <= 0.0 {
            return None;
        }

        let uv_undist = self.undistort(uv)?;
        let k = self.intrinsics.k_matrix();
        let k_inv = k.try_inverse()?;

        let p = Vec3::new(uv_undist.x, uv_undist.y, 1.0);
        let p_n = k_inv * p;
        let x = p_n.x / p_n.z;
        let y = p_n.y / p_n.z;

        Some(Pt3::new(x * depth, y * depth, depth))
    }

    /// Compute a unit-norm ray direction in camera coordinates for a pixel.
    ///
    /// This is equivalent to `self.unproject(uv, depth).coords.normalize()`
    /// for any positive depth, but does not require selecting a depth value.
    ///
    /// Returns `None` if the intrinsic matrix is singular.
    pub fn unproject_ray(&self, uv: &Vec2) -> Option<Vec3> {
        let uv_undist = self.undistort(uv)?;
        let k = self.intrinsics.k_matrix();
        let k_inv = k.try_inverse()?;

        let p = Vec3::new(uv_undist.x, uv_undist.y, 1.0);
        let p_n = k_inv * p;
        let dir = Vec3::new(p_n.x / p_n.z, p_n.y / p_n.z, 1.0);
        Some(dir.normalize())
    }
}

fn distort_normalised(model: RadialTangential, x: Real, y: Real) -> (Real, Real) {
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

fn undistort_normalised(model: RadialTangential, x_d: Real, y_d: Real) -> (Real, Real) {
    match model {
        RadialTangential::BrownConrady { k1, k2, k3, p1, p2 } => {
            // Fixed-point style iteration commonly used for Brown–Conrady:
            // start from distorted coords and iteratively "pull back".
            let mut x_u = x_d;
            let mut y_u = y_d;
            let max_iters = 10;
            let tol = 1e-12;

            for _ in 0..max_iters {
                let r2 = x_u * x_u + y_u * y_u;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                let x_t = 2.0 * p1 * x_u * y_u + p2 * (r2 + 2.0 * x_u * x_u);
                let y_t = p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * p2 * x_u * y_u;

                let x_u_new = (x_d - x_t) / radial;
                let y_u_new = (y_d - y_t) / radial;

                let dx = x_u_new - x_u;
                let dy = y_u_new - y_u;
                if dx.abs().max(dy.abs()) < tol {
                    x_u = x_u_new;
                    y_u = y_u_new;
                    break;
                }

                x_u = x_u_new;
                y_u = y_u_new;
            }

            (x_u, y_u)
        }
    }
}
