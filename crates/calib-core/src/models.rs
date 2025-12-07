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

/// Common interface for camera projection / back-projection models.
///
/// This trait is intentionally small and focuses on geometric operations.
/// Implementations are provided for [`PinholeCamera`] and
/// [`ScheimpflugCamera`], and can be used from generic calibration code.
pub trait CameraModel {
    /// Project a 3D point in camera coordinates onto the image plane.
    fn project(&self, p_c: &Pt3) -> Vec2;

    /// Unproject a pixel with a given depth into a 3D point.
    ///
    /// Returns `None` if the operation is not defined (e.g. non-positive
    /// depth or singular intrinsics / homographies).
    fn unproject(&self, uv: &Vec2, depth: Real) -> Option<Pt3>;

    /// Compute a unit ray direction from a pixel coordinate.
    ///
    /// This is equivalent to normalising any valid unprojected 3D point
    /// lying on the ray defined by `uv`.
    fn unproject_ray(&self, uv: &Vec2) -> Option<Vec3>;
}

/// Radial–tangential distortion models supported by camera models.
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

impl CameraModel for PinholeCamera {
    fn project(&self, p_c: &Pt3) -> Vec2 {
        self.project(p_c)
    }

    fn unproject(&self, uv: &Vec2, depth: Real) -> Option<Pt3> {
        self.unproject(uv, depth)
    }

    fn unproject_ray(&self, uv: &Vec2) -> Option<Vec3> {
        self.unproject_ray(uv)
    }
}

/// Pinhole camera with an oblique image plane following the Scheimpflug
/// principle.
///
/// This model composes a standard [`PinholeCamera`] with a 3×3 homography
/// acting on the image plane. The homography maps *ideal* pinhole pixel
/// coordinates into *physical* sensor coordinates, and can be used to model
/// sensor tilt (as in a tilt–shift or Scheimpflug camera).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugCamera {
    /// Underlying pinhole camera (intrinsics + lens distortion).
    pub pinhole: PinholeCamera,
    /// Homography from ideal pinhole pixels to physical sensor pixels.
    pub sensor_h: Mat3,
}

impl ScheimpflugCamera {
    /// Project a 3D point in camera coordinates onto the (tilted) sensor.
    ///
    /// This is implemented as:
    /// 1. project with the underlying [`PinholeCamera`],
    /// 2. apply the image-plane homography `sensor_h`.
    pub fn project(&self, p_c: &Pt3) -> Vec2 {
        let uv_pinhole = self.pinhole.project(p_c);
        let p = Vec3::new(uv_pinhole.x, uv_pinhole.y, 1.0);
        let q = self.sensor_h * p;
        Vec2::new(q.x / q.z, q.y / q.z)
    }

    /// Map a sensor pixel back to the underlying pinhole image coordinates.
    ///
    /// Returns `None` if `sensor_h` is not invertible.
    fn sensor_to_pinhole(&self, uv_sensor: &Vec2) -> Option<Vec2> {
        let h_inv = self.sensor_h.try_inverse()?;
        let p = Vec3::new(uv_sensor.x, uv_sensor.y, 1.0);
        let q = h_inv * p;
        Some(Vec2::new(q.x / q.z, q.y / q.z))
    }

    /// Unproject a sensor pixel and depth into a 3D point in camera coords.
    ///
    /// This inverts the sensor homography and then delegates to the underlying
    /// pinhole camera's [`PinholeCamera::unproject`] implementation.
    pub fn unproject(&self, uv_sensor: &Vec2, depth: Real) -> Option<Pt3> {
        let uv_pinhole = self.sensor_to_pinhole(uv_sensor)?;
        self.pinhole.unproject(&uv_pinhole, depth)
    }

    /// Compute a unit ray direction in camera coordinates for a sensor pixel.
    ///
    /// This inverts the sensor homography and then delegates to
    /// [`PinholeCamera::unproject_ray`].
    pub fn unproject_ray(&self, uv_sensor: &Vec2) -> Option<Vec3> {
        let uv_pinhole = self.sensor_to_pinhole(uv_sensor)?;
        self.pinhole.unproject_ray(&uv_pinhole)
    }
}

impl CameraModel for ScheimpflugCamera {
    fn project(&self, p_c: &Pt3) -> Vec2 {
        self.project(p_c)
    }

    fn unproject(&self, uv: &Vec2, depth: Real) -> Option<Pt3> {
        self.unproject(uv, depth)
    }

    fn unproject_ray(&self, uv: &Vec2) -> Option<Vec3> {
        self.unproject_ray(uv)
    }
}

/// Convenience enum covering the built-in camera model types.
///
/// This is useful when calibration code needs to operate on different models
/// without monomorphisation over the [`CameraModel`] trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenericCamera {
    /// Standard pinhole model with Brown–Conrady distortion.
    Pinhole(PinholeCamera),
    /// Pinhole model with a tilted sensor, following the Scheimpflug principle.
    Scheimpflug(ScheimpflugCamera),
}

impl CameraModel for GenericCamera {
    fn project(&self, p_c: &Pt3) -> Vec2 {
        match self {
            GenericCamera::Pinhole(cam) => cam.project(p_c),
            GenericCamera::Scheimpflug(cam) => cam.project(p_c),
        }
    }

    fn unproject(&self, uv: &Vec2, depth: Real) -> Option<Pt3> {
        match self {
            GenericCamera::Pinhole(cam) => cam.unproject(uv, depth),
            GenericCamera::Scheimpflug(cam) => cam.unproject(uv, depth),
        }
    }

    fn unproject_ray(&self, uv: &Vec2) -> Option<Vec3> {
        match self {
            GenericCamera::Pinhole(cam) => cam.unproject_ray(uv),
            GenericCamera::Scheimpflug(cam) => cam.unproject_ray(uv),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheimpflug_identity_matches_pinhole_projection() {
        let k = CameraIntrinsics {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let pinhole = PinholeCamera {
            intrinsics: k,
            distortion: Some(RadialTangential::BrownConrady {
                k1: -0.1,
                k2: 0.01,
                p1: 0.001,
                p2: -0.001,
                k3: 0.0,
            }),
        };

        let scheimpflug = ScheimpflugCamera {
            pinhole: pinhole.clone(),
            sensor_h: Mat3::identity(),
        };

        let pts = [
            Pt3::new(0.0, 0.0, 1.0),
            Pt3::new(0.1, -0.05, 1.2),
            Pt3::new(-0.2, 0.15, 2.0),
        ];

        for p in &pts {
            let u_p = pinhole.project(p);
            let u_s = scheimpflug.project(p);
            assert!((u_p.x - u_s.x).abs() < 1e-9);
            assert!((u_p.y - u_s.y).abs() < 1e-9);
        }
    }

    #[test]
    fn scheimpflug_roundtrip_project_unproject() {
        let k = CameraIntrinsics {
            fx: 600.0,
            fy: 590.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let pinhole = PinholeCamera {
            intrinsics: k,
            distortion: None,
        };

        // Simple invertible homography: translate the sensor.
        let sensor_h = Mat3::new(
            1.0, 0.0, 10.0, //
            0.0, 1.0, -5.0, //
            0.0, 0.0, 1.0,
        );

        let cam = ScheimpflugCamera { pinhole, sensor_h };

        let pts = [
            Pt3::new(0.0, 0.0, 1.0),
            Pt3::new(0.1, 0.2, 1.5),
            Pt3::new(-0.3, 0.1, 2.0),
        ];

        for p in &pts {
            let uv = cam.project(p);
            let depth = p.z;
            let p_back = cam
                .unproject(&uv, depth)
                .expect("unproject should succeed for valid depth");

            assert!((p_back.x - p.x).abs() < 1e-6, "x mismatch");
            assert!((p_back.y - p.y).abs() < 1e-6, "y mismatch");
            assert!((p_back.z - p.z).abs() < 1e-6, "z mismatch");
        }
    }
}
