//! EPnP (Efficient Perspective-n-Point) solver for camera pose estimation.
//!
//! Implements Lepetit's control-point formulation for 4+ points. Uses
//! eigen decomposition of the world point covariance to define a minimal
//! basis, then solves for control point positions in the camera frame.

use super::planar::planar_pose;
use super::pose_utils::pose_from_points;
use crate::GeometryError;
use nalgebra::{DMatrix, Vector3, linalg::SymmetricEigen};
use vision_calibration_core::{FxFyCxCy, Iso3, Mat3, Pt2, Pt3, Real, Vec3};

/// A spread along a principal axis below this fraction (as a variance ratio,
/// `1e-8` is `1e-4` in length) of the largest makes the points flat (or, for
/// two axes, collinear).
const FLAT_RATIO: Real = 1e-8;

/// EPnP pose estimation for 4+ points.
///
/// Uses a control-point formulation derived from the covariance of the
/// 3D points. Returns a single pose estimate in `T_C_W` form. Points on a
/// plane (a calibration board) are solved from the plane's homography, which
/// the control-point formulation cannot do; collinear points are an error.
///
/// # Errors
///
/// Returns [`GeometryError::InsufficientData`] if fewer than 4 correspondences
/// are given or `world.len() != image.len()`, [`GeometryError::Degenerate`] if
/// the points are collinear, or [`GeometryError::Singular`] if the intrinsics
/// `k` are non-invertible or the control-point system is degenerate.
pub fn epnp(world: &[Pt3], image: &[Pt2], k: &FxFyCxCy<Real>) -> Result<Iso3, GeometryError> {
    let n = world.len();
    if n < 4 || image.len() != n {
        return Err(GeometryError::InsufficientData { need: 4, got: n });
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx.try_inverse().ok_or(GeometryError::Singular)?;

    let mut img_norm = Vec::with_capacity(n);
    for pi in image {
        let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
        img_norm.push(Pt2::new(v.x / v.z, v.y / v.z));
    }

    let mut centroid = Vec3::zeros();
    for p in world {
        centroid += p.coords;
    }
    centroid /= n as Real;

    let mut cov = Mat3::zeros();
    for p in world {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    cov /= n as Real;

    let eig = SymmetricEigen::new(cov);
    let axes = eig.eigenvectors;
    let vals = eig.eigenvalues;

    // The control points are the centroid and the principal axes scaled by
    // the spread along them, so a flat point set has a degenerate fourth
    // control point: a plane is solved from its homography instead.
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let [l0, l1, l2] = order.map(|i| vals[i].max(0.0));
    if l0 <= 0.0 || l1 <= FLAT_RATIO * l0 {
        return Err(GeometryError::Degenerate {
            reason: "the world points are collinear".into(),
        });
    }
    if l2 <= FLAT_RATIO * l0 {
        let e1 = axes.column(order[0]).into_owned();
        let e2 = axes.column(order[1]).into_owned();
        let plane_axes = Mat3::from_columns(&[e1, e2, e1.cross(&e2)]);
        return planar_pose(world, &img_norm, &centroid, &plane_axes);
    }

    let mut control_w = [Vec3::zeros(); 4];
    control_w[0] = centroid;
    for i in 0..3 {
        let scale = vals[i].abs().sqrt();
        let axis = axes.column(i).into_owned();
        control_w[i + 1] = centroid + axis * scale;
    }

    let basis = Mat3::from_columns(&[
        control_w[1] - control_w[0],
        control_w[2] - control_w[0],
        control_w[3] - control_w[0],
    ]);
    let basis_inv = basis.try_inverse().ok_or(GeometryError::Singular)?;

    let mut alphas = Vec::with_capacity(n);
    for p in world {
        let coeff = basis_inv * (p.coords - control_w[0]);
        let a0 = 1.0 - coeff.x - coeff.y - coeff.z;
        alphas.push([a0, coeff.x, coeff.y, coeff.z]);
    }

    let mut m = DMatrix::<Real>::zeros(2 * n, 12);
    for (i, (a, uv)) in alphas.iter().zip(img_norm.iter()).enumerate() {
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        let u = uv.x;
        let v = uv.y;
        for (j, &alpha) in a.iter().enumerate().take(4) {
            let c = 3 * j;
            m[(r0, c)] = alpha;
            m[(r0, c + 2)] = -u * alpha;
            m[(r1, c + 1)] = alpha;
            m[(r1, c + 2)] = -v * alpha;
        }
    }

    // Solve `M x = 0` for the smallest right-singular vector via `AᵀA` symmetric
    // eigen (see `math::null_space`) — avoids nalgebra's hang-prone dense SVD on
    // the tall `2N×12` design matrix.
    let sol = vision_calibration_core::linalg::null_space(&m)?.vector;

    let mut control_c = [Vec3::zeros(); 4];
    for (j, cc) in control_c.iter_mut().enumerate() {
        *cc = Vec3::new(sol[3 * j], sol[3 * j + 1], sol[3 * j + 2]);
    }

    let mut sum_w = 0.0;
    let mut sum_c = 0.0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dw = (control_w[i] - control_w[j]).norm();
            let dc = (control_c[i] - control_c[j]).norm();
            sum_w += dw * dw;
            sum_c += dc * dc;
        }
    }

    if sum_c <= Real::EPSILON {
        return Err(GeometryError::Singular);
    }

    let scale = (sum_w / sum_c).sqrt();
    for cc in &mut control_c {
        *cc *= scale;
    }

    let mut camera_pts = Vec::with_capacity(n);
    for a in &alphas {
        let mut pc = Vec3::zeros();
        for (j, &alpha) in a.iter().enumerate().take(4) {
            pc += control_c[j] * alpha;
        }
        camera_pts.push(pc);
    }

    // The null vector is defined up to sign; the points lie ahead of the
    // camera, so the sign that puts them at positive depth is the solution.
    if camera_pts.iter().map(|p| p.z).sum::<Real>() < 0.0 {
        for p in &mut camera_pts {
            *p = -*p;
        }
    }

    pose_from_points(world, &camera_pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3};
    use vision_calibration_core::{Camera, IdentitySensor, NoDistortion, Pinhole};

    #[test]
    fn epnp_recovers_pose_synthetic() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);

        let rot = Rotation3::from_euler_angles(-0.1, 0.05, 0.2);
        let t = Translation3::new(0.1, -0.05, 1.2);
        let iso_gt = Isometry3::from_parts(t, rot.into());

        let mut world = Vec::new();
        let mut image = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw = Pt3::new(x as Real * 0.1, y as Real * 0.1, 0.6 + z as Real * 0.1);
                    let pc = iso_gt.transform_point(&pw);
                    let uv = cam.project_point(&pc).unwrap();
                    world.push(pw);
                    image.push(uv);
                }
            }
        }

        let est = epnp(&world, &image, &k).unwrap();

        let dt = (est.translation.vector - iso_gt.translation.vector).norm();
        let r_est = est.rotation.to_rotation_matrix();
        let r_gt = iso_gt.rotation.to_rotation_matrix();
        let r_diff = r_est.transpose() * r_gt;
        let trace = r_diff.matrix().trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang = cos_theta.acos();

        assert!(dt < 1e-3, "translation error too large: {}", dt);
        assert!(ang < 1e-3, "rotation error too large: {}", ang);
    }

    /// A shallow relief (not flat, so not the planar branch) in front of the
    /// camera: the null vector's sign must not put the points behind it.
    #[test]
    fn epnp_points_ahead_of_the_camera_for_a_shallow_object() {
        let k = FxFyCxCy {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let world: Vec<Pt3> = (0..35)
            .map(|i| {
                Pt3::new(
                    0.03 * f64::from(i % 7) - 0.09,
                    0.03 * f64::from(i / 7) - 0.06,
                    0.04 * f64::from(i * 5 % 7) / 6.0,
                )
            })
            .collect();
        let truth = Isometry3::from_parts(
            Translation3::new(0.02, -0.01, 0.7),
            Rotation3::from_euler_angles(0.25, -0.35, 0.1).into(),
        );
        let image: Vec<Pt2> = world
            .iter()
            .map(|p| {
                let c = truth * p;
                Pt2::new(c.x / c.z, c.y / c.z)
            })
            .collect();
        let est = epnp(&world, &image, &k).unwrap();
        assert!((est.translation.vector - truth.translation.vector).norm() < 1e-6);
        assert!(est.rotation.angle_to(&truth.rotation) < 1e-6);
    }

    /// Points on a plane are solved from the plane's homography.
    #[test]
    fn epnp_solves_a_planar_object_and_rejects_a_line() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 360.0,
        };
        let truth = Isometry3::from_parts(
            Translation3::new(-0.05, 0.02, 0.9),
            Rotation3::from_euler_angles(0.3, 0.2, -0.4).into(),
        );
        let project = |p: &Pt3| {
            let c = truth * p;
            Pt2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy)
        };
        let plane: Vec<Pt3> = (0..20)
            .map(|i| Pt3::new(0.04 * f64::from(i % 5), 0.04 * f64::from(i / 5), 0.0))
            .collect();
        let image: Vec<Pt2> = plane.iter().map(project).collect();
        let est = epnp(&plane, &image, &k).unwrap();
        assert!((est.translation.vector - truth.translation.vector).norm() < 1e-8);
        assert!(est.rotation.angle_to(&truth.rotation) < 1e-8);

        let line: Vec<Pt3> = (0..6)
            .map(|i| Pt3::new(0.04 * f64::from(i), 0.0, 0.0))
            .collect();
        let image: Vec<Pt2> = line.iter().map(project).collect();
        assert!(matches!(
            epnp(&line, &image, &k),
            Err(GeometryError::Degenerate { .. })
        ));
    }
}
