//! The information one view of a planar target carries about a camera's
//! parameters.
//!
//! A view observes target points `X` through the camera, each corner a
//! pixel `q(θ, T X)` with `θ` the camera's parameters and `T` the view's
//! `camera_se3_target`. Per unit pixel variance, the view's Fisher
//! information over `[θ, δ]`, with `δ` the pose's tangent coordinates, is
//! `A = Σ JᵀJ`, `J = ∂q/∂[θ, δ]`. A pose the solve estimates is not known
//! and takes part of that information: marginalizing it leaves the Schur
//! complement
//!
//! ```text
//! S = A_θθ − A_θδ A_δδ⁻¹ A_δθ
//! ```
//!
//! over `θ` alone. Views are independent given `θ`, so the information of a
//! set of views is the sum of their `S`. A pose held fixed contributes
//! `A_θθ`.
//!
//! `J` is exact: the forward-mode derivative of the camera kernels the solver
//! projects with. The pose is perturbed on the left, `Exp(δ) · T`; its
//! tangent spans the same directions as the solver's right perturbation, and
//! the Schur complement does not depend on how the marginalized block is
//! parametrized.

use nalgebra::{DMatrix, DVector, SVector, Vector3};
use num_dual::{DualDVec64, jacobian};
use vision_calibration_core::{CameraParams, Iso3, Pt3};

use crate::Error;
use crate::factors::camera_blocks::CameraBlocks;
use crate::ir::CameraModelSpec;
use crate::projection::CameraVector;

/// Whether a view's pose is estimated with the camera, or known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewPose {
    /// The solve estimates the pose: it is marginalized out of the
    /// information.
    Estimated,
    /// The solve holds the pose fixed.
    Fixed,
}

/// The information a view carries about a camera's parameters, per unit
/// pixel variance.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewInformation {
    /// The camera's parameter labels, in the order of
    /// [`camera_parameters`](crate::camera_parameters).
    pub labels: Vec<&'static str>,
    /// The information over those parameters, in px⁻² per unit of each
    /// parameter: symmetric, one row per label.
    pub matrix: DMatrix<f64>,
}

/// The information the view `camera_se3_target` of `points` (target frame,
/// metres) carries about `camera`'s parameters, per unit pixel variance; an
/// estimated pose is marginalized (see the module documentation).
///
/// Every point counts: the caller selects the ones the view observes.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a camera model the solver does not estimate
/// (a general sensor homography) or a point not in front of the camera;
/// [`Error::Numerical`] when the points do not determine an estimated pose
/// (fewer than three, or a degenerate configuration).
pub fn view_information(
    camera: &CameraParams,
    camera_se3_target: &Iso3,
    points: &[Pt3],
    pose: ViewPose,
) -> Result<ViewInformation, Error> {
    let model = CameraVector::new(camera)?;
    let n = model.theta.len();
    // Variables: θ, then the camera-frame point; the point's derivative is
    // mapped to the pose tangent after.
    let mut x = model.theta.clone().resize_vertically(n + 3, 0.0);
    let mut a = DMatrix::<f64>::zeros(n + 6, n + 6);
    for point in points {
        let p_c = camera_se3_target.transform_point(point).coords;
        if !p_c.z.is_finite() || p_c.z <= 0.0 {
            return Err(Error::invalid_input(format!(
                "target point {point} is not in front of the camera"
            )));
        }
        x.rows_mut(n, 3).copy_from(&p_c);
        let (q, j) = jacobian(|v: DVector<DualDVec64>| project(model.model, &v, n), &x);
        if !q.iter().chain(j.iter()).all(|v| v.is_finite()) {
            return Err(Error::numerical(format!(
                "the projection of target point {point} is not finite"
            )));
        }
        let j = to_pose_tangent(&j, n, &p_c);
        a.gemm_tr(1.0, &j, &j, 1.0);
    }
    let a_tt = a.view((0, 0), (n, n)).into_owned();
    let matrix = match pose {
        ViewPose::Fixed => a_tt,
        ViewPose::Estimated => {
            let a_tp = a.view((0, n), (n, 6)).into_owned();
            let a_pp = a.view((n, n), (6, 6)).into_owned();
            let a_pp_inv = invert_spd(&a_pp)
                .ok_or_else(|| Error::numerical("the view's points do not determine its pose"))?;
            let s = a_tt - &a_tp * a_pp_inv * a_tp.transpose();
            // Symmetric by construction; average out the rounding.
            (&s + s.transpose()) * 0.5
        }
    };
    Ok(ViewInformation {
        labels: model.labels,
        matrix,
    })
}

/// The pixel of camera-frame point `v[n..n + 3]` for camera parameters
/// `v[..n]`.
fn project(model: CameraModelSpec, v: &DVector<DualDVec64>, n: usize) -> SVector<DualDVec64, 2> {
    let all = v.as_slice();
    let (intrinsics, rest) = all.split_at(4);
    let (projection, rest) = rest.split_at(model.projection.dim());
    let (distortion, rest) = rest.split_at(model.distortion.dim());
    let sensor = &rest[..model.sensor.dim()];
    let camera = CameraBlocks::new(model, intrinsics, projection, distortion, sensor);
    let p = Vector3::new(v[n].clone(), v[n + 1].clone(), v[n + 2].clone());
    let (u, w) = camera.project(&p);
    SVector::<DualDVec64, 2>::new(u, w)
}

/// `J` over `[θ, p]`, `p` the camera-frame point in the last three columns,
/// to `J` over `[θ, ω, τ]`, the left-perturbation tangent: `p' = p + ω × p
/// + τ`, so `∂p'/∂ω = −[p]×` and `∂p'/∂τ = I`.
fn to_pose_tangent(
    j: &nalgebra::OMatrix<f64, nalgebra::U2, nalgebra::Dyn>,
    n: usize,
    p: &Vector3<f64>,
) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(2, n + 6);
    out.view_mut((0, 0), (2, n)).copy_from(&j.columns(0, n));
    let j_p = j.fixed_columns::<3>(n).into_owned();
    let j_omega = -j_p * p.cross_matrix();
    out.view_mut((0, n), (2, 3)).copy_from(&j_omega);
    out.view_mut((0, n + 3), (2, 3)).copy_from(&j_p);
    out
}

/// The inverse of a symmetric positive-definite matrix, factored in its
/// unit-diagonal form; `None` when it is not positive definite.
fn invert_spd(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let d: Vec<f64> = (0..n)
        .map(|i| a[(i, i)])
        .map(|v| if v > 0.0 { 1.0 / v.sqrt() } else { f64::NAN })
        .collect();
    if d.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scaled = DMatrix::from_fn(n, n, |i, j| a[(i, j)] * d[i] * d[j]);
    let chol = scaled.cholesky()?;
    // A pose the points barely determine (a near-degenerate configuration)
    // is treated as undetermined rather than inverted into noise.
    let diag = chol.l_dirty().diagonal();
    let (lo, hi) = diag.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
        (lo.min(*v), hi.max(*v))
    });
    if !lo.is_finite() || lo <= 1e-6 * hi {
        return None;
    }
    let inv = chol.inverse();
    Some(DMatrix::from_fn(n, n, |i, j| inv[(i, j)] * d[i] * d[j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{
        BrownConrady5, DistortionParams, FxFyCxCy, ProjectionParams, SensorParams,
    };

    fn camera() -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.1,
                    k2: 0.02,
                    k3: 0.0,
                    p1: 0.001,
                    p2: -0.0005,
                    iters: 8,
                },
            },
            sensor: SensorParams::Identity,
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 810.0,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    fn board() -> Vec<Pt3> {
        (0..6)
            .flat_map(|r| (0..8).map(move |c| Pt3::new(0.03 * c as f64, 0.03 * r as f64, 0.0)))
            .collect()
    }

    fn pose() -> Iso3 {
        Iso3::from_parts(
            Translation3::new(-0.1, -0.07, 0.6),
            Rotation3::from_euler_angles(0.2, -0.15, 0.05).into(),
        )
    }

    /// The pixel of `point` under camera parameters `theta` and the pose
    /// `Exp(δ) · T` (right-perturbed for this check, unlike the kernel).
    fn pixel(theta: &[f64], delta: &[f64; 6], point: &Pt3) -> [f64; 2] {
        let base = camera();
        let DistortionParams::BrownConrady5 { params } = base.distortion else {
            unreachable!()
        };
        let cam = CameraParams {
            intrinsics: FxFyCxCy {
                fx: theta[0],
                fy: theta[1],
                cx: theta[2],
                cy: theta[3],
            },
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: theta[4],
                    k2: theta[5],
                    k3: theta[6],
                    p1: theta[7],
                    p2: theta[8],
                    ..params
                },
            },
            ..base
        }
        .build()
        .unwrap();
        let tangent = Iso3::new(
            Vector3::new(delta[3], delta[4], delta[5]),
            Vector3::new(delta[0], delta[1], delta[2]),
        );
        let p = (pose() * tangent).transform_point(point);
        let q = cam.project_point_c(&p.coords).unwrap();
        [q.x, q.y]
    }

    /// Central-difference information over `[θ, δ]` (right perturbation),
    /// then its Schur complement over `θ`.
    fn numeric(pose_estimated: bool) -> DMatrix<f64> {
        let theta: Vec<f64> = CameraVector::new(&camera())
            .unwrap()
            .theta
            .iter()
            .copied()
            .collect();
        let n = theta.len();
        let mut a = DMatrix::<f64>::zeros(n + 6, n + 6);
        for point in board() {
            let mut j = DMatrix::<f64>::zeros(2, n + 6);
            for k in 0..n + 6 {
                // Step 1e-6 relative (absolute below 1): central differences
                // are accurate to about 1e-10 relative on these smooth models.
                let h = if k < n {
                    1e-6 * theta[k].abs().max(1.0)
                } else {
                    1e-6
                };
                let eval = |sign: f64| {
                    let mut t = theta.clone();
                    let mut d = [0.0; 6];
                    if k < n {
                        t[k] += sign * h;
                    } else {
                        d[k - n] = sign * h;
                    }
                    pixel(&t, &d, &point)
                };
                let (plus, minus) = (eval(1.0), eval(-1.0));
                j[(0, k)] = (plus[0] - minus[0]) / (2.0 * h);
                j[(1, k)] = (plus[1] - minus[1]) / (2.0 * h);
            }
            a += j.transpose() * &j;
        }
        let a_tt = a.view((0, 0), (n, n)).into_owned();
        if !pose_estimated {
            return a_tt;
        }
        let a_tp = a.view((0, n), (n, 6)).into_owned();
        let a_pp = a.view((n, n), (6, 6)).into_owned();
        a_tt - &a_tp * a_pp.try_inverse().unwrap() * a_tp.transpose()
    }

    fn assert_close(exact: &DMatrix<f64>, numeric: &DMatrix<f64>) {
        // Compare in the unit-diagonal frame, where every entry is O(1).
        let n = exact.nrows();
        let d: Vec<f64> = (0..n).map(|i| 1.0 / exact[(i, i)].sqrt()).collect();
        for i in 0..n {
            for j in 0..n {
                let (e, m) = (exact[(i, j)] * d[i] * d[j], numeric[(i, j)] * d[i] * d[j]);
                assert!((e - m).abs() < 1e-5, "({i}, {j}): exact {e}, numeric {m}");
            }
        }
    }

    #[test]
    fn marginal_information_matches_central_differences() {
        let info = view_information(&camera(), &pose(), &board(), ViewPose::Estimated).unwrap();
        assert_eq!(
            info.labels,
            ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]
        );
        assert_close(&info.matrix, &numeric(true));
    }

    #[test]
    fn a_fixed_pose_keeps_the_camera_block() {
        let info = view_information(&camera(), &pose(), &board(), ViewPose::Fixed).unwrap();
        assert_close(&info.matrix, &numeric(false));
    }

    #[test]
    fn a_pose_the_points_do_not_determine_is_an_error() {
        let two = &board()[..2];
        let err = view_information(&camera(), &pose(), two, ViewPose::Estimated).unwrap_err();
        assert!(matches!(err, Error::Numerical(_)), "{err}");
        // A fixed pose needs no determination.
        assert!(view_information(&camera(), &pose(), two, ViewPose::Fixed).is_ok());
    }

    #[test]
    fn a_point_behind_the_camera_is_rejected() {
        let behind = Iso3::translation(0.0, 0.0, -1.0);
        let err = view_information(&camera(), &behind, &board(), ViewPose::Estimated).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }
}
