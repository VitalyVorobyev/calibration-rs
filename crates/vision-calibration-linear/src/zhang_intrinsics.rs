//! Zhang-style planar intrinsics initialization.
//!
//! Estimates the camera matrix `K` from multiple plane homographies without
//! distortion. This is a classic closed-form initialization used before
//! non-linear refinement.

use crate::Error;
use nalgebra::{DMatrix, Vector3};
use vision_calibration_core::{FxFyCxCy, Mat3, Real};

/// Build the 6-vector v_ij(H) as in Zhang's method.
/// i, j are 0- or 1-based column indices (we'll use 0/1 for (1,2) and (1,1),(2,2)).
fn v_ij(hmtx: &Mat3, i: usize, j: usize) -> nalgebra::SVector<Real, 6> {
    let hi = hmtx.column(i);
    let hj = hmtx.column(j);

    nalgebra::SVector::<Real, 6>::from_row_slice(&[
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])
}

/// Smallest spread of the board orientations, in degrees, that the planar
/// initialization accepts ([`check_board_orientations`]).
///
/// Parallel boards measure a spread at the level of the corner noise: about
/// 0.15° for a board 40 % of the image wide with 0.2 px noise. A capture
/// tilted by 3° still initializes, with a focal length uncertain by ±4 %,
/// and real captures spread over 8° or more. Below 1° the focal length is
/// uncertain by tens of percent at best and Zhang's solution is noise.
pub(crate) const MIN_BOARD_ORIENTATION_SPREAD_DEG: Real = 1.0;

/// The spread of the board orientations across views, in radians: the
/// largest angle between a view's board plane and their mean plane.
///
/// A homography `H = K [r1 r2 t]` maps the board's line at infinity to its
/// vanishing line `h1 × h2 ∝ K⁻ᵀ r3`, so `Kᵀ (h1 × h2)` is the board normal
/// `r3`. `reference` stands in for the unknown `K`. Parallel boards share
/// one vanishing line, so their spread is zero through any reference, up to
/// noise; for tilted boards, a reference focal length equal to the image
/// extent the views cover measures the perspective the views show, which is
/// what Zhang's constraints feed on. Normals are compared as lines (`n` and
/// `−n` alike), so the sign of `H` does not matter.
///
/// Returns `None` when fewer than two views give a finite normal.
pub(crate) fn board_orientation_spread(
    homographies: &[Mat3],
    reference: &FxFyCxCy<Real>,
) -> Option<Real> {
    let kt = reference.k_matrix().transpose();
    let normals: Vec<Vector3<Real>> = homographies
        .iter()
        .filter_map(|h| {
            let n = kt * h.column(0).cross(&h.column(1));
            let norm = n.norm();
            (norm.is_finite() && norm > 0.0).then(|| n / norm)
        })
        .collect();
    if normals.len() < 2 {
        return None;
    }
    // The mean plane: the dominant eigenvector of the normals' scatter.
    let scatter = normals
        .iter()
        .fold(Mat3::zeros(), |acc, n| acc + n * n.transpose());
    let eigen = scatter.symmetric_eigen();
    let mean = eigen
        .eigenvectors
        .column(eigen.eigenvalues.imax())
        .into_owned();
    normals
        .iter()
        .map(|n| n.cross(&mean).norm().atan2(n.dot(&mean).abs()))
        .reduce(Real::max)
}

/// Reject views whose boards are parallel: a board orientation spread
/// ([`board_orientation_spread`], through `reference`) below
/// [`MIN_BOARD_ORIENTATION_SPREAD_DEG`].
///
/// # Errors
///
/// [`Error::DegenerateViews`] with the measured spread.
pub(crate) fn check_board_orientations(
    homographies: &[Mat3],
    reference: &FxFyCxCy<Real>,
) -> Result<(), Error> {
    match board_orientation_spread(homographies, reference).map(Real::to_degrees) {
        Some(spread_deg) if spread_deg < MIN_BOARD_ORIENTATION_SPREAD_DEG => {
            Err(Error::DegenerateViews {
                views: homographies.len(),
                spread_deg: Some(spread_deg),
            })
        }
        _ => Ok(()),
    }
}

/// High-level linear initialisation of intrinsics for planar calibration.
///
/// This implements the closed-form solution proposed by Zhengyou Zhang for
/// estimating the calibration matrix `K` from multiple plane homographies.
#[derive(Debug, Clone, Copy)]
pub struct PlanarIntrinsicsLinearInit;

/// Estimate camera intrinsics `K` from a set of plane homographies `H_k`.
///
/// Uses Zhang's closed-form solution (no distortion). Requires at least three
/// homographies of boards at different orientations.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if fewer than 3 homographies are given,
/// [`Error::Singular`] if the linear solve fails on non-finite input, or
/// [`Error::DegenerateViews`] if the solution is no camera.
pub fn estimate_intrinsics_from_homographies(hmtxs: &[Mat3]) -> Result<FxFyCxCy<Real>, Error> {
    PlanarIntrinsicsLinearInit::from_homographies(hmtxs)
}

impl PlanarIntrinsicsLinearInit {
    /// Estimate camera intrinsics from a slice of plane homographies.
    ///
    /// Each homography maps points from **board coordinates** (assumed to be
    /// on `Z = 0`) into image coordinates.
    ///
    /// At least three views are required, of boards at different
    /// orientations: parallel boards (all fronto-parallel, or all tilted the
    /// same way) give every view the same two constraints on `K`.
    ///
    /// The solve recovers the conic `B = K⁻ᵀ K⁻¹` up to scale, and fails when
    /// it is not definite: no camera fits the homographies. With parallel
    /// boards and noisy corners it can instead return an arbitrary `K`; the
    /// dataset-level [`estimate_intrinsics_iterative`] rejects those views
    /// before the solve, from the spread of the board orientations, which
    /// needs the image extent the homographies do not carry.
    ///
    /// [`estimate_intrinsics_iterative`]: crate::iterative_intrinsics::estimate_intrinsics_iterative
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 3 homographies are given,
    /// [`Error::Singular`] if the linear solve fails on non-finite input, or
    /// [`Error::DegenerateViews`] if the solution is no camera.
    pub fn from_homographies(hmtxs: &[Mat3]) -> Result<FxFyCxCy<Real>, Error> {
        if hmtxs.len() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: hmtxs.len(),
            });
        }

        let m = hmtxs.len();
        let mut vmtx = DMatrix::<Real>::zeros(2 * m, 6);

        for (k, hmtx) in hmtxs.iter().enumerate() {
            let v11 = v_ij(hmtx, 0, 0);
            let v22 = v_ij(hmtx, 1, 1);
            let v12 = v_ij(hmtx, 0, 1);

            // Row 2k: v_12^T
            vmtx.row_mut(2 * k).copy_from(&v12.transpose());
            // Row 2k+1: (v_11 - v_22)^T
            vmtx.row_mut(2 * k + 1).copy_from(&(v11 - v22).transpose());
        }

        // Solve `V b = 0` for the smallest right-singular vector via `AᵀA`
        // symmetric eigen (see `math::null_space`) — avoids nalgebra's
        // hang-prone dense SVD on the `2M×6` design matrix. `V` is in pixel
        // units, not normalized, but its condition number grows only
        // linearly with the focal length (≈ 3.7e4 at 16 000 px), so squaring
        // it costs no accuracy that matters (`long_focal_lengths_stay_exact`).
        let b = vision_calibration_core::linalg::null_space(&vmtx)?.vector;

        let b11 = b[0];
        let b12 = b[1];
        let b22 = b[2];
        let b13 = b[3];
        let b23 = b[4];
        let b33 = b[5];

        // From Zhang's paper:
        //
        // v0 = (B12 B13 - B11 B23) / (B11 B22 - B12^2)
        // λ = B33 - (B13^2 + v0 (B12 B13 - B11 B23)) / B11
        // α = sqrt(λ / B11)
        // β = sqrt(λ B11 / (B11 B22 - B12^2))
        // γ = -B12 α^2 β / λ
        // u0 = γ v0 / β - B13 α^2 / λ
        //
        // The general solve recovers a skew γ (needed for u0); it is
        // discarded because the camera model has no skew term.

        // `B = μ K⁻ᵀ K⁻¹` is definite: `B11 B22 − B12²` positive and
        // `λ = μ` of the sign of `B11`. A `B` that is not has no camera
        // behind it: the views leave it undetermined (parallel boards) or
        // fit no camera at all (distortion or noise larger than the
        // perspective the views show).
        let degenerate = Error::DegenerateViews {
            views: m,
            spread_deg: None,
        };
        let denom = b11 * b22 - b12 * b12;
        let denom_norm = b11 * b11 + b22 * b22;
        let definite_2x2 = denom_norm > 0.0 && denom / denom_norm > 1e-6;
        if !definite_2x2 || b11.abs() <= 1e-12 {
            return Err(degenerate);
        }

        let v0 = (b12 * b13 - b11 * b23) / denom;
        let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;

        let definite = lambda / b11 > 0.0;
        if !definite {
            return Err(degenerate);
        }

        let alpha = (lambda / b11).sqrt();
        let beta = (lambda * b11 / denom).sqrt();
        let gamma = -b12 * alpha * alpha * beta / lambda;
        let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

        Ok(FxFyCxCy {
            fx: alpha,
            fy: beta,
            cx: u0,
            cy: v0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Vector3};

    fn make_kmtx() -> (FxFyCxCy<Real>, Mat3) {
        let intr = FxFyCxCy {
            fx: 900.0,
            fy: 880.0,
            cx: 640.0,
            cy: 360.0,
        };
        let kmtx = Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0);
        (intr, kmtx)
    }

    fn synthetic_homography(kmtx: &Mat3, rot: Rotation3<Real>, t: Vector3<Real>) -> Mat3 {
        // Pose of board in camera frame
        let iso = Isometry3::from_parts(Translation3::from(t), rot.into());

        // For Z=0 plane, H = K [r1 r2 t]
        let binding = iso.rotation.to_rotation_matrix();
        let r_mat = binding.matrix();
        let r1 = r_mat.column(0);
        let r2 = r_mat.column(1);

        let mut hmtx = Mat3::zeros();
        hmtx.set_column(0, &(kmtx * r1));
        hmtx.set_column(1, &(kmtx * r2));
        hmtx.set_column(2, &(kmtx * t));
        hmtx
    }

    #[test]
    fn intrinsics_from_homographies_recovers_kmtx() {
        let (intr_gt, kmtx) = make_kmtx();

        // Three different board poses
        let hmts: Vec<Mat3> = vec![
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(0.1, 0.0, 0.05),
                Vector3::new(0.1, -0.05, 1.0),
            ),
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(-0.05, 0.15, -0.1),
                Vector3::new(-0.05, 0.1, 1.2),
            ),
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(0.2, -0.1, 0.0),
                Vector3::new(0.0, 0.0, 0.9),
            ),
        ];

        let intr_est = estimate_intrinsics_from_homographies(&hmts).unwrap();

        // Tolerances are somewhat arbitrary; adjust based on your geometry
        assert!((intr_est.fx - intr_gt.fx).abs() < 5.0, "fx mismatch");
        assert!((intr_est.fy - intr_gt.fy).abs() < 5.0, "fy mismatch");
        assert!((intr_est.cx - intr_gt.cx).abs() < 10.0, "cx mismatch");
        assert!((intr_est.cy - intr_gt.cy).abs() < 10.0, "cy mismatch");
    }

    /// Exact homographies of boards that all share the orientation `tilt`,
    /// turned in their plane and moved from view to view: parallel planes.
    fn parallel_homographies(kmtx: &Mat3, tilt: Rotation3<Real>) -> Vec<Mat3> {
        (0..6)
            .map(|k| {
                let spin = Rotation3::from_euler_angles(0.0, 0.0, 0.4 * k as Real);
                let t = Vector3::new(0.08 * (k % 3) as Real - 0.1, 0.05 * (k % 2) as Real, 1.0);
                synthetic_homography(kmtx, tilt * spin, t + 0.1 * k as Real * Vector3::z())
            })
            .collect()
    }

    /// Exact homographies of boards tilted by `tilt_rad` about +x, +y, −x
    /// and −y in turn: normals `tilt_rad` from the optical axis, around it.
    fn tilted_homographies(kmtx: &Mat3, tilt_rad: Real) -> Vec<Mat3> {
        let axes = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        axes.iter()
            .enumerate()
            .map(|(k, &(ax, ay))| {
                let rot = Rotation3::from_euler_angles(ax * tilt_rad, ay * tilt_rad, 0.0)
                    * Rotation3::from_euler_angles(0.0, 0.0, 0.3 * k as Real);
                synthetic_homography(kmtx, rot, Vector3::new(0.05 * ax, 0.05 * ay, 1.0))
            })
            .collect()
    }

    fn spread_deg(hmtxs: &[Mat3], reference: &FxFyCxCy<Real>) -> Real {
        board_orientation_spread(hmtxs, reference)
            .expect("finite normals")
            .to_degrees()
    }

    #[test]
    fn parallel_boards_have_no_spread_through_any_reference() {
        let (intr, kmtx) = make_kmtx();
        let wrong = FxFyCxCy {
            fx: 2500.0,
            fy: 2500.0,
            cx: 100.0,
            cy: 900.0,
        };
        for tilt in [
            Rotation3::identity(),
            Rotation3::from_euler_angles(0.35, 0.0, 0.0),
            Rotation3::from_euler_angles(0.2, -0.3, 0.0),
        ] {
            let hmtxs = parallel_homographies(&kmtx, tilt);
            for reference in [&intr, &wrong] {
                assert!(spread_deg(&hmtxs, reference) < 1e-9);
                assert!(matches!(
                    check_board_orientations(&hmtxs, reference),
                    Err(Error::DegenerateViews {
                        views: 6,
                        spread_deg: Some(_)
                    })
                ));
            }
        }
    }

    #[test]
    fn spread_is_the_tilt_through_the_true_camera() {
        let (intr, kmtx) = make_kmtx();
        for tilt_deg in [0.5, 3.0, 20.0] {
            let mut hmtxs = tilted_homographies(&kmtx, Real::to_radians(tilt_deg));
            // Neither the sign nor the scale of a homography matters.
            hmtxs[1] *= -3.0;
            let spread = spread_deg(&hmtxs, &intr);
            assert!(
                (spread - tilt_deg).abs() < 1e-6 * tilt_deg,
                "{spread} vs {tilt_deg}"
            );
        }
    }

    #[test]
    fn check_rejects_spreads_below_the_threshold_only() {
        let (intr, kmtx) = make_kmtx();
        let below = tilted_homographies(&kmtx, Real::to_radians(0.8));
        match check_board_orientations(&below, &intr) {
            Err(Error::DegenerateViews {
                views: 4,
                spread_deg: Some(spread),
            }) => assert!((spread - 0.8).abs() < 1e-6),
            other => panic!("expected DegenerateViews, got {other:?}"),
        }
        let above = tilted_homographies(&kmtx, Real::to_radians(1.25));
        assert!(check_board_orientations(&above, &intr).is_ok());
    }

    /// The design matrix is built from pixel-unit homographies and solved
    /// through its normal matrix, which squares its condition number. That
    /// number grows only linearly with the focal length (2.3–2.8 · f for
    /// these views with unit-norm homographies, 3.7e4 at 16 000 px), so the
    /// squared one stays far inside double precision: exact homographies,
    /// unit-norm as a DLT returns them or in their raw `K [r1 r2 t]` scale,
    /// give `K` to 1e-9 relative from 800 to 16 000 px.
    #[test]
    fn long_focal_lengths_stay_exact() {
        for (f, w, h) in [
            (800.0, 1280.0, 960.0),
            (4000.0, 4096.0, 3000.0),
            (8000.0, 6000.0, 4000.0),
            (16000.0, 8192.0, 5460.0),
            (16000.0, 2448.0, 2048.0),
        ] {
            let truth = FxFyCxCy {
                fx: f,
                fy: 1.002 * f,
                cx: 0.5 * w + 7.0,
                cy: 0.5 * h - 5.0,
            };
            let kmtx = truth.k_matrix();
            // Ten views of a 0.32 m board spanning half the image width,
            // tilted 10°–40° in varied directions.
            let depth = 2.0 * f * 0.32 / w;
            let raw: Vec<Mat3> = (0..10)
                .map(|k| {
                    let kf = k as Real;
                    let tilt = (10.0 + 30.0 * kf / 9.0).to_radians();
                    let dir = kf * 2.4;
                    let rot = Rotation3::from_euler_angles(
                        tilt * dir.cos(),
                        tilt * dir.sin(),
                        0.2 * (1.7 * kf).sin(),
                    );
                    let t = Vector3::new(0.03 * (0.9 * kf).sin(), 0.03 * (1.3 * kf).cos(), 1.0)
                        * depth
                        - rot * Vector3::new(0.16, 0.12, 0.0);
                    synthetic_homography(&kmtx, rot, t)
                })
                .collect();
            let unit: Vec<Mat3> = raw.iter().map(|h| h / h.norm()).collect();
            for hmtxs in [&raw, &unit] {
                let est = estimate_intrinsics_from_homographies(hmtxs).unwrap();
                let rel = [
                    (est.fx - truth.fx) / truth.fx,
                    (est.fy - truth.fy) / truth.fy,
                    (est.cx - truth.cx) / truth.fx,
                    (est.cy - truth.cy) / truth.fy,
                ];
                assert!(
                    rel.iter().all(|e| e.abs() < 1e-9),
                    "f = {f} px, {w}×{h}: relative errors {rel:?}"
                );
            }
        }
    }

    #[test]
    fn parallel_boards_have_no_camera() {
        let (_, kmtx) = make_kmtx();
        for tilt in [
            Rotation3::identity(),
            Rotation3::from_euler_angles(0.35, 0.0, 0.0),
        ] {
            let err = estimate_intrinsics_from_homographies(&parallel_homographies(&kmtx, tilt))
                .expect_err("parallel boards determine no camera");
            assert!(matches!(
                err,
                Error::DegenerateViews {
                    views: 6,
                    spread_deg: None
                }
            ));
            assert!(
                err.to_string()
                    .contains("tilt the board about both of its axes"),
                "{err}"
            );
        }
    }
}
