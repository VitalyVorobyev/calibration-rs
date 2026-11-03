//! Iterative linear intrinsics estimation with distortion refinement.
//!
//! This module implements an alternating optimization scheme for jointly estimating
//! camera intrinsics (fx, fy, cx, cy) and Brown-Conrady distortion coefficients
//! (k1, k2, k3, p1, p2) without requiring ground truth distortion for preprocessing.
//!
//! # Algorithm Overview
//!
//! The classic Zhang method assumes distortion-free inputs. When distortion is present,
//! directly applying Zhang to distorted pixels produces biased intrinsics estimates.
//! This module addresses the problem through iterative refinement:
//!
//! 1. **Initial estimate**: Compute K from distorted pixels (ignoring distortion)
//! 2. **Distortion estimation**: Using K from step 1, estimate distortion from homography residuals
//! 3. **Pixel undistortion**: Apply estimated distortion to correct pixel observations
//! 4. **Intrinsics refinement**: Re-estimate K from undistorted pixels
//! 5. **Iterate**: Repeat steps 2-4 for k iterations
//!
//! Typically converges in 1-3 iterations, providing initial estimates suitable for
//! non-linear bundle adjustment.
//!
//! # When to Use
//!
//! Use this solver when:
//! - You don't have ground truth distortion parameters
//! - You have multiple views of a planar calibration pattern
//! - You need both intrinsics and distortion for initialization
//!
//! For distortion-free cameras or when you already have good intrinsics,
//! use [`zhang_intrinsics`](crate::zhang_intrinsics) directly.
//!
//! # Comparison with Single-Pass Zhang
//!
//! - **Zhang alone**: Fast, but biased when distortion is significant
//! - **Iterative**: Slower (2-3× cost), but handles distortion without ground truth
//! - **After non-linear refinement**: Both converge to similar final accuracy
//!
//! # Example
//!
//! ```
//! use vision_calibration_core::{PlanarDataset, View};
//! use vision_calibration_linear::iterative_intrinsics::{
//!     IterativeIntrinsicsOptions, estimate_intrinsics_iterative,
//! };
//! # use vision_calibration_core::synthetic::{planar, poses};
//! # use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let truth = make_pinhole_camera(
//! #     FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
//! #     BrownConrady5 { k1: 0.05, k2: -0.02, ..BrownConrady5::default() },
//! # );
//! # let board = planar::grid_points(8, 6, 0.04);
//! # let specs = poses::seeded_board_pose_specs(8, 0.35, 3);
//! # let poses = poses::centered_board_poses(&specs, (0.14, 0.10), 0.6);
//! # let views = planar::project_views_all(&truth, &board, &poses)?;
//!
//! // Eight views of a planar board, seen through a lens with radial distortion.
//! let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;
//! let camera = estimate_intrinsics_iterative(&dataset, IterativeIntrinsicsOptions::default())?;
//!
//! println!("fx={:.1} fy={:.1} cx={:.1} cy={:.1}", camera.k.fx, camera.k.fy, camera.k.cx, camera.k.cy);
//! println!("k1={:.4} k2={:.4}", camera.dist.k1, camera.dist.k2);
//! // A starting point for non-linear refinement: within a few percent.
//! assert!((camera.k.fx - 800.0).abs() < 0.05 * 800.0);
//! # Ok(())
//! # }
//! ```

use crate::{
    Error,
    distortion_fit::{
        DistortionFitOptions, DistortionView, MetaHomography, estimate_distortion_from_homographies,
    },
    zhang_intrinsics::{PlanarIntrinsicsLinearInit, check_board_orientations},
};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, DistortionModel, FxFyCxCy, Mat3, PinholeCamera, PlanarDataset, Pt2, Real, Vec3,
    make_pinhole_camera,
};
use vision_geometry::homography::dlt_homography;

/// Options controlling iterative intrinsics estimation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IterativeIntrinsicsOptions {
    /// Number of refinement iterations (distortion → K → distortion → K).
    ///
    /// Each iteration:
    /// 1. Estimates distortion using current K
    /// 2. Undistorts pixels using estimated distortion
    /// 3. Re-estimates K from undistorted pixels
    ///
    /// Typical values: 1-3. Beyond 3 iterations, improvement is marginal.
    pub iterations: usize,

    /// Options for distortion fitting.
    ///
    /// Controls which distortion parameters to estimate (fix_k3, fix_tangential).
    pub distortion_opts: DistortionFitOptions,
}

impl Default for IterativeIntrinsicsOptions {
    fn default() -> Self {
        Self {
            iterations: 2, // One distortion estimate + one K re-estimate typically sufficient
            distortion_opts: DistortionFitOptions::default(),
        }
    }
}

/// Estimate intrinsics and distortion using iterative refinement.
///
/// # Algorithm
///
/// 1. Estimate initial K from distorted pixels (Zhang method, iteration 0)
/// 2. For each iteration i = 1..iterations:
///    a. Compute homographies from current pixels (distorted or undistorted)
///    b. Estimate distortion from residuals using current K
///    c. Undistort original pixels using estimated distortion
///    d. Re-estimate K from undistorted pixels
/// 3. Return final K and distortion with full history
///
/// # Arguments
///
/// * `views` - Multiple views of calibration pattern (≥3 required)
/// * `opts` - Options controlling iteration count and distortion parameters
///
/// # Returns
///
/// `IterativeIntrinsicsResult` containing final estimates and iteration history.
///
/// # Errors
///
/// - [`Error::InsufficientData`]: fewer than 3 views, or fewer than 3 whose
///   homography is well posed.
/// - [`Error::DegenerateViews`]: the board planes are parallel in every view,
///   to within 1° (all fronto-parallel, or all tilted the same way), or
///   Zhang's method finds no camera that fits the views. The spread is
///   measured through a camera with square pixels whose focal length is the
///   image extent the views cover, so it reads the perspective the views
///   show.
/// - [`Error::Singular`]: the linear solve fails on non-finite input.
///
/// A failure of a later distortion iteration is not an error: the last
/// estimate that succeeded is returned.
///
/// # Complexity
///
/// O(iterations × N_views × N_points_per_view)
///
/// Typically 2-3× slower than single-pass Zhang, but eliminates need for
/// ground truth distortion preprocessing.
pub fn estimate_intrinsics_iterative(
    dataset: &PlanarDataset,
    opts: IterativeIntrinsicsOptions,
) -> Result<PinholeCamera, Error> {
    if dataset.views.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: dataset.views.len(),
        });
    }

    let target_points_2d: Vec<Vec<Pt2>> = dataset
        .views
        .iter()
        .map(|v| v.obs.planar_points())
        .collect();

    // Iteration 0: initial K estimate from distorted pixels (ignore distortion).
    let pairs0 = valid_view_homographies(dataset.views.len(), |i| {
        dlt_homography(&target_points_2d[i], &dataset.views[i].obs.points_2d)
    })?;
    let homographies_iter0: Vec<Mat3> = pairs0.iter().map(|(_, h)| *h).collect();
    // Parallel boards leave `K` undetermined, and with noisy corners Zhang's
    // solve can return an arbitrary camera instead of failing: reject them
    // first, measured at the image extent the views cover.
    if let Some(reference) = observed_reference(dataset) {
        check_board_orientations(&homographies_iter0, &reference)?;
    }

    let mut current_intrinsics =
        PlanarIntrinsicsLinearInit::from_homographies(&homographies_iter0)?;

    let mut current_distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: opts.distortion_opts.iters,
    };

    // The distortion-refinement loop is **best-effort**: each iteration commits
    // its result only if every step is well-posed. If an iteration degenerates
    // (e.g. a poorly-conditioned camera whose distortion estimate makes the
    // undistorted homographies singular for too many views), we keep the last
    // good estimate rather than failing the whole camera — iteration 0 already
    // gave a usable distortion-free estimate that bundle adjustment refines.
    // A view that degenerates is skipped (see `valid_view_homographies`); only a
    // collapse below the minimum view count breaks the loop.
    for _iter in 0..opts.iterations {
        let k_mtx = current_intrinsics.k_matrix();
        let Some(k_inv) = k_mtx.try_inverse() else {
            break;
        };

        let Ok(pairs) = valid_view_homographies(dataset.views.len(), |i| {
            let undistorted_pixels = undistort_pixels_to_pixels(
                &dataset.views[i].obs.points_2d,
                &k_mtx,
                &k_inv,
                &current_distortion,
            );
            dlt_homography(&target_points_2d[i], &undistorted_pixels)
        }) else {
            break;
        };

        let dist_views: Vec<DistortionView> = pairs
            .iter()
            .map(|(i, h)| {
                DistortionView::new(
                    dataset.views[*i].obs.clone(),
                    MetaHomography { homography: *h },
                )
            })
            .collect();

        let Ok(new_distortion) =
            estimate_distortion_from_homographies(&k_mtx, &dist_views, opts.distortion_opts)
        else {
            break;
        };

        let Ok(pairs) = valid_view_homographies(dataset.views.len(), |i| {
            let undistorted_pixels = undistort_pixels_to_pixels(
                &dataset.views[i].obs.points_2d,
                &k_mtx,
                &k_inv,
                &new_distortion,
            );
            dlt_homography(&target_points_2d[i], &undistorted_pixels)
        }) else {
            break;
        };
        let undistorted_homographies: Vec<Mat3> = pairs.iter().map(|(_, h)| *h).collect();

        let Ok(new_intrinsics) =
            PlanarIntrinsicsLinearInit::from_homographies(&undistorted_homographies)
        else {
            break;
        };

        // Iteration fully succeeded — commit it.
        current_distortion = new_distortion;
        current_intrinsics = new_intrinsics;
    }

    Ok(make_pinhole_camera(current_intrinsics, current_distortion))
}

/// A reference camera for the board-orientation check: square pixels, the
/// principal point at the centre of the observed points' bounding box and
/// a focal length equal to its larger side. `None` without finite points
/// spanning a non-zero extent.
fn observed_reference(dataset: &PlanarDataset) -> Option<FxFyCxCy<Real>> {
    let (mut lo, mut hi) = (
        Pt2::new(Real::INFINITY, Real::INFINITY),
        Pt2::new(Real::NEG_INFINITY, Real::NEG_INFINITY),
    );
    for p in dataset.views.iter().flat_map(|v| v.obs.points_2d.iter()) {
        if p.x.is_finite() && p.y.is_finite() {
            lo = Pt2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Pt2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
    }
    let extent = (hi.x - lo.x).max(hi.y - lo.y);
    (extent.is_finite() && extent > 0.0).then(|| FxFyCxCy {
        fx: extent,
        fy: extent,
        cx: 0.5 * (lo.x + hi.x),
        cy: 0.5 * (lo.y + hi.y),
    })
}

/// Minimum number of well-posed views Zhang's linear init needs.
const MIN_VALID_VIEWS: usize = 3;

/// Build per-view homographies, **skipping** views whose DLT is singular /
/// degenerate, and return `(view_index, homography)` pairs so callers can keep
/// views and homographies aligned. Errors only if fewer than [`MIN_VALID_VIEWS`]
/// survive — so one bad view in a dense capture does not fail the camera.
fn valid_view_homographies<F, E>(n_views: usize, mut dlt: F) -> Result<Vec<(usize, Mat3)>, Error>
where
    F: FnMut(usize) -> Result<Mat3, E>,
{
    let pairs: Vec<(usize, Mat3)> = (0..n_views)
        .filter_map(|i| dlt(i).ok().map(|h| (i, h)))
        .collect();
    if pairs.len() < MIN_VALID_VIEWS {
        return Err(Error::InsufficientData {
            need: MIN_VALID_VIEWS,
            got: pairs.len(),
        });
    }
    Ok(pairs)
}

fn undistort_pixels_to_pixels(
    pixels: &[Pt2],
    k_mtx: &Mat3,
    k_inv: &Mat3,
    distortion: &BrownConrady5<Real>,
) -> Vec<Pt2> {
    pixels
        .iter()
        .map(|p| {
            let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
            let n_dist = Pt2::new(v_h.x / v_h.z, v_h.y / v_h.z);
            let n_undist = distortion.undistort(&n_dist);
            let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
            Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3, UnitQuaternion};
    use vision_calibration_core::synthetic::noise::UniformPixelNoise;
    use vision_calibration_core::synthetic::planar::{
        grid_points, project_views_all, project_views_noisy,
    };
    use vision_calibration_core::{FxFyCxCy, Iso3, View};

    fn make_ground_truth() -> (FxFyCxCy<Real>, BrownConrady5<Real>) {
        let intr = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 360.0,
        };
        let dist = BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        };
        (intr, dist)
    }

    fn make_dataset(camera: &PinholeCamera, n_views: usize) -> PlanarDataset {
        let board = grid_points(7, 7, 0.03);

        let pose_params: &[(Real, Real, Real, Real, Real, Real)] = &[
            (0.10, 0.00, 0.05, 0.10, -0.05, 1.00),
            (-0.05, 0.15, -0.10, -0.05, 0.10, 1.20),
            (0.20, -0.10, 0.00, 0.00, 0.00, 0.90),
            (0.00, 0.20, 0.10, 0.08, 0.08, 1.10),
            (-0.10, 0.10, -0.05, -0.08, -0.08, 1.05),
        ];

        let poses: Vec<Iso3> = pose_params
            .iter()
            .take(n_views)
            .map(|&(rx, ry, rz, tx, ty, tz)| {
                Iso3::from_parts(
                    Translation3::new(tx, ty, tz),
                    UnitQuaternion::from_euler_angles(rx, ry, rz),
                )
            })
            .collect();

        let views = project_views_all(camera, &board, &poses).unwrap();
        let views = views.into_iter().map(View::without_meta).collect();
        PlanarDataset::new(views).unwrap()
    }

    #[test]
    fn iterative_refinement_converges() {
        let (intr_gt, dist_gt) = make_ground_truth();
        let camera_gt = make_pinhole_camera(intr_gt, dist_gt);
        let dataset = make_dataset(&camera_gt, 4);

        let opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
        };

        let result = estimate_intrinsics_iterative(&dataset, opts).unwrap();

        // Check final estimates (linear methods: 10-30% accuracy expected for initialization)
        // This is acceptable - the purpose is to provide a reasonable starting point
        // for non-linear refinement, not to achieve final accuracy.
        let fx_err_pct = (result.k.fx - intr_gt.fx).abs() / intr_gt.fx * 100.0;
        let fy_err_pct = (result.k.fy - intr_gt.fy).abs() / intr_gt.fy * 100.0;
        let cx_err = (result.k.cx - intr_gt.cx).abs();
        let cy_err = (result.k.cy - intr_gt.cy).abs();

        assert!(fx_err_pct < 40.0, "fx error {:.1}% too large", fx_err_pct);
        assert!(fy_err_pct < 40.0, "fy error {:.1}% too large", fy_err_pct);
        assert!(cx_err < 80.0, "cx error {:.1}px too large", cx_err);
        assert!(cy_err < 150.0, "cy error {:.1}px too large", cy_err);

        // Check distortion estimates have correct sign
        assert!(
            result.dist.k1.signum() == dist_gt.k1.signum(),
            "k1 sign mismatch"
        );
    }

    #[test]
    fn iteration_improves_estimates() {
        let (intr_gt, dist_gt) = make_ground_truth();
        let camera_gt = make_pinhole_camera(intr_gt, dist_gt);
        let dataset = make_dataset(&camera_gt, 4);

        let opts0 = IterativeIntrinsicsOptions {
            iterations: 0,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
        };

        let cam0 = estimate_intrinsics_iterative(&dataset, opts0).unwrap();

        let opts2 = IterativeIntrinsicsOptions {
            iterations: 2,
            ..opts0
        };
        let cam2 = estimate_intrinsics_iterative(&dataset, opts2).unwrap();

        let rms0 = plane_homography_rms(&cam0, &dataset);
        let rms2 = plane_homography_rms(&cam2, &dataset);
        assert!(
            rms2 < rms0,
            "expected lower plane residual after iterations: rms0={rms0:.4}, rms2={rms2:.4}"
        );
    }

    fn plane_homography_rms(camera: &PinholeCamera, dataset: &PlanarDataset) -> Real {
        let k_mtx = camera.k.k_matrix();
        let k_inv = k_mtx.try_inverse().expect("K invertible for tests");

        let mut total_sq = 0.0;
        let mut total_n = 0usize;

        for view in &dataset.views {
            let world = view.obs.planar_points();
            let undistorted_pixels =
                undistort_pixels_to_pixels(&view.obs.points_2d, &k_mtx, &k_inv, &camera.dist);
            let h = dlt_homography(&world, &undistorted_pixels)
                .expect("homography should solve on synthetic data");

            for (pw, pu) in world.iter().zip(undistorted_pixels.iter()) {
                let proj_h = h * Vec3::new(pw.x, pw.y, 1.0);
                let proj = Pt2::new(proj_h.x / proj_h.z, proj_h.y / proj_h.z);
                let dx = proj.x - pu.x;
                let dy = proj.y - pu.y;
                total_sq += dx * dx + dy * dy;
                total_n += 1;
            }
        }

        (total_sq / total_n as Real).sqrt()
    }

    /// A capture of a 12 × 9 board of 30 mm squares, 0.65 m away, centred on
    /// the four image quadrants and the centre in turn and turned in its
    /// plane from view to view, through a 1000 px camera with `k1` radial
    /// distortion and ±0.35 px uniform corner noise. View `k` has the
    /// orientation `tilt(k)`.
    fn capture(k1: Real, tilt: impl Fn(usize) -> Rotation3<Real>) -> PlanarDataset {
        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 1000.0,
                fy: 1000.0,
                cx: 640.0,
                cy: 480.0,
            },
            BrownConrady5 {
                k1,
                ..BrownConrady5::default()
            },
        );
        let board = grid_points(12, 9, 0.03);
        let centre = Vec3::new(0.165, 0.12, 0.0);
        let offsets = [
            (-0.3, -0.22),
            (0.3, -0.22),
            (-0.3, 0.22),
            (0.3, 0.22),
            (0.0, 0.0),
        ];
        let poses: Vec<Iso3> = (0..10)
            .map(|k| {
                let (x, y) = offsets[k % offsets.len()];
                let spin = Rotation3::from_euler_angles(0.0, 0.0, (17.0 * k as Real).to_radians());
                let rotation = UnitQuaternion::from_rotation_matrix(&(tilt(k) * spin));
                let distance = 0.65 + 0.03 * (k / 5) as Real;
                let at = Vec3::new(x, y, 1.0).normalize() * distance - rotation * centre;
                Iso3::from_parts(Translation3::from(at), rotation)
            })
            .collect();
        let noise = UniformPixelNoise {
            seed: 7,
            max_abs_px: 0.35,
        };
        let views = project_views_noisy(&camera, &board, &poses, &noise).unwrap();
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap()
    }

    fn about(x_deg: Real, y_deg: Real) -> Rotation3<Real> {
        Rotation3::from_euler_angles(x_deg.to_radians(), y_deg.to_radians(), 0.0)
    }

    /// Tilted by `deg` about +x, +y, −x and −y in turn.
    fn turning(deg: Real) -> impl Fn(usize) -> Rotation3<Real> {
        move |k| match k % 4 {
            0 => about(deg, 0.0),
            1 => about(0.0, deg),
            2 => about(-deg, 0.0),
            _ => about(0.0, -deg),
        }
    }

    fn init(dataset: &PlanarDataset) -> Result<PinholeCamera, Error> {
        estimate_intrinsics_iterative(dataset, IterativeIntrinsicsOptions::default())
    }

    #[test]
    fn parallel_boards_are_degenerate_views() {
        type Orientation = fn(usize) -> Rotation3<Real>;
        let parallel: [(&str, Orientation); 2] = [
            ("fronto-parallel", |_| Rotation3::identity()),
            ("all tilted 20° about +x", |_| about(20.0, 0.0)),
        ];
        for k1 in [0.0, -0.15] {
            for (name, tilt) in parallel {
                let err = init(&capture(k1, tilt)).expect_err(name);
                assert!(
                    matches!(err, Error::DegenerateViews { views: 10, .. }),
                    "{name}, k1 = {k1}: {err:?}"
                );
                assert!(
                    err.to_string()
                        .contains("tilt the board about both of its axes"),
                    "{err}"
                );
            }
        }
    }

    #[test]
    fn spread_check_catches_parallel_boards_zhang_would_misread() {
        // Without distortion, the corner noise alone decides Zhang's
        // solution for parallel boards: it returns a camera (fx ≈ 54 000 px
        // for the fronto-parallel capture). The spread check rejects them
        // first, at the noise level of the board orientations.
        for tilt in [|_| Rotation3::identity(), |_| about(20.0, 0.0)] {
            let dataset = capture(0.0, tilt);
            let homographies: Vec<Mat3> = dataset
                .views
                .iter()
                .map(|v| dlt_homography(&v.obs.planar_points(), &v.obs.points_2d).unwrap())
                .collect();
            assert!(PlanarIntrinsicsLinearInit::from_homographies(&homographies).is_ok());
            match init(&dataset) {
                Err(Error::DegenerateViews {
                    spread_deg: Some(spread),
                    ..
                }) => assert!(spread < 0.25, "spread {spread}°"),
                other => panic!("expected a measured spread, got {other:?}"),
            }
        }
    }

    #[test]
    fn small_tilts_on_either_side_of_the_threshold() {
        // Measured through the observed-extent reference (≈ 1.2 × the focal
        // length here), a 0.5° tilt reads ≈ 0.6° and a 3° tilt ≈ 3.6°.
        assert!(matches!(
            init(&capture(0.0, turning(0.5))),
            Err(Error::DegenerateViews {
                spread_deg: Some(_),
                ..
            })
        ));
        let camera = init(&capture(0.0, turning(3.0))).expect("3° tilts initialize");
        assert!(
            (camera.k.fx - 1000.0).abs() < 0.05 * 1000.0,
            "fx {}",
            camera.k.fx
        );
    }

    #[test]
    fn good_capture_is_unaffected() {
        for k1 in [0.0, -0.15] {
            let camera = init(&capture(k1, turning(20.0))).expect("good capture");
            assert!(
                (camera.k.fx - 1000.0).abs() < 0.05 * 1000.0,
                "fx {}",
                camera.k.fx
            );
            assert!(
                (camera.k.fy - 1000.0).abs() < 0.05 * 1000.0,
                "fy {}",
                camera.k.fy
            );
        }
    }

    #[test]
    fn tilts_about_one_axis_both_ways_pass_init_undetermined() {
        // Two board orientations (±20° about x) give Zhang's general solve
        // four constraints on its five unknowns: not parallel boards, so not
        // rejected, but the horizontal focal length is left to the noise and
        // the distortion. Refinement then reports it through its σ.
        for k1 in [0.0, -0.15] {
            let camera = init(&capture(k1, |k| {
                about(if k % 2 == 0 { 20.0 } else { -20.0 }, 0.0)
            }))
            .expect("two orientations pass init");
            assert!(
                (camera.k.fx - 1000.0).abs() > 0.1 * 1000.0,
                "fx {}",
                camera.k.fx
            );
        }
        // A third tilt about the same axis determines the camera.
        let three = |k: usize| about([-20.0, 5.0, 25.0][k % 3], 0.0);
        let camera = init(&capture(0.0, three)).expect("three orientations");
        assert!(
            (camera.k.fx - 1000.0).abs() < 0.01 * 1000.0,
            "fx {}",
            camera.k.fx
        );
    }
}
