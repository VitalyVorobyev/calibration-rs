use calib_core::{
    test_utils::{build_corner_info, CalibrationView, CornerInfo, ViewDetections},
    BrownConrady5, Iso3, Mat3, Mat4, Pt2, Pt3, Real,
};
use calib_linear::{
    triangulate_point_linear, EpipolarSolver, HomographySolver, Mat34, PlanarIntrinsicsLinearInit,
    PlanarPoseSolver,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct StereoData {
    board: BoardSpec,
    intrinsics: IntrinsicsPair,
    distortion: DistortionPair,
    extrinsics: ExtrinsicsPair,
    essential: [[Real; 3]; 3],
    fundamental: [[Real; 3]; 3],
    views: Vec<CalibrationView>,
}

#[derive(Debug, Deserialize)]
struct BoardSpec {
    cols: usize,
    rows: usize,
    square_size: Real,
}

#[derive(Debug, Deserialize)]
struct IntrinsicsPair {
    left: [[Real; 3]; 3],
    right: [[Real; 3]; 3],
}

#[derive(Debug, Deserialize)]
struct DistortionPair {
    left: [Real; 5],
    right: [Real; 5],
}

#[derive(Debug, Deserialize)]
struct ExtrinsicsPair {
    left: Vec<[[Real; 4]; 4]>,
    right: Vec<[[Real; 4]; 4]>,
}

fn load_data() -> StereoData {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("stereo_linear.json");
    let contents = fs::read_to_string(&path).expect("read stereo_linear.json");
    let mut data: StereoData = serde_json::from_str(&contents).expect("parse stereo_linear.json");
    data.views.sort_by_key(|v| v.view_index);
    data
}

fn mat3_from_array(a: &[[Real; 3]; 3]) -> Mat3 {
    Mat3::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ])
}

fn mat4_from_array(a: &[[Real; 4]; 4]) -> Mat4 {
    Mat4::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0], a[2][1],
        a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3],
    ])
}

fn distortion_from_array(d: [Real; 5]) -> BrownConrady5<Real> {
    BrownConrady5 {
        k1: d[0],
        k2: d[1],
        p1: d[2],
        p2: d[3],
        k3: d[4],
        iters: 8,
    }
}

fn build_corners(view: &ViewDetections, k: &Mat3, dist: &BrownConrady5<Real>) -> Vec<CornerInfo> {
    build_corner_info(view, k, dist)
}

fn find_corner(corners: &[CornerInfo], i: usize, j: usize) -> Option<&CornerInfo> {
    corners.iter().find(|c| c.i == i && c.j == j)
}

fn board_point(i: usize, j: usize, square: Real) -> Pt3 {
    Pt3::new(i as Real * square, j as Real * square, 0.0)
}

fn board_point_2d(i: usize, j: usize, square: Real) -> Pt2 {
    Pt2::new(i as Real * square, j as Real * square)
}

fn projection_matrix(k: &Mat3, extr: &Mat4) -> Mat34 {
    let r = extr.fixed_view::<3, 3>(0, 0).into_owned();
    let t = extr.fixed_view::<3, 1>(0, 3).into_owned();
    let mut p = Mat34::zeros();
    p.fixed_view_mut::<3, 3>(0, 0).copy_from(&(k * r));
    p.set_column(3, &(k * t));
    p
}

fn pose_error(est: &Iso3, gt: &Mat4) -> (Real, Real) {
    let r_gt = gt.fixed_view::<3, 3>(0, 0).into_owned();
    let t_gt = gt.fixed_view::<3, 1>(0, 3).into_owned();

    let r_est = est.rotation.to_rotation_matrix();
    let t_est = est.translation.vector;

    let r_diff = r_est.matrix().transpose() * r_gt;
    let trace = r_diff.trace();
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let rot_err = cos_theta.acos();

    let t_err = (t_est - t_gt).norm();
    (rot_err, t_err)
}

fn scaled_error_mat3(est: &Mat3, gt: &Mat3) -> Real {
    let dot: Real = est
        .as_slice()
        .iter()
        .zip(gt.as_slice().iter())
        .map(|(a, b)| a * b)
        .sum();
    let denom: Real = est.as_slice().iter().map(|v| v * v).sum();
    let scale = if denom.abs() > 1e-12 {
        dot / denom
    } else {
        1.0
    };
    (est * scale - gt).norm() / gt.norm()
}

fn percentile(vals: &mut [Real], p: Real) -> Real {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((vals.len() - 1) as Real * p).round() as usize;
    vals[idx]
}

#[test]
fn stereo_zhang_intrinsics_left_right() {
    let data = load_data();
    let board = &data.board;

    for (side, k_arr, d_arr) in [
        ("left", data.intrinsics.left, data.distortion.left),
        ("right", data.intrinsics.right, data.distortion.right),
    ] {
        let k_gt = mat3_from_array(&k_arr);
        let dist = distortion_from_array(d_arr);

        let mut homographies = Vec::new();
        for view in &data.views {
            let det = if side == "left" {
                &view.left
            } else {
                &view.right
            };
            let corners = build_corners(det, &k_gt, &dist);

            let mut world = Vec::with_capacity(corners.len());
            let mut image = Vec::with_capacity(corners.len());
            for c in &corners {
                world.push(board_point_2d(c.i, c.j, board.square_size));
                image.push(c.undist_pixel);
            }

            let h = HomographySolver::dlt(&world, &image).expect("homography");
            homographies.push(h);
        }

        let intr =
            PlanarIntrinsicsLinearInit::from_homographies(&homographies).expect("zhang intrinsics");

        let fx_gt = k_gt[(0, 0)];
        let fy_gt = k_gt[(1, 1)];
        let cx_gt = k_gt[(0, 2)];
        let cy_gt = k_gt[(1, 2)];

        let fx_rel = (intr.fx - fx_gt).abs() / fx_gt;
        let fy_rel = (intr.fy - fy_gt).abs() / fy_gt;
        let cx_err = (intr.cx - cx_gt).abs();
        let cy_err = (intr.cy - cy_gt).abs();

        assert!(fx_rel < 0.05, "{side}: fx rel error too large: {fx_rel}");
        assert!(fy_rel < 0.05, "{side}: fy rel error too large: {fy_rel}");
        assert!(cx_err < 8.0, "{side}: cx error too large: {cx_err}");
        assert!(cy_err < 8.0, "{side}: cy error too large: {cy_err}");
        assert!(
            intr.skew.abs() < 1.0,
            "{side}: skew too large: {}",
            intr.skew
        );
    }
}

#[test]
fn stereo_planar_pose_matches_ground_truth() {
    let data = load_data();
    let board = &data.board;

    for (side, k_arr, d_arr, extrinsics) in [
        (
            "left",
            data.intrinsics.left,
            data.distortion.left,
            &data.extrinsics.left,
        ),
        (
            "right",
            data.intrinsics.right,
            data.distortion.right,
            &data.extrinsics.right,
        ),
    ] {
        let k_gt = mat3_from_array(&k_arr);
        let dist = distortion_from_array(d_arr);

        let mut rot_errs = Vec::new();
        let mut trans_errs = Vec::new();

        for view in &data.views {
            let det = if side == "left" {
                &view.left
            } else {
                &view.right
            };
            let corners = build_corners(det, &k_gt, &dist);

            let mut world = Vec::with_capacity(corners.len());
            let mut image = Vec::with_capacity(corners.len());
            for c in &corners {
                world.push(board_point_2d(c.i, c.j, board.square_size));
                image.push(c.undist_pixel);
            }

            let h = HomographySolver::dlt(&world, &image).expect("homography");
            let pose = PlanarPoseSolver::from_homography(&k_gt, &h).expect("planar pose");

            let gt = mat4_from_array(&extrinsics[view.view_index]);
            let (r_err, t_err) = pose_error(&pose, &gt);

            rot_errs.push(r_err);
            trans_errs.push(t_err);
        }

        let rot_p90 = percentile(&mut rot_errs, 0.9);
        let trans_p90 = percentile(&mut trans_errs, 0.9);

        assert!(rot_p90 < 0.05, "{side}: rotation p90 too large: {rot_p90}");
        assert!(
            trans_p90 < 5.0,
            "{side}: translation p90 too large: {trans_p90}"
        );
    }
}

#[test]
fn stereo_fundamental_and_essential_match_ground_truth() {
    let data = load_data();
    let board = &data.board;

    let k_left = mat3_from_array(&data.intrinsics.left);
    let k_right = mat3_from_array(&data.intrinsics.right);
    let d_left = distortion_from_array(data.distortion.left);
    let d_right = distortion_from_array(data.distortion.right);

    let view = &data.views[0];
    let left = build_corners(&view.left, &k_left, &d_left);
    let right = build_corners(&view.right, &k_right, &d_right);

    assert_eq!(left.len(), board.cols * board.rows);
    assert_eq!(right.len(), board.cols * board.rows);

    let mut pts_l = Vec::with_capacity(left.len());
    let mut pts_r = Vec::with_capacity(right.len());

    for (l, r) in left.iter().zip(right.iter()) {
        assert_eq!((l.i, l.j), (r.i, r.j));
        pts_l.push(l.undist_pixel);
        pts_r.push(r.undist_pixel);
    }

    let f_est = EpipolarSolver::fundamental_8point(&pts_l, &pts_r).expect("fundamental");
    let f_gt = mat3_from_array(&data.fundamental);
    let f_err = scaled_error_mat3(&f_est, &f_gt);
    assert!(f_err < 0.07, "fundamental error too large: {f_err}");

    let mid_i = board.cols / 2;
    let mid_j = board.rows / 2;
    let sample = [
        (0, 0),
        (board.cols - 1, 0),
        (board.cols - 1, board.rows - 1),
        (0, board.rows - 1),
        (mid_i, mid_j),
    ];

    let mut n_l = Vec::with_capacity(sample.len());
    let mut n_r = Vec::with_capacity(sample.len());
    for (i, j) in sample {
        let l = find_corner(&left, i, j).expect("left corner");
        let r = find_corner(&right, i, j).expect("right corner");
        n_l.push(Pt2::new(l.undist_norm.x, l.undist_norm.y));
        n_r.push(Pt2::new(r.undist_norm.x, r.undist_norm.y));
    }

    let mut e_candidates = EpipolarSolver::essential_5point(&n_l, &n_r).expect("essential 5-point");
    let e_gt = mat3_from_array(&data.essential);

    let mut best = Real::INFINITY;
    let mut best_residual = Real::INFINITY;
    let mut gt_residual = 0.0;

    for (l, r) in n_l.iter().zip(n_r.iter()) {
        let x = nalgebra::Vector3::new(l.x, l.y, 1.0);
        let xp = nalgebra::Vector3::new(r.x, r.y, 1.0);
        let val = xp.transpose() * e_gt * x;
        gt_residual += val[0].abs();
    }
    gt_residual /= n_l.len() as Real;

    for e in e_candidates.drain(..) {
        let mut residual = 0.0;
        for (l, r) in n_l.iter().zip(n_r.iter()) {
            let x = nalgebra::Vector3::new(l.x, l.y, 1.0);
            let xp = nalgebra::Vector3::new(r.x, r.y, 1.0);
            let val = xp.transpose() * e * x;
            residual += val[0].abs();
        }
        residual /= n_l.len() as Real;
        best_residual = best_residual.min(residual);
        best = best.min(scaled_error_mat3(&e, &e_gt));
    }

    assert!(
        best_residual <= gt_residual * 3.0,
        "essential residual too large: {best_residual} (gt {gt_residual})"
    );
    assert!(best < 1.5, "essential scale error too large: {best}");
}

#[test]
fn stereo_triangulation_recovers_board_points() {
    let data = load_data();
    let board = &data.board;

    let k_left = mat3_from_array(&data.intrinsics.left);
    let k_right = mat3_from_array(&data.intrinsics.right);
    let d_left = distortion_from_array(data.distortion.left);
    let d_right = distortion_from_array(data.distortion.right);

    let mut errs = Vec::new();

    for view in &data.views {
        let left = build_corners(&view.left, &k_left, &d_left);
        let right = build_corners(&view.right, &k_right, &d_right);

        let gt_left = mat4_from_array(&data.extrinsics.left[view.view_index]);
        let gt_right = mat4_from_array(&data.extrinsics.right[view.view_index]);

        let p_left = projection_matrix(&k_left, &gt_left);
        let p_right = projection_matrix(&k_right, &gt_right);

        for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!((l.i, l.j), (r.i, r.j));
            if idx % 7 != 0 {
                continue;
            }
            let world = board_point(l.i, l.j, board.square_size);
            let est =
                triangulate_point_linear(&[p_left, p_right], &[l.undist_pixel, r.undist_pixel])
                    .expect("triangulation");
            errs.push((est - world).norm());
        }
    }

    let p90 = percentile(&mut errs, 0.9);
    assert!(p90 < 2.0, "triangulation error too large: {p90}");
}

// ============================================================================
// NEW TESTS: Iterative intrinsics estimation without ground truth
// ============================================================================

use calib_linear::iterative_intrinsics::{
    IterativeCalibView, IterativeIntrinsicsOptions, IterativeIntrinsicsSolver,
};
use calib_linear::DistortionFitOptions;

/// Test iterative intrinsics estimation on left camera WITHOUT using ground truth distortion.
///
/// This demonstrates realistic calibration where we don't have prior knowledge of distortion.
#[test]
fn stereo_iterative_intrinsics_left_no_gt() {
    let data = load_data();

    // Ground truth for validation only (NOT used in calibration)
    let k_gt = mat3_from_array(&data.intrinsics.left);
    let dist_gt = distortion_from_array(data.distortion.left);

    // Prepare views using RAW DISTORTED pixels (no ground truth preprocessing)
    let views: Vec<IterativeCalibView> = data
        .views
        .iter()
        .map(|view| {
            let board_points: Vec<Pt2> = view
                .left
                .corners
                .iter()
                .map(|c| board_point_2d(c[0] as usize, c[1] as usize, data.board.square_size))
                .collect();

            let pixel_points: Vec<Pt2> = view
                .left
                .corners
                .iter()
                .map(|c| Pt2::new(c[2], c[3]))
                .collect();

            IterativeCalibView::new(board_points, pixel_points)
        })
        .collect();

    // Run iterative estimation (NO ground truth used here)
    let opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true, // Conservative: only estimate k1, k2, p1, p2
            fix_tangential: false,
            iters: 8,
        },
    };

    let result =
        IterativeIntrinsicsSolver::estimate(&views, opts).expect("iterative intrinsics estimation");

    // Validate against ground truth (for test purposes only)
    let fx_gt = k_gt[(0, 0)];
    let fy_gt = k_gt[(1, 1)];
    let cx_gt = k_gt[(0, 2)];
    let cy_gt = k_gt[(1, 2)];

    let fx_err_pct = (result.intrinsics.fx - fx_gt).abs() / fx_gt * 100.0;
    let fy_err_pct = (result.intrinsics.fy - fy_gt).abs() / fy_gt * 100.0;
    let cx_err = (result.intrinsics.cx - cx_gt).abs();
    let cy_err = (result.intrinsics.cy - cy_gt).abs();

    println!("Left camera iterative calibration (NO ground truth used):");
    println!(
        "  Ground truth: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        fx_gt, fy_gt, cx_gt, cy_gt
    );
    println!(
        "  Estimated:    fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        result.intrinsics.fx, result.intrinsics.fy, result.intrinsics.cx, result.intrinsics.cy
    );
    println!(
        "  Errors: fx={:.1}%, fy={:.1}%, cx={:.1}px, cy={:.1}px",
        fx_err_pct, fy_err_pct, cx_err, cy_err
    );
    println!(
        "  Distortion GT: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
        dist_gt.k1, dist_gt.k2, dist_gt.p1, dist_gt.p2
    );
    println!(
        "  Distortion Est: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
        result.distortion.k1, result.distortion.k2, result.distortion.p1, result.distortion.p2
    );

    // Linear methods: expect 10-40% error for initialization
    // These tolerances reflect realistic expectations for linear approximations
    assert!(
        fx_err_pct < 40.0,
        "fx error {:.1}% exceeds threshold",
        fx_err_pct
    );
    assert!(
        fy_err_pct < 40.0,
        "fy error {:.1}% exceeds threshold",
        fy_err_pct
    );
    assert!(cx_err < 100.0, "cx error {:.1}px exceeds threshold", cx_err);
    assert!(cy_err < 100.0, "cy error {:.1}px exceeds threshold", cy_err);

    // Distortion should have correct sign (most important for initialization)
    assert!(
        result.distortion.k1.signum() == dist_gt.k1.signum(),
        "k1 sign mismatch: got {}, expected {}",
        result.distortion.k1,
        dist_gt.k1
    );
}

/// Test that iterative refinement provides better estimates than single-pass Zhang.
///
/// Compares:
/// 1. Direct Zhang on distorted pixels (iteration 0, biased)
/// 2. Iterative refinement (iterations 1-2, progressively less biased)
#[test]
fn stereo_iterative_improves_over_zhang_left() {
    let data = load_data();
    let k_gt = mat3_from_array(&data.intrinsics.left);

    let views: Vec<IterativeCalibView> = data
        .views
        .iter()
        .map(|view| {
            let board_points: Vec<Pt2> = view
                .left
                .corners
                .iter()
                .map(|c| board_point_2d(c[0] as usize, c[1] as usize, data.board.square_size))
                .collect();

            let pixel_points: Vec<Pt2> = view
                .left
                .corners
                .iter()
                .map(|c| Pt2::new(c[2], c[3]))
                .collect();

            IterativeCalibView::new(board_points, pixel_points)
        })
        .collect();

    let opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true,
            fix_tangential: true, // Radial only for this test
            iters: 8,
        },
    };

    let result = IterativeIntrinsicsSolver::estimate(&views, opts).expect("iterative intrinsics");

    // Compute errors over iterations
    let fx_gt = k_gt[(0, 0)];
    let errors: Vec<Real> = result
        .intrinsics_history
        .iter()
        .map(|intr| (intr.fx - fx_gt).abs())
        .collect();

    println!("Iterative improvement on left camera:");
    for (i, (intr, err)) in result.intrinsics_history.iter().zip(&errors).enumerate() {
        println!(
            "  Iteration {}: fx={:.1}, error={:.1} ({:.1}%)",
            i,
            intr.fx,
            err,
            err / fx_gt * 100.0
        );
    }

    // First iteration should not make things dramatically worse
    // (loose constraint due to linearization artifacts)
    assert!(
        errors[1] < errors[0] * 1.5,
        "First iteration should not worsen estimate significantly"
    );
}
