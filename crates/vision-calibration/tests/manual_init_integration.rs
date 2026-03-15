//! Integration tests for the manual parameter initialization workflow (M10).
//!
//! Verifies that `step_set_init` with exact ground-truth seeds followed by
//! `step_optimize` achieves very tight reprojection error on synthetic data.

use vision_calibration::core::{
    BrownConrady5, FxFyCxCySkew, PlanarDataset, View, make_pinhole_camera,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsProblem, PlanarManualInit, step_optimize, step_set_init,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::planar;

fn gt_intrinsics() -> FxFyCxCySkew<f64> {
    FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
}

fn make_planar_dataset() -> PlanarDataset {
    let camera = make_pinhole_camera(gt_intrinsics(), BrownConrady5::default());
    let board_points = planar::grid_points(6, 5, 0.03);
    let poses = planar::poses_yaw_y_z(7, 0.0, 0.08, 0.55, 0.03);
    let views = planar::project_views_all(&camera, &board_points, &poses).expect("views");
    PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
}

/// Exact GT seeds → near-zero reprojection error.
#[test]
fn step_set_init_with_exact_seeds_converges_tightly() {
    let dataset = make_planar_dataset();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");

    let manual = PlanarManualInit {
        intrinsics: Some(gt_intrinsics()),
        distortion: Some(BrownConrady5::default()),
        poses: None, // auto-recovered using manual intrinsics
    };

    step_set_init(&mut session, manual, None).expect("set_init");
    step_optimize(&mut session, None).expect("optimize");

    let output = session.output().expect("output");
    assert!(
        output.mean_reproj_error < 1e-3,
        "expected tight reproj with GT seeds, got {:.6}",
        output.mean_reproj_error
    );
}

/// Perturbed seeds (~10% off) still converges to acceptable accuracy.
#[test]
fn step_set_init_with_perturbed_seeds_converges() {
    let dataset = make_planar_dataset();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");

    let gt = gt_intrinsics();
    let perturbed = FxFyCxCySkew {
        fx: gt.fx * 1.10,
        fy: gt.fy * 0.92,
        cx: gt.cx + 20.0,
        cy: gt.cy - 15.0,
        skew: 0.0,
    };

    let manual = PlanarManualInit {
        intrinsics: Some(perturbed),
        distortion: None,
        poses: None,
    };

    step_set_init(&mut session, manual, None).expect("set_init");
    step_optimize(&mut session, None).expect("optimize");

    let output = session.output().expect("output");
    assert!(
        output.mean_reproj_error < 1.0,
        "expected reasonable convergence with perturbed seeds, got {:.4}",
        output.mean_reproj_error
    );
}

/// `step_set_init` with all-None fields produces the same state as `step_init`.
#[test]
fn step_set_init_default_equivalent_to_step_init() {
    use vision_calibration::planar_intrinsics::step_init;

    let dataset = make_planar_dataset();

    let mut session_a = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session_a.set_input(dataset.clone()).expect("input");
    step_init(&mut session_a, None).expect("step_init");

    let mut session_b = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session_b.set_input(dataset).expect("input");
    step_set_init(&mut session_b, PlanarManualInit::default(), None).expect("step_set_init");

    // Both sessions should be initialized and produce comparable intrinsics.
    assert!(session_a.state.is_initialized());
    assert!(session_b.state.is_initialized());

    let k_a = session_a
        .state
        .initial_intrinsics
        .expect("initial_intrinsics a");
    let k_b = session_b
        .state
        .initial_intrinsics
        .expect("initial_intrinsics b");

    // Should produce very close results (same algorithm, same data).
    assert!(
        (k_a.fx - k_b.fx).abs() < 1.0,
        "fx mismatch: {} vs {}",
        k_a.fx,
        k_b.fx
    );
    assert!(
        (k_a.fy - k_b.fy).abs() < 1.0,
        "fy mismatch: {} vs {}",
        k_a.fy,
        k_b.fy
    );
}
