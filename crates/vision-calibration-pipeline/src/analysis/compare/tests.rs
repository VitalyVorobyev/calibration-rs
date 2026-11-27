//! Exact checks of `compare_runs` on hand-built exports: each test's σ, the
//! cases without a test, camera matching by id, rig poses independent of
//! the reference camera, the hand-eye and laser planes, the configuration
//! changes and the input checks. The statistical calibration of the tests
//! is the Monte-Carlo gates' (`tests/compare_runs_gates.rs`).

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionKind, DistortionParams, FxFyCxCy, Iso3, LaserPlane,
    ParameterCovariance, PerFeatureResiduals, ProjectionParams, SensorParams,
    TargetFeatureResidual, Uncertainty, UncertaintyUnavailable,
};
use vision_calibration_optim::LaserlineStats;

use super::*;
use crate::common::{ExportDiagnostics, HandeyeSolution, VersionedExport};
use crate::laserline_device::LaserlineDeviceExport;
use crate::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsExport};
use crate::rig_extrinsics::RigExtrinsicsExport;
use crate::rig_handeye::RigHandeyeExport;

const SIZE: [u32; 2] = [1280, 960];

fn camera(fx: f64, k1: f64) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                ..BrownConrady5::default()
            },
        },
        SensorParams::Identity,
        FxFyCxCy {
            fx,
            fy: fx,
            cx: 639.5,
            cy: 479.5,
        },
    )
}

/// A diagonal covariance of `labels` with the variances `variances`.
fn covariance(labels: &[String], variances: &[f64], view_scatter: Option<f64>) -> Uncertainty {
    let n = labels.len();
    let matrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { variances[i] } else { 0.0 })
                .collect()
        })
        .collect();
    let mut c = ParameterCovariance::new(labels.to_vec(), matrix, Vec::new(), 1000, 900);
    c.view_scatter = view_scatter;
    Uncertainty::Available(c)
}

/// Features of camera `camera` across the image, error 0.1 px.
fn residuals(cameras: usize) -> ExportDiagnostics {
    let mut target = Vec::new();
    for camera in 0..cameras {
        for view in 0..4 {
            for k in 0..30 {
                let x = 40.0 + 40.0 * k as f64;
                let y = 60.0 + 200.0 * view as f64;
                let mut r = TargetFeatureResidual::default();
                r.view = view;
                r.camera = camera;
                r.feature = k;
                r.observed_px = [x, y];
                r.projected_px = Some([x + 0.1, y]);
                r.error_px = Some(0.1);
                target.push(r);
            }
        }
    }
    let mut residuals = PerFeatureResiduals::default();
    residuals.target = target;
    ExportDiagnostics::new(residuals)
}

fn labels(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// A planar export of `camera` whose `fx` has the variance `fx_var` (and
/// `k1` a variance of 1e-6) or no covariance.
fn planar(
    camera: CameraParams,
    fx_var: Option<f64>,
    view_scatter: Option<f64>,
) -> PlanarIntrinsicsExport {
    let uncertainty = match fx_var {
        Some(v) => covariance(&labels(&["fx", "k1"]), &[v, 1e-6], view_scatter),
        None => Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::NotComputed,
        },
    };
    PlanarIntrinsicsExport {
        kind: ProblemKind::PlanarIntrinsics,
        schema_version: PlanarIntrinsicsExport::SCHEMA_VERSION,
        camera,
        cam_se3_target: Vec::new(),
        mean_reproj_error: 0.1,
        per_cam_reproj_errors: vec![0.1],
        diagnostics: residuals(1),
        uncertainty,
    }
}

fn ids(names: &[&str]) -> Vec<String> {
    labels(names)
}

fn compare_two(
    a: &dyn CalibratedCameras,
    b: &dyn CalibratedCameras,
    relation: DataRelation,
) -> RunDifference {
    let sizes = vec![SIZE; a.camera_params().len()];
    let runs = [
        ComparedRun::new("a", a, ids(&["cam"]), sizes.clone()),
        ComparedRun::new("b", b, ids(&["cam"]), sizes),
    ];
    let options = CompareRunsOptions::new().with_relations(vec![relation]);
    compare_runs(&runs, &options)
        .expect("comparison")
        .differences
        .remove(0)
}

fn parameter(difference: &RunDifference, label: &str) -> ParameterDifference {
    let ParameterComparison::SameModel { parameters } = &difference.cameras[0].parameters else {
        panic!("same model expected");
    };
    parameters
        .iter()
        .find(|p| p.label == label)
        .expect("parameter")
        .clone()
}

fn nested_other() -> DataRelation {
    DataRelation::Nested {
        smaller: RunSide::Other,
    }
}

#[test]
fn independent_sigma_is_the_root_sum_of_squares() {
    let a = planar(camera(800.0, -0.1), Some(4.0), None);
    let b = planar(camera(806.0, -0.1), Some(2.25), None);
    let fx = parameter(&compare_two(&a, &b, DataRelation::Independent), "fx");
    assert_eq!(fx.delta, 6.0);
    assert_eq!(fx.reference_sigma, Some(2.0));
    assert_eq!(fx.other_sigma, Some(1.5));
    assert_eq!(fx.delta_over_reference_sigma, Some(3.0));
    let DifferenceTest::Independent { sigma, z, p_value } = fx.test else {
        panic!("{:?}", fx.test);
    };
    assert!((sigma - 2.5).abs() < 1e-12 && (z - 2.4).abs() < 1e-12);
    // Two-sided p of z = 2.4.
    assert!((p_value - 0.016_395_071_9).abs() < 1e-9, "{p_value}");
}

#[test]
fn nested_sigma_is_the_hausman_form_on_the_larger_runs_scale() {
    // The full run states σ² = 1, the smaller 1.5625: σ_Δ² = 0.5625.
    let full = planar(camera(800.0, -0.1), Some(1.0), None);
    let small = planar(camera(801.5, -0.1), Some(1.5625), None);
    let fx = parameter(&compare_two(&full, &small, nested_other()), "fx");
    let DifferenceTest::Nested { sigma, z, .. } = fx.test else {
        panic!("{:?}", fx.test);
    };
    assert!((sigma - 0.75).abs() < 1e-12 && (z - 2.0).abs() < 1e-12);

    // The same pair with the smaller run as the reference: Δ changes sign.
    let fx = parameter(
        &compare_two(
            &small,
            &full,
            DataRelation::Nested {
                smaller: RunSide::Reference,
            },
        ),
        "fx",
    );
    let DifferenceTest::Nested { sigma, z, .. } = fx.test else {
        panic!("{:?}", fx.test);
    };
    assert!((sigma - 0.75).abs() < 1e-12 && (z + 2.0).abs() < 1e-12);

    // The full run's covariance scaled by its view scatter 2 (stated σ² =
    // 2): the smaller run's σ² is taken at that scale, 1.5625 · 2.
    let full_scattered = planar(camera(800.0, -0.1), Some(2.0), Some(2.0));
    let fx = parameter(&compare_two(&full_scattered, &small, nested_other()), "fx");
    let DifferenceTest::Nested { sigma, .. } = fx.test else {
        panic!("{:?}", fx.test);
    };
    assert!((sigma - (3.125f64 - 2.0).sqrt()).abs() < 1e-12, "{sigma}");

    // A smaller run that states the smaller variance has no Hausman σ.
    let tighter = planar(camera(801.5, -0.1), Some(0.81), None);
    let fx = parameter(&compare_two(&full, &tighter, nested_other()), "fx");
    assert_eq!(
        fx.test,
        DifferenceTest::NotAvailable {
            reason: NoTest::NestedVarianceNotPositive
        }
    );
}

#[test]
fn no_test_without_a_relation_that_supports_one() {
    let a = planar(camera(800.0, -0.1), Some(4.0), None);
    let b = planar(camera(806.0, -0.1), Some(2.25), None);
    for (relation, reason) in [
        (DataRelation::Unknown, NoTest::UnknownRelation),
        (DataRelation::SameData, NoTest::SharedData),
        (DataRelation::Overlapping, NoTest::SharedData),
        (DataRelation::Identical, NoTest::Identical),
    ] {
        let difference = compare_two(&a, &b, relation);
        let fx = parameter(&difference, "fx");
        assert_eq!(
            fx.test,
            DifferenceTest::NotAvailable { reason },
            "{relation:?}"
        );
        // Materiality is reported whatever the relation.
        assert_eq!(fx.delta_over_reference_sigma, Some(3.0));
        assert_eq!(
            difference.fit_comparable,
            matches!(relation, DataRelation::Identical | DataRelation::SameData)
        );
    }
    // No covariance in one run.
    let bare = planar(camera(806.0, -0.1), None, None);
    let fx = parameter(&compare_two(&a, &bare, DataRelation::Independent), "fx");
    assert_eq!(
        fx.test,
        DifferenceTest::NotAvailable {
            reason: NoTest::NoCovariance
        }
    );
    assert_eq!(fx.other_sigma, None);
    // A parameter one run held fixed (not in its covariance).
    let fx = parameter(&compare_two(&a, &b, DataRelation::Independent), "cx");
    assert_eq!(
        fx.test,
        DifferenceTest::NotAvailable {
            reason: NoTest::NotEstimated
        }
    );
    assert_eq!(fx.delta_over_reference_sigma, None);
}

#[test]
fn different_models_compare_by_projection_only() {
    let a = planar(camera(800.0, -0.1), Some(4.0), None);
    let mut rational = camera(800.0, -0.1);
    rational.distortion = DistortionParams::from_values(
        DistortionKind::Rational8,
        &[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("rational");
    let b = planar(rational, Some(4.0), None);
    let difference = compare_two(&a, &b, DataRelation::SameData);
    assert_eq!(difference.model_changes, ["cam"]);
    assert_eq!(
        difference.cameras[0].parameters,
        ParameterComparison::DifferentModels
    );
    // The same lens in two models projects identically.
    let ProjectionComparison::Compared {
        rms_valid_px,
        rotation_deg,
        fit_nodes,
        ..
    } = difference.cameras[0].projection
    else {
        panic!("{:?}", difference.cameras[0].projection);
    };
    assert!(rms_valid_px < 1e-9 && rotation_deg < 1e-9, "{rms_valid_px}");
    assert!(fit_nodes > 0);
}

#[test]
fn a_radial_change_shows_in_the_projection_difference() {
    let a = planar(camera(800.0, -0.10), Some(4.0), None);
    let b = planar(camera(800.0, -0.12), Some(4.0), None);
    let difference = compare_two(&a, &b, DataRelation::Independent);
    let ProjectionComparison::Compared {
        max_fit_px,
        max_valid_px,
        ..
    } = difference.cameras[0].projection
    else {
        panic!("{:?}", difference.cameras[0].projection);
    };
    assert!(max_fit_px > 1.0 && max_valid_px >= max_fit_px);
    assert!(difference.model_changes.is_empty());
}

#[test]
fn evidence_reports_each_runs_fit() {
    let a = planar(camera(800.0, -0.1), Some(4.0), Some(1.5));
    let b = planar(camera(806.0, -0.1), None, None);
    let runs = [
        ComparedRun::new("a", &a, ids(&["cam"]), vec![SIZE]),
        ComparedRun::new("b", &b, ids(&["cam"]), vec![SIZE]),
    ];
    let comparison = compare_runs(&runs, &CompareRunsOptions::new()).expect("comparison");
    let [ea, eb] = &comparison.runs[..] else {
        panic!("two runs");
    };
    assert_eq!(ea.kind, ProblemKind::PlanarIntrinsics);
    assert_eq!(
        ea.covariance,
        CovarianceStatus::Available {
            shared_parameters: 2,
            view_scatter: Some(1.5)
        }
    );
    assert!(matches!(
        eb.covariance,
        CovarianceStatus::Unavailable { .. }
    ));
    assert_eq!(ea.reference_camera.as_deref(), Some("cam"));
    let c = &ea.cameras[0];
    assert_eq!((c.views, c.features, c.unprojected), (4, 120, 0));
    let stats = c.reprojection.expect("statistics");
    assert!((stats.rms - 0.1).abs() < 1e-12);
    assert_eq!(stats.median, Some(0.1));
    assert_eq!(stats.count, 120);
    assert_eq!(c.image_size, SIZE);
    assert_eq!(comparison.differences[0].relation, DataRelation::Unknown);
}

// ── Rigs ────────────────────────────────────────────────────────────────────

fn pose(r: [f64; 3], t: [f64; 3]) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_scaled_axis(Vector3::new(r[0], r[1], r[2])),
    )
}

/// `right_se3_left`, the second camera's pose relative to the first.
fn right_se3_left() -> Iso3 {
    pose([0.01, 0.12, -0.005], [-0.12, 0.004, 0.01])
}

fn rig_labels(camera: usize, pose: &str) -> Vec<String> {
    ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .map(|c| format!("cam{camera}.{pose}.{c}"))
        .collect()
}

/// A two-camera rig export: cameras in `order` (`left`/`right`), the rig
/// frame the camera `reference`; each estimated pose with variance `var`
/// per coordinate.
fn rig(order: [&str; 2], reference: &str, mount: Iso3, var: f64) -> RigExtrinsicsExport {
    // Poses relative to the left camera, then to the reference.
    let relative = |id: &str| {
        if id == "left" {
            Iso3::identity()
        } else {
            mount
        }
    };
    let to_reference = relative(reference).inverse();
    let cam_se3_rig: Vec<Iso3> = order.iter().map(|id| relative(id) * to_reference).collect();
    let mut names = Vec::new();
    let mut variances = Vec::new();
    for (i, id) in order.iter().enumerate() {
        names.push(format!("cam{i}.fx"));
        variances.push(1.0);
        if *id != reference {
            names.extend(rig_labels(i, "cam_se3_rig"));
            variances.extend([var; 6]);
        }
    }
    RigExtrinsicsExport {
        kind: ProblemKind::RigExtrinsics,
        schema_version: RigExtrinsicsExport::SCHEMA_VERSION,
        cameras: vec![camera(800.0, -0.1), camera(800.0, -0.1)],
        cam_se3_rig,
        rig_se3_target: Vec::new(),
        mean_reproj_error: 0.1,
        per_cam_reproj_errors: vec![0.1, 0.1],
        diagnostics: residuals(2),
        uncertainty: covariance(&names, &variances, None),
    }
}

fn compare_rigs(
    a: &dyn CalibratedCameras,
    a_ids: [&str; 2],
    b: &dyn CalibratedCameras,
    b_ids: [&str; 2],
    relation: DataRelation,
) -> RunDifference {
    let runs = [
        ComparedRun::new("a", a, ids(&a_ids), vec![SIZE; 2]),
        ComparedRun::new("b", b, ids(&b_ids), vec![SIZE; 2]),
    ];
    let options = CompareRunsOptions::new().with_relations(vec![relation]);
    compare_runs(&runs, &options)
        .expect("comparison")
        .differences
        .remove(0)
}

#[test]
fn rig_pose_chi2_of_a_known_rotation() {
    let a = rig(["left", "right"], "left", right_se3_left(), 1e-6);
    // The right camera turned by 1 mrad about its z axis.
    let turned = right_se3_left() * pose([0.0, 0.0, 1e-3], [0.0; 3]);
    let b = rig(["left", "right"], "left", turned, 1e-6);
    let d = compare_rigs(
        &a,
        ["left", "right"],
        &b,
        ["left", "right"],
        DataRelation::Independent,
    );
    assert_eq!(d.rig.len(), 1);
    let p = &d.rig[0];
    assert_eq!(p.label, "right_se3_left");
    assert!((p.rotation_deg - 1e-3f64.to_degrees()).abs() < 1e-12);
    assert!((p.tangent[2] - 1e-3).abs() < 1e-12 && p.translation_norm_m < 1e-15);
    // τ = (0, 0, 1e-3, 0, 0, 0), Σ = 2e-6 I: χ² = 0.5.
    let PoseTest::Independent { chi2, dof, .. } = p.test else {
        panic!("{:?}", p.test);
    };
    assert!((chi2 - 0.5).abs() < 1e-9 && dof == 6, "{chi2}");
    assert!((p.delta_over_reference_sigma.expect("materiality") - 1.0).abs() < 1e-9);
}

#[test]
fn rig_cameras_match_by_id_in_any_order() {
    let a = rig(["left", "right"], "left", right_se3_left(), 1e-6);
    let b = rig(["right", "left"], "left", right_se3_left(), 1e-6);
    let d = compare_rigs(
        &a,
        ["left", "right"],
        &b,
        ["right", "left"],
        DataRelation::Independent,
    );
    let p = &d.rig[0];
    assert!(p.rotation_deg < 1e-12 && p.translation_norm_m < 1e-15);
    let PoseTest::Independent { chi2, .. } = p.test else {
        panic!("{:?}", p.test);
    };
    assert!(chi2 < 1e-12);
    assert!(d.unmatched_cameras.is_empty());
    assert_eq!(d.cameras.len(), 2);
    assert_eq!(d.cameras[1].id, "right");
}

#[test]
fn rig_poses_do_not_depend_on_the_reference_camera() {
    let a = rig(["left", "right"], "left", right_se3_left(), 1e-6);
    let b = rig(["left", "right"], "right", right_se3_left(), 1e-6);
    let d = compare_rigs(
        &a,
        ["left", "right"],
        &b,
        ["left", "right"],
        DataRelation::Independent,
    );
    let p = &d.rig[0];
    assert!(
        p.rotation_deg < 1e-9 && p.translation_norm_m < 1e-12,
        "{p:?}"
    );
    assert_eq!(
        p.test,
        PoseTest::NotAvailable {
            reason: NoTest::GaugeDiffers
        }
    );
}

fn handeye_rig(reference: &str, gripper_se3_left: Iso3) -> RigHandeyeExport {
    let base = rig(["left", "right"], reference, right_se3_left(), 1e-6);
    // gripper_se3_rig = gripper_se3_left · left_se3_rig.
    let left_se3_rig = base.cam_se3_rig[0];
    let mut names = Vec::new();
    let mut variances = Vec::new();
    if let Uncertainty::Available(c) = &base.uncertainty {
        names.clone_from(&c.labels);
        variances = (0..names.len()).map(|i| c.covariance[i][i]).collect();
    }
    names.extend(["rx", "ry", "rz", "tx", "ty", "tz"].map(|c| format!("gripper_se3_rig.{c}")));
    variances.extend([1e-6; 6]);
    RigHandeyeExport {
        kind: ProblemKind::RigHandeye,
        schema_version: RigHandeyeExport::SCHEMA_VERSION,
        cameras: base.cameras,
        cam_se3_rig: base.cam_se3_rig,
        rig_se3_target: Vec::new(),
        handeye: HandeyeSolution::EyeInHand {
            gripper_se3_rig: gripper_se3_left * left_se3_rig,
            base_se3_target: Iso3::identity(),
        },
        robot_refinement: None,
        mean_reproj_error: 0.1,
        per_cam_reproj_errors: vec![0.1, 0.1],
        diagnostics: residuals(2),
        uncertainty: covariance(&names, &variances, None),
    }
}

#[test]
fn the_handeye_is_compared_for_the_reference_camera() {
    let g = pose([0.1, -0.2, 0.3], [0.05, 0.02, 0.15]);
    let a = handeye_rig("left", g);
    let b = handeye_rig("left", g * pose([2e-3, 0.0, 0.0], [0.0; 3]));
    let d = compare_rigs(
        &a,
        ["left", "right"],
        &b,
        ["left", "right"],
        DataRelation::Independent,
    );
    let h = d.handeye.expect("hand-eye");
    assert_eq!(h.label, "gripper_se3_left");
    assert!((h.rotation_deg - 2e-3f64.to_degrees()).abs() < 1e-9);
    let PoseTest::Independent { chi2, .. } = h.test else {
        panic!("{:?}", h.test);
    };
    // τ = (2e-3, 0…), Σ = 2e-6 I: χ² = 2.
    assert!((chi2 - 2.0).abs() < 1e-9, "{chi2}");

    // Calibrated with the right camera as the rig frame: the same hand-eye
    // for the left camera, without a test.
    let c = handeye_rig("right", g);
    let d = compare_rigs(
        &a,
        ["left", "right"],
        &c,
        ["left", "right"],
        DataRelation::Independent,
    );
    let h = d.handeye.expect("hand-eye");
    assert!(
        h.rotation_deg < 1e-9 && h.translation_norm_m < 1e-12,
        "{h:?}"
    );
    assert_eq!(
        h.test,
        PoseTest::NotAvailable {
            reason: NoTest::GaugeDiffers
        }
    );
}

fn laser(normal: Vector3<f64>, distance: f64, d_var: f64) -> LaserlineDeviceExport {
    LaserlineDeviceExport {
        kind: ProblemKind::LaserlineDevice,
        schema_version: LaserlineDeviceExport::SCHEMA_VERSION,
        camera: camera(800.0, -0.1),
        cam_se3_target: Vec::new(),
        laser_plane: LaserPlane::new(normal, distance).expect("plane"),
        stats: LaserlineStats::new(0.1, 0.01, Vec::new(), Vec::new()),
        mean_reproj_error: 0.1,
        per_cam_reproj_errors: vec![0.1],
        uncertainty: covariance(&labels(&["fx", "plane.d"]), &[1.0, d_var], None),
        diagnostics: residuals(1),
    }
}

#[test]
fn laser_planes_compare_with_matched_signs() {
    let n = Vector3::new(0.0, 0.6, 0.8);
    let a = laser(n, -0.30, 1e-8);
    // The same plane written with the opposite sign, 0.2 mm further.
    let b = laser(-n, 0.3002, 1e-8);
    let d = compare_two(&a, &b, DataRelation::Independent);
    let p = &d.laser_planes[0];
    assert!(p.normal_angle_deg < 1e-9);
    assert!((p.distance_other_m + 0.3002).abs() < 1e-15);
    assert!((p.distance_delta_m + 2e-4).abs() < 1e-12);
    let DifferenceTest::Independent { sigma, .. } = p.distance_test else {
        panic!("{:?}", p.distance_test);
    };
    assert!((sigma - (2e-8f64).sqrt()).abs() < 1e-15);
}

// ── Configuration, inputs, wire ─────────────────────────────────────────────

#[test]
fn configuration_changes_are_listed_by_path() {
    let a = planar(camera(800.0, -0.1), Some(4.0), None);
    let b = planar(camera(806.0, -0.1), Some(2.25), None);
    let mut huber = PlanarIntrinsicsConfig::default();
    huber.optimize.reprojection_loss = vision_calibration_optim::RobustLoss::Huber { scale: 1.0 };
    let plain = CalibrationConfig::PlanarIntrinsics(PlanarIntrinsicsConfig::default());
    let runs = [
        ComparedRun::new("a", &a, ids(&["cam"]), vec![SIZE]).with_config(plain.clone()),
        ComparedRun::new("b", &b, ids(&["cam"]), vec![SIZE])
            .with_config(CalibrationConfig::PlanarIntrinsics(huber)),
        ComparedRun::new("c", &b, ids(&["cam"]), vec![SIZE]).with_config(plain),
        ComparedRun::new("d", &b, ids(&["cam"]), vec![SIZE]),
    ];
    let comparison = compare_runs(&runs, &CompareRunsOptions::new()).expect("comparison");
    let changes = comparison.differences[0]
        .config_changes
        .as_ref()
        .expect("both configured");
    assert!(!changes.is_empty());
    assert!(
        changes
            .iter()
            .all(|c| c.path.starts_with("config.optimize.reprojection_loss")),
        "{changes:?}"
    );
    assert_eq!(comparison.differences[1].config_changes, Some(Vec::new()));
    assert_eq!(comparison.differences[2].config_changes, None);
}

#[test]
fn inputs_are_checked() {
    let a = planar(camera(800.0, -0.1), Some(4.0), None);
    let r = rig(["left", "right"], "left", right_se3_left(), 1e-6);
    let one = ids(&["cam"]);
    let check = |runs: &[ComparedRun<'_>], options: &CompareRunsOptions| {
        assert!(matches!(
            compare_runs(runs, options),
            Err(Error::InvalidInput { .. })
        ));
    };
    let options = CompareRunsOptions::new();
    check(
        &[ComparedRun::new("a", &a, one.clone(), vec![SIZE])],
        &options,
    );
    // Another workflow.
    check(
        &[
            ComparedRun::new("a", &a, one.clone(), vec![SIZE]),
            ComparedRun::new("r", &r, ids(&["cam", "x"]), vec![SIZE; 2]),
        ],
        &options,
    );
    // Ids or sizes not one per camera.
    check(
        &[
            ComparedRun::new("a", &a, ids(&["cam", "x"]), vec![SIZE]),
            ComparedRun::new("b", &a, one.clone(), vec![SIZE]),
        ],
        &options,
    );
    check(
        &[
            ComparedRun::new("a", &a, one.clone(), vec![]),
            ComparedRun::new("b", &a, one.clone(), vec![SIZE]),
        ],
        &options,
    );
    // A repeated id.
    check(
        &[
            ComparedRun::new("a", &r, ids(&["x", "x"]), vec![SIZE; 2]),
            ComparedRun::new("b", &r, ids(&["x", "y"]), vec![SIZE; 2]),
        ],
        &options,
    );
    // No shared camera; a shared camera of another size.
    check(
        &[
            ComparedRun::new("a", &a, one.clone(), vec![SIZE]),
            ComparedRun::new("b", &a, ids(&["other"]), vec![SIZE]),
        ],
        &options,
    );
    check(
        &[
            ComparedRun::new("a", &a, one.clone(), vec![SIZE]),
            ComparedRun::new("b", &a, one.clone(), vec![[640, 480]]),
        ],
        &options,
    );
    // Relations not one per later run; no grid columns.
    let two = [
        ComparedRun::new("a", &a, one.clone(), vec![SIZE]),
        ComparedRun::new("b", &a, one.clone(), vec![SIZE]),
    ];
    check(
        &two,
        &CompareRunsOptions::new().with_relations(vec![DataRelation::Independent; 2]),
    );
    let mut no_columns = CompareRunsOptions::new();
    no_columns.grid_columns = 0;
    check(&two, &no_columns);
}

#[test]
fn the_result_is_deterministic_and_round_trips() {
    let a = rig(["left", "right"], "left", right_se3_left(), 1e-6);
    let b = rig(["right", "left"], "right", right_se3_left(), 2e-6);
    let runs = [
        ComparedRun::new("a", &a, ids(&["left", "right"]), vec![SIZE; 2]),
        ComparedRun::new("b", &b, ids(&["right", "left"]), vec![SIZE; 2]),
    ];
    let options = CompareRunsOptions::new().with_relations(vec![DataRelation::Nested {
        smaller: RunSide::Reference,
    }]);
    let first = compare_runs(&runs, &options).expect("comparison");
    let second = compare_runs(&runs, &options).expect("comparison");
    assert_eq!(first, second);
    let json = serde_json::to_string(&first).expect("serialize");
    let back: RunComparison = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, first);
    let json = serde_json::to_string(&options).expect("serialize");
    assert_eq!(
        serde_json::from_str::<CompareRunsOptions>(&json).expect("deserialize"),
        options
    );
    assert_eq!(
        serde_json::from_str::<CompareRunsOptions>("{}").expect("defaults"),
        CompareRunsOptions::default()
    );
}

#[test]
fn chi2_survival_matches_reference_values() {
    let z95 = 1.959_963_984_540_054_f64;
    for (x, dof, p) in [
        (z95 * z95, 1, 0.05),
        (12.591_587_243_743_977, 6, 0.05),
        (3.841_458_820_694_124, 1, 0.05),
        (6.0, 6, 0.423_190_081_126_843_1),
        (1.0, 2, (-0.5f64).exp()),
        // Q(3, 20) = e⁻²⁰ (1 + 20 + 200).
        (40.0, 6, 221.0 * (-20.0f64).exp()),
    ] {
        let got = chi2_survival(x, dof);
        assert!(
            (got - p).abs() <= 1e-12 * p.max(1e-3),
            "Q({dof}, {x}) = {got}, want {p}"
        );
    }
    assert_eq!(chi2_survival(0.0, 6), 1.0);
}
