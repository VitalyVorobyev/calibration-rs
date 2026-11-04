//! The seeded joint rig laser solve (`run_calibration_with_seed`) on the
//! solver matrix's small synthetic rig hand-eye laserline scene: an empty
//! seed is `run_calibration`, a rig hand-eye warm start computed separately
//! reproduces it bit for bit, seeded laser planes recover the truth within
//! the unseeded bounds, and a seed that does not fit the input or the config
//! is an `InvalidInput` raised before any solve. With one camera of the rig
//! (a triangulation head) the solve recovers that camera's truth.

use nalgebra::{Rotation3, Vector3};
use vision_calibration::Error;
use vision_calibration::common::{ExportKind, HandeyeSolution};
use vision_calibration::core::{Iso3, LaserPlane};
use vision_calibration::core::{RigDataset, RigView, RigViewObs};
use vision_calibration::optim::RobustLoss;
use vision_calibration::rig_handeye::{
    RigHandeyeExport, RigHandeyeInput, RigHandeyeProblem, run_calibration as run_rig_handeye,
};
use vision_calibration::rig_handeye_laserline::{
    JointManualInit, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem, run_calibration,
    run_calibration_with_seed,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_bench::solver::metrics::{GtErrors, ModelParams};
use vision_calibration_bench::solver::scenes::{
    CameraKind, OutlierSpec, Problem, Scale, Scene, SceneData, SceneSpec,
};

struct Fixture {
    input: RigHandeyeLaserlineInput,
    config: RigHandeyeLaserlineConfig,
    truth: ModelParams,
}

fn fixture() -> Fixture {
    let spec = SceneSpec {
        problem: Problem::RigHandeyeLaserline,
        camera: CameraKind::Pinhole,
        scale: Scale::Small,
        noise_px: 0.1,
        outliers: OutlierSpec::None,
        loss: RobustLoss::None,
    };
    let scene = Scene::build(&spec).expect("scene");
    let SceneData::RigHandeyeLaserline { input, config } = scene.data else {
        panic!("not a rig hand-eye laserline scene");
    };
    Fixture {
        input,
        config,
        truth: scene.truth,
    }
}

/// The rig hand-eye stage run on its own, as a staged procedure would.
fn rig_handeye_warm_start(f: &Fixture) -> RigHandeyeExport {
    let dataset: RigHandeyeInput = RigDataset {
        num_cameras: f.input.num_cameras,
        views: f
            .input
            .views
            .iter()
            .map(|v| RigView {
                meta: v.meta.clone(),
                obs: RigViewObs {
                    cameras: v.obs.cameras.clone(),
                },
            })
            .collect(),
    };
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_input(dataset).unwrap();
    session.set_config(f.config.handeye.clone()).unwrap();
    run_rig_handeye(&mut session).unwrap();
    session.export().unwrap()
}

fn session(f: &Fixture) -> CalibrationSession<RigHandeyeLaserlineProblem> {
    let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
    session.set_config(f.config.clone()).unwrap();
    session.set_input(f.input.clone()).unwrap();
    session
}

fn seeded(f: &Fixture, seed: JointManualInit) -> RigHandeyeLaserlineExport {
    let mut s = session(f);
    run_calibration_with_seed(&mut s, seed).unwrap();
    s.export().unwrap()
}

fn seed(rig_handeye: Option<RigHandeyeExport>, planes: Option<Vec<LaserPlane>>) -> JointManualInit {
    let mut seed = JointManualInit::default();
    seed.rig_handeye = rig_handeye;
    seed.planes_cam = planes;
    seed
}

fn json(export: &RigHandeyeLaserlineExport) -> serde_json::Value {
    serde_json::to_value(export).unwrap()
}

/// The unseeded run's ground-truth bounds (the solver matrix's).
fn assert_recovers_truth(f: &Fixture, export: &RigHandeyeLaserlineExport) {
    let est = ModelParams {
        cameras: export.cameras.clone(),
        cam_se3_rig: export.cam_se3_rig.clone(),
        handeye: Some(*export.handeye.handeye().pose()),
        planes_cam: export.laser_planes_cam.clone(),
        ..ModelParams::default()
    };
    let gt = GtErrors::between(&est, &f.truth);
    assert!(gt.focal_rel.unwrap() < 0.015, "{gt:?}");
    assert!(gt.handeye_rot_deg.unwrap() < 1.0, "{gt:?}");
    assert!(gt.laser_normal_deg.unwrap() < 2.0, "{gt:?}");
}

/// The truth's planes, each tilted by 3° and moved by 5 mm: a design value.
fn design_planes(f: &Fixture) -> Vec<LaserPlane> {
    let tilt = Rotation3::from_axis_angle(&Vector3::x_axis(), 3.0_f64.to_radians());
    f.truth
        .planes_cam
        .iter()
        .map(|p| {
            LaserPlane::new(
                (tilt * p.normal().into_inner()).normalize(),
                p.distance() + 0.005,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn empty_seed_and_a_separate_warm_start_reproduce_run_calibration() {
    let f = fixture();
    let mut s = session(&f);
    run_calibration(&mut s).unwrap();
    let unseeded = s.export().unwrap();
    assert_recovers_truth(&f, &unseeded);

    assert_eq!(
        json(&seeded(&f, JointManualInit::default())),
        json(&unseeded)
    );

    // The warm start `run_calibration` computes, handed in from outside.
    let staged = seeded(&f, seed(Some(rig_handeye_warm_start(&f)), None));
    assert_eq!(json(&staged), json(&unseeded));
}

#[test]
fn seeded_planes_recover_the_truth() {
    let f = fixture();
    let mut s = session(&f);
    run_calibration(&mut s).unwrap();
    let unseeded = s.export().unwrap();

    let planes_only = seeded(&f, seed(None, Some(design_planes(&f))));
    assert_recovers_truth(&f, &planes_only);
    // The planes reached the solve: it started elsewhere.
    assert_ne!(json(&planes_only), json(&unseeded));

    let both = seeded(
        &f,
        seed(Some(rig_handeye_warm_start(&f)), Some(design_planes(&f))),
    );
    assert_recovers_truth(&f, &both);
    assert_eq!(json(&both), json(&planes_only));
}

#[test]
fn seed_round_trips_through_json() {
    let f = fixture();
    let original = seed(Some(rig_handeye_warm_start(&f)), Some(design_planes(&f)));
    let text = serde_json::to_string(&original).unwrap();
    let restored: JointManualInit = serde_json::from_str(&text).unwrap();
    assert_eq!(
        serde_json::to_value(&restored).unwrap(),
        serde_json::to_value(&original).unwrap()
    );
    // An empty seed is an empty object, and fields left out are `None`.
    let empty: JointManualInit = serde_json::from_str("{}").unwrap();
    assert!(empty.rig_handeye.is_none() && empty.planes_cam.is_none());
}

#[test]
fn a_seed_that_does_not_fit_is_invalid_input_before_any_solve() {
    let f = fixture();
    let export = rig_handeye_warm_start(&f);
    let planes = design_planes(&f);

    let with_export = |edit: &dyn Fn(&mut RigHandeyeExport)| {
        let mut e = export.clone();
        edit(&mut e);
        seed(Some(e), None)
    };
    let cases: Vec<(&str, JointManualInit)> = vec![
        ("kind", with_export(&|e| e.kind = ExportKind::RigExtrinsics)),
        (
            "cameras",
            with_export(&|e| {
                e.cameras.pop();
                e.cam_se3_rig.pop();
            }),
        ),
        (
            "rig_se3_target",
            with_export(&|e| {
                e.rig_se3_target.pop();
            }),
        ),
        (
            "robot_deltas",
            with_export(&|e| e.robot_deltas = Some(vec![[0.0; 6]])),
        ),
        (
            "handeye.mode",
            with_export(&|e| {
                e.handeye = HandeyeSolution::EyeToHand {
                    rig_se3_base: Iso3::identity(),
                    gripper_se3_target: Iso3::identity(),
                }
            }),
        ),
        ("planes", seed(None, Some(planes[..1].to_vec()))),
    ];
    for (what, bad) in cases {
        let mut s = session(&f);
        let err = run_calibration_with_seed(&mut s, bad).unwrap_err();
        let Error::InvalidInput { reason } = &err else {
            panic!("{what}: expected InvalidInput, got {err:?}");
        };
        assert!(reason.contains(what), "{what}: {reason}");
        assert!(!s.has_output(), "{what}");
        // The seed check is the only operation the session saw.
        let log = s.log();
        assert_eq!(log.len(), 1, "{what}: {log:?}");
        assert!(!log[0].success, "{what}");
    }

    // A warm start in another camera model than the config's.
    let mut s = session(&f);
    let mut config = f.config.clone();
    config.handeye.camera.model = CameraKind::Fisheye.model();
    s.set_config(config).unwrap();
    let err = run_calibration_with_seed(&mut s, seed(Some(export), None)).unwrap_err();
    let Error::InvalidInput { reason } = &err else {
        panic!("model: expected InvalidInput, got {err:?}");
    };
    assert!(reason.contains("camera 0"), "{reason}");
    assert!(!s.has_output());
}

/// `f` with only camera 0 and its laser: one camera carrying its laser on a
/// robot. Camera 0 is the rig frame, so the truth's hand-eye is that camera's.
fn head(f: &Fixture) -> Fixture {
    let mut input = f.input.clone();
    input.num_cameras = 1;
    for v in &mut input.views {
        v.obs.cameras.truncate(1);
        v.obs.laser_pixels.truncate(1);
    }
    let mut truth = f.truth.clone();
    truth.cameras.truncate(1);
    truth.cam_se3_rig.truncate(1);
    truth.planes_cam.truncate(1);
    Fixture {
        input,
        config: f.config.clone(),
        truth,
    }
}

#[test]
fn one_camera_carrying_its_laser_recovers_its_truth() {
    let f = head(&fixture());
    let mut s = session(&f);
    run_calibration(&mut s).unwrap();
    let export = s.export().unwrap();
    assert_eq!(export.cameras.len(), 1);
    assert_eq!(export.laser_planes_cam.len(), 1);
    assert_recovers_truth(&f, &export);

    // A seeded warm start: the one-camera rig hand-eye, and a design plane.
    let staged = seeded(
        &f,
        seed(
            Some(rig_handeye_warm_start(&f)),
            Some(design_planes(&f)[..1].to_vec()),
        ),
    );
    assert_recovers_truth(&f, &staged);

    // No camera at all is still refused.
    let mut none = f.input.clone();
    none.num_cameras = 0;
    for v in &mut none.views {
        v.obs.cameras.clear();
        v.obs.laser_pixels.clear();
    }
    let mut s = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
    assert!(matches!(
        s.set_input(none).unwrap_err(),
        Error::InsufficientData { need: 1, got: 0 }
    ));
}
