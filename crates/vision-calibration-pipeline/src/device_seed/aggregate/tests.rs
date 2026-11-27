use super::*;
use nalgebra::Vector3;
use vision_calibration_core::{
    BrownConrady5, FxFyCxCy, ScheimpflugParams, make_pinhole_camera, pinhole_camera_params,
};

/// A calibrated Brown-Conrady lens of a 640 × 480 camera.
fn lens(fx: f64, k: [f64; 3], tilt_x: Option<f64>) -> SeedLens {
    let intrinsics = FxFyCxCy {
        fx,
        fy: fx + 0.5,
        cx: 320.0,
        cy: 240.0,
    };
    let distortion = BrownConrady5 {
        k1: k[0],
        k2: k[1],
        k3: k[2],
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let mut camera = pinhole_camera_params(&make_pinhole_camera(intrinsics, distortion));
    if let Some(tilt_x) = tilt_x {
        camera = camera.with_sensor(SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x,
                tilt_y: 0.0,
            },
        });
    }
    SeedLens::Calibrated { camera }
}

fn rotation(axis: Vector3<f64>, degrees: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(axis.normalize() * degrees.to_radians())
}

fn pose(r: UnitQuaternion<f64>, t: [f64; 3]) -> Iso3 {
    Iso3::from_parts(Translation3::new(t[0], t[1], t[2]), r)
}

/// The type's cam1 mount: 0.1 m to the side, turned 5° about y.
fn nominal_mount() -> Iso3 {
    pose(rotation(Vector3::y(), 5.0), [-0.1, 0.0, 0.0])
}

fn nominal_handeye() -> Iso3 {
    pose(
        rotation(Vector3::new(1.0, 1.0, 0.0), 30.0),
        [0.0, 0.05, 0.2],
    )
}

/// One unit of a two-camera eye-in-hand rig with laser planes: the type's
/// values with `fx` and `k1` offset, the cam1 mount and the hand-eye turned
/// by `turn` and shifted by `shift` metres along x.
fn unit(fx: f64, k1: f64, turn: UnitQuaternion<f64>, shift: f64) -> DeviceSeed {
    let mut cam0 = SeedCamera::new("cam0", [640, 480], lens(fx, [k1, 0.0, 0.0], Some(0.02)));
    cam0.cam_se3_rig = Some(Iso3::identity());
    cam0.laser_plane = Some(LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), 0.3).unwrap());
    let mut cam1 = SeedCamera::new("cam1", [640, 480], lens(fx + 10.0, [k1, 0.0, 0.0], None));
    let offset = pose(turn, [shift, 0.0, 0.0]);
    cam1.cam_se3_rig = Some(offset * nominal_mount());
    cam1.laser_plane = Some(LaserPlane::new(Vector3::new(0.0, -0.6, 0.8), -0.25).unwrap());
    let mut seed = DeviceSeed::new(vec![cam0, cam1]);
    seed.handeye = Some(SeedHandeye::EyeInHand {
        gripper_se3_rig: offset * nominal_handeye(),
    });
    seed
}

/// Four units around the type, ±1° about x and y, and one outlier: fx off
/// by 20 %, k1 of the other sign, its mount and hand-eye turned 10° about z
/// and 5 cm off.
fn units_with_outlier() -> Vec<DeviceSeed> {
    vec![
        unit(800.0, -0.20, rotation(Vector3::x(), 1.0), 0.002),
        unit(802.0, -0.21, rotation(Vector3::x(), -1.0), -0.002),
        unit(960.0, 0.10, rotation(Vector3::z(), 10.0), 0.05),
        unit(801.0, -0.19, rotation(Vector3::y(), 1.0), 0.001),
        unit(799.0, -0.20, rotation(Vector3::y(), -1.0), -0.001),
    ]
}

fn json(seed: &DeviceSeed) -> String {
    serde_json::to_string(seed).unwrap()
}

fn calibrated(camera: &SeedCamera) -> &CameraParams {
    match &camera.lens {
        SeedLens::Calibrated { camera } => camera,
        SeedLens::Nominal { .. } => panic!("the aggregate is calibrated"),
    }
}

fn spread<'a>(aggregate: &'a SeedAggregate, label: &str) -> &'a SeedSpread {
    aggregate
        .spread
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no spread `{label}`"))
}

#[test]
fn the_summary_of_a_known_series() {
    let odd = summarize("x".into(), vec![3.0, 1.0, 2.0, 100.0, 4.0]);
    assert_eq!(odd.values, [3.0, 1.0, 2.0, 100.0, 4.0]);
    assert_eq!(odd.median, 3.0);
    // |x − 3| = 0, 2, 1, 97, 1: the middle one is 1.
    assert_eq!(odd.mad_sigma, MAD_TO_SIGMA);
    assert_eq!((odd.min, odd.max), (1.0, 100.0));

    let even = summarize("x".into(), vec![10.0, 1.0, 3.0, 2.0]);
    assert_eq!(even.median, 2.5);
    // |x − 2.5| = 0.5, 0.5, 1.5, 7.5: the middle two average to 1.
    assert_eq!(even.mad_sigma, MAD_TO_SIGMA);

    let one = summarize("x".into(), vec![7.0]);
    assert_eq!(
        (one.median, one.mad_sigma, one.min, one.max),
        (7.0, 0.0, 7.0, 7.0)
    );
    // 1.4826 is 1 / Φ⁻¹(3/4).
    assert!((MAD_TO_SIGMA * 0.674_489_750_196_081_7 - 1.0).abs() < 1e-12);
}

#[test]
fn the_median_ignores_an_outlier_unit() {
    let units = units_with_outlier();
    let aggregate = DeviceSeed::aggregate(&units).unwrap();
    assert_eq!(aggregate.units, 5);
    assert!(aggregate.omitted.is_empty(), "{:?}", aggregate.omitted);
    let seed = &aggregate.seed;
    let ids: Vec<&str> = seed.cameras.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["cam0", "cam1"]);

    // Each scalar within the four inliers' range.
    let cam0 = calibrated(&seed.cameras[0]);
    assert!(
        (799.0..=802.0).contains(&cam0.intrinsics.fx),
        "{}",
        cam0.intrinsics.fx
    );
    let fx = spread(&aggregate, "cam0.fx");
    assert_eq!(fx.values, [800.0, 802.0, 960.0, 801.0, 799.0]);
    assert_eq!(fx.median, 801.0);
    assert_eq!((fx.min, fx.max), (799.0, 960.0));
    // |fx − 801| = 1, 1, 159, 0, 2: the middle one is 1.
    assert!((fx.mad_sigma - MAD_TO_SIGMA).abs() < 1e-12);
    let k1 = spread(&aggregate, "cam0.k1");
    assert!((-0.21..=-0.19).contains(&k1.median), "{}", k1.median);
    assert_eq!(spread(&aggregate, "cam0.tilt_x").median, 0.02);
    assert_eq!(calibrated(&seed.cameras[1]).intrinsics.fx, 811.0);

    // The reference camera's mount is the identity; cam1's is the type's.
    assert_eq!(seed.cameras[0].cam_se3_rig, Some(Iso3::identity()));
    let mount = seed.cameras[1].cam_se3_rig.unwrap();
    let truth = nominal_mount();
    let angle = (truth.rotation.inverse() * mount.rotation)
        .angle()
        .to_degrees();
    assert!(angle < 0.05, "mount {angle}° from the type's");
    assert!((mount.translation.vector - truth.translation.vector).norm() < 2.5e-3);
    for axis in ["rx", "ry", "rz", "tx", "ty", "tz"] {
        spread(&aggregate, &format!("cam1.cam_se3_rig.{axis}"));
    }
    let Some(SeedHandeye::EyeInHand { gripper_se3_rig }) = seed.handeye else {
        panic!("the hand-eye is aggregated");
    };
    // The tangent median is componentwise: with the outlier pulling the
    // chordal mean, the inliers' turns (about the gripper's x and y, not
    // the tangent axes) leave it within their spread (1°), not at their
    // mean.
    let truth = nominal_handeye();
    let angle = (truth.rotation.inverse() * gripper_se3_rig.rotation)
        .angle()
        .to_degrees();
    assert!(angle < 0.5, "hand-eye {angle}° from the type's");
    spread(&aggregate, "handeye.rz");

    let plane = seed.cameras[1].laser_plane.as_ref().unwrap();
    // The sign is made `distance > 0` first.
    assert_eq!(plane.distance(), 0.25);
    assert!((plane.normal().into_inner() - Vector3::new(0.0, 0.6, -0.8)).norm() < 1e-12);
    assert_eq!(spread(&aggregate, "cam1.laser_plane.distance").median, 0.25);
}

#[test]
fn the_rotation_median_of_inliers_is_their_mean() {
    // Four units ±2° about x and y around the type, no outlier: their
    // chordal mean is the type's mount, and the tangent median keeps it.
    let units = vec![
        unit(800.0, -0.2, rotation(Vector3::x(), 2.0), 0.0),
        unit(800.0, -0.2, rotation(Vector3::x(), -2.0), 0.0),
        unit(800.0, -0.2, rotation(Vector3::y(), 2.0), 0.0),
        unit(800.0, -0.2, rotation(Vector3::y(), -2.0), 0.0),
    ];
    let aggregate = DeviceSeed::aggregate(&units).unwrap();
    let mount = aggregate.seed.cameras[1].cam_se3_rig.unwrap();
    let angle = (nominal_mount().rotation.inverse() * mount.rotation)
        .angle()
        .to_degrees();
    assert!(angle < 0.01, "{angle}°");
    let Some(SeedHandeye::EyeInHand { gripper_se3_rig }) = aggregate.seed.handeye else {
        panic!("eye-in-hand");
    };
    let angle = (nominal_handeye().rotation.inverse() * gripper_se3_rig.rotation)
        .angle()
        .to_degrees();
    assert!(angle < 0.01, "{angle}°");
}

#[test]
fn the_order_of_the_seeds_does_not_change_the_seed() {
    let units = units_with_outlier();
    let forward = DeviceSeed::aggregate(&units).unwrap();
    let mut reversed = units.clone();
    reversed.reverse();
    reversed.swap(0, 2);
    let backward = DeviceSeed::aggregate(&reversed).unwrap();
    assert_eq!(json(&forward.seed), json(&backward.seed));
    // A seed's camera order does not matter either.
    let mut shuffled = units;
    shuffled[1].cameras.reverse();
    assert_eq!(
        json(&forward.seed),
        json(&DeviceSeed::aggregate(&shuffled).unwrap().seed)
    );
}

#[test]
fn each_seed_is_taken_in_the_frame_of_the_smallest_camera_id() {
    let units = units_with_outlier();
    let before = DeviceSeed::aggregate(&units).unwrap();
    // The first unit stated in another rig frame: every mount and the
    // hand-eye moved by the same change of frame, as a run whose reference
    // camera was cam1 would state it.
    let mut moved = units.clone();
    let change = pose(
        rotation(Vector3::new(0.3, -1.0, 0.2), 40.0),
        [0.3, -0.2, 0.1],
    );
    for camera in &mut moved[0].cameras {
        camera.cam_se3_rig = camera.cam_se3_rig.map(|c| c * change);
    }
    if let Some(SeedHandeye::EyeInHand { gripper_se3_rig }) = &mut moved[0].handeye {
        *gripper_se3_rig *= change;
    }
    let after = DeviceSeed::aggregate(&moved).unwrap();
    let pose_gap = |a: &Iso3, b: &Iso3| {
        (a.rotation.inverse() * b.rotation).angle()
            + (a.translation.vector - b.translation.vector).norm()
    };
    let mount = |s: &SeedAggregate| s.seed.cameras[1].cam_se3_rig.unwrap();
    assert!(pose_gap(&mount(&before), &mount(&after)) < 1e-12);
    let handeye = |s: &SeedAggregate| match s.seed.handeye {
        Some(SeedHandeye::EyeInHand { gripper_se3_rig }) => gripper_se3_rig,
        _ => panic!("eye-in-hand"),
    };
    assert!(pose_gap(&handeye(&before), &handeye(&after)) < 1e-12);
}

#[test]
fn one_seed_aggregates_to_itself() {
    let mut camera = SeedCamera::new("cam0", [640, 480], lens(800.0, [-0.2, 0.05, 0.0], None));
    camera.laser_plane = Some(LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), 0.3).unwrap());
    let mut seed = DeviceSeed::new(vec![camera]);
    seed.handeye = Some(SeedHandeye::EyeToHand {
        rig_se3_base: nominal_handeye(),
    });
    let aggregate = DeviceSeed::aggregate(std::slice::from_ref(&seed)).unwrap();
    let one = &aggregate.seed;
    assert_eq!(one.cameras[0].cam_se3_rig, None);
    assert_eq!(
        serde_json::to_value(&one.cameras[0].lens).unwrap(),
        serde_json::to_value(&seed.cameras[0].lens).unwrap()
    );
    let plane = one.cameras[0].laser_plane.as_ref().unwrap();
    assert!((plane.normal().into_inner() - Vector3::new(0.0, 0.6, 0.8)).norm() < 1e-15);
    let Some(SeedHandeye::EyeToHand { rig_se3_base }) = one.handeye else {
        panic!("eye-to-hand");
    };
    let truth = nominal_handeye();
    assert!((truth.rotation.inverse() * rig_se3_base.rotation).angle() < 1e-12);
    assert!(aggregate.spread.iter().all(|s| s.mad_sigma == 0.0));
}

#[test]
fn what_only_some_seeds_have_is_omitted() {
    let mut units = units_with_outlier();
    units[1].cameras[0].laser_plane = None;
    let aggregate = DeviceSeed::aggregate(&units).unwrap();
    assert_eq!(aggregate.omitted, ["cam0.laser_plane"]);
    assert!(aggregate.seed.cameras[0].laser_plane.is_none());
    assert!(aggregate.seed.cameras[1].laser_plane.is_some());

    // A seed without mounts (and so without a hand-eye): the aggregate has
    // neither.
    let mut units = units_with_outlier();
    for camera in &mut units[2].cameras {
        camera.cam_se3_rig = None;
    }
    units[2].handeye = None;
    let aggregate = DeviceSeed::aggregate(&units).unwrap();
    assert_eq!(
        aggregate.omitted,
        ["cam0.cam_se3_rig", "cam1.cam_se3_rig", "handeye"]
    );
    assert!(aggregate.seed.handeye.is_none());
    assert!(
        aggregate
            .seed
            .cameras
            .iter()
            .all(|c| c.cam_se3_rig.is_none())
    );
}

#[test]
fn seeds_of_another_device_type_are_refused() {
    let units = units_with_outlier();
    let refused = |units: &[DeviceSeed]| DeviceSeed::aggregate(units).unwrap_err();

    assert!(matches!(
        refused(&[]),
        DeviceSeedError::AggregateMismatch { .. }
    ));

    let mut other = units.clone();
    other[3].cameras[1].id = "cam2".into();
    assert!(matches!(
        refused(&other),
        DeviceSeedError::AggregateMismatch { .. }
    ));

    let mut other = units.clone();
    other[3].cameras[1].image_size = [1280, 960];
    let err = refused(&other).to_string();
    assert!(err.contains("cam1") && err.contains("1280"), "{err}");

    let mut other = units.clone();
    other[3].cameras[1].lens = lens(800.0, [-0.2, 0.0, 0.0], Some(0.0));
    let err = refused(&other).to_string();
    assert!(err.contains("model"), "{err}");

    let mut other = units.clone();
    other[3].handeye = Some(SeedHandeye::EyeToHand {
        rig_se3_base: nominal_handeye(),
    });
    let err = refused(&other).to_string();
    assert!(err.contains("modes"), "{err}");

    let mut other = units;
    other[3].cameras[0].lens = SeedLens::Nominal {
        intrinsics: FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        },
        sensor: ScheimpflugParams::default(),
    };
    assert!(matches!(
        refused(&other),
        DeviceSeedError::NominalNotAggregated { camera } if camera == "cam0"
    ));
}

#[test]
fn a_median_lens_that_folds_is_refused() {
    // Three lenses valid over a 640 × 480 image at f = 500 px (normalized
    // radius up to 0.8), whose radial coefficients' medians (−0.7, −0.7,
    // −1.5) fold well inside it: each unit's large coefficient keeps it
    // unfolded, and the median takes the small ones.
    let seed = |k: [f64; 3]| {
        let mut camera = match lens(500.0, k, None) {
            SeedLens::Calibrated { camera } => camera,
            SeedLens::Nominal { .. } => unreachable!(),
        };
        camera.intrinsics.fy = 500.0;
        DeviceSeed::new(vec![SeedCamera::new(
            "cam0",
            [640, 480],
            SeedLens::Calibrated { camera },
        )])
    };
    let units = [
        seed([-0.7, 2.2, -1.6]),
        seed([2.7, -0.9, -1.5]),
        seed([-1.2, -0.7, 2.7]),
    ];
    let err = DeviceSeed::aggregate(&units).unwrap_err();
    assert!(
        matches!(&err, DeviceSeedError::AggregateFolds { camera } if camera == "cam0"),
        "{err}"
    );
}
