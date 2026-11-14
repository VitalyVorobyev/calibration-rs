//! A calibrated camera with a residual field projects consistently through
//! every path that evaluates a calibrated camera: the runtime model, the
//! projection derivatives, rig triangulation, stereo rectification, and a
//! downstream workflow (the rig laser calibration) that holds the field
//! fixed and echoes it in its export. A workflow configuration cannot ask
//! for a field it would not estimate.
//!
//! Every check uses noise-free data from cameras with a field of a few
//! pixels, so a path that ignored the field would miss by pixels or by the
//! corresponding depth and plane error.

#![allow(missing_docs)]

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use vision_calibration::common::RigLaserlineView;
use vision_calibration::core::{
    BSplineField, BrownConrady5, CalibratedRig, CameraModelSpec, CameraParams, CorrespondenceView,
    DistortionKind, DistortionParams, FieldKind, FieldParams, FxFyCxCy, Iso3, LaserPlane,
    ParameterCovariance, PointCovariance, ProjectionParams, Pt2, Pt3, Real, SensorKind,
    SensorParams, Uncertainty,
};
use vision_calibration::mvg::rectification::{RectifyOptions, rectify};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::Captures;
use vision_calibration::optim::{camera_parameters, projection_uncertainty};
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;
use vision_calibration::rig_laserline_device::{
    RigLaserlineDataset, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
    RigLaserlineDeviceProblem, RigUpstreamCalibration,
};
use vision_calibration::session::CalibrationSession;

const IMAGE: [u32; 2] = [1280, 720];

/// A smooth field of up to `amplitude` px, 6 × 4 cells; `phase` makes the
/// two cameras' fields differ.
fn field(amplitude: Real, phase: Real) -> FieldParams {
    let cells = [6, 4];
    let mut f = BSplineField::zeros(cells, IMAGE).unwrap();
    let nx = cells[0] as usize + 3;
    for (k, (du, dv)) in f.du.iter_mut().zip(f.dv.iter_mut()).enumerate() {
        let (i, j) = ((k % nx) as Real, (k / nx) as Real);
        *du = amplitude * (0.6 * i + 0.4 * j + phase).sin();
        *dv = amplitude * (0.5 * i - 0.7 * j + phase).cos();
    }
    FieldParams::BSpline { params: f }
}

fn camera(field: FieldParams) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.1,
                k2: 0.02,
                iters: 20,
                ..Default::default()
            },
        },
        SensorParams::Identity,
        FxFyCxCy {
            fx: 900.0,
            fy: 895.0,
            cx: 640.0,
            cy: 360.0,
        },
    )
    .with_field(field)
}

/// Camera 1 12 cm to the right of camera 0, toed in by 2°.
fn cam_se3_rig() -> Vec<Iso3> {
    vec![
        Iso3::identity(),
        Isometry3::from_parts(
            Translation3::new(-0.12, 0.0, 0.0),
            Rotation3::from_euler_angles(0.0, -0.035, 0.0).into(),
        ),
    ]
}

fn field_rig() -> CalibratedRig {
    CalibratedRig::new(
        vec![camera(field(2.5, 0.0)), camera(field(2.0, 1.3))],
        cam_se3_rig(),
        Uncertainty::default(),
    )
    .unwrap()
}

fn points() -> Vec<Pt3> {
    vec![
        Pt3::new(0.0, 0.0, 0.8),
        Pt3::new(0.15, -0.08, 0.9),
        Pt3::new(-0.2, 0.12, 1.1),
        Pt3::new(0.05, 0.18, 0.7),
    ]
}

fn pixel(rig: &CalibratedRig, c: usize, p: &Pt3) -> Pt2 {
    rig.model(c)
        .unwrap()
        .project_point(&(rig.cam_se3_rig()[c] * p))
        .unwrap()
}

#[test]
fn projection_uncertainty_runs_through_the_field() {
    let cam = camera(field(2.5, 0.0));
    // The held field is part of the model, not a parameter.
    let labels: Vec<String> = camera_parameters(&cam)
        .unwrap()
        .into_iter()
        .map(|(l, _)| l.to_string())
        .collect();
    assert!(!labels.iter().any(|l| l.starts_with("du_")));
    let n = labels.len();
    let covariance = ParameterCovariance {
        labels,
        covariance: (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1e-4 } else { 0.0 }).collect())
            .collect(),
        views: Vec::new(),
        fixed: vec!["field".into()],
        noise: Vec::new(),
        num_residuals: 0,
        dof: 0,
        view_scatter: None,
        second_order: None,
    };
    let with = projection_uncertainty(&cam, &covariance, 0, IMAGE).unwrap();
    assert!(with.sigma.iter().all(Option::is_some));
    assert!(with.center.is_finite() && with.center > 0.0);
}

#[test]
fn triangulation_honours_the_field() {
    let rig = field_rig();
    let noise = PointCovariance::isotropic(0.1).unwrap();
    let tracks: Vec<Vec<RigObservation>> = points()
        .iter()
        .map(|p| {
            (0..2)
                .map(|c| RigObservation::new(0, c, pixel(&rig, c, p), noise))
                .collect()
        })
        .collect();
    let result = triangulate(
        &rig,
        Captures::Rig,
        &tracks,
        &vision_calibration::control::RunControl::default(),
    )
    .unwrap();
    for (got, want) in result.points.iter().zip(points()) {
        let d = (got.point - want).norm();
        assert!(d < 1e-8, "triangulated point off by {d} m");
    }

    // The same pixels through cameras without the field land elsewhere.
    let plain = CalibratedRig::new(
        vec![camera(FieldParams::None), camera(FieldParams::None)],
        cam_se3_rig(),
        Uncertainty::default(),
    )
    .unwrap();
    let off = triangulate(
        &plain,
        Captures::Rig,
        &tracks,
        &vision_calibration::control::RunControl::default(),
    )
    .unwrap();
    let d = (off.points[0].point - points()[0]).norm();
    assert!(d > 1e-4, "the field moved the point by only {d} m");
}

#[test]
fn rectification_honours_the_field() {
    let rig = field_rig();
    let rect = rectify(&rig, [0, 1], &RectifyOptions::default()).unwrap();
    for p in points() {
        let raw = [pixel(&rig, 0, &p), pixel(&rig, 1, &p)];
        let left = rect.rectified_pixel(0, &raw[0]).unwrap();
        let right = rect.rectified_pixel(1, &raw[1]).unwrap();
        // A point lands on one rectified row in both images.
        assert!(
            (left.y - right.y).abs() < 1e-8,
            "rows {} vs {}",
            left.y,
            right.y
        );
        // And the rectified pixel maps back to the raw one.
        for (side, r) in [(0, left), (1, right)] {
            let back = rect.raw_pixel(side, &r).unwrap();
            assert!((back - raw[side]).norm() < 1e-8);
        }
    }
}

fn board_poses() -> Vec<Iso3> {
    [
        (0.0, 0.0),
        (0.25, 0.0),
        (-0.25, 0.0),
        (0.0, 0.25),
        (0.0, -0.25),
        (0.18, 0.18),
        (-0.18, 0.18),
        (0.18, -0.18),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(rx, ry))| {
        Isometry3::from_parts(
            Translation3::new(-0.09, -0.09, 0.6 + 0.03 * (i % 3) as Real),
            Rotation3::from_euler_angles(rx, ry, 0.05 * i as Real).into(),
        )
    })
    .collect()
}

fn board_points() -> Vec<Pt3> {
    (0..7)
        .flat_map(|y| (0..9).map(move |x| Pt3::new(x as Real * 0.025, y as Real * 0.03, 0.0)))
        .collect()
}

fn laser_plane_rig() -> LaserPlane {
    let normal = Vector3::new(0.1, 0.9, 0.3).normalize();
    let centre = board_poses()[0].transform_point(&Pt3::new(0.1, 0.09, 0.0));
    LaserPlane::new(normal, -normal.dot(&centre.coords)).unwrap()
}

/// The laser views of the field rig: board corners and the stripe of each
/// camera's plane, projected through the cameras with their fields.
fn laser_views(rig: &CalibratedRig) -> Vec<RigLaserlineView> {
    let plane = laser_plane_rig();
    board_poses()
        .iter()
        .map(|rig_se3_target| {
            let mut view = RigLaserlineView {
                cameras: Vec::new(),
                laser_pixels: Vec::new(),
            };
            for c in 0..2 {
                let model = rig.model(c).unwrap();
                let cam_se3_target = rig.cam_se3_rig()[c] * rig_se3_target;
                let (mut p3, mut p2) = (Vec::new(), Vec::new());
                for pw in board_points() {
                    if let Some(px) = model.project_point(&cam_se3_target.transform_point(&pw)) {
                        p3.push(pw);
                        p2.push(px);
                    }
                }
                view.cameras
                    .push(Some(CorrespondenceView::new(p3, p2).unwrap()));
                let plane_cam = plane.transform_by(&rig.cam_se3_rig()[c]);
                let n_t = cam_se3_target.rotation.inverse() * plane_cam.normal().into_inner();
                let d_t = plane_cam.distance()
                    + plane_cam.normal().dot(&cam_se3_target.translation.vector);
                let stripe: Vec<Pt2> = (0..41)
                    .filter_map(|i| {
                        let x = i as Real * 0.005;
                        let y = -(n_t.x * x + d_t) / n_t.y;
                        (0.0..=0.18).contains(&y).then_some(())?;
                        model.project_point(&cam_se3_target.transform_point(&Pt3::new(x, y, 0.0)))
                    })
                    .collect();
                view.laser_pixels.push(Some(stripe));
            }
            view
        })
        .collect()
}

fn rig_laser_export(
    rig: &CalibratedRig,
    upstream_cameras: Vec<CameraParams>,
) -> RigLaserlineDeviceExport {
    let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
    session
        .set_input(RigLaserlineDeviceInput {
            dataset: RigLaserlineDataset::new(laser_views(rig), 2).unwrap(),
            upstream: RigUpstreamCalibration {
                cameras: upstream_cameras,
                cam_se3_rig: rig.cam_se3_rig().to_vec(),
                rig_se3_target: board_poses(),
                uncertainty: Uncertainty::default(),
            },
        })
        .unwrap();
    vision_calibration::rig_laserline_device::run_calibration(&mut session).unwrap();
    session.export().unwrap()
}

/// The angle between two planes and the difference of their distances, with
/// the normals oriented alike (a plane's sign is a convention).
fn plane_error(a: &LaserPlane, b: &LaserPlane) -> (Real, Real) {
    let dot = a.normal().dot(&b.normal());
    let sign = dot.signum();
    let angle = dot.abs().min(1.0).acos();
    (angle, (sign * a.distance() - b.distance()).abs())
}

#[test]
fn a_downstream_workflow_holds_and_echoes_the_field() {
    let rig = field_rig();
    let export = rig_laser_export(&rig, rig.cameras().to_vec());
    let truth = laser_plane_rig();
    for (c, got) in export.laser_planes_rig.iter().enumerate() {
        let (angle, dd) = plane_error(got, &truth);
        assert!(angle < 1e-6 && dd < 1e-6, "camera {c}: {angle} rad, {dd} m");
    }
    assert!(
        export.mean_reproj_error < 1e-6,
        "reprojection {} px",
        export.mean_reproj_error
    );
    for (got, want) in export.cameras.iter().zip(rig.cameras()) {
        assert_eq!(got.field, want.field);
    }
    // The export round-trips with its fields.
    let json = serde_json::to_string(&export).unwrap();
    let back: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.cameras[1].field, rig.cameras()[1].field);

    // The same data against cameras without their fields: the planes are off.
    let plain = rig_laser_export(
        &rig,
        vec![camera(FieldParams::None), camera(FieldParams::None)],
    );
    let (angle, dd) = plane_error(&plain.laser_planes_rig[0], &truth);
    assert!(
        angle > 1e-4 || dd > 1e-4,
        "without the field the plane is still within {angle} rad, {dd} m"
    );
}

#[test]
fn a_workflow_config_cannot_ask_for_a_field() {
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model =
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity).with_field(
            FieldKind::BSpline {
                cells: [6, 4],
                image_size: IMAGE,
            },
        );
    let err = config.camera.validate().unwrap_err().to_string();
    assert!(err.contains("residual field"), "{err}");
}
