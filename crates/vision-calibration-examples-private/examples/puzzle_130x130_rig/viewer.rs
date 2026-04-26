//! Example-local calibration viewer artifact exporter.
//!
//! This is intentionally kept in the private example layer. The JSON shape is a
//! debug/workbench artifact, not a public `vision-calibration` API contract.

use anyhow::{Context, Result};
use nalgebra::{Point2, Point3, Translation3, Unit, UnitQuaternion, Vector3};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use vision_calibration::rig_scheimpflug_handeye::RigScheimpflugHandeyeExport;
use vision_calibration_core::{Camera, Iso3, Pinhole, Pt2, Real};
use vision_calibration_examples_private::PoseEntry;
use vision_calibration_optim::{
    HandEyeMode, LaserPlane, RigHandeyeLaserlineEstimate, RigHandeyeLaserlineParams,
    RigHandeyeLaserlinePerCamStats,
};

use crate::{DetectedDatasets, NUM_CAMERAS};

pub(crate) struct ViewerExportInput<'a> {
    pub out_dir: &'a Path,
    pub data_dir: &'a Path,
    pub poses: &'a [PoseEntry],
    pub tile_w: u32,
    pub tile_h: u32,
    pub detected: &'a DetectedDatasets,
    pub rig_export: &'a RigScheimpflugHandeyeExport,
    pub joint_initial: &'a RigHandeyeLaserlineParams,
    pub joint_initial_stats: &'a [RigHandeyeLaserlinePerCamStats],
    pub joint_est: &'a RigHandeyeLaserlineEstimate,
}

type ViewerCameraModel = Camera<
    Real,
    Pinhole,
    vision_calibration_core::BrownConrady5<Real>,
    vision_calibration_core::HomographySensor<Real>,
    vision_calibration_core::FxFyCxCySkew<Real>,
>;

struct StageBuildInput<'a> {
    id: &'a str,
    label: &'a str,
    mean_reproj_error_px: f64,
    final_cost: Option<f64>,
    detected: &'a DetectedDatasets,
    params: &'a RigHandeyeLaserlineParams,
    mode: HandEyeMode,
    robot_deltas: Option<&'a [[Real; 6]]>,
    per_cam_stats: &'a [RigHandeyeLaserlinePerCamStats],
}

pub(crate) fn write_viewer_artifacts(input: ViewerExportInput<'_>) -> Result<()> {
    fs::create_dir_all(input.out_dir).with_context(|| {
        format!(
            "create calibration viewer output dir {}",
            input.out_dir.display()
        )
    })?;
    copy_images(input.out_dir, input.data_dir, input.poses)?;

    let stage2 = build_stage(StageBuildInput {
        id: "stage2",
        label: "Rig + Scheimpflug + hand-eye BA",
        mean_reproj_error_px: input.rig_export.mean_reproj_error,
        final_cost: None,
        detected: input.detected,
        params: input.joint_initial,
        mode: input.rig_export.handeye_mode,
        robot_deltas: input.rig_export.robot_deltas.as_deref(),
        per_cam_stats: input.joint_initial_stats,
    });
    let stage3 = build_stage(StageBuildInput {
        id: "stage3",
        label: "Rig laser plane fit",
        mean_reproj_error_px: input.rig_export.mean_reproj_error,
        final_cost: None,
        detected: input.detected,
        params: input.joint_initial,
        mode: input.rig_export.handeye_mode,
        robot_deltas: input.rig_export.robot_deltas.as_deref(),
        per_cam_stats: input.joint_initial_stats,
    });
    let stage4 = build_stage(StageBuildInput {
        id: "stage4",
        label: "Joint rig + hand-eye + laser BA",
        mean_reproj_error_px: input.joint_est.mean_reproj_error_px,
        final_cost: Some(input.joint_est.report.final_cost),
        detected: input.detected,
        params: &input.joint_est.params,
        mode: input.rig_export.handeye_mode,
        robot_deltas: input.joint_est.robot_deltas.as_deref(),
        per_cam_stats: &input.joint_est.per_cam_stats,
    });

    let manifest = ViewerManifest {
        schema_version: 1,
        generator: "vision-calibration-examples-private puzzle_130x130_rig".to_string(),
        dataset: DatasetInfo {
            name: "130x130_puzzle".to_string(),
            source_dir: input.data_dir.display().to_string(),
            num_cameras: NUM_CAMERAS,
            board_rows: crate::BOARD_ROWS,
            board_cols: crate::BOARD_COLS,
            cell_size_mm: crate::CELL_SIZE_MM,
            full_image_size: [input.tile_w * NUM_CAMERAS as u32, input.tile_h],
            tile_size: [input.tile_w, input.tile_h],
        },
        frame_conventions: FrameConventions {
            cam_se3_rig: "T_C_R".to_string(),
            cam_to_rig: "T_R_C".to_string(),
            eye_to_hand_chain: "T_C_T = T_C_R * T_R_B * T_B_G * T_G_T".to_string(),
            robot_delta: "T_B_G_corr = exp(delta) * T_B_G".to_string(),
        },
        poses: input
            .poses
            .iter()
            .enumerate()
            .map(|(idx, pose)| PoseRecord {
                index: idx,
                snap_type: pose.snap_type.clone(),
                target_image: format!("images/{}", pose.target_image),
                laser_image: pose
                    .has_laser()
                    .then(|| format!("images/{}", pose.laser_image)),
                base_se3_gripper: iso_record(&pose.base_se3_gripper()),
            })
            .collect(),
        cameras: input
            .joint_est
            .params
            .cameras
            .iter()
            .zip(input.joint_est.params.sensors.iter())
            .enumerate()
            .map(|(idx, (camera, sensor))| CameraRecord {
                index: idx,
                intrinsics: [
                    camera.k.fx,
                    camera.k.fy,
                    camera.k.cx,
                    camera.k.cy,
                    camera.k.skew,
                ],
                distortion: [
                    camera.dist.k1,
                    camera.dist.k2,
                    camera.dist.k3,
                    camera.dist.p1,
                    camera.dist.p2,
                ],
                scheimpflug: [sensor.tilt_x, sensor.tilt_y],
                cam_se3_rig: iso_record(&input.joint_est.params.cam_to_rig[idx].inverse()),
                cam_to_rig: iso_record(&input.joint_est.params.cam_to_rig[idx]),
                laser_plane_camera: plane_record(&input.joint_est.params.planes_cam[idx]),
                laser_plane_rig: plane_record(&input.joint_est.planes_rig[idx]),
            })
            .collect(),
        handeye_mode: format!("{:?}", input.rig_export.handeye_mode),
        handeye: iso_record(&input.joint_est.params.handeye),
        target_ref: iso_record(&input.joint_est.params.target_ref),
        stages: vec![stage2, stage3, stage4],
    };

    let manifest_path = input.out_dir.join("viewer_manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&manifest_path, json)
        .with_context(|| format!("write viewer manifest {}", manifest_path.display()))?;
    Ok(())
}

fn copy_images(out_dir: &Path, data_dir: &Path, poses: &[PoseEntry]) -> Result<()> {
    let image_dir = out_dir.join("images");
    fs::create_dir_all(&image_dir)
        .with_context(|| format!("create viewer image dir {}", image_dir.display()))?;
    let mut copied = BTreeSet::new();
    for pose in poses {
        for image in [&pose.target_image, &pose.laser_image] {
            if copied.insert(image.clone()) {
                fs::copy(data_dir.join(image), image_dir.join(image))
                    .with_context(|| format!("copy viewer image {image}"))?;
            }
        }
    }
    Ok(())
}

fn build_stage(input: StageBuildInput<'_>) -> ViewerStage {
    let mut target_features = Vec::new();
    let mut laser_features = Vec::new();
    let camera_models = camera_models(input.params);

    for (pose_idx, view) in input.detected.joint_views.iter().enumerate() {
        let robot_pose = if let Some(deltas) = input.robot_deltas {
            corrected_robot_pose(view.meta.base_se3_gripper, deltas[pose_idx])
        } else {
            view.meta.base_se3_gripper
        };

        for (cam_idx, camera_model) in camera_models
            .iter()
            .enumerate()
            .take(view.obs.cameras.len())
        {
            let cam_se3_target =
                cam_se3_target_from_robot(input.params, input.mode, robot_pose, cam_idx);
            if let Some(obs) = &view.obs.cameras[cam_idx] {
                for (feature_idx, (pt3, observed)) in
                    obs.points_3d.iter().zip(obs.points_2d.iter()).enumerate()
                {
                    let p_camera = cam_se3_target.transform_point(pt3);
                    let projected = camera_model.project_point(&p_camera);
                    let (projected_px, error_px) = if let Some(projected) = projected {
                        let err = (projected - *observed).norm();
                        (Some([projected.x, projected.y]), Some(err))
                    } else {
                        (None, None)
                    };
                    target_features.push(TargetFeatureRecord {
                        pose: pose_idx,
                        camera: cam_idx,
                        feature: feature_idx,
                        target_xyz_m: [pt3.x, pt3.y, pt3.z],
                        observed_px: [observed.x, observed.y],
                        projected_px,
                        error_px,
                    });
                }
            }

            if let Some(pixels) = &view.obs.laser_pixels[cam_idx] {
                let laser_line_px = laser_line_endpoints_px(
                    camera_model,
                    &cam_se3_target,
                    &input.params.planes_cam[cam_idx],
                );
                for (feature_idx, px) in pixels.iter().enumerate() {
                    let residual_m = laser_point_to_plane_residual_m(
                        camera_model,
                        &cam_se3_target,
                        &input.params.planes_cam[cam_idx],
                        px,
                    );
                    let residual_px =
                        laser_line_px.map(|line| point_line_distance([px.x, px.y], line));
                    laser_features.push(LaserFeatureRecord {
                        pose: pose_idx,
                        camera: cam_idx,
                        feature: feature_idx,
                        observed_px: [px.x, px.y],
                        residual_m,
                        residual_px,
                        projected_line_px: laser_line_px,
                    });
                }
            }
        }
    }

    ViewerStage {
        id: input.id.to_string(),
        label: input.label.to_string(),
        mean_reproj_error_px: input.mean_reproj_error_px,
        final_cost: input.final_cost,
        robot_deltas: input
            .robot_deltas
            .map(<[[Real; 6]]>::to_vec)
            .unwrap_or_default(),
        per_camera_stats: input.per_cam_stats.to_vec(),
        geometry: stage_geometry(input.params),
        target_features,
        laser_features,
    }
}

fn stage_geometry(params: &RigHandeyeLaserlineParams) -> StageGeometry {
    StageGeometry {
        handeye: iso_record(&params.handeye),
        target_ref: iso_record(&params.target_ref),
        cameras: params
            .cameras
            .iter()
            .zip(params.sensors.iter())
            .enumerate()
            .map(|(idx, (camera, sensor))| {
                let plane_rig = params.planes_cam[idx].transform_by(&params.cam_to_rig[idx]);
                StageCameraRecord {
                    index: idx,
                    intrinsics: [
                        camera.k.fx,
                        camera.k.fy,
                        camera.k.cx,
                        camera.k.cy,
                        camera.k.skew,
                    ],
                    distortion: [
                        camera.dist.k1,
                        camera.dist.k2,
                        camera.dist.k3,
                        camera.dist.p1,
                        camera.dist.p2,
                    ],
                    scheimpflug: [sensor.tilt_x, sensor.tilt_y],
                    cam_se3_rig: iso_record(&params.cam_to_rig[idx].inverse()),
                    cam_to_rig: iso_record(&params.cam_to_rig[idx]),
                    laser_plane_camera: plane_record(&params.planes_cam[idx]),
                    laser_plane_rig: plane_record(&plane_rig),
                }
            })
            .collect(),
    }
}

fn camera_models(params: &RigHandeyeLaserlineParams) -> Vec<ViewerCameraModel> {
    params
        .cameras
        .iter()
        .zip(params.sensors.iter())
        .map(|(camera, sensor)| Camera::new(Pinhole, camera.dist, sensor.compile(), camera.k))
        .collect()
}

fn cam_se3_target_from_robot(
    params: &RigHandeyeLaserlineParams,
    mode: HandEyeMode,
    robot_pose: Iso3,
    cam_idx: usize,
) -> Iso3 {
    let cam_se3_rig = params.cam_to_rig[cam_idx].inverse();
    match mode {
        HandEyeMode::EyeToHand => cam_se3_rig * params.handeye * robot_pose * params.target_ref,
        HandEyeMode::EyeInHand => {
            cam_se3_rig * params.handeye.inverse() * robot_pose.inverse() * params.target_ref
        }
    }
}

fn corrected_robot_pose(robot_pose: Iso3, delta: [Real; 6]) -> Iso3 {
    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);
    let angle = rot_vec.norm();
    let delta_rot = if angle > 1e-12 {
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(rot_vec), angle)
    } else {
        UnitQuaternion::identity()
    };
    let delta_iso = Iso3::from_parts(Translation3::from(trans_vec), delta_rot);
    delta_iso * robot_pose
}

fn laser_point_to_plane_residual_m(
    camera: &impl CameraProject,
    cam_se3_target: &Iso3,
    plane: &LaserPlane,
    px: &Pt2,
) -> Option<f64> {
    let ray = camera.backproject(px);
    let ray_dir_camera = ray.normalize();
    let target_se3_cam = cam_se3_target.inverse();
    let ray_origin_target = target_se3_cam.transform_point(&Point3::origin());
    let ray_dir_target = target_se3_cam.rotation.transform_vector(&ray_dir_camera);
    if ray_dir_target.z.abs() < 1.0e-12 {
        return None;
    }
    let t = -ray_origin_target.z / ray_dir_target.z;
    if t < 0.0 {
        return None;
    }
    let point_target = ray_origin_target + ray_dir_target * t;
    let point_camera = cam_se3_target.transform_point(&point_target);
    Some(plane.point_distance(&point_camera))
}

fn laser_line_endpoints_px(
    camera: &impl CameraProject,
    cam_se3_target: &Iso3,
    plane: &LaserPlane,
) -> Option<[[f64; 2]; 2]> {
    let target_normal_c = cam_se3_target.rotation * Vector3::z_axis().into_inner();
    let target_d_c = -target_normal_c.dot(&cam_se3_target.translation.vector);
    let laser_normal = plane.normal.into_inner();
    let direction = laser_normal.cross(&target_normal_c);
    if direction.norm() < 1.0e-12 {
        return None;
    }
    let direction = direction.normalize();
    let p0 = line_origin(
        &laser_normal,
        plane.distance,
        &target_normal_c,
        target_d_c,
        &direction,
    )?;
    let mut projected = Vec::new();
    for s in [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12] {
        let p = Point3::from(p0 + direction * s);
        if let Some(px) = camera.project(&p) {
            projected.push([px.x, px.y]);
        }
    }
    if projected.len() < 2 {
        return None;
    }
    Some([projected[0], *projected.last().expect("len checked")])
}

fn line_origin(
    n1: &Vector3<f64>,
    d1: f64,
    n2: &Vector3<f64>,
    d2: f64,
    v: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let abs_v = [v.x.abs(), v.y.abs(), v.z.abs()];
    let axis = abs_v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)?;
    match axis {
        2 => solve_2x2(n1.x, n1.y, -d1, n2.x, n2.y, -d2).map(|(x, y)| Vector3::new(x, y, 0.0)),
        1 => solve_2x2(n1.x, n1.z, -d1, n2.x, n2.z, -d2).map(|(x, z)| Vector3::new(x, 0.0, z)),
        _ => solve_2x2(n1.y, n1.z, -d1, n2.y, n2.z, -d2).map(|(y, z)| Vector3::new(0.0, y, z)),
    }
}

fn solve_2x2(a00: f64, a01: f64, b0: f64, a10: f64, a11: f64, b1: f64) -> Option<(f64, f64)> {
    let det = a00 * a11 - a01 * a10;
    if det.abs() < 1.0e-12 {
        return None;
    }
    Some(((b0 * a11 - a01 * b1) / det, (a00 * b1 - b0 * a10) / det))
}

fn point_line_distance(point: [f64; 2], line: [[f64; 2]; 2]) -> f64 {
    let ax = line[0][0];
    let ay = line[0][1];
    let bx = line[1][0];
    let by = line[1][1];
    let vx = bx - ax;
    let vy = by - ay;
    let wx = point[0] - ax;
    let wy = point[1] - ay;
    let denom = (vx * vx + vy * vy).sqrt();
    if denom <= 1.0e-12 {
        return 0.0;
    }
    (vx * wy - vy * wx).abs() / denom
}

trait CameraProject {
    fn project(&self, p: &Point3<f64>) -> Option<Point2<f64>>;
    fn backproject(&self, px: &Pt2) -> Vector3<f64>;
}

impl CameraProject for ViewerCameraModel {
    fn project(&self, p: &Point3<f64>) -> Option<Point2<f64>> {
        self.project_point(p)
    }

    fn backproject(&self, px: &Pt2) -> Vector3<f64> {
        self.backproject_pixel(px).point
    }
}

fn iso_record(iso: &Iso3) -> TransformRecord {
    let q = iso.rotation.into_inner();
    let t = iso.translation.vector;
    let m = iso.to_homogeneous();
    TransformRecord {
        translation_m: [t.x, t.y, t.z],
        quaternion_xyzw: [q.i, q.j, q.k, q.w],
        matrix4_row_major: [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)], m[(2, 3)]],
            [m[(3, 0)], m[(3, 1)], m[(3, 2)], m[(3, 3)]],
        ],
    }
}

fn plane_record(plane: &LaserPlane) -> PlaneRecord {
    let n = plane.normal.into_inner();
    PlaneRecord {
        normal: [n.x, n.y, n.z],
        distance_m: plane.distance,
    }
}

#[derive(Debug, Clone, Serialize)]
struct ViewerManifest {
    schema_version: u32,
    generator: String,
    dataset: DatasetInfo,
    frame_conventions: FrameConventions,
    poses: Vec<PoseRecord>,
    cameras: Vec<CameraRecord>,
    handeye_mode: String,
    handeye: TransformRecord,
    target_ref: TransformRecord,
    stages: Vec<ViewerStage>,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetInfo {
    name: String,
    source_dir: String,
    num_cameras: usize,
    board_rows: u32,
    board_cols: u32,
    cell_size_mm: f64,
    full_image_size: [u32; 2],
    tile_size: [u32; 2],
}

#[derive(Debug, Clone, Serialize)]
struct FrameConventions {
    cam_se3_rig: String,
    cam_to_rig: String,
    eye_to_hand_chain: String,
    robot_delta: String,
}

#[derive(Debug, Clone, Serialize)]
struct PoseRecord {
    index: usize,
    snap_type: String,
    target_image: String,
    laser_image: Option<String>,
    base_se3_gripper: TransformRecord,
}

#[derive(Debug, Clone, Serialize)]
struct CameraRecord {
    index: usize,
    intrinsics: [f64; 5],
    distortion: [f64; 5],
    scheimpflug: [f64; 2],
    cam_se3_rig: TransformRecord,
    cam_to_rig: TransformRecord,
    laser_plane_camera: PlaneRecord,
    laser_plane_rig: PlaneRecord,
}

#[derive(Debug, Clone, Serialize)]
struct TransformRecord {
    translation_m: [f64; 3],
    quaternion_xyzw: [f64; 4],
    matrix4_row_major: [[f64; 4]; 4],
}

#[derive(Debug, Clone, Serialize)]
struct PlaneRecord {
    normal: [f64; 3],
    distance_m: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerStage {
    id: String,
    label: String,
    mean_reproj_error_px: f64,
    final_cost: Option<f64>,
    robot_deltas: Vec<[Real; 6]>,
    per_camera_stats: Vec<RigHandeyeLaserlinePerCamStats>,
    geometry: StageGeometry,
    target_features: Vec<TargetFeatureRecord>,
    laser_features: Vec<LaserFeatureRecord>,
}

#[derive(Debug, Clone, Serialize)]
struct StageGeometry {
    handeye: TransformRecord,
    target_ref: TransformRecord,
    cameras: Vec<StageCameraRecord>,
}

#[derive(Debug, Clone, Serialize)]
struct StageCameraRecord {
    index: usize,
    intrinsics: [f64; 5],
    distortion: [f64; 5],
    scheimpflug: [f64; 2],
    cam_se3_rig: TransformRecord,
    cam_to_rig: TransformRecord,
    laser_plane_camera: PlaneRecord,
    laser_plane_rig: PlaneRecord,
}

#[derive(Debug, Clone, Serialize)]
struct TargetFeatureRecord {
    pose: usize,
    camera: usize,
    feature: usize,
    target_xyz_m: [f64; 3],
    observed_px: [f64; 2],
    projected_px: Option<[f64; 2]>,
    error_px: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LaserFeatureRecord {
    pose: usize,
    camera: usize,
    feature: usize,
    observed_px: [f64; 2],
    residual_m: Option<f64>,
    residual_px: Option<f64>,
    projected_line_px: Option<[[f64; 2]; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_manifest_serializes() {
        let manifest = ViewerManifest {
            schema_version: 1,
            generator: "test".to_string(),
            dataset: DatasetInfo {
                name: "synthetic".to_string(),
                source_dir: ".".to_string(),
                num_cameras: 0,
                board_rows: 0,
                board_cols: 0,
                cell_size_mm: 1.0,
                full_image_size: [0, 0],
                tile_size: [0, 0],
            },
            frame_conventions: FrameConventions {
                cam_se3_rig: "T_C_R".to_string(),
                cam_to_rig: "T_R_C".to_string(),
                eye_to_hand_chain: "T_C_T = T_C_R * T_R_B * T_B_G * T_G_T".to_string(),
                robot_delta: "T_B_G_corr = exp(delta) * T_B_G".to_string(),
            },
            poses: Vec::new(),
            cameras: Vec::new(),
            handeye_mode: "EyeToHand".to_string(),
            handeye: iso_record(&Iso3::identity()),
            target_ref: iso_record(&Iso3::identity()),
            stages: Vec::new(),
        };

        let json = serde_json::to_string(&manifest).expect("serialize manifest");
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("T_C_R"));
    }
}
