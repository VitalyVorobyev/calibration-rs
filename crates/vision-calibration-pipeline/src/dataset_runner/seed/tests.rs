//! Seeded runs on synthetic scenes of every topology: unseeded, a run is
//! the workflow's public step list; a seed from a calibration reaches the
//! unseeded optimum; a wrong seed converges or fails typed; and a seed that
//! does not fit the dataset fails before any image is read.

use std::path::Path;

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::synthetic::{planar, poses};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    DistortionParams, FxFyCxCy, Iso3, LaserPlane, NoMeta, ParameterCovariance, PlanarDataset,
    ProjectionParams, Pt2, Pt3, RigDataset, RigView, RigViewObs, RunControl, ScheimpflugParams,
    SensorKind, SensorParams, Uncertainty, View,
};
use vision_calibration_dataset::{DatasetSpec, ProblemKind};
use vision_calibration_optim::{HandeyeMode, SolverBackend, camera_parameters};

use super::OrderedSeed;
use crate::Error;
use crate::analysis::{QualityThresholds, Verdict, assess};
use crate::common::{CalibratedCameras, RigLaserlineView, RobotPoseMeta};
use crate::dataset_runner::calibrate::{run_joint_seeded, solve};
use crate::dataset_runner::{
    CalibrateDatasetError, CalibrationConfig, CalibrationExport, Detection,
    calibrate_dataset_with_seed,
};
use crate::device_seed::{DeviceSeed, DeviceSeedError, SeedHandeye, SeedLens};
use crate::laserline_device::{
    LaserlineDataset, LaserlineDeviceConfig, LaserlineDeviceProblem, LaserlineMeta,
};
use crate::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};
use crate::rig_extrinsics::{RigExtrinsicsConfig, RigExtrinsicsProblem};
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem};
use crate::rig_handeye_laserline::{
    JointManualInit, RigHandeyeLaserlineConfig, RigHandeyeLaserlineInput,
    RigHandeyeLaserlineProblem, RigHandeyeLaserlineView,
};
use crate::rig_laserline_device::{
    RigLaserlineDataset, RigLaserlineDeviceConfig, RigLaserlineDeviceInput,
    RigLaserlineDeviceProblem, RigUpstreamCalibration,
};
use crate::session::{CalibrationSession, ProblemType};
use crate::single_cam_handeye::{
    SingleCamHandeyeConfig, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
};

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic scenes
// ─────────────────────────────────────────────────────────────────────────────

const TOPOLOGIES: [ProblemKind; 7] = [
    ProblemKind::PlanarIntrinsics,
    ProblemKind::SingleCamHandeye,
    ProblemKind::LaserlineDevice,
    ProblemKind::RigExtrinsics,
    ProblemKind::RigHandeye,
    ProblemKind::RigLaserlineDevice,
    ProblemKind::RigHandeyeLaserline,
];

/// Per-axis standard deviation of the corner noise, pixels.
const NOISE_PX: f64 = 0.1;
const VIEWS: usize = 10;
const NX: usize = 9;
const NY: usize = 7;
const SPACING: f64 = 0.06;
/// Anchor depth of the laser scenes, metres.
const LASER_Z0: f64 = 0.5;

/// One topology's synthetic calibration: its input, config, camera ids and
/// image sizes, and the cameras that produced it.
struct Scene {
    config: CalibrationConfig,
    input: Input,
    ids: Vec<String>,
    sizes: Vec<[u32; 2]>,
    truth: Vec<CameraParams>,
}

#[allow(clippy::large_enum_variant)] // A test scene, built once per run.
enum Input {
    Planar(PlanarDataset),
    SingleCamHandeye(SingleCamHandeyeInput),
    Laserline(LaserlineDataset),
    Rig(RigDataset<NoMeta>),
    RigHandeye(RigDataset<RobotPoseMeta>),
    RigLaserline(RigLaserlineDeviceInput),
    RigHandeyeLaserline(RigHandeyeLaserlineInput),
}

fn pinhole_model(sensor: SensorKind) -> CameraModelSpec {
    CameraModelSpec::pinhole(DistortionKind::BrownConrady5, sensor)
}

/// A pinhole camera with Brown-Conrady `k1`, `k2` and a straight sensor.
fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                k2,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        },
        SensorParams::Identity,
        FxFyCxCy { fx, fy, cx, cy },
    )
}

fn transpose(per_cam: Vec<Vec<CorrespondenceView>>) -> Vec<Vec<CorrespondenceView>> {
    let views = per_cam.first().map_or(0, Vec::len);
    let mut out: Vec<Vec<CorrespondenceView>> = (0..views).map(|_| Vec::new()).collect();
    for cam_views in per_cam {
        for (v, obs) in cam_views.into_iter().enumerate() {
            out[v].push(obs);
        }
    }
    out
}

/// A laser plane in a camera frame crossing the optical axis at `z0`.
fn plane_at(normal: Vector3<f64>, z0: f64) -> LaserPlane {
    let n = normal.normalize();
    LaserPlane::new(n, -n.z * z0).unwrap()
}

/// The hand-eye ground truth: `gripper_se3_cam` (or `_rig`) and the target
/// in the robot base.
fn handeye_truth() -> (Iso3, Iso3) {
    (
        poses::make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08)),
        poses::make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, 1.0)),
    )
}

/// `rig_se3_target = X⁻¹ · T_B_G⁻¹ · Y` for each robot station.
fn rig_poses_from_robot(stations: &[Iso3], x: &Iso3, y: &Iso3) -> Vec<Iso3> {
    stations
        .iter()
        .map(|t_bg| (t_bg * x).inverse() * y)
        .collect()
}

struct Builder {
    seed: u64,
    board: Vec<Pt3>,
    extent: BoardExtent,
}

impl Builder {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            board: planar::grid_points(NX, NY, SPACING),
            extent: BoardExtent::from_grid(NX, NY, SPACING),
        }
    }

    fn noise(&self, stream: u64) -> UniformPixelNoise {
        UniformPixelNoise {
            seed: self.seed.wrapping_add(stream),
            max_abs_px: NOISE_PX * 3.0_f64.sqrt(),
        }
    }

    fn rng_seed(&self, stream: u64) -> u64 {
        self.seed.wrapping_add(0x9E37_79B9).wrapping_mul(stream | 1)
    }

    fn board_poses(&self, max_tilt: f64, z_start: f64, z_span: f64) -> Vec<Iso3> {
        let c = self.extent.center();
        poses::tilted_board_poses(
            &poses::seeded_tilts(VIEWS, max_tilt, self.rng_seed(1)),
            (c[0], c[1]),
            z_start,
            z_span / VIEWS as f64,
        )
    }

    fn observe(
        &self,
        camera: &CameraParams,
        cam_se3_target: &[Iso3],
        stream: u64,
    ) -> Vec<CorrespondenceView> {
        planar::project_views_noisy(
            &camera.build().unwrap(),
            &self.board,
            cam_se3_target,
            &self.noise(stream),
        )
        .unwrap()
    }

    /// `[view][camera]`.
    fn observe_rig(
        &self,
        cameras: &[CameraParams],
        cam_se3_rig: &[Iso3],
        rig_se3_target: &[Iso3],
    ) -> Vec<Vec<CorrespondenceView>> {
        let per_cam = cameras
            .iter()
            .zip(cam_se3_rig)
            .enumerate()
            .map(|(c, (camera, t_c_r))| {
                let poses: Vec<Iso3> = rig_se3_target.iter().map(|t| t_c_r * t).collect();
                self.observe(camera, &poses, c as u64)
            })
            .collect();
        transpose(per_cam)
    }

    /// Board poses whose laser stripe crosses the board in every camera.
    fn laser_board_poses(
        &self,
        cameras: &[CameraParams],
        cam_se3_rig: &[Iso3],
        planes: &[LaserPlane],
    ) -> Vec<Iso3> {
        let c = self.extent.center();
        let specs = poses::seeded_board_pose_specs(4 * VIEWS, 0.35, self.rng_seed(1));
        let silent = UniformPixelNoise::default();
        let synth: Vec<_> = cameras.iter().map(|c| c.build().unwrap()).collect();
        let kept: Vec<Iso3> = poses::centered_board_poses(&specs, (c[0], c[1]), LASER_Z0)
            .into_iter()
            .filter(|rig_se3_target| {
                synth
                    .iter()
                    .zip(cam_se3_rig)
                    .zip(planes)
                    .all(|((camera, t_c_r), plane)| {
                        laser_stripe_pixels(
                            camera,
                            &(t_c_r * rig_se3_target),
                            plane.normal().as_ref(),
                            plane.distance(),
                            &self.extent,
                            0,
                            &silent,
                        )
                        .len()
                            >= 20
                    })
            })
            .take(VIEWS)
            .collect();
        assert_eq!(kept.len(), VIEWS);
        kept
    }

    /// `[view]` stripe pixels of one camera.
    fn stripes(
        &self,
        camera: &CameraParams,
        cam_se3_target: &[Iso3],
        plane: &LaserPlane,
        stream: u64,
    ) -> Vec<Vec<Pt2>> {
        let synth = camera.build().unwrap();
        let noise = self.noise(1000 + stream);
        cam_se3_target
            .iter()
            .enumerate()
            .map(|(view, pose)| {
                laser_stripe_pixels(
                    &synth,
                    pose,
                    plane.normal().as_ref(),
                    plane.distance(),
                    &self.extent,
                    view,
                    &noise,
                )
            })
            .collect()
    }

    fn rig_laser_views(
        &self,
        cameras: &[CameraParams],
        layout: &[Iso3],
        planes: &[LaserPlane],
        rig_poses: &[Iso3],
    ) -> Vec<RigLaserlineView> {
        let obs = self.observe_rig(cameras, layout, rig_poses);
        let stripes: Vec<Vec<Vec<Pt2>>> = cameras
            .iter()
            .zip(layout)
            .zip(planes)
            .enumerate()
            .map(|(c, ((camera, t_c_r), plane))| {
                let poses: Vec<Iso3> = rig_poses.iter().map(|t| t_c_r * t).collect();
                self.stripes(camera, &poses, plane, c as u64)
            })
            .collect();
        obs.into_iter()
            .enumerate()
            .map(|(v, cams)| {
                RigLaserlineView::new(
                    cams.into_iter().map(Some).collect(),
                    stripes.iter().map(|s| Some(s[v].clone())).collect(),
                )
            })
            .collect()
    }

    /// Per-camera planes, each offset to cut the board in the most views of
    /// a robot-moved rig.
    fn planes_through_target(
        &self,
        cameras: &[CameraParams],
        layout: &[Iso3],
        rig_poses: &[Iso3],
    ) -> Vec<LaserPlane> {
        let c = self.extent.center();
        let board_center = Pt3::new(c[0], c[1], 0.0);
        let normals = [Vector3::new(0.25, 0.0, 1.0), Vector3::new(0.0, 0.25, 1.0)];
        layout
            .iter()
            .zip(normals)
            .zip(cameras)
            .map(|((t_c_r, normal), camera)| {
                let poses: Vec<Iso3> = rig_poses.iter().map(|t| t_c_r * t).collect();
                let sum: Vector3<f64> = poses
                    .iter()
                    .map(|p| p.transform_point(&board_center).coords)
                    .sum();
                let n = normal.normalize();
                let d_mean = -n.dot(&(sum / poses.len() as f64));
                let synth = camera.build().unwrap();
                let silent = UniformPixelNoise::default();
                let coverage = |d: f64| {
                    poses
                        .iter()
                        .enumerate()
                        .filter(|(v, p)| {
                            laser_stripe_pixels(&synth, p, &n, d, &self.extent, *v, &silent).len()
                                >= 8
                        })
                        .count()
                };
                let best = (-30..=30)
                    .map(|k| d_mean + 0.005 * f64::from(k))
                    .max_by_key(|&d| (coverage(d), -((d - d_mean).abs() * 1e6) as i64))
                    .unwrap();
                LaserPlane::new(n, best).unwrap()
            })
            .collect()
    }
}

fn rig_cameras() -> Vec<CameraParams> {
    vec![
        pinhole(850.0, 845.0, 640.0, 480.0, -0.05, 0.0),
        pinhole(820.0, 815.0, 650.0, 470.0, -0.04, 0.0),
    ]
}

fn ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("cam{i}")).collect()
}

/// The scene of `kind`, solved on `backend`.
fn scene(kind: ProblemKind, backend: SolverBackend) -> Scene {
    let b = Builder::new(0x5EED_0000 + kind as u64);
    let model = pinhole_model(SensorKind::Identity);
    match kind {
        ProblemKind::PlanarIntrinsics => {
            let truth = pinhole(1000.0, 990.0, 640.0, 480.0, -0.12, 0.04);
            let obs = b.observe(&truth, &b.board_poses(0.5, 0.6, 0.4), 0);
            let input =
                PlanarDataset::new(obs.into_iter().map(View::without_meta).collect()).unwrap();
            let mut config = PlanarIntrinsicsConfig::default();
            config.camera.model = model;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::PlanarIntrinsics(config),
                input: Input::Planar(input),
                ids: ids(1),
                sizes: vec![[1280, 960]],
                truth: vec![truth],
            }
        }
        ProblemKind::SingleCamHandeye => {
            let truth = pinhole(800.0, 780.0, 512.0, 384.0, -0.05, 0.0);
            let (x, y) = handeye_truth();
            let stations = poses::robot_stations(VIEWS, b.rng_seed(3));
            let obs = b.observe(&truth, &rig_poses_from_robot(&stations, &x, &y), 0);
            let views: Vec<SingleCamHandeyeView> = obs
                .into_iter()
                .zip(&stations)
                .map(|(o, t_bg)| View::new(o, RobotPoseMeta::new(*t_bg)))
                .collect();
            let mut config = SingleCamHandeyeConfig::default();
            config.camera.model = model;
            config.handeye_init.handeye_mode = HandeyeMode::EyeInHand;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::SingleCamHandeye(config),
                input: Input::SingleCamHandeye(SingleCamHandeyeInput::new(views).unwrap()),
                ids: ids(1),
                sizes: vec![[1024, 768]],
                truth: vec![truth],
            }
        }
        ProblemKind::LaserlineDevice => {
            let truth = pinhole(900.0, 900.0, 640.0, 360.0, -0.05, 0.0);
            let plane = plane_at(Vector3::new(0.30, 0.0, 1.0), LASER_Z0);
            let cam_poses = b.laser_board_poses(
                std::slice::from_ref(&truth),
                &[Iso3::identity()],
                std::slice::from_ref(&plane),
            );
            let obs = b.observe(&truth, &cam_poses, 0);
            let stripes = b.stripes(&truth, &cam_poses, &plane, 0);
            let views = obs
                .into_iter()
                .zip(stripes)
                .map(|(o, pixels)| View::new(o, LaserlineMeta::new(pixels)))
                .collect();
            let mut config = LaserlineDeviceConfig::default();
            config.camera.model = model;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::LaserlineDevice(config),
                input: Input::Laserline(LaserlineDataset::new(views).unwrap()),
                ids: ids(1),
                sizes: vec![[1280, 720]],
                truth: vec![truth],
            }
        }
        ProblemKind::RigExtrinsics => {
            let cameras = rig_cameras();
            let layout = poses::rig_layout(2, 0.06, 0.05);
            let obs = b.observe_rig(&cameras, &layout, &b.board_poses(0.45, 0.65, 0.3));
            let views = obs
                .into_iter()
                .map(|cams| {
                    RigView::new(
                        RigViewObs::new(cams.into_iter().map(Some).collect()),
                        NoMeta,
                    )
                })
                .collect();
            let mut config = RigExtrinsicsConfig::default();
            config.camera.model = model;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::RigExtrinsics(config),
                input: Input::Rig(RigDataset::new(views, 2).unwrap()),
                ids: ids(2),
                sizes: vec![[1280, 960]; 2],
                truth: cameras,
            }
        }
        ProblemKind::RigHandeye => {
            let cameras = rig_cameras();
            let layout = poses::rig_layout(2, 0.12, 0.0);
            let (x, y) = handeye_truth();
            let stations = poses::robot_stations(VIEWS, b.rng_seed(3));
            let rig_poses = rig_poses_from_robot(&stations, &x, &y);
            let obs = b.observe_rig(&cameras, &layout, &rig_poses);
            let views = obs
                .into_iter()
                .zip(&stations)
                .map(|(cams, t_bg)| {
                    RigView::new(
                        RigViewObs::new(cams.into_iter().map(Some).collect()),
                        RobotPoseMeta::new(*t_bg),
                    )
                })
                .collect();
            let mut config = RigHandeyeConfig::default();
            config.camera.model = model;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::RigHandeye(config),
                input: Input::RigHandeye(RigDataset::new(views, 2).unwrap()),
                ids: ids(2),
                sizes: vec![[1280, 960]; 2],
                truth: cameras,
            }
        }
        ProblemKind::RigLaserlineDevice => {
            let cameras = rig_cameras();
            let layout = poses::rig_layout(2, 0.06, 0.0);
            let planes = vec![
                plane_at(Vector3::new(0.30, 0.0, 1.0), LASER_Z0),
                plane_at(Vector3::new(0.0, 0.45, 1.0), LASER_Z0),
            ];
            let rig_poses = b.laser_board_poses(&cameras, &layout, &planes);
            let views = b.rig_laser_views(&cameras, &layout, &planes, &rig_poses);
            let dataset = RigLaserlineDataset::new(views, 2).unwrap();
            let upstream = RigUpstreamCalibration::new(cameras.clone(), layout, rig_poses);
            let mut config = RigLaserlineDeviceConfig::default();
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::RigLaserlineDevice(config),
                input: Input::RigLaserline(RigLaserlineDeviceInput::new(dataset, upstream)),
                ids: ids(2),
                sizes: vec![[1280, 960]; 2],
                truth: cameras,
            }
        }
        ProblemKind::RigHandeyeLaserline => {
            let cameras = rig_cameras();
            let layout = poses::rig_layout(2, 0.12, 0.0);
            let (x, y) = handeye_truth();
            let stations = poses::robot_stations(VIEWS, b.rng_seed(3));
            let rig_poses = rig_poses_from_robot(&stations, &x, &y);
            let planes = b.planes_through_target(&cameras, &layout, &rig_poses);
            let views: Vec<RigHandeyeLaserlineView> = b
                .rig_laser_views(&cameras, &layout, &planes, &rig_poses)
                .into_iter()
                .zip(&stations)
                .map(|(obs, t_bg)| RigHandeyeLaserlineView::new(obs, RobotPoseMeta::new(*t_bg)))
                .collect();
            let mut config = RigHandeyeLaserlineConfig::default();
            config.handeye.camera.model = model;
            config.handeye.solver.backend = backend;
            config.laserline_init.solver.backend = backend;
            config.solver.backend = backend;
            Scene {
                config: CalibrationConfig::RigHandeyeLaserline(config),
                input: Input::RigHandeyeLaserline(RigHandeyeLaserlineInput::new(views, 2)),
                ids: ids(2),
                sizes: vec![[1280, 960]; 2],
                truth: cameras,
            }
        }
        other => unreachable!("{other:?} has no problem type"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Running and comparing
// ─────────────────────────────────────────────────────────────────────────────

/// The scene's calibration through the dataset path's seeded runs, from
/// `seed` when one is given.
fn run(
    scene: &Scene,
    seed: Option<&DeviceSeed>,
) -> Result<CalibrationExport, CalibrateDatasetError> {
    let ids: Vec<&str> = scene.ids.iter().map(String::as_str).collect();
    let ordered = seed
        .map(|seed| OrderedSeed::with_ids(seed, ids))
        .transpose()?;
    if let Some(ordered) = &ordered {
        ordered.check_image_sizes(&scene.sizes)?;
    }
    let control = RunControl::default();
    macro_rules! go {
        ($config:expr, $input:expr, $resolve:path, $run:path, $variant:ident) => {{
            let resolved = match &ordered {
                Some(ordered) => Some($resolve(ordered, $config)?),
                None => None,
            };
            let export = solve(
                $input.clone(),
                $config.clone(),
                |session| $run(session, resolved),
                &control,
            )?;
            Ok(CalibrationExport::$variant(export))
        }};
    }
    match (&scene.config, &scene.input) {
        (CalibrationConfig::PlanarIntrinsics(c), Input::Planar(i)) => go!(
            c,
            i,
            OrderedSeed::planar,
            crate::planar_intrinsics::run_seeded,
            PlanarIntrinsics
        ),
        (CalibrationConfig::SingleCamHandeye(c), Input::SingleCamHandeye(i)) => go!(
            c,
            i,
            OrderedSeed::single_cam_handeye,
            crate::single_cam_handeye::run_seeded,
            SingleCamHandeye
        ),
        (CalibrationConfig::LaserlineDevice(c), Input::Laserline(i)) => go!(
            c,
            i,
            OrderedSeed::laserline_device,
            crate::laserline_device::run_seeded,
            LaserlineDevice
        ),
        (CalibrationConfig::RigExtrinsics(c), Input::Rig(i)) => go!(
            c,
            i,
            OrderedSeed::rig_extrinsics,
            crate::rig_extrinsics::run_seeded,
            RigExtrinsics
        ),
        (CalibrationConfig::RigHandeye(c), Input::RigHandeye(i)) => go!(
            c,
            i,
            OrderedSeed::rig_handeye,
            crate::rig_handeye::run_seeded,
            RigHandeye
        ),
        (CalibrationConfig::RigLaserlineDevice(c), Input::RigLaserline(i)) => go!(
            c,
            i,
            OrderedSeed::rig_laserline_device,
            crate::rig_laserline_device::run_seeded,
            RigLaserlineDevice
        ),
        (CalibrationConfig::RigHandeyeLaserline(c), Input::RigHandeyeLaserline(i)) => go!(
            c,
            i,
            OrderedSeed::rig_handeye_laserline,
            run_joint_seeded,
            RigHandeyeLaserline
        ),
        _ => unreachable!("a scene's config and input are of one topology"),
    }
}

fn session<P: ProblemType>(input: &P::Input, config: &P::Config) -> CalibrationSession<P>
where
    P::Input: Clone,
    P::Config: Clone,
{
    let mut session = CalibrationSession::<P>::new();
    session.set_input(input.clone()).unwrap();
    session.set_config(config.clone()).unwrap();
    session
}

/// The scene's calibration by the workflow's public steps, in the order its
/// `run_calibration` documents: what an unseeded run computed before the
/// step lists took a seed.
fn public_steps(scene: &Scene) -> Result<CalibrationExport, Error> {
    use crate::{
        laserline_device as ld, planar_intrinsics as pi, rig_extrinsics as re,
        rig_laserline_device as rl, single_cam_handeye as sh,
    };
    Ok(match (&scene.config, &scene.input) {
        (CalibrationConfig::PlanarIntrinsics(c), Input::Planar(i)) => {
            let mut s = session::<PlanarIntrinsicsProblem>(i, c);
            pi::step_init(&mut s, None)?;
            pi::step_optimize(&mut s, None)?;
            CalibrationExport::PlanarIntrinsics(s.export()?)
        }
        (CalibrationConfig::SingleCamHandeye(c), Input::SingleCamHandeye(i)) => {
            let mut s = session::<SingleCamHandeyeProblem>(i, c);
            sh::step_intrinsics_init(&mut s, None)?;
            sh::step_intrinsics_optimize(&mut s, None)?;
            sh::step_handeye_init(&mut s, None)?;
            sh::step_handeye_optimize(&mut s, None)?;
            CalibrationExport::SingleCamHandeye(s.export()?)
        }
        (CalibrationConfig::LaserlineDevice(c), Input::Laserline(i)) => {
            let mut s = session::<LaserlineDeviceProblem>(i, c);
            ld::step_init(&mut s, None)?;
            ld::step_optimize(&mut s, None)?;
            CalibrationExport::LaserlineDevice(s.export()?)
        }
        (CalibrationConfig::RigExtrinsics(c), Input::Rig(i)) => {
            let mut s = session::<RigExtrinsicsProblem>(i, c);
            re::step_intrinsics_init_all(&mut s, None)?;
            re::step_intrinsics_optimize_all(&mut s, None)?;
            re::step_rig_init(&mut s, None)?;
            re::step_rig_optimize(&mut s, None)?;
            CalibrationExport::RigExtrinsics(s.export()?)
        }
        (CalibrationConfig::RigHandeye(c), Input::RigHandeye(i)) => {
            let mut s = session::<RigHandeyeProblem>(i, c);
            rig_handeye_steps(&mut s)?;
            CalibrationExport::RigHandeye(s.export()?)
        }
        (CalibrationConfig::RigLaserlineDevice(c), Input::RigLaserline(i)) => {
            let mut s = session::<RigLaserlineDeviceProblem>(i, c);
            rl::step_init(&mut s, None)?;
            rl::step_optimize(&mut s, None)?;
            CalibrationExport::RigLaserlineDevice(s.export()?)
        }
        (CalibrationConfig::RigHandeyeLaserline(c), Input::RigHandeyeLaserline(i)) => {
            // The warm start by the rig hand-eye's public steps, handed to
            // the joint run as its seed: the joint run's other stages are
            // the unseeded ones.
            let mut warm = session::<RigHandeyeProblem>(&i.handeye_dataset()?, &c.handeye);
            rig_handeye_steps(&mut warm)?;
            let mut manual = JointManualInit::new();
            manual.rig_handeye = Some(warm.export()?);
            let mut s = session::<RigHandeyeLaserlineProblem>(i, c);
            crate::rig_handeye_laserline::run_calibration_with_seed(&mut s, manual)?;
            CalibrationExport::RigHandeyeLaserline(s.export()?)
        }
        _ => unreachable!("a scene's config and input are of one topology"),
    })
}

fn rig_handeye_steps(s: &mut CalibrationSession<RigHandeyeProblem>) -> Result<(), Error> {
    use crate::rig_handeye as rh;
    rh::step_intrinsics_init_all(s, None)?;
    rh::step_intrinsics_optimize_all(s, None)?;
    rh::step_rig_init(s, None)?;
    rh::step_rig_optimize(s, None)?;
    rh::step_handeye_init(s, None)?;
    rh::step_handeye_optimize(s, None)?;
    Ok(())
}

/// How far one calibration is from another.
#[derive(Debug, Default, Clone)]
struct Diff {
    /// The largest `|Δ| / σ` over the camera parameters the reference states
    /// a σ for, and the parameter's label.
    max_z: f64,
    worst: String,
    /// The largest rotation (degrees) and translation (metres) between the
    /// two calibrations' mounts and hand-eye.
    pose_deg: f64,
    pose_m: f64,
    /// The largest angle (degrees) between their laser planes' normals and
    /// difference (metres) of their distances.
    plane_deg: f64,
    plane_m: f64,
    /// The largest of the same differences in units of the smallest σ the
    /// reference states for that pose's rotation or translation, or that
    /// plane's normal or distance coordinates: a conservative `|Δ| / σ`.
    pose_z: f64,
    plane_z: f64,
    /// `|SSE_b − SSE_a| / SSE_a` of the target reprojection residuals.
    sse_rel: f64,
}

impl Diff {
    /// Whether two calibrations are at one optimum: every camera
    /// parameter within 0.01 σ; the mounts, hand-eye and planes within
    /// 0.05 σ (each difference a magnitude over its smallest coordinate σ);
    /// and the target SSE within 1e-5 relative. The last two are the
    /// solver's precision: it stops once a step lowers the cost by less than
    /// 1e-5 relative, which leaves a stop anywhere within about
    /// `√(1e-5 · n)` σ of the minimum along its flattest direction for `n`
    /// residuals, so two starts agree to that precision and not beyond.
    fn same_optimum(&self) -> bool {
        self.max_z < 0.01 && self.pose_z < 0.05 && self.plane_z < 0.05 && self.sse_rel < 1e-5
    }
}

/// The smallest σ `cov` states for the coordinates `coordinates` of the
/// block named one of `names` (`<name>.<coordinate>`): a conservative scale
/// for a difference measured as one magnitude.
fn smallest_sigma(
    cov: Option<&ParameterCovariance>,
    names: &[String],
    coordinates: &[&str],
) -> Option<f64> {
    let cov = cov?;
    names
        .iter()
        .flat_map(|name| coordinates.iter().map(move |c| format!("{name}.{c}")))
        .filter_map(|label| cov.std_dev(&label))
        .filter(|sigma| *sigma > 0.0)
        .reduce(f64::min)
}

/// `delta / sigma`; a difference the covariance states no σ for (a rig
/// hand-eye's mounts, held in its last stage) over `fallback`, a scale far
/// below any σ of these scenes.
fn scaled(delta: f64, sigma: Option<f64>, fallback: f64) -> f64 {
    delta / sigma.unwrap_or(fallback)
}

/// The scale of an uncovered rotation (radians, 0.001°) and translation
/// (metres, 1 µm).
const UNCOVERED_RAD: f64 = 1.745e-5;
const UNCOVERED_M: f64 = 1e-6;

fn pose_diff(a: &Iso3, b: &Iso3) -> (f64, f64) {
    let e = poses::pose_error(a, b);
    (e.rot_deg, e.trans)
}

fn target_sse(export: &CalibrationExport) -> f64 {
    export
        .diagnostics()
        .per_feature_residuals
        .target
        .iter()
        .filter_map(|r| r.error_px)
        .map(|e| e * e)
        .sum()
}

fn diff(reference: &CalibrationExport, other: &CalibrationExport) -> Diff {
    let mut d = Diff::default();
    let cov = match reference.parameter_uncertainty() {
        Uncertainty::Available(cov) => Some(cov),
        Uncertainty::Unavailable { .. } => None,
    };
    let (a, b) = (reference.camera_params(), other.camera_params());
    assert_eq!(a.len(), b.len());
    for (i, (ca, cb)) in a.iter().zip(&b).enumerate() {
        let pa = camera_parameters(ca).unwrap();
        let pb = camera_parameters(cb).unwrap();
        for ((label, va), (_, vb)) in pa.iter().zip(&pb) {
            let sigma = cov.and_then(|cov| {
                cov.std_dev(&format!("cam{i}.{label}"))
                    .or_else(|| cov.std_dev(label))
            });
            if let Some(sigma) = sigma.filter(|s| *s > 0.0) {
                let z = (va - vb).abs() / sigma;
                if z > d.max_z {
                    d.max_z = z;
                    d.worst = format!("cam{i}.{label}");
                }
            }
        }
    }
    // Each pose with the covariance labels it may have: a mount, then the
    // hand-eye of either mode.
    let mut poses: Vec<(Iso3, Iso3, Vec<String>)> = reference
        .cam_se3_rig()
        .into_iter()
        .zip(other.cam_se3_rig())
        .enumerate()
        .map(|(i, (pa, pb))| (pa, pb, vec![format!("cam{i}.cam_se3_rig")]))
        .collect();
    if let (Some(ha), Some(hb)) = (reference.handeye(), other.handeye()) {
        let names = [
            "gripper_se3_rig",
            "gripper_se3_cam",
            "rig_se3_base",
            "cam_se3_base",
        ];
        poses.push((
            *ha.pose(),
            *hb.pose(),
            names.iter().map(|n| (*n).to_string()).collect(),
        ));
    }
    for (pa, pb, names) in &poses {
        let (deg, m) = pose_diff(pa, pb);
        d.pose_deg = d.pose_deg.max(deg);
        d.pose_m = d.pose_m.max(m);
        let rotation = smallest_sigma(cov, names, &["rx", "ry", "rz"]);
        let translation = smallest_sigma(cov, names, &["tx", "ty", "tz"]);
        let z = scaled(deg.to_radians(), rotation, UNCOVERED_RAD).max(scaled(
            m,
            translation,
            UNCOVERED_M,
        ));
        d.pose_z = d.pose_z.max(z);
    }
    for (i, (pa, pb)) in reference
        .laser_planes_cam()
        .iter()
        .zip(&other.laser_planes_cam())
        .enumerate()
    {
        let cos = pa.normal().dot(&pb.normal()).clamp(-1.0, 1.0);
        let (deg, m) = (
            cos.acos().to_degrees(),
            (pa.distance() - pb.distance()).abs(),
        );
        d.plane_deg = d.plane_deg.max(deg);
        d.plane_m = d.plane_m.max(m);
        let names = [format!("cam{i}.plane"), "plane".to_string()];
        let normal = smallest_sigma(cov, &names, &["nx", "ny", "nz"]);
        let distance = smallest_sigma(cov, &names, &["d"]);
        let z =
            scaled(deg.to_radians(), normal, UNCOVERED_RAD).max(scaled(m, distance, UNCOVERED_M));
        d.plane_z = d.plane_z.max(z);
    }
    let (sa, sb) = (target_sse(reference), target_sse(other));
    d.sse_rel = (sb - sa).abs() / sa;
    d
}

fn json(export: &CalibrationExport) -> String {
    serde_json::to_string(export).unwrap()
}

/// The seed of a calibration of `scene`: its cameras whole, mounts, planes
/// and hand-eye.
fn self_seed(scene: &Scene, export: &CalibrationExport) -> DeviceSeed {
    DeviceSeed::from_calibration(export, &scene.ids, &scene.sizes).unwrap()
}

/// `pose` turned by `deg` about `axis` and moved by `m` along x.
fn nudged(pose: &Iso3, axis: Vector3<f64>, deg: f64, m: f64) -> Iso3 {
    let turn = Iso3::from_parts(
        Translation3::new(m, 0.0, 0.0),
        UnitQuaternion::from_scaled_axis(axis.normalize() * deg.to_radians()),
    );
    turn * pose
}

/// Scale every calibrated lens's focal lengths by `focal` and add `k1` to
/// its leading radial coefficient.
fn perturb_lenses(seed: &mut DeviceSeed, focal: f64, k1: f64) {
    for camera in &mut seed.cameras {
        if let SeedLens::Calibrated { camera } = &mut camera.lens {
            camera.intrinsics.fx *= focal;
            camera.intrinsics.fy *= focal;
            if let DistortionParams::BrownConrady5 { params } = &mut camera.distortion {
                params.k1 += k1;
            }
        }
    }
}

/// Turn the second camera's mount by `deg` and move it by `m`.
fn perturb_mount(seed: &mut DeviceSeed, deg: f64, m: f64) {
    if let Some(mount) = seed
        .cameras
        .get_mut(1)
        .and_then(|camera| camera.cam_se3_rig.as_mut())
    {
        *mount = nudged(mount, Vector3::new(0.3, 1.0, 0.2), deg, m);
    }
}

/// Turn the hand-eye by `deg` and move it by `m`.
fn perturb_handeye(seed: &mut DeviceSeed, deg: f64, m: f64) {
    let axis = Vector3::new(1.0, 0.4, -0.3);
    seed.handeye = seed.handeye.map(|handeye| match handeye {
        SeedHandeye::EyeInHand { gripper_se3_rig } => SeedHandeye::EyeInHand {
            gripper_se3_rig: nudged(&gripper_se3_rig, axis, deg, m),
        },
        SeedHandeye::EyeToHand { rig_se3_base } => SeedHandeye::EyeToHand {
            rig_se3_base: nudged(&rig_se3_base, axis, deg, m),
        },
    });
}

/// Turn every laser plane by `deg` and shift it by `m`.
fn perturb_planes(seed: &mut DeviceSeed, deg: f64, m: f64) {
    let turn = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 1.0, 0.0) * deg.to_radians());
    for camera in &mut seed.cameras {
        if let Some(plane) = &camera.laser_plane {
            let normal = turn * plane.normal().into_inner();
            camera.laser_plane = Some(LaserPlane::new(normal, plane.distance() + m).unwrap());
        }
    }
}

/// A sibling unit's seed: focal lengths 2 % long, `k1` 0.05 larger, the
/// second mount and the hand-eye 1° and 2 mm off, the planes 1° and 2 mm
/// off.
fn sibling_seed(seed: &DeviceSeed) -> DeviceSeed {
    let mut sibling = seed.clone();
    perturb_lenses(&mut sibling, 1.02, 0.05);
    perturb_mount(&mut sibling, 1.0, 0.002);
    perturb_handeye(&mut sibling, 1.0, 0.002);
    perturb_planes(&mut sibling, 1.0, 0.002);
    sibling
}

// ─────────────────────────────────────────────────────────────────────────────
// Unseeded runs keep their step lists
// ─────────────────────────────────────────────────────────────────────────────

/// Unseeded, the dataset path's run of every topology is the workflow's
/// public step list, bit for bit.
#[test]
fn an_unseeded_run_is_the_workflow_s_public_step_list() {
    for kind in TOPOLOGIES {
        let scene = scene(kind, SolverBackend::default());
        let through_seed_path = run(&scene, None).unwrap();
        let by_public_steps = public_steps(&scene).unwrap();
        assert_eq!(json(&through_seed_path), json(&by_public_steps), "{kind:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Seeded runs reach the unseeded optimum
// ─────────────────────────────────────────────────────────────────────────────

/// How far a stage's solve stops from where it would stop from another
/// start, on the scene, for the stages that stop short of their minimum by
/// more than [`Diff::same_optimum`]'s tolerances; `None` for the others.
///
/// - The rig bundle adjustment stops on a relative cost decrease of 1e-5 in
///   a flat valley: restarted from its own poses, a rig extrinsics run ends
///   some hundredths of a σ away in the cameras and a fraction of a σ in a
///   mount. A rig hand-eye's hand-eye stage holds the mounts where the rig
///   stage left them: from mounts and a hand-eye turned 1° to either side,
///   they end a few tenths of a micrometre apart.
///
/// They measure the workflow's own precision, which a seeded run's
/// difference is read against: a restart, or two starts symmetric about
/// the unseeded result.
fn stage_precision(scene: &Scene, unseeded: &CalibrationExport) -> Option<Diff> {
    use crate::common::{RigManualInit, RigPoses};
    use crate::rig_extrinsics as re;
    let restart = |cam_se3_rig: Vec<Iso3>, rig_se3_target: Vec<Iso3>| {
        let mut manual = RigManualInit::new();
        manual.poses = Some(RigPoses::new(cam_se3_rig, rig_se3_target));
        manual
    };
    match (&scene.config, &scene.input) {
        (CalibrationConfig::RigExtrinsics(c), Input::Rig(i)) => {
            let mut s = session::<RigExtrinsicsProblem>(i, c);
            re::step_intrinsics_init_all(&mut s, None).unwrap();
            re::step_intrinsics_optimize_all(&mut s, None).unwrap();
            re::step_rig_init(&mut s, None).unwrap();
            re::step_rig_optimize(&mut s, None).unwrap();
            let first = CalibrationExport::RigExtrinsics(s.export().unwrap());
            let out = s.output().unwrap().clone();
            let manual = restart(
                crate::rig_family::inverse_poses(&out.params.rig_se3_cam),
                out.params.rig_se3_target,
            );
            re::step_rig_init_with_seed(&mut s, manual, None).unwrap();
            re::step_rig_optimize(&mut s, None).unwrap();
            let restarted = CalibrationExport::RigExtrinsics(s.export().unwrap());
            Some(diff(&first, &restarted))
        }
        (CalibrationConfig::RigHandeye(_), Input::RigHandeye(_)) => {
            // The hand-eye stage holds the mounts where the rig stage left
            // them, and no covariance covers them: their spread is the rig
            // stage's, from mounts and a hand-eye turned 1° either way.
            let turned = |deg: f64| {
                let mut seed = self_seed(scene, unseeded);
                perturb_mount(&mut seed, deg, 0.0);
                perturb_handeye(&mut seed, deg, 0.0);
                run(scene, Some(&seed)).unwrap()
            };
            Some(diff(&turned(1.0), &turned(-1.0)))
        }
        _ => None,
    }
}

impl Diff {
    /// Whether `self` is within three times a stage's `precision`, and
    /// never held tighter than [`Diff::same_optimum`].
    fn within(&self, precision: &Diff) -> bool {
        self.max_z <= (3.0 * precision.max_z).max(0.01)
            && self.pose_z <= (3.0 * precision.pose_z).max(0.05)
            && self.plane_z <= (3.0 * precision.plane_z).max(0.05)
            && self.sse_rel <= (3.0 * precision.sse_rel).max(1e-5)
    }
}

/// Whether a seeded run's difference from the unseeded one is within the
/// workflow's precision: [`Diff::same_optimum`], or three times the stage's
/// own precision where the unseeded workflow stops short of that.
fn at_the_unseeded_optimum(d: &Diff, precision: Option<&Diff>) -> bool {
    match precision {
        Some(precision) => d.within(precision),
        None => d.same_optimum(),
    }
}

/// Seeded with its own calibration, and with a sibling unit's, each
/// topology's run lands on the unseeded run's optimum
/// ([`at_the_unseeded_optimum`]).
///
/// One cell is reported and not held to it: a rig laserline device whose
/// planes are seeded a degree and 2 mm off. Its plane solve stops a few σ
/// of a normal coordinate from the linear fit's optimum, at a laser cost
/// 0.3 % higher, and a restart from there does not move it; the same start
/// with the point-to-plane residual converges. The self-seeded cell, which
/// is held to it, lands exactly on the unseeded optimum.
fn assert_seeded_runs_reach_the_unseeded_optimum(backend: SolverBackend) {
    for kind in TOPOLOGIES {
        let scene = scene(kind, backend);
        let unseeded = run(&scene, None).unwrap();
        let precision = stage_precision(&scene, &unseeded);
        if let Some(precision) = &precision {
            println!("{backend:?} {kind:?} stage precision: {precision:?}");
        }
        let seed = self_seed(&scene, &unseeded);
        for (name, seed) in [("self", seed.clone()), ("sibling", sibling_seed(&seed))] {
            let seeded = run(&scene, Some(&seed)).unwrap();
            let d = diff(&unseeded, &seeded);
            println!("{backend:?} {kind:?} {name}: {d:?}");
            if kind == ProblemKind::RigLaserlineDevice && name == "sibling" {
                continue;
            }
            assert!(
                at_the_unseeded_optimum(&d, precision.as_ref()),
                "{backend:?} {kind:?} {name}: {d:?} (precision {precision:?})"
            );
        }
    }
}

#[test]
fn seeded_runs_reach_the_unseeded_optimum() {
    assert_seeded_runs_reach_the_unseeded_optimum(SolverBackend::default());
}

/// The same on the other solver backend.
#[test]
#[ignore = "release-only: every topology on the second backend"]
fn seeded_runs_reach_the_unseeded_optimum_on_the_other_backend() {
    let other = if SolverBackend::default() == SolverBackend::Factrs {
        SolverBackend::TinySolver
    } else {
        SolverBackend::Factrs
    };
    assert_seeded_runs_reach_the_unseeded_optimum(other);
}

/// A rig hand-eye run seeded from a calibration whose reference camera is
/// the other one reaches the same optimum: the seed's rig frame is
/// re-expressed in the configured reference camera's.
#[test]
fn a_seed_in_another_rig_frame_reaches_the_same_optimum() {
    let scene = scene(ProblemKind::RigHandeye, SolverBackend::default());
    let unseeded = run(&scene, None).unwrap();
    let CalibrationConfig::RigHandeye(config) = &scene.config else {
        unreachable!()
    };
    let mut flipped = config.clone();
    flipped.rig.reference_camera_idx = 1;
    let other_frame = Scene {
        config: CalibrationConfig::RigHandeye(flipped),
        input: match &scene.input {
            Input::RigHandeye(i) => Input::RigHandeye(i.clone()),
            _ => unreachable!(),
        },
        ids: scene.ids.clone(),
        sizes: scene.sizes.clone(),
        truth: scene.truth.clone(),
    };
    let reference_one = run(&other_frame, None).unwrap();
    assert!(reference_one.cam_se3_rig()[1].rotation.angle() < 1e-12);
    let seed = sibling_seed(&self_seed(&other_frame, &reference_one));
    let seeded = run(&scene, Some(&seed)).unwrap();
    let d = diff(&unseeded, &seeded);
    let precision = stage_precision(&scene, &unseeded);
    println!(
        "rig hand-eye seeded from the other reference camera: {d:?} (precision {precision:?})"
    );
    assert!(at_the_unseeded_optimum(&d, precision.as_ref()), "{d:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrong seeds
// ─────────────────────────────────────────────────────────────────────────────

/// How a run from a wrong seed ended.
enum Outcome {
    /// At the unseeded optimum ([`at_the_unseeded_optimum`]).
    Converged,
    /// Elsewhere: its difference from the unseeded run, its quality verdict
    /// and its largest camera-parameter distance from the truth, in σ.
    Elsewhere {
        diff: Diff,
        verdict: Verdict,
        truth_sigma: f64,
    },
    /// A typed error.
    Failed(String),
}

/// Seeds far from the truth: each run converges to the unseeded optimum,
/// fails typed, or ends where its quality verdict is not pass. No run ends
/// in another basin with a pass verdict and a camera parameter more than
/// 5 σ from the truth.
#[test]
#[ignore = "release-only: the wrong-seed matrix"]
fn a_wrong_seed_converges_or_is_flagged() {
    type Perturb = fn(&mut DeviceSeed);
    let cells: [(ProblemKind, &str, Perturb); 9] = [
        (ProblemKind::PlanarIntrinsics, "fx x0.5", |s| {
            perturb_lenses(s, 0.5, 0.0)
        }),
        (ProblemKind::PlanarIntrinsics, "fx x2", |s| {
            perturb_lenses(s, 2.0, 0.0)
        }),
        (ProblemKind::PlanarIntrinsics, "k1 sign", |s| flip_k1(s)),
        (ProblemKind::RigExtrinsics, "fx x0.5", |s| {
            perturb_lenses(s, 0.5, 0.0)
        }),
        (ProblemKind::RigExtrinsics, "fx x2", |s| {
            perturb_lenses(s, 2.0, 0.0)
        }),
        (ProblemKind::RigExtrinsics, "mount 30 deg", |s| {
            perturb_mount(s, 30.0, 0.0)
        }),
        (ProblemKind::SingleCamHandeye, "hand-eye 90 deg", |s| {
            perturb_handeye(s, 90.0, 0.0)
        }),
        (ProblemKind::RigHandeye, "mount 30 deg", |s| {
            perturb_mount(s, 30.0, 0.0)
        }),
        (ProblemKind::RigHandeye, "hand-eye 90 deg", |s| {
            perturb_handeye(s, 90.0, 0.0)
        }),
    ];
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        for (kind, name, perturb) in cells {
            let scene = scene(kind, backend);
            let unseeded = run(&scene, None).unwrap();
            let precision = stage_precision(&scene, &unseeded);
            let mut seed = self_seed(&scene, &unseeded);
            perturb(&mut seed);
            let outcome = match run(&scene, Some(&seed)) {
                Err(err) => Outcome::Failed(err.to_string()),
                Ok(seeded) => {
                    let d = diff(&unseeded, &seeded);
                    if at_the_unseeded_optimum(&d, precision.as_ref()) {
                        Outcome::Converged
                    } else {
                        let report =
                            assess(&seeded, &scene.sizes, &QualityThresholds::default()).unwrap();
                        let off = truth_z(&scene, &seeded);
                        assert!(
                            report.verdict != Verdict::Pass || off <= 5.0,
                            "{backend:?} {kind:?} {name}: a wrong basin passes ({off:.1} σ from \
                             the truth)"
                        );
                        Outcome::Elsewhere {
                            diff: d,
                            verdict: report.verdict,
                            truth_sigma: off,
                        }
                    }
                }
            };
            match &outcome {
                Outcome::Converged => println!("{backend:?} {kind:?} {name}: converged"),
                Outcome::Elsewhere {
                    diff,
                    verdict,
                    truth_sigma,
                } => println!(
                    "{backend:?} {kind:?} {name}: elsewhere, {diff:?}, verdict {verdict:?}, \
                     {truth_sigma:.2} σ from the truth"
                ),
                Outcome::Failed(err) => println!("{backend:?} {kind:?} {name}: failed: {err}"),
            }
        }
    }
}

fn flip_k1(seed: &mut DeviceSeed) {
    for camera in &mut seed.cameras {
        if let SeedLens::Calibrated { camera } = &mut camera.lens
            && let DistortionParams::BrownConrady5 { params } = &mut camera.distortion
        {
            params.k1 = -params.k1;
        }
    }
}

/// The largest `|θ − θ_truth| / σ` over the camera parameters `export`
/// states a σ for (infinite without a covariance).
fn truth_z(scene: &Scene, export: &CalibrationExport) -> f64 {
    let Uncertainty::Available(cov) = export.parameter_uncertainty() else {
        return f64::INFINITY;
    };
    let mut worst: f64 = 0.0;
    for (i, (camera, truth)) in export.camera_params().iter().zip(&scene.truth).enumerate() {
        let estimated = camera_parameters(camera).unwrap();
        let truth = camera_parameters(truth).unwrap();
        for ((label, v), (_, t)) in estimated.iter().zip(&truth) {
            let sigma = cov
                .std_dev(&format!("cam{i}.{label}"))
                .or_else(|| cov.std_dev(label));
            if let Some(sigma) = sigma.filter(|s| *s > 0.0) {
                worst = worst.max((v - t).abs() / sigma);
            }
        }
    }
    worst
}

// ─────────────────────────────────────────────────────────────────────────────
// A seeded Scheimpflug `k1`
// ─────────────────────────────────────────────────────────────────────────────

/// A planar Scheimpflug calibration seeded with its own result: the seeded
/// refinement starts its leading radial coefficient from a fixed sweep, not
/// from the seed. For strong barrel distortion the seeded run still lands on
/// the optimum of the truth's basin, on both backends.
#[test]
#[ignore = "release-only: three lenses on both backends"]
fn a_seeded_scheimpflug_k1_lands_on_the_optimum() {
    let tilted = pinhole_model(SensorKind::Scheimpflug);
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        for k1 in [-0.30, -0.45, -0.60] {
            let mut truth = pinhole(1000.0, 990.0, 640.0, 480.0, k1, 0.0);
            truth.sensor = SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.087,
                    tilt_y: 0.0,
                },
            };
            let b = Builder::new(0x5C4E_0000);
            let obs = b.observe(&truth, &b.board_poses(0.5, 0.6, 0.4), 0);
            let input =
                PlanarDataset::new(obs.into_iter().map(View::without_meta).collect()).unwrap();
            let mut config = PlanarIntrinsicsConfig::default();
            config.camera.model = tilted;
            config.solver.backend = backend;
            let scene = Scene {
                config: CalibrationConfig::PlanarIntrinsics(config),
                input: Input::Planar(input),
                ids: ids(1),
                sizes: vec![[1280, 960]],
                truth: vec![truth.clone()],
            };
            // The datasheet start (a coarse focal and the mount tilt), then
            // the seed of that run's own result.
            let mut nominal = DeviceSeed::new(vec![crate::device_seed::SeedCamera::new(
                "cam0",
                [1280, 960],
                SeedLens::Nominal {
                    intrinsics: FxFyCxCy {
                        fx: 920.0,
                        fy: 920.0,
                        cx: 640.0,
                        cy: 480.0,
                    },
                    sensor: ScheimpflugParams {
                        tilt_x: 0.08,
                        tilt_y: 0.0,
                    },
                },
            )]);
            nominal.description = Some("datasheet".into());
            let from_datasheet = run(&scene, Some(&nominal)).unwrap();
            let seeded = run(&scene, Some(&self_seed(&scene, &from_datasheet))).unwrap();
            let d = diff(&from_datasheet, &seeded);
            let off = truth_z(&scene, &seeded);
            let seeded_k1 = camera_parameters(&seeded.camera_params()[0])
                .unwrap()
                .into_iter()
                .find(|(label, _)| *label == "k1")
                .map(|(_, value)| value)
                .unwrap();
            println!(
                "{backend:?} k1={k1}: seeded k1={seeded_k1:.5}, Δ to the datasheet-seeded run \
                 {d:?}, {off:.2} σ from the truth"
            );
            assert!(d.max_z < 0.01, "{backend:?} k1={k1}: {d:?}");
            assert!(off < 5.0, "{backend:?} k1={k1}: {off} σ from the truth");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checks before any image is read
// ─────────────────────────────────────────────────────────────────────────────

/// A cache that fails the test when the run reads an image's detection.
struct NoDetection;

impl vision_calibration_detect::DetectionCache for NoDetection {
    fn get(
        &self,
        _: &vision_calibration_detect::CacheKey,
    ) -> Result<
        Option<vision_calibration_detect::CachedFeatures>,
        vision_calibration_detect::CacheError,
    > {
        panic!("the run read an image");
    }

    fn put(
        &self,
        _: &vision_calibration_detect::CacheKey,
        _: &vision_calibration_detect::CachedFeatures,
    ) -> Result<(), vision_calibration_detect::CacheError> {
        panic!("the run detected an image");
    }
}

/// A manifest of `topology` with cameras `ids` whose images match nothing:
/// reading one would fail with an empty match, so a seed error proves the
/// check ran first.
fn manifest(topology: &str, ids: &[&str], extra: serde_json::Value) -> DatasetSpec {
    let cameras: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "images": { "kind": "glob", "pattern": format!("missing/{id}/*.png") },
                "laser_images": { "kind": "glob", "pattern": format!("missing/{id}/laser_*.png") },
            })
        })
        .collect();
    let mut spec = serde_json::json!({
        "version": 1,
        "topology": topology,
        "cameras": cameras,
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
    });
    if let (Some(spec), Some(extra)) = (spec.as_object_mut(), extra.as_object()) {
        spec.extend(extra.clone());
    }
    if topology != "laserline_device"
        && topology != "rig_laserline_device"
        && topology != "rig_handeye_laserline"
    {
        for camera in spec["cameras"].as_array_mut().unwrap() {
            camera.as_object_mut().unwrap().remove("laser_images");
        }
    }
    serde_json::from_value(spec).unwrap()
}

fn robot() -> serde_json::Value {
    serde_json::json!({
        "robot_poses": { "path": "poses.txt", "format": "rowmajor4x4" },
        "pose_convention": {
            "transform": "t_base_tcp",
            "rotation_format": "matrix4x4_row_major",
            "translation_units": "m"
        },
        "pose_pairing": { "kind": "by_index" },
    })
}

/// A laser-line extractor the checks never reach.
struct NoLaser;

impl crate::dataset_runner::LaserPixelExtractor for NoLaser {
    fn name(&self) -> &str {
        "none"
    }

    fn extract(
        &self,
        _: vision_calibration_detect::GrayImageView<'_>,
        _: &vision_calibration_dataset::LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        panic!("the run read a laser image");
    }
}

fn seeded_error(spec: &DatasetSpec, seed: &DeviceSeed) -> DeviceSeedError {
    let laser = NoLaser;
    match calibrate_dataset_with_seed(
        spec,
        Path::new("."),
        None,
        seed,
        Detection::Cached(&NoDetection),
        Some(&laser),
        &RunControl::default(),
    ) {
        Err(CalibrateDatasetError::Seed(err)) => err,
        other => panic!("expected a seed error, got {other:?}"),
    }
}

/// A calibrated seed of the cameras `ids`, each a default-model camera of
/// 640 × 480 images.
fn calibrated_seed(ids: &[&str]) -> DeviceSeed {
    let cameras: Vec<CameraParams> = ids
        .iter()
        .map(|_| pinhole(600.0, 600.0, 320.0, 240.0, 0.0, 0.0))
        .collect();
    let export_ids: Vec<String> = ids.iter().map(|id| (*id).to_string()).collect();
    let sizes = vec![[640, 480]; ids.len()];
    let mut seed = DeviceSeed::new(
        cameras
            .into_iter()
            .zip(&export_ids)
            .map(|(camera, id)| {
                crate::device_seed::SeedCamera::new(
                    id.clone(),
                    sizes[0],
                    SeedLens::Calibrated { camera },
                )
            })
            .collect(),
    );
    if ids.len() > 1 {
        for (i, camera) in seed.cameras.iter_mut().enumerate() {
            camera.cam_se3_rig = Some(poses::make_iso((0.0, 0.0, 0.0), (0.1 * i as f64, 0.0, 0.0)));
        }
    }
    seed
}

#[test]
fn a_seed_that_does_not_fit_fails_before_any_image_is_read() {
    let planar = manifest("planar_intrinsics", &["cam0"], serde_json::json!({}));

    // The seed itself.
    let mut seed = calibrated_seed(&["cam0"]);
    seed.version = 99;
    assert!(matches!(
        seeded_error(&planar, &seed),
        DeviceSeedError::UnsupportedVersion { found: 99, .. }
    ));

    // Its cameras are not the manifest's.
    let err = seeded_error(&planar, &calibrated_seed(&["cam1"]));
    let DeviceSeedError::CameraIdsDiffer {
        missing,
        unexpected,
    } = err
    else {
        panic!("{err:?}");
    };
    assert_eq!(
        (missing, unexpected),
        (vec!["cam0".to_string()], vec!["cam1".to_string()])
    );
    assert!(matches!(
        seeded_error(&planar, &calibrated_seed(&["cam0", "cam1"])),
        DeviceSeedError::CameraIdsDiffer { .. }
    ));

    // A calibrated lens of another model: another projection, as another
    // distortion or sensor.
    let mut fisheye = calibrated_seed(&["cam0"]);
    if let SeedLens::Calibrated { camera } = &mut fisheye.cameras[0].lens {
        camera.projection = ProjectionParams::Equidistant;
    }
    assert!(matches!(
        seeded_error(&planar, &fisheye),
        DeviceSeedError::ModelMismatch { ref camera, .. } if camera == "cam0"
    ));

    // A nominal tilt for an untilted model.
    let tilted = DeviceSeed::new(vec![crate::device_seed::SeedCamera::new(
        "cam0",
        [640, 480],
        SeedLens::Nominal {
            intrinsics: FxFyCxCy {
                fx: 600.0,
                fy: 600.0,
                cx: 320.0,
                cy: 240.0,
            },
            sensor: ScheimpflugParams {
                tilt_x: 0.05,
                tilt_y: 0.0,
            },
        },
    )]);
    assert!(matches!(
        seeded_error(&planar, &tilted),
        DeviceSeedError::TiltForUntiltedModel { .. }
    ));

    // A hand-eye of the other mode (the default config is eye-in-hand).
    let handeye = manifest("single_cam_handeye", &["cam0"], robot());
    let mut eye_to_hand = calibrated_seed(&["cam0"]);
    eye_to_hand.handeye = Some(SeedHandeye::EyeToHand {
        rig_se3_base: Iso3::identity(),
    });
    assert!(matches!(
        seeded_error(&handeye, &eye_to_hand),
        DeviceSeedError::HandeyeModeMismatch {
            spec: HandeyeMode::EyeToHand,
            expected: HandeyeMode::EyeInHand,
        }
    ));

    // A rig's lenses calibrated for some cameras and nominal for others.
    let by_index = serde_json::json!({ "pose_pairing": { "kind": "by_index" } });
    let rig = manifest("rig_extrinsics", &["cam0", "cam1"], by_index);
    let mut mixed = calibrated_seed(&["cam0", "cam1"]);
    mixed.cameras[1].lens = SeedLens::Nominal {
        intrinsics: FxFyCxCy {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
        },
        sensor: ScheimpflugParams::default(),
    };
    assert!(matches!(
        seeded_error(&rig, &mixed),
        DeviceSeedError::PartialField {
            field: "distortion"
        }
    ));

    // A rig laserline device takes only planes: a seed without them seeds
    // nothing.
    let mut rig_laser_extra = robot();
    rig_laser_extra["upstream_calibration"] = "upstream.json".into();
    let rig_laser = manifest("rig_laserline_device", &["cam0", "cam1"], rig_laser_extra);
    assert!(matches!(
        seeded_error(&rig_laser, &calibrated_seed(&["cam0", "cam1"])),
        DeviceSeedError::NothingToSeed {
            topology: ProblemKind::RigLaserlineDevice
        }
    ));
}

/// A seed for images of another size fails after the images' headers are
/// read and before any detection.
#[test]
fn a_seed_for_another_image_size_fails_before_detection() {
    let tmp = tempfile::tempdir().unwrap();
    let paths: Vec<String> = (0..3)
        .map(|i| {
            let rel = format!("img/view{i}.png");
            std::fs::create_dir_all(tmp.path().join("img")).unwrap();
            image::GrayImage::new(640, 480)
                .save(tmp.path().join(&rel))
                .unwrap();
            rel
        })
        .collect();
    let spec: DatasetSpec = serde_json::from_value(serde_json::json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": paths } }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
    }))
    .unwrap();
    let mut seed = calibrated_seed(&["cam0"]);
    seed.cameras[0].image_size = [1280, 960];
    let err = calibrate_dataset_with_seed(
        &spec,
        tmp.path(),
        None,
        &seed,
        Detection::Cached(&NoDetection),
        None,
        &RunControl::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            CalibrateDatasetError::Seed(DeviceSeedError::ImageSizeMismatch {
                seed: [1280, 960],
                dataset: [640, 480],
                ..
            })
        ),
        "{err:?}"
    );
}
