//! Deterministic synthetic scene matrix for the solver benchmark.
//!
//! A scene is `problem × camera × scale × pixel noise × outlier spec`. Every
//! scene is generated from a seed derived from its id, carries the ground
//! truth that produced it, and records exactly which target observations were
//! displaced into outliers.

use std::fmt;

use anyhow::{Result, bail};
use nalgebra::Vector3;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use vision_calibration::common::{RigLaserlineView, RobotPoseMeta};
use vision_calibration::core::{
    BrownConrady5, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4, NoMeta, PlanarDataset, ProjectionKind,
    ProjectionParams, Pt2, Pt3, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorParams,
    View,
};
use vision_calibration::laserline_device::LaserlineMeta;
use vision_calibration::laserline_device::{LaserlineDeviceConfig, LaserlineDeviceInput};
use vision_calibration::optim::{HandeyeMode, LaserPlane, RobustLoss, SolverBackend};
use vision_calibration::planar_intrinsics::{IntrinsicsManualInit, PlanarIntrinsicsConfig};
use vision_calibration::rig_extrinsics::{
    IntrinsicsManualInit as RigIntrinsicsManualInit, RigExtrinsicsConfig, RigExtrinsicsInput,
};
use vision_calibration::rig_handeye::{RigHandeyeConfig, RigHandeyeInput};
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineConfig, RigHandeyeLaserlineInput, RigHandeyeLaserlineView,
};
use vision_calibration::rig_laserline_device::{
    RigLaserlineDataset, RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigUpstreamCalibration,
};
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeConfig, SingleCamHandeyeInput, SingleCamHandeyeView,
};
use vision_calibration::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::planar;
use vision_calibration::synthetic::poses;
use vision_calibration_core::CameraModel;
use vision_calibration_optim::LaserlineResidualType;

use super::metrics::ModelParams;

/// Pixel-noise levels of the full matrix (per-axis standard deviation, px).
pub const NOISE_LEVELS_PX: [f64; 2] = [0.1, 0.5];
/// Fraction of target observations displaced into outliers.
pub const OUTLIER_FRACTION: f64 = 0.05;
/// Outlier displacement magnitude range, pixels.
pub const OUTLIER_SHIFT_PX: (f64, f64) = (10.0, 30.0);
/// Scale of the Huber / Cauchy loss used on contaminated scenes.
pub const OUTLIER_LOSS_SCALE: f64 = 1.0;

/// The seven problem types; a tilted sensor or a fisheye lens is a
/// [`CameraKind`] of a scene, not a problem.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum Problem {
    /// Planar intrinsics.
    PlanarIntrinsics,
    /// Single camera hand-eye.
    SingleCamHandeye,
    /// Laser plane + camera on a planar target.
    LaserlineDevice,
    /// Multi-camera rig extrinsics.
    RigExtrinsics,
    /// Multi-camera rig hand-eye.
    RigHandeye,
    /// Laser planes for a frozen rig.
    RigLaserlineDevice,
    /// Joint rig hand-eye + laser planes.
    RigHandeyeLaserline,
}

impl Problem {
    /// Every problem type, in table order.
    pub const ALL: [Problem; 7] = [
        Problem::PlanarIntrinsics,
        Problem::SingleCamHandeye,
        Problem::LaserlineDevice,
        Problem::RigExtrinsics,
        Problem::RigHandeye,
        Problem::RigLaserlineDevice,
        Problem::RigHandeyeLaserline,
    ];

    /// Stable snake_case name (also the CLI and serde spelling).
    pub fn name(self) -> &'static str {
        match self {
            Self::PlanarIntrinsics => "planar_intrinsics",
            Self::SingleCamHandeye => "single_cam_handeye",
            Self::LaserlineDevice => "laserline_device",
            Self::RigExtrinsics => "rig_extrinsics",
            Self::RigHandeye => "rig_handeye",
            Self::RigLaserlineDevice => "rig_laserline_device",
            Self::RigHandeyeLaserline => "rig_handeye_laserline",
        }
    }

    /// Whether the solve consults a robust loss on the target observations.
    /// The frozen-rig laser problem fits only laser planes, so contaminating
    /// its target observations would change nothing.
    pub fn uses_target_loss(self) -> bool {
        self != Self::RigLaserlineDevice
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Camera flavour of the scene: the model its cameras are generated with
/// and calibrated as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraKind {
    /// Pinhole projection, Brown-Conrady distortion, straight sensor.
    Pinhole,
    /// Pinhole projection, Brown-Conrady distortion, Scheimpflug-tilted
    /// sensor.
    Scheimpflug,
    /// Kannala-Brandt fisheye: equidistant projection, `KannalaBrandt4`
    /// distortion, straight sensor. Boards come closer, so the corners
    /// reach 50–60° off-axis.
    Fisheye,
}

impl fmt::Display for CameraKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pinhole => "pinhole",
            Self::Scheimpflug => "scheimpflug",
            Self::Fisheye => "fisheye",
        })
    }
}

impl CameraKind {
    /// The camera model spec the scene's cameras are calibrated as.
    pub fn model(self) -> CameraModelSpec {
        match self {
            Self::Pinhole => CameraModelSpec::pinhole(
                DistortionKind::BrownConrady5,
                vision_calibration::core::SensorKind::Identity,
            ),
            Self::Scheimpflug => CameraModelSpec::pinhole(
                DistortionKind::BrownConrady5,
                vision_calibration::core::SensorKind::Scheimpflug,
            ),
            Self::Fisheye => CameraModelSpec::new(
                ProjectionKind::Equidistant,
                DistortionKind::KannalaBrandt4,
                vision_calibration::core::SensorKind::Identity,
            ),
        }
    }
}

/// Scene size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scale {
    /// Few views, sparse board.
    Small,
    /// Typical calibration session.
    Medium,
    /// Dense board, many views.
    Large,
}

/// Views and board density of a [`Scale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleSpec {
    /// Number of views (robot stations / board poses).
    pub views: usize,
    /// Board corners along x.
    pub nx: usize,
    /// Board corners along y.
    pub ny: usize,
    /// Corner spacing, metres.
    pub spacing: f64,
}

/// The single scale table. Every board spans about 48 × 36 cm, so at the
/// scenes' 0.5–1 m working distance it fills most of the image: corners reach
/// the image periphery, where distortion is observable.
pub const SCALE_TABLE: [(Scale, ScaleSpec); 3] = [
    (
        Scale::Small,
        ScaleSpec {
            views: 8,
            nx: 9,
            ny: 7,
            spacing: 0.06,
        },
    ),
    (
        Scale::Medium,
        ScaleSpec {
            views: 20,
            nx: 13,
            ny: 9,
            spacing: 0.04,
        },
    ),
    (
        Scale::Large,
        ScaleSpec {
            views: 36,
            nx: 19,
            ny: 13,
            spacing: 0.48 / 18.0,
        },
    ),
];

impl Scale {
    /// All scales, smallest first.
    pub const ALL: [Scale; 3] = [Scale::Small, Scale::Medium, Scale::Large];

    /// Look this scale up in [`SCALE_TABLE`].
    pub fn spec(self) -> ScaleSpec {
        SCALE_TABLE
            .iter()
            .find(|(s, _)| *s == self)
            .map(|(_, spec)| *spec)
            .expect("every scale is in the table")
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        })
    }
}

/// Outlier contamination of the target observations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutlierSpec {
    /// No outliers.
    None,
    /// `fraction` of the target observations displaced by a random direction
    /// times a uniform `[10, 30]` px magnitude.
    Injected {
        /// Fraction of target observations, in `(0, 1)`.
        fraction: f64,
    },
}

/// One cell of the scene matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneSpec {
    /// Problem type.
    pub problem: Problem,
    /// Camera flavour.
    pub camera: CameraKind,
    /// Scene size.
    pub scale: Scale,
    /// Per-axis pixel-noise standard deviation.
    pub noise_px: f64,
    /// Outlier contamination.
    pub outliers: OutlierSpec,
    /// Robust loss the solve is configured with.
    pub loss: RobustLoss,
}

impl SceneSpec {
    /// Stable scene key: `problem/camera/scale/n<σ>/<clean|huber|cauchy|…>`.
    pub fn id(&self) -> String {
        let contamination = match (self.outliers, self.loss) {
            (OutlierSpec::None, RobustLoss::None) => "clean".to_string(),
            (OutlierSpec::None, loss) => format!("clean+{}", loss_name(loss)),
            (OutlierSpec::Injected { .. }, RobustLoss::None) => "outliers+l2".to_string(),
            (OutlierSpec::Injected { .. }, loss) => loss_name(loss).to_string(),
        };
        format!(
            "{}/{}/{}/n{:.1}/{}",
            self.problem, self.camera, self.scale, self.noise_px, contamination
        )
    }

    /// Deterministic seed: FNV-1a of the scene id.
    ///
    /// A tilted planar scene hashes its id with the problem spelled
    /// `scheimpflug_intrinsics`, the key its data was first generated under,
    /// so the scene — and the regression-suite entry frozen from it — keeps
    /// its data.
    pub fn seed(&self) -> u64 {
        let id = self.id();
        let key = match (self.problem, self.camera) {
            (Problem::PlanarIntrinsics, CameraKind::Scheimpflug) => id.replacen(
                Problem::PlanarIntrinsics.name(),
                "scheimpflug_intrinsics",
                1,
            ),
            _ => id,
        };
        key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }
}

fn loss_name(loss: RobustLoss) -> &'static str {
    match loss {
        RobustLoss::None => "l2",
        RobustLoss::Huber { .. } => "huber",
        RobustLoss::Cauchy { .. } => "cauchy",
        RobustLoss::Arctan { .. } => "arctan",
    }
}

/// An injected outlier: which target observation was displaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OutlierRef {
    /// View index.
    pub pose: usize,
    /// Camera index (`0` for single-camera problems).
    pub camera: usize,
    /// Feature index within the view's correspondences.
    pub feature: usize,
}

impl OutlierRef {
    /// `(pose, camera, feature)` lookup key matching the export's residual records.
    pub fn key(&self) -> (usize, usize, usize) {
        (self.pose, self.camera, self.feature)
    }
}

/// Everything a driver needs to run one scene.
#[derive(Debug, Clone)]
pub enum SceneData {
    /// Planar intrinsics.
    Planar {
        /// Input dataset.
        input: PlanarDataset,
        /// Solver configuration.
        config: PlanarIntrinsicsConfig,
    },
    /// Planar intrinsics of a tilted (Scheimpflug) camera with a coarse
    /// device-spec seed.
    Scheimpflug {
        /// Input dataset.
        input: PlanarDataset,
        /// Solver configuration (a Scheimpflug camera model).
        config: PlanarIntrinsicsConfig,
        /// Coarse focal + nominal tilt seed.
        seed: IntrinsicsManualInit,
    },
    /// Single-camera hand-eye.
    SingleCamHandeye {
        /// Input views.
        input: SingleCamHandeyeInput,
        /// Solver configuration.
        config: SingleCamHandeyeConfig,
    },
    /// Laserline device.
    Laserline {
        /// Input views.
        input: LaserlineDeviceInput,
        /// Solver configuration.
        config: LaserlineDeviceConfig,
    },
    /// Rig extrinsics.
    Rig {
        /// Input dataset.
        input: RigExtrinsicsInput,
        /// Solver configuration.
        config: RigExtrinsicsConfig,
        /// Per-camera intrinsics-stage seed (nominal tilts for the
        /// Scheimpflug rig; empty for the pinhole rig).
        seed: RigIntrinsicsManualInit,
    },
    /// Rig hand-eye.
    RigHandeye {
        /// Input dataset.
        input: RigHandeyeInput,
        /// Solver configuration.
        config: RigHandeyeConfig,
    },
    /// Rig laserline device (frozen upstream).
    RigLaserline {
        /// Input (dataset + upstream calibration).
        input: RigLaserlineDeviceInput,
        /// Solver configuration.
        config: RigLaserlineDeviceConfig,
    },
    /// Joint rig hand-eye + laser.
    RigHandeyeLaserline {
        /// Input views.
        input: RigHandeyeLaserlineInput,
        /// Solver configuration.
        config: RigHandeyeLaserlineConfig,
    },
}

impl SceneData {
    /// Solve every stage of the scene's problem with `backend`.
    pub fn set_backend(&mut self, backend: SolverBackend) {
        match self {
            SceneData::Planar { config, .. } => config.solver.backend = backend,
            SceneData::Scheimpflug { config, .. } => config.solver.backend = backend,
            SceneData::SingleCamHandeye { config, .. } => config.solver.backend = backend,
            SceneData::Laserline { config, .. } => config.solver.backend = backend,
            SceneData::Rig { config, .. } => config.solver.backend = backend,
            SceneData::RigHandeye { config, .. } => config.solver.backend = backend,
            SceneData::RigLaserline { config, .. } => config.solver.backend = backend,
            SceneData::RigHandeyeLaserline { config, .. } => {
                config.handeye.solver.backend = backend;
                config.laserline_init.solver.backend = backend;
                config.joint_optimize.solver.backend = backend;
            }
        }
    }
}

/// A generated scene.
#[derive(Debug, Clone)]
pub struct Scene {
    /// The matrix cell.
    pub spec: SceneSpec,
    /// Ground-truth parameters.
    pub truth: ModelParams,
    /// Each camera's image `[width, height]`, pixels.
    pub image_sizes: Vec<[u32; 2]>,
    /// Injected outliers (sorted by `(pose, camera, feature)`).
    pub outliers: Vec<OutlierRef>,
    /// Problem input + configuration.
    pub data: SceneData,
}

/// Which scenes a run covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// Each problem × camera variant at small scale, σ = 0.1 px, no
    /// outliers.
    Quick,
    /// The whole product.
    Full,
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Quick => "quick",
            Self::Full => "full",
        })
    }
}

/// Problem × camera combinations of the matrix: every problem with a
/// pinhole and a fisheye camera, and a tilted sensor where the scenes seed
/// one.
const VARIANTS: [(Problem, CameraKind); 16] = [
    (Problem::PlanarIntrinsics, CameraKind::Pinhole),
    (Problem::PlanarIntrinsics, CameraKind::Scheimpflug),
    (Problem::SingleCamHandeye, CameraKind::Pinhole),
    (Problem::LaserlineDevice, CameraKind::Pinhole),
    (Problem::RigExtrinsics, CameraKind::Pinhole),
    (Problem::RigExtrinsics, CameraKind::Scheimpflug),
    (Problem::RigHandeye, CameraKind::Pinhole),
    (Problem::RigLaserlineDevice, CameraKind::Pinhole),
    (Problem::RigHandeyeLaserline, CameraKind::Pinhole),
    (Problem::PlanarIntrinsics, CameraKind::Fisheye),
    (Problem::SingleCamHandeye, CameraKind::Fisheye),
    (Problem::LaserlineDevice, CameraKind::Fisheye),
    (Problem::RigExtrinsics, CameraKind::Fisheye),
    (Problem::RigHandeye, CameraKind::Fisheye),
    (Problem::RigLaserlineDevice, CameraKind::Fisheye),
    (Problem::RigHandeyeLaserline, CameraKind::Fisheye),
];

/// The scene specs of a preset, in a stable order.
pub fn scene_specs(preset: Preset) -> Vec<SceneSpec> {
    let clean = |problem, camera, scale, noise_px| SceneSpec {
        problem,
        camera,
        scale,
        noise_px,
        outliers: OutlierSpec::None,
        loss: RobustLoss::None,
    };
    match preset {
        Preset::Quick => VARIANTS
            .iter()
            .map(|&(p, s)| clean(p, s, Scale::Small, NOISE_LEVELS_PX[0]))
            .collect(),
        Preset::Full => {
            let mut out = Vec::new();
            for &(problem, camera) in &VARIANTS {
                for scale in Scale::ALL {
                    for noise_px in NOISE_LEVELS_PX {
                        out.push(clean(problem, camera, scale, noise_px));
                        if !problem.uses_target_loss() {
                            continue;
                        }
                        for loss in [
                            RobustLoss::Huber {
                                scale: OUTLIER_LOSS_SCALE,
                            },
                            RobustLoss::Cauchy {
                                scale: OUTLIER_LOSS_SCALE,
                            },
                        ] {
                            out.push(SceneSpec {
                                outliers: OutlierSpec::Injected {
                                    fraction: OUTLIER_FRACTION,
                                },
                                loss,
                                ..clean(problem, camera, scale, noise_px)
                            });
                        }
                    }
                }
            }
            out
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Outlier injection
// ─────────────────────────────────────────────────────────────────────────────

/// Choose which of `counts.iter().sum()` observations become outliers and how
/// far each is displaced.
///
/// `counts[i]` is the feature count of observation set `i` (one per
/// view × camera). Exactly `ceil(fraction · total)` distinct observations are
/// drawn with a seeded partial Fisher–Yates shuffle; the result is sorted by
/// `(set, feature)` and each entry carries its pixel displacement.
pub fn select_outliers(
    counts: &[usize],
    fraction: f64,
    seed: u64,
) -> Vec<(usize, usize, [f64; 2])> {
    let flat: Vec<(usize, usize)> = counts
        .iter()
        .enumerate()
        .flat_map(|(set, &n)| (0..n).map(move |f| (set, f)))
        .collect();
    let k = ((fraction * flat.len() as f64).ceil() as usize).min(flat.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..flat.len()).collect();
    for i in 0..k {
        let j = rng.random_range(i..idx.len());
        idx.swap(i, j);
    }
    let mut chosen: Vec<(usize, usize)> = idx[..k].iter().map(|&i| flat[i]).collect();
    chosen.sort_unstable();
    chosen
        .into_iter()
        .map(|(set, feature)| {
            let angle = rng.random_range(0.0..std::f64::consts::TAU);
            let mag = rng.random_range(OUTLIER_SHIFT_PX.0..=OUTLIER_SHIFT_PX.1);
            (set, feature, [mag * angle.cos(), mag * angle.sin()])
        })
        .collect()
}

/// Displace the selected observations of `obs` (`[view][camera]`) in place and
/// return their identities.
fn inject_outliers(
    obs: &mut [Vec<CorrespondenceView>],
    spec: &SceneSpec,
    seed: u64,
) -> Vec<OutlierRef> {
    let OutlierSpec::Injected { fraction } = spec.outliers else {
        return Vec::new();
    };
    let cameras = obs.first().map_or(0, Vec::len);
    let counts: Vec<usize> = obs
        .iter()
        .flat_map(|v| v.iter().map(|c| c.points_2d.len()))
        .collect();
    select_outliers(&counts, fraction, seed)
        .into_iter()
        .map(|(set, feature, shift)| {
            let (pose, camera) = (set / cameras, set % cameras);
            let px = &mut obs[pose][camera].points_2d[feature];
            *px = Pt2::new(px.x + shift[0], px.y + shift[1]);
            OutlierRef {
                pose,
                camera,
                feature,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene construction
// ─────────────────────────────────────────────────────────────────────────────

fn synth_camera(c: &CameraParams) -> Result<CameraModel> {
    Ok(c.build()?)
}

/// A pinhole camera with Brown-Conrady `k1`, `k2` (the rest zero) and a
/// straight sensor.
fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                k2,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy { fx, fy, cx, cy },
    }
}

/// A Kannala-Brandt fisheye camera: equidistant projection, `KannalaBrandt4`
/// coefficients `k`, straight sensor.
fn fisheye(fx: f64, fy: f64, cx: f64, cy: f64, k: [f64; 4]) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: k[0],
                k2: k[1],
                k3: k[2],
                k4: k[3],
                iters: 8,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy { fx, fy, cx, cy },
    }
}

/// `KannalaBrandt4` coefficients of the fisheye scenes' cameras: a lens
/// whose image radius grows monotonically past 70° off-axis.
const FISHEYE_KB4: [[f64; 4]; 2] = [
    [-0.015, 0.008, -0.004, 0.0008],
    [-0.012, 0.006, -0.003, 0.0005],
];

/// `camera` with a Scheimpflug sensor of tilt `params`.
fn tilted(mut camera: CameraParams, params: ScheimpflugParams) -> CameraParams {
    camera.sensor = SensorParams::Scheimpflug { params };
    camera
}

/// Shared per-scene state.
struct Builder {
    spec: SceneSpec,
    seed: u64,
    scale: ScaleSpec,
    board: Vec<Pt3>,
    extent: BoardExtent,
}

impl Builder {
    fn new(spec: &SceneSpec) -> Self {
        let scale = spec.scale.spec();
        Self {
            spec: spec.clone(),
            seed: spec.seed(),
            scale,
            board: planar::grid_points(scale.nx, scale.ny, scale.spacing),
            extent: BoardExtent::from_grid(scale.nx, scale.ny, scale.spacing),
        }
    }

    fn views(&self) -> usize {
        self.scale.views
    }

    fn fisheye(&self) -> bool {
        self.spec.camera == CameraKind::Fisheye
    }

    /// Board half extent: the offset that centres a corner-anchored board.
    fn half_extent(&self) -> (f64, f64) {
        let c = self.extent.center();
        (c[0], c[1])
    }

    /// Uniform-noise stream whose per-axis standard deviation is `noise_px`.
    fn noise(&self, stream: u64) -> UniformPixelNoise {
        UniformPixelNoise {
            seed: self.seed.wrapping_add(stream),
            max_abs_px: self.spec.noise_px * 3.0_f64.sqrt(),
        }
    }

    fn rng_seed(&self, stream: u64) -> u64 {
        self.seed.wrapping_add(0x9E37_79B9).wrapping_mul(stream | 1)
    }

    /// Board poses in front of a camera with seeded tilts and a depth ramp.
    ///
    /// Tilts of ±0.45–0.5 rad (about ±27°) make the focal length observable; a
    /// near-frontal set leaves it in a flat valley with the board distance.
    fn board_poses(&self, max_tilt: f64, z_start: f64, z_span: f64) -> Vec<Iso3> {
        let n = self.views();
        poses::tilted_board_poses(
            &poses::seeded_tilts(n, max_tilt, self.rng_seed(1)),
            self.half_extent(),
            z_start,
            z_span / n as f64,
        )
    }

    /// Noisy projections of the board, `[view]`, for one camera.
    fn observe(
        &self,
        camera: &CameraParams,
        cam_se3_target: &[Iso3],
        stream: u64,
    ) -> Result<Vec<CorrespondenceView>> {
        Ok(planar::project_views_noisy(
            &synth_camera(camera)?,
            &self.board,
            cam_se3_target,
            &self.noise(stream),
        )?)
    }

    /// Noisy projections for a rig: `[view][camera]`.
    fn observe_rig(
        &self,
        cameras: &[CameraParams],
        cam_se3_rig: &[Iso3],
        rig_se3_target: &[Iso3],
    ) -> Result<Vec<Vec<CorrespondenceView>>> {
        let per_cam: Vec<Vec<CorrespondenceView>> = cameras
            .iter()
            .zip(cam_se3_rig)
            .enumerate()
            .map(|(c, (camera, t_c_r))| {
                let poses: Vec<Iso3> = rig_se3_target.iter().map(|t| t_c_r * t).collect();
                self.observe(camera, &poses, c as u64)
            })
            .collect::<Result<_>>()?;
        Ok(transpose(per_cam))
    }

    /// Board poses (`rig_se3_target`) centred on the optical axis at
    /// `z0` whose laser stripe crosses the board in every camera.
    ///
    /// A seeded pose can leave the board nearly parallel to a laser plane,
    /// which gives no stripe; such poses are skipped. Seeded candidates are
    /// drawn in a fixed order and the first `views()` that pass are kept, so
    /// the result stays deterministic.
    fn laser_board_poses(
        &self,
        cameras: &[CameraParams],
        cam_se3_rig: &[Iso3],
        planes: &[LaserPlane],
        z0: f64,
    ) -> Result<Vec<Iso3>> {
        /// Stripe samples a view needs in every camera.
        const MIN_STRIPE_PIXELS: usize = 20;
        let n = self.views();
        let c = self.extent.center();
        let specs = poses::seeded_board_pose_specs(4 * n, LASER_MAX_TILT, self.rng_seed(1));
        let silent = UniformPixelNoise::default();
        let synth: Vec<CameraModel> = cameras.iter().map(synth_camera).collect::<Result<_>>()?;
        let kept: Vec<Iso3> = poses::centered_board_poses(&specs, (c[0], c[1]), z0)
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
                            plane.normal.as_ref(),
                            plane.distance,
                            &self.extent,
                            0,
                            &silent,
                        )
                        .len()
                            >= MIN_STRIPE_PIXELS
                    })
            })
            .take(n)
            .collect();
        if kept.len() < n {
            bail!("only {} of {n} laser views cross the board", kept.len());
        }
        Ok(kept)
    }

    /// Laser stripe pixels for one camera, `[view]`.
    fn stripes(
        &self,
        camera: &CameraParams,
        cam_se3_target: &[Iso3],
        plane: &LaserPlane,
        stream: u64,
    ) -> Result<Vec<Vec<Pt2>>> {
        let synth = synth_camera(camera)?;
        let noise = self.noise(1000 + stream);
        Ok(cam_se3_target
            .iter()
            .enumerate()
            .map(|(view, pose)| {
                laser_stripe_pixels(
                    &synth,
                    pose,
                    plane.normal.as_ref(),
                    plane.distance,
                    &self.extent,
                    view,
                    &noise,
                )
            })
            .collect())
    }

    fn outliers(&self, obs: &mut [Vec<CorrespondenceView>]) -> Vec<OutlierRef> {
        inject_outliers(obs, &self.spec, self.rng_seed(2))
    }

    /// The laser residual the scene solves with: a fisheye laser workflow
    /// needs the point-to-plane residual (a line's image is not straight).
    fn laser_residual(&self, default: LaserlineResidualType) -> LaserlineResidualType {
        if self.fisheye() {
            LaserlineResidualType::PointToPlane
        } else {
            default
        }
    }

    /// Anchor depth of the laser scenes: closer for a fisheye camera, so the
    /// board spans a wide field.
    fn laser_z0(&self) -> f64 {
        if self.fisheye() {
            FISHEYE_LASER_Z0
        } else {
            LASER_Z0
        }
    }
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

/// Laser plane in a camera frame crossing the optical axis at depth `z0`.
fn plane_at(normal: Vector3<f64>, z0: f64) -> LaserPlane {
    let n = normal.normalize();
    LaserPlane::new(n, -n.z * z0)
}

impl Scene {
    /// Generate the scene for `spec`.
    ///
    /// # Errors
    ///
    /// Fails if the spec is not a supported combination or the synthetic
    /// geometry is not projectable.
    pub fn build(spec: &SceneSpec) -> Result<Self> {
        let b = Builder::new(spec);
        let supported = match spec.camera {
            CameraKind::Pinhole | CameraKind::Fisheye => true,
            CameraKind::Scheimpflug => matches!(
                spec.problem,
                Problem::PlanarIntrinsics | Problem::RigExtrinsics
            ),
        };
        if !supported {
            bail!("unsupported combination {}", spec.id());
        }
        match spec.problem {
            Problem::PlanarIntrinsics => match spec.camera {
                CameraKind::Pinhole | CameraKind::Fisheye => planar_scene(&b),
                CameraKind::Scheimpflug => scheimpflug_scene(&b),
            },
            Problem::SingleCamHandeye => single_handeye_scene(&b),
            Problem::LaserlineDevice => laserline_scene(&b),
            Problem::RigExtrinsics => rig_scene(&b),
            Problem::RigHandeye => rig_handeye_scene(&b),
            Problem::RigLaserlineDevice => rig_laserline_scene(&b),
            Problem::RigHandeyeLaserline => rig_handeye_laserline_scene(&b),
        }
    }
}

fn planar_scene(b: &Builder) -> Result<Scene> {
    // A fisheye board comes within 0.22–0.42 m, so its corners reach about
    // 55° off-axis (more on the tilted views).
    let (truth_cam, (z_start, z_span)) = if b.fisheye() {
        (
            fisheye(420.0, 418.0, 640.0, 480.0, FISHEYE_KB4[0]),
            (0.22, 0.2),
        )
    } else {
        (
            pinhole(1000.0, 990.0, 640.0, 480.0, -0.12, 0.04),
            (0.6, 0.4),
        )
    };
    let poses = b.board_poses(0.5, z_start, z_span);
    let mut obs = transpose(vec![b.observe(&truth_cam, &poses, 0)?]);
    let outliers = b.outliers(&mut obs);
    let input = PlanarDataset::new(
        obs.into_iter()
            .map(|mut v| View::without_meta(v.remove(0)))
            .collect(),
    )?;
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model = b.spec.camera.model();
    config.solver.robust_loss = b.spec.loss;
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]],
        truth: ModelParams {
            cameras: vec![truth_cam],
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::Planar { input, config },
    })
}

fn scheimpflug_scene(b: &Builder) -> Result<Scene> {
    let truth_cam = tilted(
        pinhole(1000.0, 990.0, 640.0, 480.0, -0.12, 0.04),
        ScheimpflugParams {
            tilt_x: -0.05,
            tilt_y: 0.02,
        },
    );
    let poses = b.board_poses(0.5, 0.6, 0.4);
    let mut obs = transpose(vec![b.observe(&truth_cam, &poses, 0)?]);
    let outliers = b.outliers(&mut obs);
    let input = PlanarDataset::new(
        obs.into_iter()
            .map(|mut v| View::without_meta(v.remove(0)))
            .collect(),
    )?;
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model = b.spec.camera.model();
    config.solver.robust_loss = b.spec.loss;
    // Coarse device-spec seed: focal ~8 % low, principal point at the image
    // centre, the nominal mount tilt, no distortion.
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: 920.0,
        fy: 920.0,
        cx: 640.0,
        cy: 480.0,
    });
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: -0.04,
        tilt_y: 0.0,
    });
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]],
        truth: ModelParams {
            cameras: vec![truth_cam],
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::Scheimpflug {
            input,
            config,
            seed,
        },
    })
}

/// Hand-eye ground truth shared by the single-camera and rig scenes:
/// `gripper_se3_camera` (or `_rig`) and the target pose in the robot base.
/// A fisheye camera sees the target from about 0.3 m instead of 1 m, so the
/// board spans a wide field.
fn handeye_truth(camera: CameraKind) -> (Iso3, Iso3) {
    let target_z = match camera {
        CameraKind::Fisheye => 0.35,
        CameraKind::Pinhole | CameraKind::Scheimpflug => 1.0,
    };
    (
        poses::make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08)),
        poses::make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, target_z)),
    )
}

/// `rig_se3_target = X⁻¹ · T_B_G⁻¹ · Y` for each robot station.
fn rig_poses_from_robot(stations: &[Iso3], x: &Iso3, y: &Iso3) -> Vec<Iso3> {
    stations
        .iter()
        .map(|t_bg| (t_bg * x).inverse() * y)
        .collect()
}

fn single_handeye_scene(b: &Builder) -> Result<Scene> {
    let truth_cam = if b.fisheye() {
        fisheye(340.0, 336.0, 512.0, 384.0, FISHEYE_KB4[0])
    } else {
        pinhole(800.0, 780.0, 512.0, 384.0, 0.0, 0.0)
    };
    let (x, y) = handeye_truth(b.spec.camera);
    let stations = poses::robot_stations(b.views(), b.rng_seed(3));
    let cam_poses = rig_poses_from_robot(&stations, &x, &y);
    let mut obs = transpose(vec![b.observe(&truth_cam, &cam_poses, 0)?]);
    let outliers = b.outliers(&mut obs);
    let views: Vec<SingleCamHandeyeView> = obs
        .into_iter()
        .zip(&stations)
        .map(|(mut v, t_bg)| {
            View::new(
                v.remove(0),
                RobotPoseMeta {
                    base_se3_gripper: *t_bg,
                },
            )
        })
        .collect();
    let mut config = SingleCamHandeyeConfig::default();
    config.camera.model = b.spec.camera.model();
    config.handeye_init.handeye_mode = HandeyeMode::EyeInHand;
    config.solver.robust_loss = b.spec.loss;
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1024, 768]],
        truth: ModelParams {
            cameras: vec![truth_cam],
            handeye: Some(x),
            base_se3_target: Some(y),
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::SingleCamHandeye {
            input: SingleCamHandeyeInput::new(views)?,
            config,
        },
    })
}

/// Anchor depth of the laser-plane scenes.
const LASER_Z0: f64 = 0.5;
/// Anchor depth of the fisheye laser-plane scenes: the board's corners reach
/// about 50° off-axis.
const FISHEYE_LASER_Z0: f64 = 0.25;
/// Pitch/yaw range of the laser-plane scenes' board poses, radians: tilted
/// enough to observe the focal length, and the board centre stays on the
/// optical axis so every stripe crosses the board.
const LASER_MAX_TILT: f64 = 0.35;

fn laserline_scene(b: &Builder) -> Result<Scene> {
    let truth_cam = if b.fisheye() {
        fisheye(380.0, 380.0, 640.0, 360.0, FISHEYE_KB4[0])
    } else {
        pinhole(900.0, 900.0, 640.0, 360.0, 0.0, 0.0)
    };
    let z0 = b.laser_z0();
    let plane = plane_at(Vector3::new(0.30, 0.0, 1.0), z0);
    let cam_poses = b.laser_board_poses(
        std::slice::from_ref(&truth_cam),
        &[Iso3::identity()],
        std::slice::from_ref(&plane),
        z0,
    )?;
    let mut obs = transpose(vec![b.observe(&truth_cam, &cam_poses, 0)?]);
    let outliers = b.outliers(&mut obs);
    let stripes = b.stripes(&truth_cam, &cam_poses, &plane, 0)?;
    let views = obs
        .into_iter()
        .zip(stripes)
        .map(|(mut v, laser_pixels)| {
            View::new(
                v.remove(0),
                LaserlineMeta {
                    laser_pixels,
                    laser_weights: Vec::new(),
                },
            )
        })
        .collect();
    let mut config = LaserlineDeviceConfig::default();
    config.camera.model = b.spec.camera.model();
    config.optimize.laser_residual_type = b.laser_residual(config.optimize.laser_residual_type);
    config.optimize.calib_loss = b.spec.loss;
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 720]],
        truth: ModelParams {
            cameras: vec![truth_cam],
            planes_cam: vec![plane],
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::Laserline {
            input: views,
            config,
        },
    })
}

/// Two-camera rig intrinsics of the rig scenes (with a per-camera tilt for the
/// Scheimpflug variant).
fn rig_cameras(camera: CameraKind) -> Vec<CameraParams> {
    match camera {
        CameraKind::Pinhole => vec![
            pinhole(850.0, 845.0, 640.0, 480.0, -0.05, 0.0),
            pinhole(820.0, 815.0, 650.0, 470.0, -0.04, 0.0),
        ],
        CameraKind::Scheimpflug => vec![
            tilted(
                pinhole(850.0, 845.0, 640.0, 480.0, -0.05, 0.0),
                ScheimpflugParams::default(),
            ),
            tilted(
                pinhole(820.0, 815.0, 650.0, 470.0, -0.04, 0.0),
                ScheimpflugParams {
                    tilt_x: 0.06,
                    tilt_y: -0.04,
                },
            ),
        ],
        CameraKind::Fisheye => vec![
            fisheye(400.0, 398.0, 640.0, 480.0, FISHEYE_KB4[0]),
            fisheye(390.0, 388.0, 650.0, 470.0, FISHEYE_KB4[1]),
        ],
    }
}

fn rig_scene(b: &Builder) -> Result<Scene> {
    let camera = b.spec.camera;
    let cameras = rig_cameras(camera);
    let layout = match camera {
        CameraKind::Pinhole | CameraKind::Fisheye => poses::rig_layout(2, 0.06, 0.05),
        CameraKind::Scheimpflug => poses::rig_layout(2, 0.12, 0.0),
    };
    let (z_start, z_span) = match camera {
        CameraKind::Pinhole => (0.65, 0.3),
        CameraKind::Scheimpflug => (0.55, 0.2),
        CameraKind::Fisheye => (0.25, 0.2),
    };
    let rig_poses = b.board_poses(0.45, z_start, z_span);
    let mut obs = b.observe_rig(&cameras, &layout, &rig_poses)?;
    let outliers = b.outliers(&mut obs);
    let views: Vec<RigView<NoMeta>> = obs
        .into_iter()
        .map(|cams| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: cams.into_iter().map(Some).collect(),
            },
        })
        .collect();
    let input = RigDataset::new(views, cameras.len())?;
    let mut config = RigExtrinsicsConfig::default();
    config.solver.robust_loss = b.spec.loss;
    config.camera.model = camera.model();
    // Every camera tilted; the model's default fix lists hold all but the
    // low-order radial terms.
    if camera == CameraKind::Scheimpflug {
        config.solver.max_iters = 80;
    }
    // From a zero tilt the per-camera Scheimpflug init can settle in the
    // tilt / principal-point alias; seed each camera with its nominal mount
    // tilt (a device-spec seed, about 1° off), as the Scheimpflug intrinsics
    // scene does.
    let mut seed = RigIntrinsicsManualInit::default();
    if camera == CameraKind::Scheimpflug {
        seed.per_cam_sensors = Some(vec![
            ScheimpflugParams::default(),
            ScheimpflugParams {
                tilt_x: 0.05,
                tilt_y: -0.03,
            },
        ]);
    }
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]; 2],
        truth: ModelParams {
            cameras,
            cam_se3_rig: layout,
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::Rig {
            input,
            config,
            seed,
        },
    })
}

fn rig_handeye_scene(b: &Builder) -> Result<Scene> {
    let cameras = rig_cameras(b.spec.camera);
    let layout = poses::rig_layout(2, 0.12, 0.0);
    let (x, y) = handeye_truth(b.spec.camera);
    let stations = poses::robot_stations(b.views(), b.rng_seed(3));
    let rig_poses = rig_poses_from_robot(&stations, &x, &y);
    let mut obs = b.observe_rig(&cameras, &layout, &rig_poses)?;
    let outliers = b.outliers(&mut obs);
    let views: Vec<RigView<RobotPoseMeta>> = obs
        .into_iter()
        .zip(&stations)
        .map(|(cams, t_bg)| RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: *t_bg,
            },
            obs: RigViewObs {
                cameras: cams.into_iter().map(Some).collect(),
            },
        })
        .collect();
    let input = RigDataset::new(views, cameras.len())?;
    let mut config = RigHandeyeConfig::default();
    config.camera.model = b.spec.camera.model();
    config.solver.robust_loss = b.spec.loss;
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]; 2],
        truth: ModelParams {
            cameras,
            cam_se3_rig: layout,
            handeye: Some(x),
            base_se3_target: Some(y),
            ..ModelParams::default()
        },
        outliers,
        data: SceneData::RigHandeye { input, config },
    })
}

/// Per-camera laser planes of the rig laser scenes (camera frame).
fn rig_planes(z0: f64) -> Vec<LaserPlane> {
    vec![
        plane_at(Vector3::new(0.30, 0.0, 1.0), z0),
        plane_at(Vector3::new(0.0, 0.45, 1.0), z0),
    ]
}

/// Per-camera laser planes for a robot-moved board.
///
/// The board wanders about the view as the robot moves, so a fixed plane cuts
/// it in only some views. For each camera the plane normal is fixed and the
/// offset is the one (scanned around the mean board position) that cuts the
/// board in the most views, ties going to the offset nearest the mean.
fn planes_through_target(
    b: &Builder,
    cameras: &[CameraParams],
    layout: &[Iso3],
    rig_poses: &[Iso3],
) -> Result<Vec<LaserPlane>> {
    let c = b.extent.center();
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
            let synth = synth_camera(camera)?;
            let silent = UniformPixelNoise::default();
            let coverage = |d: f64| {
                poses
                    .iter()
                    .enumerate()
                    .filter(|(v, p)| {
                        laser_stripe_pixels(&synth, p, &n, d, &b.extent, *v, &silent).len() >= 8
                    })
                    .count()
            };
            let best = (-30..=30)
                .map(|k| d_mean + 0.005 * f64::from(k))
                .max_by_key(|&d| (coverage(d), -((d - d_mean).abs() * 1e6) as i64))
                .expect("non-empty scan");
            Ok(LaserPlane::new(n, best))
        })
        .collect()
}

fn rig_laserline_scene(b: &Builder) -> Result<Scene> {
    let cameras = rig_cameras(b.spec.camera);
    let layout = poses::rig_layout(2, 0.06, 0.0);
    let z0 = b.laser_z0();
    let planes = rig_planes(z0);
    let rig_poses = b.laser_board_poses(&cameras, &layout, &planes, z0)?;
    let obs = b.observe_rig(&cameras, &layout, &rig_poses)?;
    let views = rig_laser_views(b, &cameras, &layout, &planes, &rig_poses, obs)?;
    let dataset = RigLaserlineDataset::new(views, cameras.len())?;
    let upstream = RigUpstreamCalibration {
        cameras,
        cam_se3_rig: layout,
        rig_se3_target: rig_poses,
    };
    let mut config = RigLaserlineDeviceConfig::default();
    config.laser_residual_type = b.laser_residual(config.laser_residual_type);
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]; 2],
        // Camera and rig geometry are frozen upstream inputs, not estimates.
        truth: ModelParams {
            planes_cam: planes,
            ..ModelParams::default()
        },
        outliers: Vec::new(),
        data: SceneData::RigLaserline {
            input: RigLaserlineDeviceInput {
                dataset,
                upstream,
                initial_planes_cam: None,
            },
            config,
        },
    })
}

/// Pair per-camera target observations with laser stripes.
fn rig_laser_views(
    b: &Builder,
    cameras: &[CameraParams],
    layout: &[Iso3],
    planes: &[LaserPlane],
    rig_poses: &[Iso3],
    obs: Vec<Vec<CorrespondenceView>>,
) -> Result<Vec<RigLaserlineView>> {
    let stripes: Vec<Vec<Vec<Pt2>>> = cameras
        .iter()
        .zip(layout)
        .zip(planes)
        .enumerate()
        .map(|(c, ((camera, t_c_r), plane))| {
            let poses: Vec<Iso3> = rig_poses.iter().map(|t| t_c_r * t).collect();
            b.stripes(camera, &poses, plane, c as u64)
        })
        .collect::<Result<_>>()?;
    Ok(obs
        .into_iter()
        .enumerate()
        .map(|(v, cams)| RigLaserlineView {
            cameras: cams.into_iter().map(Some).collect(),
            laser_pixels: stripes.iter().map(|s| Some(s[v].clone())).collect(),
        })
        .collect())
}

fn rig_handeye_laserline_scene(b: &Builder) -> Result<Scene> {
    let cameras = rig_cameras(b.spec.camera);
    let layout = poses::rig_layout(2, 0.12, 0.0);
    let (x, y) = handeye_truth(b.spec.camera);
    let stations = poses::robot_stations(b.views(), b.rng_seed(3));
    let rig_poses = rig_poses_from_robot(&stations, &x, &y);
    let planes = planes_through_target(b, &cameras, &layout, &rig_poses)?;
    let mut obs = b.observe_rig(&cameras, &layout, &rig_poses)?;
    let outliers = b.outliers(&mut obs);
    let laser_views = rig_laser_views(b, &cameras, &layout, &planes, &rig_poses, obs)?;
    let views: Vec<RigHandeyeLaserlineView> = laser_views
        .into_iter()
        .zip(&stations)
        .map(|(obs, t_bg)| RigHandeyeLaserlineView {
            obs,
            meta: RobotPoseMeta {
                base_se3_gripper: *t_bg,
            },
        })
        .collect();
    let mut config = RigHandeyeLaserlineConfig::default();
    config.handeye.camera.model = b.spec.camera.model();
    config.joint_optimize.calib_loss = b.spec.loss;
    config.handeye.solver.robust_loss = b.spec.loss;
    Ok(Scene {
        spec: b.spec.clone(),
        image_sizes: vec![[1280, 960]; 2],
        truth: ModelParams {
            cameras,
            cam_se3_rig: layout,
            handeye: Some(x),
            base_se3_target: Some(y),
            planes_cam: planes,
        },
        outliers,
        data: SceneData::RigHandeyeLaserline {
            input: RigHandeyeLaserlineInput {
                views,
                num_cameras: 2,
            },
            config,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlier_selection_is_deterministic_and_exact() {
        let counts = [10, 20, 0, 30];
        let a = select_outliers(&counts, 0.05, 42);
        let b = select_outliers(&counts, 0.05, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3); // ceil(0.05 · 60)
        assert_ne!(a, select_outliers(&counts, 0.05, 43));
        // Sorted, distinct, in range, displacement magnitude in [10, 30] px.
        assert!(a.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)));
        for (set, feature, shift) in &a {
            assert!(*feature < counts[*set]);
            let mag = shift[0].hypot(shift[1]);
            assert!((10.0 - 1e-9..=30.0 + 1e-9).contains(&mag), "{mag}");
        }
    }

    #[test]
    fn scene_ids_and_seeds_are_stable_and_unique() {
        let full = scene_specs(Preset::Full);
        let mut ids: Vec<_> = full.iter().map(SceneSpec::id).collect();
        let n = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), n, "scene ids must be unique");
        assert_eq!(full[0].id(), "planar_intrinsics/pinhole/small/n0.1/clean");
        assert_eq!(full[0].seed(), full[0].seed());
        assert_eq!(
            full[18].id(),
            "planar_intrinsics/scheimpflug/small/n0.1/clean",
            "a tilted sensor is a planar scene's sensor, not a problem"
        );
        let seeds: std::collections::BTreeSet<_> = full.iter().map(SceneSpec::seed).collect();
        assert_eq!(seeds.len(), n);
        assert_eq!(scene_specs(Preset::Quick).len(), 16);
        // 14 variants with target losses × 6 (scale, noise) × 3 contaminations
        // + the two frozen-rig laser variants × 6 clean scenes.
        assert_eq!(n, 264);
    }

    #[test]
    fn problem_names_agree_across_display_cli_and_serde() {
        use clap::ValueEnum;
        assert_eq!(Problem::value_variants(), &Problem::ALL);
        for p in Problem::ALL {
            assert_eq!(Problem::from_str(p.name(), false).unwrap(), p);
            assert_eq!(serde_json::to_value(p).unwrap(), p.name());
        }
        assert!(Problem::from_str("nope", false).is_err());
        for preset in [Preset::Quick, Preset::Full] {
            assert_eq!(
                Preset::from_str(&preset.to_string(), false).unwrap(),
                preset
            );
        }
    }

    #[test]
    fn outlier_scenes_record_their_outliers() {
        let spec = SceneSpec {
            problem: Problem::PlanarIntrinsics,
            camera: CameraKind::Pinhole,
            scale: Scale::Small,
            noise_px: 0.1,
            outliers: OutlierSpec::Injected {
                fraction: OUTLIER_FRACTION,
            },
            loss: RobustLoss::Huber { scale: 1.0 },
        };
        let a = Scene::build(&spec).unwrap();
        let b = Scene::build(&spec).unwrap();
        assert_eq!(a.outliers, b.outliers);
        assert_eq!(a.outliers.len(), 26); // ceil(0.05 · 8 · 63)
    }
}
