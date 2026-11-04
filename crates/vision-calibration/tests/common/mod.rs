//! The synthetic planar calibration the facade-only problem tests share:
//! a camera, a board, its poses, deterministic noise, and the depth
//! variance of a pose block.

// Each test binary that includes this module uses a part of it.
#![allow(dead_code)]

pub mod laser;

use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    BrownConrady5, CalibratedRig, CameraParams, CorrespondenceView, FxFyCxCy, Iso3, NoMeta,
    ParameterCovariance, PinholeCamera, Pt2, Pt3, RigDataset, RigView, RigViewObs, Uncertainty,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsInput, RigExtrinsicsProblem, run_calibration,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::poses::make_iso;
use vision_calibration::synthetic::{planar, poses};

/// The pixel noise of every synthetic observation.
pub const PIXEL_SIGMA: f64 = 0.3;

/// A deterministic standard normal draw for `key`.
pub fn gauss(key: u64) -> f64 {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let uniform = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u1, u2) = (uniform(2 * key), uniform(2 * key + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The true camera, board and board poses.
pub struct Truth {
    pub camera: CameraParams,
    pub board: Vec<Pt3>,
    pub poses: Vec<Iso3>,
}

/// 8 views of a 9×6 board with 30 mm squares at about 0.45 m, seen by an
/// 820 px camera with Brown–Conrady distortion.
pub fn truth() -> Truth {
    let k = FxFyCxCy {
        fx: 820.0,
        fy: 810.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.12,
        k2: 0.03,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    Truth {
        camera: pinhole_camera_params(&make_pinhole_camera(k, distortion)),
        board: planar::grid_points(9, 6, 0.03),
        poses: poses::tilted_board_poses(
            &poses::seeded_tilts(8, 0.4, 7),
            (0.12, 0.075),
            0.45,
            0.02,
        ),
    }
}

/// The pixels of `points` seen through `camera_se3_target`, with noise of
/// σ [`PIXEL_SIGMA`] keyed by `seed` and `view`.
pub fn project(
    camera: &CameraParams,
    camera_se3_target: &Iso3,
    points: &[Pt3],
    seed: u64,
    view: usize,
) -> Vec<Pt2> {
    let model = camera.build().unwrap();
    let key = |i: usize, axis: u64| (seed << 32) + 2 * (1000 * view + i) as u64 + axis;
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q = model
                .project_point(&camera_se3_target.transform_point(p))
                .unwrap();
            Pt2::new(
                q.x + PIXEL_SIGMA * gauss(key(i, 0)),
                q.y + PIXEL_SIGMA * gauss(key(i, 1)),
            )
        })
        .collect()
}

/// Noisy pixels of every board view, from `seed`.
pub fn observe_board(truth: &Truth, seed: u64) -> Vec<Vec<Pt2>> {
    truth
        .poses
        .iter()
        .enumerate()
        .map(|(v, pose)| project(&truth.camera, pose, &truth.board, seed, v))
        .collect()
}

/// The stated variance of the translation of pose block `name` along the
/// camera's z axis.
///
/// A pose's covariance is over its right-perturbation tangent: `T·Exp(δ)`
/// moves the translation by `R·δt`, a step in the target's frame. Rotated
/// into the camera frame, the depth's variance is `r₃ᵀ Σ_t r₃` with `r₃`
/// the third row of `R`.
pub fn depth_variance(cov: &ParameterCovariance, name: &str, pose: &Iso3) -> f64 {
    let at = |c: &str| {
        let label = format!("{name}.{c}");
        cov.labels.iter().position(|l| *l == label).unwrap()
    };
    let t = [at("tx"), at("ty"), at("tz")];
    let r = pose.rotation.to_rotation_matrix();
    let row = [r[(2, 0)], r[(2, 1)], r[(2, 2)]];
    (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| row[a] * cov.covariance[t[a]][t[b]] * row[b])
        .sum()
}

/// The ratio of the scatter of `estimates` to the root mean of `variances`.
pub fn scatter_over_stated(estimates: &[f64], variances: &[f64]) -> f64 {
    let n = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let scatter = (estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let stated = (variances.iter().sum::<f64>() / n).sqrt();
    scatter / stated
}

// ---------------------------------------------------------------------------
// A synthetic stereo rig, calibrated from noisy board views
// ---------------------------------------------------------------------------

/// The two true cameras of the stereo rig.
pub fn stereo_cameras() -> [PinholeCamera; 2] {
    let make =
        |fx, fy, cx, cy| make_pinhole_camera(FxFyCxCy { fx, fy, cx, cy }, BrownConrady5::default());
    [
        make(850.0, 845.0, 640.0, 480.0),
        make(820.0, 815.0, 650.0, 470.0),
    ]
}

/// Camera 1's true pose in the rig (camera 0's) frame.
pub fn stereo_cam1_se3_rig() -> Iso3 {
    make_iso((0.0, 0.05, 0.0), (-0.12, 0.0, 0.0))
}

/// The `(pitch, yaw)` tilts of the calibration views, rad.
fn stereo_board_poses_tilts() -> [(f64, f64); 8] {
    [
        (0.00, 0.00),
        (0.16, -0.06),
        (-0.14, 0.10),
        (0.08, 0.18),
        (-0.06, -0.16),
        (0.20, 0.12),
        (-0.18, -0.10),
        (0.11, -0.20),
    ]
}

/// The calibration views: board poses with diverse tilt, 0.6 to 0.8 m away.
pub fn stereo_board_poses() -> Vec<Iso3> {
    poses::tilted_board_poses(&stereo_board_poses_tilts(), (0.105, 0.075), 0.60, 0.03)
}

/// Calibration views that fill the field: the tilts of
/// [`stereo_board_poses`] ×2.5 (up to 0.5 rad), the board moved off the
/// centre by 12 cm sideways and 9 cm up or down, or both.
///
/// The centred, mildly tilted views of [`stereo_board_poses`] barely tell a
/// camera's principal point from its rotation in the rig (σ(cx) ≈ 5 px
/// against σ(ry) ≈ 9 mrad of camera 1), and the second-order terms of those
/// large, cancelling errors make the calibration's error 2.5–3× its stated σ
/// along the two combinations the views do pin down. A test that feeds the
/// calibration's full covariance onward as a Gaussian, such as a prior,
/// needs a calibration in the regime where its covariance describes the
/// joint error, which these views give.
pub fn stereo_board_poses_spread() -> Vec<Iso3> {
    let offsets = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
    ];
    let tilts: Vec<(f64, f64)> = stereo_board_poses_tilts()
        .iter()
        .map(|&(pitch, yaw)| (2.5 * pitch, 2.5 * yaw))
        .collect();
    poses::tilted_board_poses(&tilts, (0.105, 0.075), 0.60, 0.03)
        .into_iter()
        .zip(offsets)
        .map(|(pose, (x, y))| Iso3::translation(0.12 * x, 0.09 * y, 0.0) * pose)
        .collect()
}

/// The pixels of `p` (camera 0's frame) in camera `c`, plus Gaussian noise of
/// σ [`PIXEL_SIGMA`] keyed by `key`.
pub fn stereo_noisy_pixel(cameras: &[PinholeCamera; 2], c: usize, p: &Pt3, key: u64) -> Pt2 {
    let cam_se3_rig = [Iso3::identity(), stereo_cam1_se3_rig()][c];
    let q = cameras[c].project_point(&(cam_se3_rig * p)).unwrap();
    Pt2::new(
        q.x + PIXEL_SIGMA * gauss(2 * key),
        q.y + PIXEL_SIGMA * gauss(2 * key + 1),
    )
}

/// The noisy calibration dataset of trial `seed`, with the board at `poses`
/// (`rig_se3_target`).
pub fn calibration_input(
    cameras: &[PinholeCamera; 2],
    poses: &[Iso3],
    seed: u64,
) -> RigExtrinsicsInput {
    let board = planar::grid_points(8, 6, 0.03);
    let views: Vec<RigView<NoMeta>> = poses
        .iter()
        .enumerate()
        .map(|(v, rig_se3_target)| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: (0..2)
                    .map(|c| {
                        let pixels = board
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                let key = (seed << 32) + (c * 1_000_000 + v * 1000 + i) as u64;
                                stereo_noisy_pixel(cameras, c, &(rig_se3_target * p), key)
                            })
                            .collect();
                        Some(CorrespondenceView::new(board.clone(), pixels).unwrap())
                    })
                    .collect(),
            },
        })
        .collect();
    RigDataset::new(views, 2).unwrap()
}

/// The stereo rig calibrated by the public rig-extrinsics workflow from the
/// noisy views [`stereo_board_poses`] of trial `seed`, with its covariance.
pub fn calibrated_stereo_rig(cameras: &[PinholeCamera; 2], seed: u64) -> CalibratedRig {
    calibrated_stereo_rig_from(cameras, &stereo_board_poses(), seed)
}

/// [`calibrated_stereo_rig`] from the board at `poses` (`rig_se3_target`).
pub fn calibrated_stereo_rig_from(
    cameras: &[PinholeCamera; 2],
    poses: &[Iso3],
    seed: u64,
) -> CalibratedRig {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_config(RigExtrinsicsConfig::default()).unwrap();
    session
        .set_input(calibration_input(cameras, poses, seed))
        .unwrap();
    run_calibration(&mut session).unwrap();
    session.export().unwrap().calibrated_rig().unwrap()
}

/// Ten points of a held-out capture, 0.6 to 0.8 m ahead of camera 0.
pub fn stereo_test_points() -> Vec<Pt3> {
    (0..10)
        .map(|i| {
            let (a, b) = (i as f64, (i * 7 % 10) as f64);
            Pt3::new(
                -0.09 + 0.02 * a,
                -0.07 + 0.016 * b,
                0.6 + 0.02 * ((i * 3) % 10) as f64,
            )
        })
        .collect()
}

/// The same calibration with its covariance shrunk to nothing: the σ of the
/// observations alone.
pub fn without_calibration_share(rig: &CalibratedRig) -> CalibratedRig {
    let mut covariance = rig.uncertainty().covariance().expect("covariance").clone();
    let n = covariance.labels.len();
    covariance.covariance = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1e-20 } else { 0.0 }).collect())
        .collect();
    let bare = CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::Available(covariance),
    )
    .unwrap();
    let planes: Vec<_> = (0..rig.num_cameras())
        .map(|i| rig.laser_plane(i).cloned())
        .collect();
    if planes.iter().any(Option::is_some) {
        bare.with_laser_planes(planes).unwrap()
    } else {
        bare
    }
}
