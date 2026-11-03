//! The synthetic laserline-device dataset the facade tests share: a board,
//! its poses around a laser plane, and the corners and laser-stripe pixels
//! they produce.

use vision_calibration::core::LaserPlane;
use vision_calibration::core::{
    BrownConrady5, CorrespondenceView, FxFyCxCy, Iso3, PinholeCamera, Pt2, Pt3, Vec2, View,
    make_pinhole_camera,
};
use vision_calibration::laserline_device::{LaserlineMeta, LaserlineView};
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::{laser, poses};

// Board extent in target coordinates: a 9×7 grid of 30 mm cells spans
// [0, 0.24] × [0, 0.18] m; its centre sits at (0.12, 0.09).
pub const BOARD_NX: usize = 9;
pub const BOARD_NY: usize = 7;
pub const BOARD_SPACING: f64 = 0.03;
pub const BOARD_MAX_X: f64 = (BOARD_NX - 1) as f64 * BOARD_SPACING;
pub const BOARD_MAX_Y: f64 = (BOARD_NY - 1) as f64 * BOARD_SPACING;
pub const BOARD_CX: f64 = BOARD_MAX_X / 2.0;
pub const BOARD_CY: f64 = BOARD_MAX_Y / 2.0;

pub fn board() -> Vec<Pt3> {
    vision_calibration::synthetic::planar::grid_points(BOARD_NX, BOARD_NY, BOARD_SPACING)
}

/// Six diverse board poses for a plane anchored at optical-axis depth `z0`.
/// Each board is centred on the optical axis at `z0 ± jitter` (jitter ≤
/// 25 mm) with mixed pitch/yaw/**roll**. Roll rotates the stripe *within* the
/// board without pushing it off-board, so the per-view stripes span the
/// plane's 2D extent instead of lying on a single line — the pose diversity
/// that makes the laser plane observable (see the math note,
/// §Identifiability). The stripe stays on-board because the depth spread is
/// kept small relative to `board_half_extent · sin(tilt)`.
pub fn poses(z0: f64) -> Vec<Iso3> {
    // pitch, yaw, roll (rad), depth jitter (m)
    let spec = poses::BoardPoseSpec::new;
    let specs = [
        spec(0.00, 0.00, 0.00, 0.000),
        spec(0.12, -0.05, 0.35, -0.012),
        spec(-0.10, 0.09, -0.40, 0.010),
        spec(0.06, 0.14, 0.60, 0.014),
        spec(-0.13, -0.07, -0.25, -0.014),
        spec(0.09, -0.15, 0.45, 0.008),
    ];
    poses::centered_board_poses(&specs, (BOARD_CX, BOARD_CY), z0)
}

/// On-board laser stripe for one view (see
/// [`vision_calibration::synthetic::laser`]), with deterministic pixel noise.
pub fn laser_pixels_for_view(
    camera: &PinholeCamera,
    cam_se3_target: &Iso3,
    plane: &LaserPlane,
    view_idx: usize,
    noise: &UniformPixelNoise,
) -> Vec<Pt2> {
    let board = laser::BoardExtent::from_grid(BOARD_NX, BOARD_NY, BOARD_SPACING);
    laser::laser_stripe_pixels(
        camera,
        cam_se3_target,
        plane.normal().as_ref(),
        plane.distance(),
        &board,
        view_idx,
        noise,
    )
}

/// Build a full synthetic laserline dataset (target corners + laser pixels)
/// for a given GT plane, intrinsics, and noise level.
pub fn build_dataset(
    plane: &LaserPlane,
    gt_intr: &FxFyCxCy<f64>,
    board: &[Pt3],
    poses: &[Iso3],
    noise: &UniformPixelNoise,
) -> Vec<LaserlineView> {
    let distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let camera = make_pinhole_camera(*gt_intr, distortion);

    let mut views = Vec::new();
    for (view_idx, pose) in poses.iter().enumerate() {
        let mut pts_3d = Vec::with_capacity(board.len());
        let mut pts_2d = Vec::with_capacity(board.len());
        for (pt_idx, pw) in board.iter().enumerate() {
            let p_cam = pose.transform_point(pw);
            if let Some(px) = camera.project_point(&p_cam) {
                let noisy = noise.apply(view_idx, pt_idx, Vec2::new(px.x, px.y));
                pts_3d.push(*pw);
                pts_2d.push(Pt2::new(noisy.x, noisy.y));
            }
        }
        let laser_pixels = laser_pixels_for_view(&camera, pose, plane, view_idx, noise);
        let obs = CorrespondenceView::new(pts_3d, pts_2d).expect("corner correspondences");
        views.push(View::new(
            obs,
            LaserlineMeta {
                laser_pixels,
                laser_weights: Vec::new(),
            },
        ));
    }
    views
}

/// The true camera of the laser tests.
pub fn laser_intrinsics() -> FxFyCxCy<f64> {
    FxFyCxCy {
        fx: 900.0,
        fy: 900.0,
        cx: 640.0,
        cy: 360.0,
    }
}

/// The true laser plane: 16.7 degrees about the y axis, crossing the optical
/// axis at 0.5 m.
pub fn laser_truth() -> LaserPlane {
    let n = nalgebra::Vector3::new(0.30, 0.0, 1.0).normalize();
    LaserPlane::new(n, -n.z * 0.5).unwrap()
}

/// The dataset of trial `seed`: the board corners and the laser-stripe
/// pixels, each with Gaussian noise of σ [`PIXEL_SIGMA`](super::PIXEL_SIGMA).
pub fn noisy_laser_dataset(seed: u64) -> Vec<LaserlineView> {
    let exact = UniformPixelNoise::default();
    let mut views = build_dataset(
        &laser_truth(),
        &laser_intrinsics(),
        &board(),
        &poses(0.5),
        &exact,
    );
    let key =
        |view: usize, i: usize, axis: u64| (seed << 32) + 2 * (view * 10_000 + i) as u64 + axis;
    for (v, view) in views.iter_mut().enumerate() {
        for (i, px) in view.obs.points_2d.iter_mut().enumerate() {
            px.x += super::PIXEL_SIGMA * super::gauss(key(v, i, 0));
            px.y += super::PIXEL_SIGMA * super::gauss(key(v, i, 1));
        }
        // Laser pixels draw from a key space disjoint from the corners'.
        for (i, px) in view.meta.laser_pixels.iter_mut().enumerate() {
            px.x += super::PIXEL_SIGMA * super::gauss(key(v, 5_000 + i, 0));
            px.y += super::PIXEL_SIGMA * super::gauss(key(v, 5_000 + i, 1));
        }
    }
    views
}
