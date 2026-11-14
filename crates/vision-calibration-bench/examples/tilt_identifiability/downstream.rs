//! What a camera estimate is worth downstream, in terms no choice of camera
//! frame can change.
//!
//! A tilted camera's frame is a choice (the lens axis, the sensor normal or
//! anything a fit settles on), so its estimate is judged by what it predicts,
//! never by its raw rotations:
//!
//! - **held-out reprojection**: noise-free corners of views the fit never
//!   saw, each view's pose refitted with the camera held;
//! - **rig transfer and triangulation**: a second, untilted and exactly known
//!   camera beside it; the extrinsics are solved from the training views with
//!   both cameras held, then a held-out view's pose from camera 1 predicts
//!   camera 2's pixels, and its corners are triangulated in camera 2's frame
//!   (which no fit of camera 1 can rotate);
//! - **hand-eye prediction**: camera 1 on a robot with exact poses; the
//!   hand-eye and the target pose are solved from the training views with the
//!   camera held, then predict the corners' pixels at held-out robot poses.
//!
//! Each is measured on noise-free held-out data, so it is the calibration's
//! error alone.

use anyhow::{Result, anyhow};
use nalgebra::{
    DVector, Isometry3, Matrix3, Matrix6, Point2, Point3, Quaternion, RealField, Rotation3,
    Translation3, UnitQuaternion, Vector3, Vector6,
};
use vision_calibration_core::{Iso3, PlanarDataset};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{Noise, Problem, Residual, SolveOptions, se3};

use crate::model::{Cam, perturb, project, tilt_rotation};
use crate::scene::{
    CamSpec, Coverage, Rng, Scene, ViewDesign, ViewObs, build_scene, camera_params,
};
use crate::solve::Outcome;

/// A tilted camera with its training and held-out views, a rig partner and a
/// robot.
#[derive(Clone, Debug)]
pub struct Bench {
    pub train: Scene,
    /// Held-out views over the whole image, noise-free.
    pub test: Scene,
    /// The rig's second camera: untilted, known exactly.
    pub cam2: Cam,
    pub c2_se3_c1: Iso3,
    /// Camera 2's noise-free corners of the training and held-out views.
    pub train2: Vec<ViewObs>,
    pub test2: Vec<ViewObs>,
    pub gripper_se3_cam: Iso3,
    pub base_se3_target: Iso3,
}

/// Camera 2 of the rig: 1280×960, 60° field, a mild barrel lens.
fn partner_spec() -> CamSpec {
    CamSpec {
        size: (1280.0, 960.0),
        hfov_deg: 60.0,
        corner_distortion: -0.03,
        tilt_deg: (0.0, 0.0),
        p: (0.0, 0.0),
    }
}

/// The corners of `views` (target poses `poses` in camera 1) that camera 2
/// sees, noise-free.
fn partner_views(
    cam2: &Cam,
    size: (f64, f64),
    c2_se3_c1: &Iso3,
    poses: &[Iso3],
    views: &[ViewObs],
) -> Vec<ViewObs> {
    poses
        .iter()
        .zip(views)
        .map(|(pose, v)| {
            let mut out = ViewObs {
                board: Vec::new(),
                pixels: Vec::new(),
            };
            for b in &v.board {
                let x = c2_se3_c1 * (pose * Point3::new(b[0], b[1], b[2]));
                if let Some(px) = project(&cam2.k, &cam2.d, &cam2.tilt, &x.coords)
                    && px[0] >= 0.0
                    && px[0] <= size.0
                    && px[1] >= 0.0
                    && px[1] <= size.1
                {
                    out.board.push(*b);
                    out.pixels.push(px);
                }
            }
            out
        })
        .collect()
}

impl Bench {
    pub fn new(spec: CamSpec, design: &ViewDesign) -> Result<Self> {
        let train = build_scene(spec, design)?;
        let test = build_scene(
            spec,
            &ViewDesign {
                views: 12,
                coverage: Coverage::Full,
                seed: design.seed + 1000,
                ..ViewDesign::baseline()
            },
        )?;
        let p2 = partner_spec();
        let cam2 = p2.camera();
        // Camera 2 sits 0.25 to the side (the boards are about 1 away) and
        // converges on camera 1's axis at that distance.
        let centre = Vector3::new(0.25, 0.0, 0.0);
        let axis = (Vector3::z() - centre).normalize();
        let rot = UnitQuaternion::rotation_between(&Vector3::z(), &axis)
            .ok_or_else(|| anyhow!("partner axis"))?;
        let c1_se3_c2 = Iso3::from_parts(Translation3::from(centre), rot);
        let c2_se3_c1 = c1_se3_c2.inverse();
        let train2 = partner_views(&cam2, p2.size, &c2_se3_c1, &train.poses, &train.views);
        let test2 = partner_views(&cam2, p2.size, &c2_se3_c1, &test.poses, &test.views);
        let gripper_se3_cam = Iso3::from_parts(
            Translation3::new(0.05, -0.03, 0.12),
            UnitQuaternion::from_scaled_axis(Vector3::new(1.0, 1.0, 0.0).normalize() * 0.35),
        );
        let base_se3_target = Iso3::from_parts(
            Translation3::new(0.5, 0.2, 0.0),
            UnitQuaternion::from_scaled_axis(Vector3::new(2.97, 0.1, 0.0)),
        );
        Ok(Self {
            train,
            test,
            cam2,
            c2_se3_c1,
            train2,
            test2,
            gripper_se3_cam,
            base_se3_target,
        })
    }

    /// Camera 2's training corners with pixel noise `sigma`.
    pub fn partner_data(&self, sigma: f64, seed: u64) -> Vec<ViewObs> {
        let mut rng = Rng::new(seed ^ 0xC2C2);
        self.train2
            .iter()
            .map(|v| ViewObs {
                board: v.board.clone(),
                pixels: v
                    .pixels
                    .iter()
                    .map(|p| [p[0] + sigma * rng.normal(), p[1] + sigma * rng.normal()])
                    .collect(),
            })
            .collect()
    }

    /// The robot's `base_se3_gripper` that puts the target at `cam_se3_target`.
    fn robot_pose(&self, cam_se3_target: &Iso3) -> Iso3 {
        self.base_se3_target * cam_se3_target.inverse() * self.gripper_se3_cam.inverse()
    }
}

/// The values of a camera at scalar type `T`.
fn cam_t<T: RealField>(cam: &Cam) -> (Vec<T>, Vec<T>, Vec<T>) {
    let c = |x: &f64| T::from_f64(*x).unwrap();
    (
        cam.k.iter().map(c).collect(),
        cam.d.iter().map(c).collect(),
        cam.tilt.iter().map(c).collect(),
    )
}

fn iso_t<T: RealField>(iso: &Iso3) -> Isometry3<T> {
    let q = iso.rotation.quaternion();
    let t = iso.translation.vector;
    let c = |x: f64| T::from_f64(x).unwrap();
    Isometry3::from_parts(
        Translation3::new(c(t.x), c(t.y), c(t.z)),
        UnitQuaternion::new_unchecked(Quaternion::new(c(q.w), c(q.i), c(q.j), c(q.k))),
    )
}

/// How a corner reaches the held camera.
#[derive(Clone)]
enum Chain {
    /// Camera 2 of the rig: blocks `[c2_se3_c1, c1_se3_target]`.
    Rig,
    /// Camera 1: block `[c1_se3_target]`.
    Direct,
    /// Camera 1 on the robot: blocks `[cam_se3_gripper, base_se3_target]`,
    /// the view's `gripper_se3_base` known.
    HandEye { gripper_se3_base: Iso3 },
}

/// One corner seen by a held camera.
struct HeldCorner {
    cam: Cam,
    chain: Chain,
    pw: [f64; 3],
    uv: [f64; 2],
}

impl Residual for HeldCorner {
    fn dim(&self) -> usize {
        2
    }

    fn block_dims(&self) -> Vec<usize> {
        match self.chain {
            Chain::Direct => vec![7],
            _ => vec![7, 7],
        }
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let c = |x: f64| T::from_f64(x).unwrap();
        let pw = Point3::new(c(self.pw[0]), c(self.pw[1]), c(self.pw[2]));
        let x = match &self.chain {
            Chain::Direct => se3(&blocks[0]) * pw,
            Chain::Rig => se3(&blocks[0]) * (se3(&blocks[1]) * pw),
            Chain::HandEye { gripper_se3_base } => {
                se3(&blocks[0]) * (iso_t::<T>(gripper_se3_base) * (se3(&blocks[1]) * pw))
            }
        };
        let (k, d, t) = cam_t::<T>(&self.cam);
        match project(&k, &d, &t, &x.coords) {
            Some([u, v]) => DVector::from_vec(vec![u - c(self.uv[0]), v - c(self.uv[1])]),
            None => DVector::from_element(2, c(f64::NAN)),
        }
    }
}

/// The rotation from the true lens-axis frame to a fit's camera frame: the
/// best rotation `R` with `fit_i ≈ R·true_i` over the views.
pub fn frame_rotation(fit: &[Iso3], truth: &[Iso3]) -> Rotation3<f64> {
    let mut m = Matrix3::zeros();
    for (a, b) in fit.iter().zip(truth) {
        m += a.rotation.to_rotation_matrix().matrix()
            * b.rotation.to_rotation_matrix().matrix().transpose();
    }
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u.expect("requested"), svd.v_t.expect("requested"));
    let mut d = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Rotation3::from_matrix_unchecked(u * d * v_t)
}

/// A pose refitted to noise-free corners with the camera held
/// (Levenberg–Marquardt on central differences).
pub fn refit_pose(cam: &Cam, board: &[[f64; 3]], uv: &[[f64; 2]], init: Iso3) -> Iso3 {
    let resid = |pose: &Iso3, out: &mut Vec<f64>| {
        out.clear();
        for (b, p) in board.iter().zip(uv) {
            let x = pose * Point3::new(b[0], b[1], b[2]);
            match project(&cam.k, &cam.d, &cam.tilt, &x.coords) {
                Some(px) => out.extend_from_slice(&[px[0] - p[0], px[1] - p[1]]),
                None => out.extend_from_slice(&[1e3, 1e3]),
            }
        }
    };
    let cost = |r: &[f64]| r.iter().map(|x| x * x).sum::<f64>();
    let mut pose = init;
    let (mut r, mut rp, mut rm) = (Vec::new(), Vec::new(), Vec::new());
    resid(&pose, &mut r);
    let mut c0 = cost(&r);
    let mut lambda = 1e-4;
    let depth = pose.translation.vector.norm().max(1e-3);
    for _ in 0..60 {
        let n = r.len();
        let mut jac = vec![[0.0; 6]; n];
        for k in 0..6 {
            let h = if k < 3 { 1e-6 } else { 1e-6 * depth };
            let mut d = [0.0; 6];
            d[k] = h;
            resid(&perturb(&pose, &d), &mut rp);
            d[k] = -h;
            resid(&perturb(&pose, &d), &mut rm);
            for i in 0..n {
                jac[i][k] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        for i in 0..n {
            let row = Vector6::from_row_slice(&jac[i]);
            jtj += row * row.transpose();
            jtr += row * r[i];
        }
        let mut improved = false;
        for _ in 0..10 {
            let mut a = jtj;
            for k in 0..6 {
                a[(k, k)] *= 1.0 + lambda;
            }
            let Some(step) = a.lu().solve(&(-jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let delta = [step[0], step[1], step[2], step[3], step[4], step[5]];
            let cand = perturb(&pose, &delta);
            resid(&cand, &mut rp);
            let c1 = cost(&rp);
            if c1 < c0 {
                let small = step.norm() < 1e-12;
                pose = cand;
                std::mem::swap(&mut r, &mut rp);
                let rel = (c0 - c1) / c0.max(1e-300);
                c0 = c1;
                lambda = (lambda * 0.3).max(1e-9);
                improved = !small && rel > 1e-14;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    pose
}

/// A camera estimate's errors downstream; see the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    /// Held-out reprojection RMS after a pose refit, px.
    pub heldout: f64,
    /// Rig transfer RMS into camera 2, px.
    pub transfer: f64,
    /// Rig triangulation RMS in camera 2's frame, per mille of the distance.
    pub triangulation: f64,
    /// Hand-eye prediction RMS at held-out robot poses, px.
    pub handeye: f64,
    /// Rotation of the fit's camera frame from the lens-axis frame, degrees.
    pub frame_lens_deg: f64,
    /// Rotation of the fit's camera frame from the sensor-normal frame,
    /// degrees.
    pub frame_sensor_deg: f64,
}

impl Metrics {
    pub const NAMES: [&'static str; 4] = [
        "held-out px",
        "rig transfer px",
        "triangulation ‰",
        "hand-eye px",
    ];

    pub fn values(&self) -> [f64; 4] {
        [
            self.heldout,
            self.transfer,
            self.triangulation,
            self.handeye,
        ]
    }
}

fn rms(sum_sq: f64, n: usize) -> f64 {
    if n == 0 {
        f64::NAN
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// The midpoint of two rays `o1 + s·d1`, `t·d2` (camera 2's centre at the
/// origin).
fn midpoint(o1: &Vector3<f64>, d1: &Vector3<f64>, d2: &Vector3<f64>) -> Option<Vector3<f64>> {
    // Minimize |o1 + s d1 − t d2|².
    let (a, b, c) = (d1.dot(d1), d1.dot(d2), d2.dot(d2));
    let (d, e) = (d1.dot(o1), d2.dot(o1));
    let den = a * c - b * b;
    if den.abs() < 1e-14 {
        return None;
    }
    let s = (b * e - c * d) / den;
    let t = (a * e - b * d) / den;
    Some(0.5 * ((o1 + d1 * s) + d2 * t))
}

/// Evaluate a fitted camera 1 (`fit`, with its training poses) given the
/// training corners it was fitted to (`data1`) and camera 2's (`data2`).
pub fn evaluate(
    bench: &Bench,
    fit: &Outcome,
    data1: &PlanarDataset,
    data2: &[ViewObs],
) -> Result<Metrics> {
    let cam = fit.cam;
    let frame = frame_rotation(&fit.poses, &bench.train.poses);
    let frame_iso = Iso3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_rotation_matrix(&frame),
    );
    let truth = &bench.train.cam;
    let r_tilt = Rotation3::from_matrix_unchecked(tilt_rotation(truth.tilt[0], truth.tilt[1]));
    // Angles through the quaternion: `Rotation3::angle` is NaN at rounding
    // distance from the identity.
    let angle = |r: Rotation3<f64>| {
        UnitQuaternion::from_rotation_matrix(&r)
            .angle()
            .to_degrees()
    };
    let frame_lens_deg = angle(frame);
    let frame_sensor_deg = angle(frame * r_tilt.inverse());

    // Held-out reprojection.
    let mut test_poses = Vec::new();
    let (mut ss, mut n) = (0.0, 0usize);
    for (pose, view) in bench.test.poses.iter().zip(&bench.test.views) {
        let p = refit_pose(&cam, &view.board, &view.pixels, frame_iso * pose);
        for (b, px) in view.board.iter().zip(&view.pixels) {
            let x = p * Point3::new(b[0], b[1], b[2]);
            let q = project(&cam.k, &cam.d, &cam.tilt, &x.coords).unwrap_or([f64::NAN; 2]);
            ss += (q[0] - px[0]).powi(2) + (q[1] - px[1]).powi(2);
            n += 1;
        }
        test_poses.push(p);
    }
    let heldout = rms(ss, n);

    // The rig: extrinsics and the training poses with both cameras held.
    let mut opts = SolveOptions::default();
    opts.max_iters = 200;
    let mut problem = Problem::new();
    let ext = problem.add_pose("c2_se3_c1", &(bench.c2_se3_c1 * frame_iso.inverse()))?;
    let mut pose_ids = Vec::new();
    for (i, p) in fit.poses.iter().enumerate() {
        pose_ids.push(problem.add_pose(format!("pose{i}"), p)?);
    }
    for (vi, view) in data1.views.iter().enumerate() {
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            problem.add_residual(
                HeldCorner {
                    cam,
                    chain: Chain::Direct,
                    pw: [p3.x, p3.y, p3.z],
                    uv: [p2.x, p2.y],
                },
                &[pose_ids[vi]],
                RobustLoss::None,
                Noise::Estimated("px".into()),
            )?;
        }
        for (b, px) in data2[vi].board.iter().zip(&data2[vi].pixels) {
            problem.add_residual(
                HeldCorner {
                    cam: bench.cam2,
                    chain: Chain::Rig,
                    pw: *b,
                    uv: *px,
                },
                &[ext, pose_ids[vi]],
                RobustLoss::None,
                Noise::Estimated("px".into()),
            )?;
        }
    }
    let sol = vision_calibration_optim::problem::solve(&problem, &opts)?;
    let c2_se3_c1 = sol.pose(ext).expect("own block");
    let (mut ss_t, mut n_t, mut ss_x, mut n_x) = (0.0, 0usize, 0.0, 0usize);
    let model1 = crate::scene::camera_params(&cam).build()?;
    let model2 = camera_params(&bench.cam2).build()?;
    let o1 = c2_se3_c1.translation.vector;
    for (vi, view2) in bench.test2.iter().enumerate() {
        let p1 = test_poses[vi];
        let true_pose = bench.test.poses[vi];
        let view1 = &bench.test.views[vi];
        for (b, px2) in view2.board.iter().zip(&view2.pixels) {
            let pw = Point3::new(b[0], b[1], b[2]);
            let x2 = c2_se3_c1 * (p1 * pw);
            let q = project(&bench.cam2.k, &bench.cam2.d, &bench.cam2.tilt, &x2.coords)
                .unwrap_or([f64::NAN; 2]);
            ss_t += (q[0] - px2[0]).powi(2) + (q[1] - px2[1]).powi(2);
            n_t += 1;
            // Triangulate when camera 1 sees the corner too.
            if let Some(j) = view1.board.iter().position(|c| c == b) {
                let px1 = view1.pixels[j];
                let d1 = c2_se3_c1.rotation
                    * model1
                        .backproject_pixel(&Point2::new(px1[0], px1[1]))
                        .direction;
                let d2 = model2
                    .backproject_pixel(&Point2::new(px2[0], px2[1]))
                    .direction;
                if let Some(x) = midpoint(&o1, &d1, &d2) {
                    let truth_x = bench.c2_se3_c1 * (true_pose * pw);
                    ss_x += ((x - truth_x.coords).norm() / truth_x.coords.norm() * 1e3).powi(2);
                    n_x += 1;
                }
            }
        }
    }

    // Hand-eye with exact robot poses.
    let mut problem = Problem::new();
    let cam_se3_gripper = problem.add_pose(
        "cam_se3_gripper",
        &(frame_iso * bench.gripper_se3_cam.inverse()),
    )?;
    let base_se3_target = problem.add_pose("base_se3_target", &bench.base_se3_target)?;
    for (vi, view) in data1.views.iter().enumerate() {
        let gripper_se3_base = bench.robot_pose(&bench.train.poses[vi]).inverse();
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            problem.add_residual(
                HeldCorner {
                    cam,
                    chain: Chain::HandEye { gripper_se3_base },
                    pw: [p3.x, p3.y, p3.z],
                    uv: [p2.x, p2.y],
                },
                &[cam_se3_gripper, base_se3_target],
                RobustLoss::None,
                Noise::Estimated("px".into()),
            )?;
        }
    }
    let sol = vision_calibration_optim::problem::solve(&problem, &opts)?;
    let c_g = sol.pose(cam_se3_gripper).expect("own block");
    let b_t = sol.pose(base_se3_target).expect("own block");
    let (mut ss_h, mut n_h) = (0.0, 0usize);
    for (pose, view) in bench.test.poses.iter().zip(&bench.test.views) {
        let g_b = bench.robot_pose(pose).inverse();
        for (b, px) in view.board.iter().zip(&view.pixels) {
            let x = c_g * (g_b * (b_t * Point3::new(b[0], b[1], b[2])));
            let q = project(&cam.k, &cam.d, &cam.tilt, &x.coords).unwrap_or([f64::NAN; 2]);
            ss_h += (q[0] - px[0]).powi(2) + (q[1] - px[1]).powi(2);
            n_h += 1;
        }
    }
    Ok(Metrics {
        heldout,
        transfer: rms(ss_t, n_t),
        triangulation: rms(ss_x, n_x),
        handeye: rms(ss_h, n_h),
        frame_lens_deg,
        frame_sensor_deg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FreeSet;
    use crate::solve::{backend_opts, solve_direct};
    use vision_calibration_optim::SolverBackend;

    /// The true camera, fitted to noise-free data, predicts every held-out
    /// quantity exactly; the frame is the lens-axis frame.
    #[test]
    fn truth_predicts_exactly() {
        let spec = CamSpec {
            tilt_deg: (8.0, -3.0),
            ..CamSpec::baseline()
        };
        let bench = Bench::new(spec, &ViewDesign::baseline()).unwrap();
        let data = bench.train.dataset(0.0, 1);
        let fit = solve_direct(
            &data,
            &bench.train.cam,
            &bench.train.poses,
            FreeSet::Default,
            None,
            backend_opts(SolverBackend::TinySolver, 50, true),
        )
        .unwrap();
        let m = evaluate(&bench, &fit, &data, &bench.partner_data(0.0, 1)).unwrap();
        assert!(m.heldout < 1e-6, "{m:?}");
        assert!(m.transfer < 1e-6 && m.handeye < 1e-6, "{m:?}");
        assert!(m.triangulation < 1e-6, "{m:?}");
        assert!(m.frame_lens_deg < 1e-6, "{m:?}");
        assert!((m.frame_sensor_deg - 8.5).abs() < 0.2, "{m:?}");
    }
}
