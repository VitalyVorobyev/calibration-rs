use crate::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};
use calib_core::{
    Iso3, PinholeCamera, Pt3, Vec2, Real,
    RadialTangential, CameraIntrinsics,
};
use nalgebra::{DMatrix, DVector, Point3, Vector3, UnitQuaternion};

/// Observations for a single image/view of the planar target.
#[derive(Debug, Clone)]
pub struct PlanarViewObservations {
    /// 3D points in board coordinates (typically z = 0).
    pub points_3d: Vec<Pt3>,
    /// Corresponding detected image points (pixels).
    pub points_2d: Vec<Vec2>,
}

impl PlanarViewObservations {
    pub fn new(points_3d: Vec<Pt3>, points_2d: Vec<Vec2>) -> Self {
        assert_eq!(
            points_3d.len(),
            points_2d.len(),
            "3D / 2D point counts must match"
        );
        Self { points_3d, points_2d }
    }

    pub fn len(&self) -> usize {
        self.points_3d.len()
    }
}

/// Non-linear refinement problem for planar intrinsics (and per-view poses).
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsProblem {
    pub views: Vec<PlanarViewObservations>,
}

impl PlanarIntrinsicsProblem {
    pub fn new(views: Vec<PlanarViewObservations>) -> Self {
        assert!(
            !views.is_empty(),
            "need at least one view for calibration"
        );
        for (i, v) in views.iter().enumerate() {
            assert!(
                v.len() >= 4,
                "view {} has too few points (need >=4)",
                i
            );
        }
        Self { views }
    }

    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    pub fn param_dim(&self) -> usize {
        10 + 6 * self.num_views()
    }

    pub fn residual_dim(&self) -> usize {
        self.views.iter().map(|v| 2 * v.len()).sum()
    }
}

/// Pack initial intrinsics, distortion and poses into parameter vector.
pub fn pack_initial_params(
    camera: &PinholeCamera,
    poses_board_to_cam: &[Iso3],
) -> DVector<Real> {
    assert!(
        !poses_board_to_cam.is_empty(),
        "need at least one pose"
    );
    let n_views = poses_board_to_cam.len();
    let dim = 10 + 6 * n_views;
    let mut x = DVector::zeros(dim);

    let k = &camera.intrinsics;
    x[0] = k.fx;
    x[1] = k.fy;
    x[2] = k.cx;
    x[3] = k.cy;
    x[4] = k.skew;

    let (k1, k2, p1, p2, k3) = match camera.distortion {
        Some(RadialTangential::BrownConrady {
            k1, k2, p1, p2, k3,
        }) => (k1, k2, p1, p2, k3),
        None => (0.0, 0.0, 0.0, 0.0, 0.0),
    };

    x[5] = k1;
    x[6] = k2;
    x[7] = p1;
    x[8] = p2;
    x[9] = k3;

    for (i, pose) in poses_board_to_cam.iter().enumerate() {
        let idx = 10 + 6 * i;

        // axis-angle from rotation
        let axis_angle = pose.rotation.scaled_axis();
        x[idx + 0] = axis_angle.x;
        x[idx + 1] = axis_angle.y;
        x[idx + 2] = axis_angle.z;

        let t = pose.translation.vector;
        x[idx + 3] = t.x;
        x[idx + 4] = t.y;
        x[idx + 5] = t.z;
    }

    x
}

/// Helper: decode parameter vector into camera + per-view poses.
fn decode_params(
    prob: &PlanarIntrinsicsProblem,
    x: &DVector<Real>,
) -> (PinholeCamera, Vec<Iso3>) {
    let n_views = prob.num_views();
    assert_eq!(x.len(), 10 + 6 * n_views);

    let fx = x[0];
    let fy = x[1];
    let cx = x[2];
    let cy = x[3];
    let skew = x[4];

    let k1 = x[5];
    let k2 = x[6];
    let p1 = x[7];
    let p2 = x[8];
    let k3 = x[9];

    let intrinsics = CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        skew,
    };
    let distortion = Some(RadialTangential::BrownConrady {
        k1,
        k2,
        p1,
        p2,
        k3,
    });

    let camera = PinholeCamera {
        intrinsics,
        distortion,
    };

    let mut poses = Vec::with_capacity(n_views);
    for i in 0..n_views {
        let idx = 10 + 6 * i;
        let wx = x[idx + 0];
        let wy = x[idx + 1];
        let wz = x[idx + 2];
        let tx = x[idx + 3];
        let ty = x[idx + 4];
        let tz = x[idx + 5];

        let axis_angle = Vector3::new(wx, wy, wz);
        let rq = UnitQuaternion::from_scaled_axis(axis_angle);
        let trans = Vector3::new(tx, ty, tz);

        let iso = Iso3::from_parts(trans.into(), rq);
        poses.push(iso);
    }

    (camera, poses)
}

impl NllsProblem for PlanarIntrinsicsProblem {
    fn residuals(&self, x: &DVector<Real>) -> DVector<Real> {
        let (camera, poses) = decode_params(self, x);

        let mut r = DVector::zeros(self.residual_dim());
        let mut offset = 0;

        for (view_idx, view) in self.views.iter().enumerate() {
            let pose = &poses[view_idx];

            for j in 0..view.points_3d.len() {
                let pw = view.points_3d[j];
                let z0 = pw.z;
                debug_assert!(
                    z0.abs() < 1e-9,
                    "planar assumption: z ≈ 0"
                );

                // board → camera
                let p_cam: Point3<Real> = pose.transform_point(&pw);
                let proj = camera.project(&p_cam);

                let meas = view.points_2d[j];
                r[offset + 0] = meas.x - proj.x;
                r[offset + 1] = meas.y - proj.y;

                offset += 2;
            }
        }

        r
    }

    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real> {
        // Finite differences for now; replace with analytic Jacobian later.
        let m = self.residual_dim();
        let n = x.len();
        let mut j = DMatrix::zeros(m, n);

        let base_r = self.residuals(x);
        let eps = 1e-6;

        for k in 0..n {
            let mut x_pert = x.clone();
            x_pert[k] += eps;
            let r_plus = self.residuals(&x_pert);
            let diff = (r_plus - &base_r) / eps;
            j.set_column(k, &diff);
        }

        j
    }
}

/// High-level API: refine camera intrinsics & per-view poses.
///
/// Returns (refined_camera, refined_poses, report).
pub fn refine_planar_intrinsics<B: NllsSolverBackend>(
    backend: &B,
    problem: &PlanarIntrinsicsProblem,
    initial_params: DVector<Real>,
    opts: &SolveOptions,
) -> (PinholeCamera, Vec<Iso3>, SolveReport) {
    assert_eq!(
        initial_params.len(),
        problem.param_dim(),
        "initial parameter vector has wrong dimension"
    );

    let (x_opt, report) = backend.solve(problem, initial_params, opts);
    let (camera, poses) = decode_params(problem, &x_opt);
    (camera, poses, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_lm::LmBackend;
    use calib_core::{Pt2, Pt3};

    #[test]
    fn synthetic_planar_intrinsics_refinement() {
        // Ground-truth camera
        let k_gt = CameraIntrinsics {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = RadialTangential::BrownConrady {
            k1: -0.1,
            k2: 0.01,
            p1: 0.001,
            p2: -0.001,
            k3: 0.0,
        };
        let cam_gt = PinholeCamera {
            intrinsics: k_gt,
            distortion: Some(dist_gt),
        };

        // Simple 6x4 board with 30mm spacing on the plane z=0.
        let nx = 6;
        let ny = 4;
        let spacing = 0.03_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * spacing;
                let y = j as f64 * spacing;
                board_points.push(Pt3::new(x, y, 0.0));
            }
        }

        // Two synthetic views with different poses.
        let mut views = Vec::new();
        let mut poses_gt = Vec::new();

        for view_idx in 0..2 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rq = UnitQuaternion::from_scaled_axis(axis * angle);
            let rot = rq.to_rotation_matrix();
            let trans = Vector3::new(0.0, 0.0, 0.5 + 0.2 * view_idx as f64);
            let pose = Iso3::from_parts(trans.into(), rot);

            poses_gt.push(pose);

            let mut img_points = Vec::new();
            for pw in &board_points {
                let p_cam = pose.transform_point(pw);
                let proj = cam_gt.project(&p_cam);
                img_points.push(proj);
            }

            views.push(PlanarViewObservations::new(
                board_points.clone(),
                img_points,
            ));
        }

        let problem = PlanarIntrinsicsProblem::new(views);

        // Initial guess: slightly wrong intrinsics, no distortion, poses = GT.
        let cam_init = PinholeCamera {
            intrinsics: CameraIntrinsics {
                fx: 780.0,
                fy: 760.0,
                cx: 630.0,
                cy: 350.0,
                skew: 0.0,
            },
            distortion: None,
        };

        let x0 = pack_initial_params(&cam_init, &poses_gt);
        let backend = LmBackend::default();
        let opts = SolveOptions::default();

        let (cam_refined, poses_refined, report) =
            refine_planar_intrinsics(&backend, &problem, x0, &opts);

        println!("LM report: {:?}", report);
        println!("Refined intrinsics: {:?}", cam_refined.intrinsics);

        // Very simple checks – just ensure we're close to GT.
        assert!((cam_refined.intrinsics.fx - k_gt.fx).abs() < 5.0);
        assert!((cam_refined.intrinsics.fy - k_gt.fy).abs() < 5.0);
        assert!((cam_refined.intrinsics.cx - k_gt.cx).abs() < 5.0);
        assert!((cam_refined.intrinsics.cy - k_gt.cy).abs() < 5.0);

        assert!(report.converged);
        assert!(report.final_cost < 1e-6);
        assert_eq!(poses_refined.len(), poses_gt.len());
    }
}
