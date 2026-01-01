use crate::robust::RobustKernel;
use crate::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};
use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2,
};
use nalgebra::{DMatrix, DVector, Point3, UnitQuaternion, Vector3};

pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

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
        Self {
            points_3d,
            points_2d,
        }
    }

    pub fn len(&self) -> usize {
        self.points_3d.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points_3d.is_empty()
    }
}

/// Non-linear refinement problem for planar intrinsics (and per-view poses).
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsProblem {
    pub views: Vec<PlanarViewObservations>,
    pub robust_kernel: RobustKernel,
}

impl PlanarIntrinsicsProblem {
    pub fn new(views: Vec<PlanarViewObservations>) -> Self {
        assert!(!views.is_empty(), "need at least one view for calibration");
        for (i, v) in views.iter().enumerate() {
            assert!(v.len() >= 4, "view {} has too few points (need >=4)", i);
        }
        Self {
            views,
            robust_kernel: RobustKernel::None,
        }
    }

    pub fn with_kernel(mut self, kernel: RobustKernel) -> Self {
        self.robust_kernel = kernel;
        self
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

    fn eval_residuals_unweighted_into(&self, x: &DVector<Real>, out: &mut DVector<Real>) {
        let (camera, poses) = decode_params(self, x);
        debug_assert_eq!(out.len(), self.residual_dim());

        let mut offset = 0;
        for (view_idx, view) in self.views.iter().enumerate() {
            let pose = &poses[view_idx];

            for (pw, meas) in view.points_3d.iter().zip(view.points_2d.iter()) {
                let z0 = pw.z;
                debug_assert!(z0.abs() < 1e-9, "planar assumption: z ≈ 0");

                // board -> camera
                let p_cam: Point3<Real> = pose.transform_point(pw);
                let Some(proj) = camera.project_point(&p_cam) else {
                    out[offset] = 1.0e6;
                    out[offset + 1] = 1.0e6;
                    offset += 2;
                    continue;
                };

                let ru = meas.x - proj.x;
                let rv = meas.y - proj.y;
                out[offset] = ru;
                out[offset + 1] = rv;
                offset += 2;
            }
        }

        debug_assert_eq!(offset, out.len());
    }

    fn residuals_unweighted_impl(&self, x: &DVector<Real>) -> DVector<Real> {
        let mut r = DVector::zeros(self.residual_dim());
        self.eval_residuals_unweighted_into(x, &mut r);
        r
    }

    fn robust_row_scales_from_unweighted(&self, r_unweighted: &DVector<Real>) -> DVector<Real> {
        debug_assert_eq!(r_unweighted.len(), self.residual_dim());
        let mut scales = DVector::from_element(r_unweighted.len(), 1.0);
        if matches!(self.robust_kernel, RobustKernel::None) {
            return scales;
        }

        let mut offset = 0;
        for view in &self.views {
            for _ in 0..view.points_3d.len() {
                let ru = r_unweighted[offset];
                let rv = r_unweighted[offset + 1];
                let r2 = ru * ru + rv * rv;
                let s = self.robust_kernel.sqrt_weight(r2);
                scales[offset] = s;
                scales[offset + 1] = s;
                offset += 2;
            }
        }

        debug_assert_eq!(offset, scales.len());
        scales
    }

    fn residuals_weighted(&self, x: &DVector<Real>) -> DVector<Real> {
        let mut r = self.residuals_unweighted_impl(x);
        let scales = self.robust_row_scales_from_unweighted(&r);
        r.component_mul_assign(&scales);
        r
    }

    fn jacobian_unweighted_fd(&self, x: &DVector<Real>) -> DMatrix<Real> {
        let m = self.residual_dim();
        let n = x.len();
        let mut j = DMatrix::zeros(m, n);

        let base_r = self.residuals_unweighted_impl(x);
        let mut x_pert = x.clone();
        let mut r_plus = DVector::zeros(m);
        let mut diff = DVector::zeros(m);

        for k in 0..n {
            let orig = x_pert[k];
            let eps = 1e-6 * (1.0 + orig.abs());
            x_pert[k] = orig + eps;

            self.eval_residuals_unweighted_into(&x_pert, &mut r_plus);
            diff.copy_from(&r_plus);
            diff -= &base_r;
            diff /= eps;
            j.set_column(k, &diff);

            x_pert[k] = orig;
        }

        j
    }

    fn jacobian_weighted(&self, x: &DVector<Real>) -> DMatrix<Real> {
        let r_unweighted = self.residuals_unweighted_impl(x);
        let scales = self.robust_row_scales_from_unweighted(&r_unweighted);
        let mut j = self.jacobian_unweighted_fd(x);
        debug_assert_eq!(scales.len(), j.nrows());
        for (mut row, scale) in j.row_iter_mut().zip(scales.iter()) {
            if *scale != 1.0 {
                row.scale_mut(*scale);
            }
        }
        j
    }
}

/// Pack initial intrinsics, distortion and poses into parameter vector.
pub fn pack_initial_params(camera: &PinholeCamera, poses_board_to_cam: &[Iso3]) -> DVector<Real> {
    assert!(!poses_board_to_cam.is_empty(), "need at least one pose");
    let n_views = poses_board_to_cam.len();
    let dim = 10 + 6 * n_views;
    let mut x = DVector::zeros(dim);

    let k = &camera.k;
    x[0] = k.fx;
    x[1] = k.fy;
    x[2] = k.cx;
    x[3] = k.cy;
    x[4] = k.skew;

    let dist = &camera.dist;
    x[5] = dist.k1;
    x[6] = dist.k2;
    x[7] = dist.p1;
    x[8] = dist.p2;
    x[9] = dist.k3;

    for (i, pose) in poses_board_to_cam.iter().enumerate() {
        let idx = 10 + 6 * i;

        // axis-angle from rotation
        let axis_angle = pose.rotation.scaled_axis();
        x[idx] = axis_angle.x;
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
fn decode_params(prob: &PlanarIntrinsicsProblem, x: &DVector<Real>) -> (PinholeCamera, Vec<Iso3>) {
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

    let intrinsics = FxFyCxCySkew {
        fx,
        fy,
        cx,
        cy,
        skew,
    };
    let distortion = BrownConrady5 {
        k1,
        k2,
        k3,
        p1,
        p2,
        iters: 8,
    };

    let camera = Camera::new(Pinhole, distortion, IdentitySensor, intrinsics);

    let mut poses = Vec::with_capacity(n_views);
    for i in 0..n_views {
        let idx = 10 + 6 * i;
        let wx = x[idx];
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
    fn num_params(&self) -> usize {
        self.param_dim()
    }

    fn num_residuals(&self) -> usize {
        self.residual_dim()
    }

    fn residuals_unweighted(&self, x: &DVector<Real>) -> DVector<Real> {
        self.residuals_unweighted_impl(x)
    }

    fn jacobian_unweighted(&self, x: &DVector<Real>) -> DMatrix<Real> {
        // Finite differences for now; replace with analytic Jacobian later.
        self.jacobian_unweighted_fd(x)
    }

    fn robust_row_scales(&self, r_unweighted: &DVector<Real>) -> DVector<Real> {
        self.robust_row_scales_from_unweighted(r_unweighted)
    }

    fn residuals(&self, x: &DVector<Real>) -> DVector<Real> {
        self.residuals_weighted(x)
    }

    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real> {
        self.jacobian_weighted(x)
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
    use crate::problem::SolveOptions;
    use crate::robust::RobustKernel;
    use calib_core::{BrownConrady5, FxFyCxCySkew, Pt2, Pt3, Real};
    use nalgebra::{UnitQuaternion, Vector3};

    struct SyntheticScenario {
        problem: PlanarIntrinsicsProblem,
        poses_gt: Vec<Iso3>,
        cam_gt: PinholeCamera,
        cam_init: PinholeCamera,
    }

    fn make_camera(k: FxFyCxCySkew<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
        Camera::new(Pinhole, dist, IdentitySensor, k)
    }

    fn build_synthetic_scenario(noise_amplitude: f64) -> SyntheticScenario {
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: -0.1,
            k2: 0.01,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        };
        let cam_gt = make_camera(k_gt, dist_gt);

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

        let mut views = Vec::new();
        let mut poses_gt = Vec::new();

        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rq = UnitQuaternion::from_scaled_axis(axis * angle);
            let rot = rq.to_rotation_matrix();
            let trans = Vector3::new(0.0, 0.0, 0.5 + 0.2 * view_idx as f64);
            let pose = Iso3::from_parts(trans.into(), rot.into());

            poses_gt.push(pose);

            let mut img_points = Vec::new();
            for (pt_idx, pw) in board_points.iter().enumerate() {
                let p_cam = pose.transform_point(pw);
                let proj = cam_gt.project_point(&p_cam).unwrap();
                let mut coords = Pt2::new(proj.x, proj.y).coords;

                if noise_amplitude > 0.0 {
                    let sign = if (view_idx + pt_idx) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = noise_amplitude * sign;
                    coords.x += delta;
                    coords.y -= delta;
                }

                img_points.push(coords);
            }

            views.push(PlanarViewObservations::new(
                board_points.clone(),
                img_points,
            ));
        }

        let problem = PlanarIntrinsicsProblem::new(views);
        let cam_init = make_camera(
            FxFyCxCySkew {
                fx: 780.0,
                fy: 760.0,
                cx: 630.0,
                cy: 350.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        );

        SyntheticScenario {
            problem,
            poses_gt,
            cam_gt,
            cam_init,
        }
    }

    #[test]
    fn synthetic_planar_intrinsics_refinement_converges() {
        let SyntheticScenario {
            problem,
            poses_gt,
            cam_gt,
            cam_init,
        } = build_synthetic_scenario(0.0);
        let k_gt = cam_gt.k;

        let x0 = pack_initial_params(&cam_init, &poses_gt);
        let backend = LmBackend;
        let opts = SolveOptions::default();

        let (cam_refined, poses_refined, report) =
            refine_planar_intrinsics(&backend, &problem, x0, &opts);

        // Very simple checks – just ensure we're close to GT.
        assert!((cam_refined.k.fx - k_gt.fx).abs() < 5.0);
        assert!((cam_refined.k.fy - k_gt.fy).abs() < 5.0);
        assert!((cam_refined.k.cx - k_gt.cx).abs() < 5.0);
        assert!((cam_refined.k.cy - k_gt.cy).abs() < 5.0);

        assert!(report.converged);
        assert!(report.final_cost < 1e-6);
        assert_eq!(poses_refined.len(), poses_gt.len());
    }

    #[test]
    fn synthetic_planar_intrinsics_with_noise_still_works() {
        let SyntheticScenario {
            problem,
            poses_gt,
            cam_gt,
            cam_init,
        } = build_synthetic_scenario(0.001);
        let k_gt = cam_gt.k;

        let x0 = pack_initial_params(&cam_init, &poses_gt);
        let backend = LmBackend;
        let opts = SolveOptions::default();

        let (cam_refined, poses_refined, report) =
            refine_planar_intrinsics(&backend, &problem, x0, &opts);

        assert!(report.converged);
        assert!(
            report.final_cost < 1e-4,
            "final cost too high with noise: {}",
            report.final_cost
        );
        assert_eq!(poses_refined.len(), poses_gt.len());

        let init_fx_err = (cam_init.k.fx - k_gt.fx).abs();
        let init_fy_err = (cam_init.k.fy - k_gt.fy).abs();
        let refined_fx_err = (cam_refined.k.fx - k_gt.fx).abs();
        let refined_fy_err = (cam_refined.k.fy - k_gt.fy).abs();

        assert!(
            refined_fx_err < init_fx_err,
            "fx error did not improve: init {} vs refined {}",
            init_fx_err,
            refined_fx_err
        );
        assert!(
            refined_fy_err < init_fy_err,
            "fy error did not improve: init {} vs refined {}",
            init_fy_err,
            refined_fy_err
        );
    }

    #[test]
    fn planar_intrinsics_cost_improves_over_initial() {
        let SyntheticScenario {
            problem,
            poses_gt,
            cam_gt: _,
            cam_init,
        } = build_synthetic_scenario(0.0);

        let x0 = pack_initial_params(&cam_init, &poses_gt);
        let r_init = problem.residuals(&x0);
        let initial_cost = 0.5 * r_init.dot(&r_init);

        let backend = LmBackend;
        let opts = SolveOptions::default();
        let x0_for_solver = x0.clone();
        let (cam_refined, poses_refined, report) =
            refine_planar_intrinsics(&backend, &problem, x0_for_solver, &opts);

        let refined_params = pack_initial_params(&cam_refined, &poses_refined);
        let r_final = problem.residuals(&refined_params);
        let final_cost = 0.5 * r_final.dot(&r_final);

        assert!(
            final_cost < 0.1 * initial_cost,
            "final cost {} not sufficiently smaller than initial {}",
            final_cost,
            initial_cost
        );
        assert!(
            (report.final_cost - final_cost).abs() < 1e-9,
            "reported final cost {} disagrees with recomputed {}",
            report.final_cost,
            final_cost
        );
    }

    #[test]
    fn irls_jacobian_matches_row_scaled_unweighted() {
        let SyntheticScenario {
            problem,
            poses_gt,
            cam_gt: _,
            cam_init,
        } = build_synthetic_scenario(0.0);

        let problem = problem.with_kernel(RobustKernel::Cauchy { c: 1.0e-6 });
        let x = pack_initial_params(&cam_init, &poses_gt);

        let r_unweighted = problem.residuals_unweighted(&x);
        let scales = problem.robust_row_scales_from_unweighted(&r_unweighted);
        assert!(
            scales.iter().any(|s| *s < 0.999),
            "expected some robust down-weighting"
        );

        let j_unweighted = problem.jacobian_unweighted_fd(&x);
        let mut expected = j_unweighted.clone();
        for (mut row, scale) in expected.row_iter_mut().zip(scales.iter()) {
            if *scale != 1.0 {
                row.scale_mut(*scale);
            }
        }

        let actual = problem.jacobian(&x);
        let max_abs: Real = actual
            .iter()
            .zip(expected.iter())
            .fold(0.0, |acc, (a, b)| acc.max((a - b).abs()));

        assert!(
            max_abs < 1e-10,
            "weighted Jacobian mismatch, max abs diff {}",
            max_abs
        );
    }

    #[test]
    fn synthetic_planar_intrinsics_with_outliers_robust_better_than_l2() {
        // Ground-truth camera
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: -0.1,
            k2: 0.01,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        };
        let cam_gt = make_camera(k_gt, dist_gt);

        // Board setup
        let nx = 6;
        let ny = 4;
        let spacing = 0.03_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        let mut poses_gt = Vec::new();
        let outlier_stride = 12;
        let outlier_offset = 20.0;

        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rq = UnitQuaternion::from_scaled_axis(axis * angle);
            let rot = rq.to_rotation_matrix();
            let trans = Vector3::new(0.0, 0.0, 0.5 + 0.2 * view_idx as f64);
            let pose = Iso3::from_parts(trans.into(), rot.into());
            poses_gt.push(pose);

            let mut img_points = Vec::new();
            for (pt_idx, pw) in board_points.iter().enumerate() {
                let p_cam = pose.transform_point(pw);
                let proj = cam_gt.project_point(&p_cam).unwrap();
                let mut coords = Pt2::new(proj.x, proj.y).coords;

                if pt_idx % outlier_stride == 0 {
                    coords.x += outlier_offset;
                    coords.y += outlier_offset;
                }

                img_points.push(coords);
            }

            views.push(PlanarViewObservations::new(
                board_points.clone(),
                img_points,
            ));
        }

        let problem_l2 = PlanarIntrinsicsProblem::new(views.clone());
        let problem_robust =
            PlanarIntrinsicsProblem::new(views).with_kernel(RobustKernel::Huber { delta: 2.0 });

        let cam_init = make_camera(
            FxFyCxCySkew {
                fx: 780.0,
                fy: 760.0,
                cx: 630.0,
                cy: 350.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        );

        let x0 = pack_initial_params(&cam_init, &poses_gt);
        let x0_l2 = x0.clone();
        let x0_robust = x0;

        let backend = LmBackend;
        let opts = SolveOptions::default();

        let (cam_l2, _, report_l2) = refine_planar_intrinsics(&backend, &problem_l2, x0_l2, &opts);
        let (cam_robust, _, report_robust) =
            refine_planar_intrinsics(&backend, &problem_robust, x0_robust, &opts);

        assert!(report_l2.converged);
        assert!(report_robust.converged);

        let err_total = |cam: &PinholeCamera| -> Real {
            (cam.k.fx - k_gt.fx).abs()
                + (cam.k.fy - k_gt.fy).abs()
                + (cam.k.cx - k_gt.cx).abs()
                + (cam.k.cy - k_gt.cy).abs()
        };

        let err_l2 = err_total(&cam_l2);
        let err_robust = err_total(&cam_robust);

        assert!(
            err_robust < err_l2,
            "robust intrinsics error {} should be smaller than L2 {}",
            err_robust,
            err_l2
        );
    }
}
