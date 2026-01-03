//! Planar intrinsics optimization built on tiny-solver.
//!
//! Each observation contributes a residual block with two residuals (u, v),
//! enabling robust loss to operate per point rather than per view.

use crate::factors::reprojection::ReprojPointFactor;
use crate::params::intrinsics::Intrinsics4;
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use crate::solver::tiny::{solve, TinySolveOptions};
use anyhow::{anyhow, ensure, Result};
use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2,
};
use nalgebra::DVector;
use std::collections::HashMap;
use std::sync::Arc;
use tiny_solver::loss_functions::{ArctanLoss, CauchyLoss, HuberLoss, Loss};
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::problem::Problem;

/// Camera type returned by planar intrinsics optimization.
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// Observations for a single view of a planar target.
#[derive(Debug, Clone)]
pub struct PlanarViewObservations {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
    pub weights: Option<Vec<f64>>,
}

impl PlanarViewObservations {
    /// Construct observations without per-point weights.
    pub fn new(points_3d: Vec<Pt3>, points_2d: Vec<Vec2>) -> Result<Self> {
        ensure!(
            points_3d.len() == points_2d.len(),
            "3D / 2D point counts must match"
        );
        Ok(Self {
            points_3d,
            points_2d,
            weights: None,
        })
    }

    /// Construct observations with per-point weights.
    pub fn new_with_weights(
        points_3d: Vec<Pt3>,
        points_2d: Vec<Vec2>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        ensure!(
            points_3d.len() == points_2d.len(),
            "3D / 2D point counts must match"
        );
        ensure!(
            weights.len() == points_3d.len(),
            "weight count must match point count"
        );
        ensure!(
            weights.iter().all(|w| *w >= 0.0),
            "weights must be non-negative"
        );
        Ok(Self {
            points_3d,
            points_2d,
            weights: Some(weights),
        })
    }

    pub fn len(&self) -> usize {
        self.points_3d.len()
    }

    pub fn weight(&self, idx: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[idx])
    }
}

/// A planar dataset consisting of multiple views.
#[derive(Debug, Clone)]
pub struct PlanarDataset {
    pub views: Vec<PlanarViewObservations>,
}

impl PlanarDataset {
    pub fn new(views: Vec<PlanarViewObservations>) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view for calibration");
        for (i, view) in views.iter().enumerate() {
            ensure!(view.len() >= 4, "view {} has too few points (need >=4)", i);
        }
        Ok(Self { views })
    }

    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

/// Initial values for planar intrinsics optimization.
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsInit {
    pub intrinsics: Intrinsics4,
    pub poses: Vec<Iso3>,
}

impl PlanarIntrinsicsInit {
    pub fn new(intrinsics: Intrinsics4, poses: Vec<Iso3>) -> Result<Self> {
        ensure!(!poses.is_empty(), "need at least one pose");
        Ok(Self { intrinsics, poses })
    }

    pub fn from_camera_and_poses(camera: &PinholeCamera, poses: Vec<Iso3>) -> Result<Self> {
        let intrinsics = Intrinsics4 {
            fx: camera.k.fx,
            fy: camera.k.fy,
            cx: camera.k.cx,
            cy: camera.k.cy,
        };
        Self::new(intrinsics, poses)
    }
}

/// Robust loss applied per point residual block.
#[derive(Debug, Clone)]
pub enum RobustLoss {
    None,
    Huber { scale: f64 },
    Cauchy { scale: f64 },
    Arctan { tol: f64 },
}

impl Default for RobustLoss {
    fn default() -> Self {
        Self::None
    }
}

impl RobustLoss {
    fn to_loss(&self) -> Result<Option<Box<dyn Loss + Send>>> {
        match *self {
            RobustLoss::None => Ok(None),
            RobustLoss::Huber { scale } => {
                ensure!(scale > 0.0, "Huber scale must be positive");
                Ok(Some(Box::new(HuberLoss::new(scale))))
            }
            RobustLoss::Cauchy { scale } => {
                ensure!(scale > 0.0, "Cauchy scale must be positive");
                Ok(Some(Box::new(CauchyLoss::new(scale))))
            }
            RobustLoss::Arctan { tol } => {
                ensure!(tol > 0.0, "Arctan tolerance must be positive");
                Ok(Some(Box::new(ArctanLoss::new(tol))))
            }
        }
    }
}

/// Solve options specific to planar intrinsics.
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsSolveOptions {
    pub robust_loss: RobustLoss,
    pub fix_fx: bool,
    pub fix_fy: bool,
    pub fix_cx: bool,
    pub fix_cy: bool,
    pub fix_poses: Vec<usize>,
}

impl Default for PlanarIntrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_fx: false,
            fix_fy: false,
            fix_cx: false,
            fix_cy: false,
            fix_poses: Vec::new(),
        }
    }
}

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsResult {
    pub camera: PinholeCamera,
    pub poses: Vec<Iso3>,
    pub final_cost: f64,
}

/// Build a tiny-solver problem and initial parameter map.
pub fn build_planar_intrinsics_problem(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsInit,
    opts: &PlanarIntrinsicsSolveOptions,
) -> Result<(Problem, HashMap<String, DVector<f64>>)> {
    ensure!(
        dataset.num_views() == initial.poses.len(),
        "pose count ({}) must match number of views ({})",
        initial.poses.len(),
        dataset.num_views()
    );
    for &idx in &opts.fix_poses {
        ensure!(
            idx < dataset.num_views(),
            "fixed pose index {} out of range ({} views)",
            idx,
            dataset.num_views()
        );
    }

    let mut problem = Problem::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();
    initial_map.insert("cam".to_string(), initial.intrinsics.to_dvec());

    if opts.fix_fx {
        problem.fix_variable("cam", 0);
    }
    if opts.fix_fy {
        problem.fix_variable("cam", 1);
    }
    if opts.fix_cx {
        problem.fix_variable("cam", 2);
    }
    if opts.fix_cy {
        problem.fix_variable("cam", 3);
    }

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_key = format!("pose/{}", view_idx);
        initial_map.insert(pose_key.clone(), iso3_to_se3_dvec(&initial.poses[view_idx]));

        if opts.fix_poses.contains(&view_idx) {
            for idx in 0..7 {
                problem.fix_variable(&pose_key, idx);
            }
        } else {
            problem.set_variable_manifold(&pose_key, Arc::new(SE3Manifold));
        }

        for (pt_idx, (pw, uv)) in view.points_3d.iter().zip(view.points_2d.iter()).enumerate() {
            let factor = ReprojPointFactor {
                pw: pw.clone(),
                uv: uv.clone(),
                w: view.weight(pt_idx),
            };
            let loss = opts.robust_loss.to_loss()?;
            problem.add_residual_block(2, &["cam", pose_key.as_str()], Box::new(factor), loss);
        }
    }

    Ok((problem, initial_map))
}

/// Optimize planar intrinsics using tiny-solver.
pub fn optimize_planar_intrinsics(
    dataset: PlanarDataset,
    initial: PlanarIntrinsicsInit,
    opts: PlanarIntrinsicsSolveOptions,
    solver: TinySolveOptions,
) -> Result<PlanarIntrinsicsResult> {
    let (problem, initial_map) = build_planar_intrinsics_problem(&dataset, &initial, &opts)?;
    let solution = solve(&problem, initial_map, &solver)?;

    let cam_vec = solution
        .get("cam")
        .ok_or_else(|| anyhow!("missing camera parameters in solution"))?;
    let intrinsics = Intrinsics4::from_dvec(cam_vec.as_view())?;

    let mut poses = Vec::with_capacity(dataset.num_views());
    for i in 0..dataset.num_views() {
        let key = format!("pose/{}", i);
        let pose_vec = solution
            .get(&key)
            .ok_or_else(|| anyhow!("missing pose {} in solution", i))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
        IdentitySensor,
        intrinsics.to_core(),
    );

    let param_blocks = problem.initialize_parameter_blocks(&solution);
    let residuals = problem.compute_residuals(&param_blocks, true);
    let final_cost = 0.5 * residuals.as_ref().squared_norm_l2();

    Ok(PlanarIntrinsicsResult {
        camera,
        poses,
        final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::tiny::TinySolveOptions;
    use calib_core::{Pt2, Pt3};
    use nalgebra::{UnitQuaternion, Vector3};

    struct SyntheticScenario {
        dataset: PlanarDataset,
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
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
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

            views.push(PlanarViewObservations::new(board_points.clone(), img_points).unwrap());
        }

        let dataset = PlanarDataset::new(views).unwrap();
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
            dataset,
            poses_gt,
            cam_gt,
            cam_init,
        }
    }

    #[test]
    fn synthetic_planar_intrinsics_refinement_converges() {
        let SyntheticScenario {
            dataset,
            poses_gt,
            cam_gt,
            cam_init,
        } = build_synthetic_scenario(0.0);
        let k_gt = cam_gt.k;

        let init = PlanarIntrinsicsInit::from_camera_and_poses(&cam_init, poses_gt).unwrap();
        let opts = PlanarIntrinsicsSolveOptions::default();
        let solver = TinySolveOptions::default();

        let result = optimize_planar_intrinsics(dataset, init, opts, solver).unwrap();

        assert!((result.camera.k.fx - k_gt.fx).abs() < 5.0);
        assert!((result.camera.k.fy - k_gt.fy).abs() < 5.0);
        assert!((result.camera.k.cx - k_gt.cx).abs() < 5.0);
        assert!((result.camera.k.cy - k_gt.cy).abs() < 5.0);
    }

    #[test]
    fn synthetic_planar_intrinsics_with_outliers_robust_better_than_l2() {
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let cam_gt = make_camera(k_gt, dist_gt);

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

            views.push(PlanarViewObservations::new(board_points.clone(), img_points).unwrap());
        }

        let dataset = PlanarDataset::new(views).unwrap();
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

        let init = PlanarIntrinsicsInit::from_camera_and_poses(&cam_init, poses_gt).unwrap();
        let solver = TinySolveOptions::default();

        let l2_opts = PlanarIntrinsicsSolveOptions::default();
        let robust_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            ..PlanarIntrinsicsSolveOptions::default()
        };

        let l2 = optimize_planar_intrinsics(dataset.clone(), init.clone(), l2_opts, solver.clone())
            .unwrap();
        let robust = optimize_planar_intrinsics(dataset, init, robust_opts, solver).unwrap();

        let err_total = |cam: &PinholeCamera| -> Real {
            (cam.k.fx - k_gt.fx).abs()
                + (cam.k.fy - k_gt.fy).abs()
                + (cam.k.cx - k_gt.cx).abs()
                + (cam.k.cy - k_gt.cy).abs()
        };

        let err_l2 = err_total(&l2.camera);
        let err_robust = err_total(&robust.camera);

        assert!(
            err_robust < err_l2,
            "robust intrinsics error {} should be smaller than L2 {}",
            err_robust,
            err_l2
        );
    }

    #[test]
    fn intrinsics_masking_keeps_fixed_params() {
        let SyntheticScenario {
            dataset,
            poses_gt,
            cam_init,
            ..
        } = build_synthetic_scenario(0.0);

        let init = PlanarIntrinsicsInit::from_camera_and_poses(&cam_init, poses_gt).unwrap();
        let opts = PlanarIntrinsicsSolveOptions {
            fix_fx: true,
            fix_fy: true,
            ..PlanarIntrinsicsSolveOptions::default()
        };
        let solver = TinySolveOptions::default();

        let result = optimize_planar_intrinsics(dataset, init.clone(), opts, solver).unwrap();

        assert!(
            (result.camera.k.fx - init.intrinsics.fx).abs() < 1e-12,
            "fx should remain fixed"
        );
        assert!(
            (result.camera.k.fy - init.intrinsics.fy).abs() < 1e-12,
            "fy should remain fixed"
        );
    }
}
