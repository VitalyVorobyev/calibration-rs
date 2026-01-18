//! Planar intrinsics optimization using the backend-agnostic IR.
//!
//! Each observation contributes a residual block with two residuals (u, v),
//! enabling robust loss to operate per point rather than per view.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use anyhow::{anyhow, ensure, Result};
use calib_core::{
    BrownConrady5, Camera, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, IdentitySensor,
    IntrinsicsFixMask, Iso3, Pinhole, Real,
};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use crate::ir::RobustLoss;

/// Camera type returned by planar intrinsics optimization.
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// A planar dataset consisting of multiple views.
///
/// Each view observes a planar calibration target in pixel coordinates.
#[derive(Debug, Clone)]
pub struct PlanarDataset {
    pub views: Vec<CorrespondenceView>,
}

impl PlanarDataset {
    pub fn new(views: Vec<CorrespondenceView>) -> Result<Self> {
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
///
/// Poses are board-to-camera transforms for each view.
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsInit {
    pub intrinsics: FxFyCxCySkew<Real>,
    pub distortion: BrownConrady5<Real>,
    pub poses: Vec<Iso3>,
}

impl PlanarIntrinsicsInit {
    pub fn new(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        poses: Vec<Iso3>,
    ) -> Result<Self> {
        ensure!(!poses.is_empty(), "need at least one pose");
        Ok(Self {
            intrinsics,
            distortion,
            poses,
        })
    }

    /// Create with zero distortion (pinhole model only).
    pub fn new_pinhole(intrinsics: FxFyCxCySkew<Real>, poses: Vec<Iso3>) -> Result<Self> {
        Self::new(
            intrinsics,
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
            poses,
        )
    }

    pub fn from_camera_and_poses(camera: &PinholeCamera, poses: Vec<Iso3>) -> Result<Self> {
        let intrinsics = camera.k;
        let distortion = camera.dist;
        Self::new(intrinsics, distortion, poses)
    }
}

/// Solve options specific to planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Mask for fixing intrinsics parameters.
    pub fix_intrinsics: IntrinsicsFixMask,
    /// Mask for fixing distortion parameters (k3 fixed by default).
    pub fix_distortion: DistortionFixMask,
    /// Indices of poses to keep fixed.
    pub fix_poses: Vec<usize>,
}

impl Default for PlanarIntrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: DistortionFixMask::default(), // k3 fixed by default
            fix_poses: Vec::new(),
        }
    }
}

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone)]
pub struct PlanarIntrinsicsResult {
    /// Refined camera with intrinsics and distortion.
    pub camera: PinholeCamera,
    /// Refined board-to-camera poses.
    pub poses: Vec<Iso3>,
    /// Final robustified cost.
    pub final_cost: f64,
}

/// Build the backend-agnostic IR and initial values for planar intrinsics.
///
/// This is the canonical problem builder reused by all backends.
pub fn build_planar_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsInit,
    opts: &PlanarIntrinsicsSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
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

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    let cam_id = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_intrinsics.to_indices()),
        None,
    );
    initial_map.insert("cam".to_string(), pack_intrinsics(&initial.intrinsics)?);

    // Add distortion parameter block
    let dist_id = ir.add_param_block(
        "dist",
        DISTORTION_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_distortion.to_indices()),
        None,
    );
    initial_map.insert("dist".to_string(), pack_distortion(&initial.distortion));

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_key = format!("pose/{}", view_idx);
        let fixed = if opts.fix_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let pose_id = ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(pose_key.clone(), iso3_to_se3_dvec(&initial.poses[view_idx]));

        for (pt_idx, (pw, uv)) in view.points_3d.iter().zip(view.points_2d.iter()).enumerate() {
            let factor = FactorKind::ReprojPointPinhole4Dist5 {
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: view.weight(pt_idx),
            };
            let residual = ResidualBlock {
                params: vec![cam_id, dist_id, pose_id],
                loss: opts.robust_loss,
                factor,
                residual_dim: 2,
            };
            ir.add_residual_block(residual);
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

/// Optimize planar intrinsics using the default tiny-solver backend.
pub fn optimize_planar_intrinsics(
    dataset: PlanarDataset,
    initial: PlanarIntrinsicsInit,
    opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsResult> {
    optimize_planar_intrinsics_with_backend(
        dataset,
        initial,
        opts,
        BackendKind::TinySolver,
        backend_opts,
    )
}

/// Optimize planar intrinsics using the selected backend.
pub fn optimize_planar_intrinsics_with_backend(
    dataset: PlanarDataset,
    initial: PlanarIntrinsicsInit,
    opts: PlanarIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsResult> {
    let (ir, initial_map) = build_planar_intrinsics_ir(&dataset, &initial, &opts)?;
    let solution = solve_with_backend(backend, &ir, &initial_map, &backend_opts)?;

    let cam_vec = solution
        .params
        .get("cam")
        .ok_or_else(|| anyhow!("missing camera parameters in solution"))?;
    let intrinsics = unpack_intrinsics(cam_vec.as_view())?;

    let dist_vec = solution
        .params
        .get("dist")
        .ok_or_else(|| anyhow!("missing distortion parameters in solution"))?;
    let distortion = unpack_distortion(dist_vec.as_view())?;

    let mut poses = Vec::with_capacity(dataset.num_views());
    for i in 0..dataset.num_views() {
        let key = format!("pose/{}", i);
        let pose_vec = solution
            .params
            .get(&key)
            .ok_or_else(|| anyhow!("missing pose {} in solution", i))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let camera = Camera::new(Pinhole, distortion, IdentitySensor, intrinsics);

    Ok(PlanarIntrinsicsResult {
        camera,
        poses,
        final_cost: solution.final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendSolveOptions;
    use calib_core::{Pt2, Pt3, Vec2};
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

            views.push(CorrespondenceView::new(board_points.clone(), img_points).unwrap());
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
        let solver = BackendSolveOptions::default();

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

            views.push(CorrespondenceView::new(board_points.clone(), img_points).unwrap());
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
        let solver = BackendSolveOptions::default();

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
            fix_intrinsics: IntrinsicsFixMask {
                fx: true,
                fy: true,
                ..Default::default()
            },
            ..PlanarIntrinsicsSolveOptions::default()
        };
        let solver = BackendSolveOptions::default();

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

    #[test]
    fn ir_validation_catches_missing_param() {
        let mut ir = ProblemIR::new();
        let cam_id = ir.add_param_block(
            "cam",
            INTRINSICS_DIM,
            ManifoldKind::Euclidean,
            FixedMask::all_free(),
            None,
        );
        let residual = ResidualBlock {
            params: vec![cam_id, crate::ir::ParamId(42)],
            loss: RobustLoss::None,
            factor: FactorKind::ReprojPointPinhole4 {
                pw: [0.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                w: 1.0,
            },
            residual_dim: 2,
        };
        ir.add_residual_block(residual);

        let err = ir.validate().unwrap_err().to_string();
        assert!(
            err.contains("references missing param"),
            "unexpected validation error: {}",
            err
        );
    }

    #[test]
    fn synthetic_planar_with_distortion_converges() {
        // Test that distortion optimization works with known ground truth
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: -0.2, // Barrel distortion
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        };
        let cam_gt = make_camera(k_gt, dist_gt);

        // Generate synthetic planar target (10x7 grid, 3cm spacing)
        let board_points: Vec<Pt3> = (0..10)
            .flat_map(|i| (0..7).map(move |j| Pt3::new(j as Real * 0.03, i as Real * 0.03, 0.0)))
            .collect();

        let mut views = vec![];
        let mut poses_gt = vec![];

        // Create 10 views at different poses
        for i in 0..10 {
            let angle = (i as Real * 10.0).to_radians();
            let dist_from_board = 0.5 + i as Real * 0.05;
            let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.5, 0.0);
            let trans = Vector3::new(0.1, 0.1, dist_from_board);
            let pose = Iso3::from_parts(trans.into(), rot);
            poses_gt.push(pose);

            // Project points through ground truth camera
            let mut points_2d = vec![];
            let mut points_3d = vec![];

            for pw in &board_points {
                let pc = pose.transform_point(pw).coords;
                if pc.z > 0.1 {
                    if let Some(uv) = cam_gt.project_point_c(&pc) {
                        points_2d.push(Vec2::new(uv.x, uv.y));
                        points_3d.push(*pw);
                    }
                }
            }

            views.push(CorrespondenceView::new(points_3d, points_2d).unwrap());
        }

        let dataset = PlanarDataset::new(views).unwrap();

        // Initialize with noisy intrinsics and zero distortion
        let init = PlanarIntrinsicsInit {
            intrinsics: FxFyCxCySkew {
                fx: 780.0, // -20 error
                fy: 760.0, // -20 error
                cx: 630.0, // -10 error
                cy: 350.0, // -10 error
                skew: 0.0,
            },
            distortion: BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
            poses: poses_gt.clone(),
        };

        let opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::None,
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions::default();
        let result = optimize_planar_intrinsics(dataset, init, opts, backend_opts).unwrap();

        // Verify convergence to ground truth
        println!(
            "Final camera: fx={}, fy={}, cx={}, cy={}",
            result.camera.k.fx, result.camera.k.fy, result.camera.k.cx, result.camera.k.cy
        );
        println!(
            "Final distortion: k1={}, k2={}, k3={}, p1={}, p2={}",
            result.camera.dist.k1,
            result.camera.dist.k2,
            result.camera.dist.k3,
            result.camera.dist.p1,
            result.camera.dist.p2
        );

        assert!(
            (result.camera.k.fx - k_gt.fx).abs() < 5.0,
            "fx off by {}",
            result.camera.k.fx - k_gt.fx
        );
        assert!(
            (result.camera.k.fy - k_gt.fy).abs() < 5.0,
            "fy off by {}",
            result.camera.k.fy - k_gt.fy
        );
        assert!(
            (result.camera.k.cx - k_gt.cx).abs() < 3.0,
            "cx off by {}",
            result.camera.k.cx - k_gt.cx
        );
        assert!(
            (result.camera.k.cy - k_gt.cy).abs() < 3.0,
            "cy off by {}",
            result.camera.k.cy - k_gt.cy
        );

        assert!(
            (result.camera.dist.k1 - dist_gt.k1).abs() < 0.01,
            "k1 off by {}",
            result.camera.dist.k1 - dist_gt.k1
        );
        assert!(
            (result.camera.dist.k2 - dist_gt.k2).abs() < 0.01,
            "k2 off by {}",
            result.camera.dist.k2 - dist_gt.k2
        );
        assert!(
            (result.camera.dist.p1 - dist_gt.p1).abs() < 0.001,
            "p1 off by {}",
            result.camera.dist.p1 - dist_gt.p1
        );
        assert!(
            (result.camera.dist.p2 - dist_gt.p2).abs() < 0.001,
            "p2 off by {}",
            result.camera.dist.p2 - dist_gt.p2
        );
    }

    #[test]
    fn distortion_parameter_masking_works() {
        // Test selective fixing of distortion parameters
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: -0.15,
            k2: 0.04,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let cam_gt = make_camera(k_gt, dist_gt);

        // Generate smaller synthetic dataset
        let board_points: Vec<Pt3> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt3::new(j as Real * 0.03, i as Real * 0.03, 0.0)))
            .collect();

        let mut views = vec![];
        let mut poses_gt = vec![];

        for i in 0..5 {
            let angle = (i as Real * 15.0).to_radians();
            let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.3, 0.0);
            let trans = Vector3::new(0.0, 0.0, 0.6);
            let pose = Iso3::from_parts(trans.into(), rot);
            poses_gt.push(pose);

            let mut points_2d = vec![];
            let mut points_3d = vec![];

            for pw in &board_points {
                let pc = pose.transform_point(pw).coords;
                if pc.z > 0.1 {
                    if let Some(uv) = cam_gt.project_point_c(&pc) {
                        points_2d.push(Vec2::new(uv.x, uv.y));
                        points_3d.push(*pw);
                    }
                }
            }

            views.push(CorrespondenceView::new(points_3d, points_2d).unwrap());
        }

        let dataset = PlanarDataset::new(views).unwrap();

        let init = PlanarIntrinsicsInit {
            intrinsics: FxFyCxCySkew {
                fx: 790.0,
                fy: 770.0,
                cx: 635.0,
                cy: 355.0,
                skew: 0.0,
            },
            distortion: BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
            poses: poses_gt,
        };

        // Fix k3, p1, p2 (they are zero in ground truth)
        let opts = PlanarIntrinsicsSolveOptions {
            fix_distortion: DistortionFixMask {
                k3: true,
                p1: true,
                p2: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let result =
            optimize_planar_intrinsics(dataset, init, opts, BackendSolveOptions::default())
                .unwrap();

        // Fixed params should stay at initial values
        assert_eq!(result.camera.dist.k3, 0.0, "k3 should stay fixed at 0");
        assert_eq!(result.camera.dist.p1, 0.0, "p1 should stay fixed at 0");
        assert_eq!(result.camera.dist.p2, 0.0, "p2 should stay fixed at 0");

        // k1, k2 should converge
        assert!(
            (result.camera.dist.k1 - dist_gt.k1).abs() < 0.01,
            "k1 should converge, off by {}",
            result.camera.dist.k1 - dist_gt.k1
        );
        assert!(
            (result.camera.dist.k2 - dist_gt.k2).abs() < 0.01,
            "k2 should converge, off by {}",
            result.camera.dist.k2 - dist_gt.k2
        );
    }
}
