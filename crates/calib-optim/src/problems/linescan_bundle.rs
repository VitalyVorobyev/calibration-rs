//! Linescan bundle adjustment: joint optimization of camera intrinsics, distortion,
//! target poses, and laser plane parameters.
//!
//! This problem builder combines standard planar calibration observations (corners)
//! with laser line observations to estimate laser plane parameters in camera frame.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::BrownConrady5Params;
use crate::params::intrinsics::Intrinsics4;
use crate::params::laser_plane::LaserPlane;
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use anyhow::{anyhow, ensure, Result};
use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2,
};
use nalgebra::DVector;
use std::collections::HashMap;

/// Camera type returned by linescan optimization.
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// Observations for a single view with both calibration features and laser line.
#[derive(Debug, Clone)]
pub struct LinescanViewObservations {
    /// Standard calibration observations (corners on target)
    pub calib_points_3d: Vec<Pt3>,
    pub calib_pixels: Vec<Vec2>,

    /// Laser line pixel observations
    pub laser_pixels: Vec<Vec2>,

    /// Per-point weights for calibration observations
    pub calib_weights: Option<Vec<f64>>,
    /// Per-point weights for laser observations
    pub laser_weights: Option<Vec<f64>>,
}

impl LinescanViewObservations {
    /// Create observations with calibration points and laser pixels.
    pub fn new(
        calib_points_3d: Vec<Pt3>,
        calib_pixels: Vec<Vec2>,
        laser_pixels: Vec<Vec2>,
    ) -> Result<Self> {
        ensure!(
            calib_points_3d.len() == calib_pixels.len(),
            "calibration 3D / 2D point counts must match"
        );
        ensure!(
            !laser_pixels.is_empty(),
            "need at least one laser pixel observation"
        );
        Ok(Self {
            calib_points_3d,
            calib_pixels,
            laser_pixels,
            calib_weights: None,
            laser_weights: None,
        })
    }

    /// Create observations with per-point weights.
    pub fn new_with_weights(
        calib_points_3d: Vec<Pt3>,
        calib_pixels: Vec<Vec2>,
        laser_pixels: Vec<Vec2>,
        calib_weights: Vec<f64>,
        laser_weights: Vec<f64>,
    ) -> Result<Self> {
        ensure!(
            calib_points_3d.len() == calib_pixels.len(),
            "calibration 3D / 2D point counts must match"
        );
        ensure!(
            calib_weights.len() == calib_points_3d.len(),
            "calibration weight count must match point count"
        );
        ensure!(
            !laser_pixels.is_empty(),
            "need at least one laser pixel observation"
        );
        ensure!(
            laser_weights.len() == laser_pixels.len(),
            "laser weight count must match pixel count"
        );
        ensure!(
            calib_weights.iter().all(|w| *w >= 0.0),
            "calibration weights must be non-negative"
        );
        ensure!(
            laser_weights.iter().all(|w| *w >= 0.0),
            "laser weights must be non-negative"
        );
        Ok(Self {
            calib_points_3d,
            calib_pixels,
            laser_pixels,
            calib_weights: Some(calib_weights),
            laser_weights: Some(laser_weights),
        })
    }

    pub fn num_calib_points(&self) -> usize {
        self.calib_points_3d.len()
    }

    pub fn num_laser_pixels(&self) -> usize {
        self.laser_pixels.len()
    }

    pub fn calib_weight(&self, idx: usize) -> f64 {
        self.calib_weights.as_ref().map_or(1.0, |w| w[idx])
    }

    pub fn laser_weight(&self, idx: usize) -> f64 {
        self.laser_weights.as_ref().map_or(1.0, |w| w[idx])
    }
}

/// Dataset for linescan bundle adjustment.
#[derive(Debug, Clone)]
pub struct LinescanDataset {
    pub views: Vec<LinescanViewObservations>,
    pub num_planes: usize,
}

impl LinescanDataset {
    /// Create dataset with a single laser plane.
    pub fn new_single_plane(views: Vec<LinescanViewObservations>) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view for calibration");
        for (i, view) in views.iter().enumerate() {
            ensure!(
                view.num_calib_points() >= 4,
                "view {} has too few calibration points (need >=4)",
                i
            );
            ensure!(
                view.num_laser_pixels() >= 3,
                "view {} has too few laser pixels (need >=3)",
                i
            );
        }
        Ok(Self {
            views,
            num_planes: 1,
        })
    }

    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

/// Initial values for linescan bundle adjustment.
#[derive(Debug, Clone)]
pub struct LinescanInit {
    pub intrinsics: Intrinsics4,
    pub distortion: BrownConrady5Params,
    pub poses: Vec<Iso3>,
    pub planes: Vec<LaserPlane>,
}

impl LinescanInit {
    pub fn new(
        intrinsics: Intrinsics4,
        distortion: BrownConrady5Params,
        poses: Vec<Iso3>,
        planes: Vec<LaserPlane>,
    ) -> Result<Self> {
        ensure!(!poses.is_empty(), "need at least one pose");
        ensure!(!planes.is_empty(), "need at least one plane");
        Ok(Self {
            intrinsics,
            distortion,
            poses,
            planes,
        })
    }

    pub fn from_camera(
        camera: &PinholeCamera,
        poses: Vec<Iso3>,
        planes: Vec<LaserPlane>,
    ) -> Result<Self> {
        let intrinsics = Intrinsics4 {
            fx: camera.k.fx,
            fy: camera.k.fy,
            cx: camera.k.cx,
            cy: camera.k.cy,
        };
        let distortion = BrownConrady5Params::from_core(&camera.dist);
        Self::new(intrinsics, distortion, poses, planes)
    }
}

/// Solve options for linescan bundle adjustment.
#[derive(Debug, Clone)]
pub struct LinescanSolveOptions {
    /// Robust loss applied to calibration reprojection residuals
    pub calib_loss: RobustLoss,
    /// Robust loss applied to laser plane residuals
    pub laser_loss: RobustLoss,
    /// Fix camera intrinsics during optimization
    pub fix_intrinsics: bool,
    /// Fix distortion parameters during optimization
    pub fix_distortion: bool,
    /// Fix k3 distortion parameter (common for typical lenses)
    pub fix_k3: bool,
    /// Indices of poses to fix (e.g., [0] to fix first pose for gauge freedom)
    pub fix_poses: Vec<usize>,
    /// Indices of planes to fix
    pub fix_planes: Vec<usize>,
}

impl Default for LinescanSolveOptions {
    fn default() -> Self {
        Self {
            calib_loss: RobustLoss::Huber { scale: 1.0 },
            laser_loss: RobustLoss::Huber { scale: 0.01 }, // Smaller scale for plane residuals
            fix_intrinsics: false,
            fix_distortion: false,
            fix_k3: true,
            fix_poses: vec![0], // Fix first pose by default
            fix_planes: vec![],
        }
    }
}

/// Result of linescan bundle adjustment.
#[derive(Debug, Clone)]
pub struct LinescanResult {
    pub camera: PinholeCamera,
    pub poses: Vec<Iso3>,
    pub planes: Vec<LaserPlane>,
    pub final_cost: f64,
}

/// Build IR for linescan bundle adjustment.
fn build_linescan_ir(
    dataset: &LinescanDataset,
    initial: &LinescanInit,
    opts: &LinescanSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        dataset.num_views() == initial.poses.len(),
        "dataset has {} views but {} initial poses",
        dataset.num_views(),
        initial.poses.len()
    );
    ensure!(
        dataset.num_planes == initial.planes.len(),
        "dataset expects {} planes but got {} initial planes",
        dataset.num_planes,
        initial.planes.len()
    );

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    // Add intrinsics parameter block
    let intrinsics_fixed = if opts.fix_intrinsics {
        FixedMask::all_fixed(4)
    } else {
        FixedMask::all_free()
    };
    let intrinsics_id = ir.add_param_block(
        "intrinsics",
        4,
        ManifoldKind::Euclidean,
        intrinsics_fixed,
        None,
    );
    initial_map.insert("intrinsics".to_string(), initial.intrinsics.to_dvec());

    // Add distortion parameter block
    let distortion_fixed = if opts.fix_distortion {
        FixedMask::all_fixed(5)
    } else if opts.fix_k3 {
        FixedMask::fix_indices(&[2]) // k3 is at index 2
    } else {
        FixedMask::all_free()
    };
    let distortion_id = ir.add_param_block(
        "distortion",
        5,
        ManifoldKind::Euclidean,
        distortion_fixed,
        None,
    );
    initial_map.insert("distortion".to_string(), initial.distortion.to_dvec());

    // Add pose parameter blocks (one per view)
    let mut pose_ids = Vec::new();
    for (i, pose) in initial.poses.iter().enumerate() {
        let fixed = if opts.fix_poses.contains(&i) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let name = format!("pose_{}", i);
        let id = ir.add_param_block(&name, 7, ManifoldKind::SE3, fixed, None);
        pose_ids.push(id);
        initial_map.insert(name, iso3_to_se3_dvec(pose));
    }

    // Add laser plane parameter blocks
    let mut plane_ids = Vec::new();
    for (i, plane) in initial.planes.iter().enumerate() {
        let fixed = if opts.fix_planes.contains(&i) {
            FixedMask::all_fixed(4)
        } else {
            FixedMask::all_free()
        };
        let name = format!("plane_{}", i);
        let id = ir.add_param_block(&name, 4, ManifoldKind::Euclidean, fixed, None);
        plane_ids.push(id);
        initial_map.insert(name, plane.to_dvec());
    }

    // Add residual blocks
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_id = pose_ids[view_idx];

        // Calibration reprojection residuals
        for (pt_idx, (pt_3d, pt_2d)) in view
            .calib_points_3d
            .iter()
            .zip(&view.calib_pixels)
            .enumerate()
        {
            let w = view.calib_weight(pt_idx);
            ir.add_residual_block(ResidualBlock {
                params: vec![intrinsics_id, distortion_id, pose_id],
                loss: opts.calib_loss,
                factor: FactorKind::ReprojPointPinhole4Dist5 {
                    pw: [pt_3d.x, pt_3d.y, pt_3d.z],
                    uv: [pt_2d.x, pt_2d.y],
                    w,
                },
                residual_dim: 2,
            });
        }

        // Laser plane residuals (one per plane, typically just one)
        for &plane_id in plane_ids.iter().take(dataset.num_planes) {
            for (laser_idx, laser_pixel) in view.laser_pixels.iter().enumerate() {
                let w = view.laser_weight(laser_idx);
                ir.add_residual_block(ResidualBlock {
                    params: vec![intrinsics_id, distortion_id, pose_id, plane_id],
                    loss: opts.laser_loss,
                    factor: FactorKind::LaserPlanePixel {
                        laser_pixel: [laser_pixel.x, laser_pixel.y],
                        w,
                    },
                    residual_dim: 1,
                });
            }
        }
    }

    Ok((ir, initial_map))
}

/// Extract solution from backend result.
fn extract_solution(
    solution: crate::backend::BackendSolution,
    num_poses: usize,
    num_planes: usize,
) -> Result<LinescanResult> {
    let intrinsics = Intrinsics4::from_dvec(
        solution
            .params
            .get("intrinsics")
            .ok_or_else(|| anyhow!("missing intrinsics in solution"))?
            .as_view(),
    )?;

    let distortion = BrownConrady5Params::from_dvec(
        solution
            .params
            .get("distortion")
            .ok_or_else(|| anyhow!("missing distortion in solution"))?
            .as_view(),
    )?;

    let mut poses = Vec::new();
    for i in 0..num_poses {
        let name = format!("pose_{}", i);
        let pose_vec = solution
            .params
            .get(&name)
            .ok_or_else(|| anyhow!("missing {} in solution", name))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let mut planes = Vec::new();
    for i in 0..num_planes {
        let name = format!("plane_{}", i);
        let plane_vec = solution
            .params
            .get(&name)
            .ok_or_else(|| anyhow!("missing {} in solution", name))?;
        planes.push(LaserPlane::from_dvec(plane_vec.as_view())?);
    }

    let camera = Camera::new(
        Pinhole,
        distortion.to_core(),
        IdentitySensor,
        intrinsics.to_core(),
    );

    Ok(LinescanResult {
        camera,
        poses,
        planes,
        final_cost: solution.final_cost,
    })
}

/// Optimize linescan calibration with joint bundle adjustment.
///
/// This function jointly optimizes camera intrinsics, distortion parameters,
/// camera-to-target poses, and laser plane parameters using both calibration
/// feature observations and laser line observations.
///
/// # Example
///
/// ```ignore
/// use calib_optim::problems::linescan_bundle::*;
///
/// let dataset = LinescanDataset::new_single_plane(views)?;
/// let initial = LinescanInit::new(intrinsics, distortion, poses, planes)?;
/// let opts = LinescanSolveOptions::default();
/// let backend_opts = BackendSolveOptions::default();
///
/// let result = optimize_linescan(&dataset, &initial, &opts, &backend_opts)?;
/// println!("Laser plane: normal={:?}, distance={}",
///          result.planes[0].normal, result.planes[0].distance);
/// ```
pub fn optimize_linescan(
    dataset: &LinescanDataset,
    initial: &LinescanInit,
    opts: &LinescanSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<LinescanResult> {
    let (ir, initial_map) = build_linescan_ir(dataset, initial, opts)?;
    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, backend_opts)?;
    extract_solution(solution, dataset.num_views(), dataset.num_planes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_validation_catches_missing_param() {
        let views = vec![LinescanViewObservations {
            calib_points_3d: vec![Pt3::new(0.0, 0.0, 0.0); 4],
            calib_pixels: vec![Vec2::new(100.0, 100.0); 4],
            laser_pixels: vec![Vec2::new(200.0, 200.0); 5],
            calib_weights: None,
            laser_weights: None,
        }];
        let dataset = LinescanDataset::new_single_plane(views).unwrap();

        let intrinsics = Intrinsics4 {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
        };
        let distortion = BrownConrady5Params::zeros();
        let poses = vec![Iso3::identity()];
        let planes = vec![LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5)];
        let initial = LinescanInit::new(intrinsics, distortion, poses, planes).unwrap();

        let opts = LinescanSolveOptions::default();
        let (ir, mut initial_map) = build_linescan_ir(&dataset, &initial, &opts).unwrap();

        // Remove intrinsics from initial values
        initial_map.remove("intrinsics");

        // Should fail compilation due to missing parameter
        use crate::backend::{OptimBackend, TinySolverBackend};
        let backend = TinySolverBackend;
        let backend_opts = BackendSolveOptions::default();
        let result = backend.solve(&ir, &initial_map, &backend_opts);
        assert!(result.is_err());
    }
}
