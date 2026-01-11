//! Linescan bundle adjustment: joint optimization of camera intrinsics, distortion,
//! target poses, and laser plane parameters.
//!
//! This problem builder combines standard planar calibration observations (corners)
//! with laser line observations to estimate laser plane parameters in camera frame.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
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
    pub intrinsics: FxFyCxCySkew<Real>,
    pub distortion: BrownConrady5<Real>,
    pub poses: Vec<Iso3>,
    pub planes: Vec<LaserPlane>,
}

impl LinescanInit {
    pub fn new(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
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
        let intrinsics = camera.k;
        let distortion = camera.dist;
        Self::new(intrinsics, distortion, poses, planes)
    }
}

/// Type of laser plane residual to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaserResidualType {
    /// Point-to-plane distance (original approach).
    ///
    /// Undistorts pixel, back-projects to 3D ray, intersects with target plane,
    /// computes 3D point in camera frame, measures signed distance to laser plane.
    /// Residual: 1D distance in meters.
    PointToPlane,
    /// Line-distance in normalized plane (alternative approach).
    ///
    /// Computes 3D intersection line of laser plane and target plane, projects
    /// line onto z=1 normalized camera plane, undistorts laser pixels to normalized
    /// coordinates, measures perpendicular distance from pixel to projected line,
    /// scales by sqrt(fx*fy) for pixel-comparable residual.
    /// Residual: 1D distance in pixels.
    LineDistNormalized,
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
    /// Indices of poses to fix (e.g., \[0\] to fix first pose for gauge freedom)
    pub fix_poses: Vec<usize>,
    /// Indices of planes to fix
    pub fix_planes: Vec<usize>,
    /// Laser residual type: point-to-plane distance or line-distance in normalized plane
    pub laser_residual_type: LaserResidualType,
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
            laser_residual_type: LaserResidualType::LineDistNormalized, // New default
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
        FixedMask::all_fixed(INTRINSICS_DIM)
    } else {
        FixedMask::all_free()
    };
    let intrinsics_id = ir.add_param_block(
        "intrinsics",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        intrinsics_fixed,
        None,
    );
    initial_map.insert(
        "intrinsics".to_string(),
        pack_intrinsics(&initial.intrinsics)?,
    );

    // Add distortion parameter block
    let distortion_fixed = if opts.fix_distortion {
        FixedMask::all_fixed(DISTORTION_DIM)
    } else if opts.fix_k3 {
        FixedMask::fix_indices(&[2]) // k3 is at index 2
    } else {
        FixedMask::all_free()
    };
    let distortion_id = ir.add_param_block(
        "distortion",
        DISTORTION_DIM,
        ManifoldKind::Euclidean,
        distortion_fixed,
        None,
    );
    initial_map.insert(
        "distortion".to_string(),
        pack_distortion(&initial.distortion),
    );

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

    // Add laser plane parameter blocks (unit normal on S2 + scalar distance)
    let mut plane_ids = Vec::new();
    for (i, plane) in initial.planes.iter().enumerate() {
        let fixed_plane = opts.fix_planes.contains(&i);
        let normal_fixed = if fixed_plane {
            FixedMask::all_fixed(3)
        } else {
            FixedMask::all_free()
        };
        let distance_fixed = if fixed_plane {
            FixedMask::all_fixed(1)
        } else {
            FixedMask::all_free()
        };

        let normal_name = format!("plane_{}_normal", i);
        let distance_name = format!("plane_{}_distance", i);
        let normal_id = ir.add_param_block(&normal_name, 3, ManifoldKind::S2, normal_fixed, None);
        let distance_id = ir.add_param_block(
            &distance_name,
            1,
            ManifoldKind::Euclidean,
            distance_fixed,
            None,
        );
        plane_ids.push((normal_id, distance_id));
        initial_map.insert(normal_name, plane.normal_to_dvec());
        initial_map.insert(distance_name, plane.distance_to_dvec());
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
        for &(plane_normal_id, plane_distance_id) in plane_ids.iter().take(dataset.num_planes) {
            for (laser_idx, laser_pixel) in view.laser_pixels.iter().enumerate() {
                let w = view.laser_weight(laser_idx);

                // Select factor type based on options
                let factor = match opts.laser_residual_type {
                    LaserResidualType::PointToPlane => FactorKind::LaserPlanePixel {
                        laser_pixel: [laser_pixel.x, laser_pixel.y],
                        w,
                    },
                    LaserResidualType::LineDistNormalized => FactorKind::LaserLineDist2D {
                        laser_pixel: [laser_pixel.x, laser_pixel.y],
                        w,
                    },
                };

                ir.add_residual_block(ResidualBlock {
                    params: vec![
                        intrinsics_id,
                        distortion_id,
                        pose_id,
                        plane_normal_id,
                        plane_distance_id,
                    ],
                    loss: opts.laser_loss,
                    factor,
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
    let intrinsics = unpack_intrinsics(
        solution
            .params
            .get("intrinsics")
            .ok_or_else(|| anyhow!("missing intrinsics in solution"))?
            .as_view(),
    )?;

    let distortion = unpack_distortion(
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
        let normal_name = format!("plane_{}_normal", i);
        let distance_name = format!("plane_{}_distance", i);
        let plane_normal = solution
            .params
            .get(&normal_name)
            .ok_or_else(|| anyhow!("missing {} in solution", normal_name))?;
        let plane_distance = solution
            .params
            .get(&distance_name)
            .ok_or_else(|| anyhow!("missing {} in solution", distance_name))?;
        planes.push(LaserPlane::from_split_dvec(
            plane_normal.as_view(),
            plane_distance.as_view(),
        )?);
    }

    let camera = Camera::new(Pinhole, distortion, IdentitySensor, intrinsics);

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

        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
            skew: 0.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
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

    #[test]
    fn ir_uses_s2_for_plane_normal() {
        let views = vec![LinescanViewObservations {
            calib_points_3d: vec![Pt3::new(0.0, 0.0, 0.0); 4],
            calib_pixels: vec![Vec2::new(100.0, 100.0); 4],
            laser_pixels: vec![Vec2::new(200.0, 200.0); 5],
            calib_weights: None,
            laser_weights: None,
        }];
        let dataset = LinescanDataset::new_single_plane(views).unwrap();

        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
            skew: 0.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let poses = vec![Iso3::identity()];
        let planes = vec![LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5)];
        let initial = LinescanInit::new(intrinsics, distortion, poses, planes).unwrap();

        let opts = LinescanSolveOptions::default();
        let (ir, _) = build_linescan_ir(&dataset, &initial, &opts).unwrap();

        let normal_id = ir.param_by_name("plane_0_normal").unwrap();
        let distance_id = ir.param_by_name("plane_0_distance").unwrap();

        let normal_param = &ir.params[normal_id.0];
        let distance_param = &ir.params[distance_id.0];

        assert_eq!(normal_param.dim, 3);
        assert_eq!(normal_param.manifold, ManifoldKind::S2);
        assert_eq!(distance_param.dim, 1);
        assert_eq!(distance_param.manifold, ManifoldKind::Euclidean);
    }
}
