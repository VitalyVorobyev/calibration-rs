//! Laserline bundle adjustment: joint optimization of camera intrinsics, distortion,
//! target poses, and laser plane parameters.
//!
//! This problem builder combines standard planar calibration observations (corners)
//! with laser line observations to estimate laser plane parameters in camera frame.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions, SolveReport};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
use crate::params::laser_plane::LaserPlane;
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use anyhow::{anyhow, ensure, Result};
use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt2, Real, ScheimpflugParams, View};
use nalgebra::{DVector, DVectorView};
use std::collections::HashMap;

/// Metadata for a laserline view (laser pixels + optional weights).
#[derive(Debug, Clone)]
pub struct LaserlineMeta {
    /// Laser line pixel observations.
    pub laser_pixels: Vec<Pt2>,
    /// Optional per-pixel weights.
    pub laser_weights: Option<Vec<f64>>,
}

impl LaserlineMeta {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.laser_pixels.is_empty(),
            "need at least one laser pixel observation"
        );
        if let Some(weights) = &self.laser_weights {
            ensure!(
                weights.len() == self.laser_pixels.len(),
                "laser weight count must match pixel count"
            );
            ensure!(
                weights.iter().all(|w| *w >= 0.0),
                "laser weights must be non-negative"
            );
        }
        Ok(())
    }

    fn laser_weight(&self, idx: usize) -> f64 {
        self.laser_weights.as_ref().map_or(1.0, |w| w[idx])
    }
}

/// A single view with calibration correspondences and laserline pixels.
pub type LaserlineView = View<LaserlineMeta>;

/// Dataset for laserline bundle adjustment.
pub type LaserlineDataset = Vec<LaserlineView>;

/// Initial values for laserline bundle adjustment.
#[derive(Debug, Clone)]
pub struct LaserlineParams {
    pub intrinsics: FxFyCxCySkew<Real>,
    pub distortion: BrownConrady5<Real>,
    /// Scheimpflug sensor parameters (use defaults for identity sensor).
    pub sensor: ScheimpflugParams,
    pub poses: Vec<Iso3>,
    pub plane: LaserPlane,
}

impl LaserlineParams {
    pub fn new(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        sensor: ScheimpflugParams,
        poses: Vec<Iso3>,
        plane: LaserPlane,
    ) -> Result<Self> {
        ensure!(!poses.is_empty(), "need at least one pose");
        Ok(Self {
            intrinsics,
            distortion,
            sensor,
            poses,
            plane,
        })
    }
}

/// Result of laserline bundle adjustment.
#[derive(Debug, Clone)]
pub struct LaserlineEstimate {
    pub params: LaserlineParams,
    pub report: SolveReport,
}

/// Type of laser plane residual to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LaserlineResidualType {
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
    #[default]
    LineDistNormalized,
}

/// Solve options for laserline bundle adjustment.
#[derive(Debug, Clone)]
pub struct LaserlineSolveOptions {
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
    /// Fix Scheimpflug sensor parameters during optimization
    pub fix_sensor: bool,
    /// Indices of poses to fix (e.g., \[0\] to fix first pose for gauge freedom)
    pub fix_poses: Vec<usize>,
    /// Indices of planes to fix
    pub fix_plane: bool,
    /// Laser residual type: point-to-plane distance or line-distance in normalized plane
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for LaserlineSolveOptions {
    fn default() -> Self {
        Self {
            calib_loss: RobustLoss::Huber { scale: 1.0 },
            laser_loss: RobustLoss::Huber { scale: 0.01 }, // Smaller scale for plane residuals
            fix_intrinsics: false,
            fix_distortion: false,
            fix_k3: true,
            fix_sensor: true,
            fix_poses: vec![0], // Fix first pose by default
            fix_plane: false,
            laser_residual_type: LaserlineResidualType::LineDistNormalized, // New default
        }
    }
}

fn pack_scheimpflug(sensor: &ScheimpflugParams) -> DVector<f64> {
    DVector::from_vec(vec![sensor.tilt_x, sensor.tilt_y])
}

fn unpack_scheimpflug(sensor: DVectorView<'_, f64>) -> Result<ScheimpflugParams> {
    ensure!(
        sensor.len() == 2,
        "Scheimpflug sensor params require 2D vector, got {}",
        sensor.len()
    );
    Ok(ScheimpflugParams {
        tilt_x: sensor[0],
        tilt_y: sensor[1],
    })
}

/// Extract solution from backend result.
fn extract_solution(
    solution: crate::backend::BackendSolution,
    num_poses: usize,
) -> Result<LaserlineEstimate> {
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

    let normal_name = "plane_normal".to_owned();
    let distance_name = "plane_distance".to_owned();
    let plane_normal = solution
        .params
        .get(&normal_name)
        .ok_or_else(|| anyhow!("missing {} in solution", normal_name))?;
    let plane_distance = solution
        .params
        .get(&distance_name)
        .ok_or_else(|| anyhow!("missing {} in solution", distance_name))?;

    let plane = LaserPlane::from_split_dvec(plane_normal.as_view(), plane_distance.as_view())?;
    let sensor = unpack_scheimpflug(
        solution
            .params
            .get("sensor")
            .ok_or_else(|| anyhow!("missing sensor in solution"))?
            .as_view(),
    )?;

    Ok(LaserlineEstimate {
        params: LaserlineParams::new(intrinsics, distortion, sensor, poses, plane)?,
        report: solution.solve_report,
    })
}

/// Optimize laserline calibration with joint bundle adjustment.
///
/// This function jointly optimizes camera intrinsics, distortion parameters,
/// camera-to-target poses, and laser plane parameters using both calibration
/// feature observations and laser line observations.
///
/// # Example
///
/// ```ignore
/// use calib_optim::problems::laserline_bundle::*;
/// use calib_core::ScheimpflugParams;
///
/// let dataset = views;
/// let sensor = ScheimpflugParams::default(); // identity sensor
/// let initial = LaserlineParams::new(intrinsics, distortion, sensor, poses, plane)?;
/// let opts = LaserlineSolveOptions::default();
/// let backend_opts = BackendSolveOptions::default();
///
/// let result = optimize_laserline(&dataset, &initial, &opts, &backend_opts)?;
/// println!(
///     "Laser plane: normal={:?}, distance={}",
///     result.params.plane.normal,
///     result.params.plane.distance
/// );
/// ```
pub fn optimize_laserline(
    dataset: &LaserlineDataset,
    initial: &LaserlineParams,
    opts: &LaserlineSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<LaserlineEstimate> {
    let (ir, initial_map) = build_laserline_ir(dataset, initial, opts)?;
    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, backend_opts)?;
    extract_solution(solution, dataset.len())
}

/// Build IR for laserline bundle adjustment.
fn build_laserline_ir(
    dataset: &LaserlineDataset,
    initial: &LaserlineParams,
    opts: &LaserlineSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(!dataset.is_empty(), "need at least one view");
    ensure!(
        dataset.len() == initial.poses.len(),
        "dataset has {} views but {} initial poses",
        dataset.len(),
        initial.poses.len()
    );
    for (idx, view) in dataset.iter().enumerate() {
        view.meta
            .validate()
            .map_err(|e| anyhow!("view {}: {}", idx, e))?;
    }

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

    // Add sensor parameter block (Scheimpflug)
    let sensor_fixed = if opts.fix_sensor {
        FixedMask::all_fixed(2)
    } else {
        FixedMask::all_free()
    };
    let sensor_id =
        ir.add_param_block("sensor", 2, ManifoldKind::Euclidean, sensor_fixed, None);
    initial_map.insert("sensor".to_string(), pack_scheimpflug(&initial.sensor));

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
    let normal_fixed = if opts.fix_plane {
        FixedMask::all_fixed(3)
    } else {
        FixedMask::all_free()
    };
    let distance_fixed = if opts.fix_plane {
        FixedMask::all_fixed(1)
    } else {
        FixedMask::all_free()
    };

    let normal_name = "plane_normal";
    let distance_name = "plane_distance";
    let normal_id = ir.add_param_block(normal_name, 3, ManifoldKind::S2, normal_fixed, None);
    let distance_id = ir.add_param_block(
        distance_name,
        1,
        ManifoldKind::Euclidean,
        distance_fixed,
        None,
    );
    initial_map.insert(normal_name.to_owned(), initial.plane.normal_to_dvec());
    initial_map.insert(distance_name.to_owned(), initial.plane.distance_to_dvec());

    // Add residual blocks
    for (view_idx, view) in dataset.iter().enumerate() {
        let pose_id = pose_ids[view_idx];

        // Calibration reprojection residuals
        for (pt_idx, (pt_3d, pt_2d)) in view
            .obs
            .points_3d
            .iter()
            .zip(&view.obs.points_2d)
            .enumerate()
        {
            let w = view.obs.weights.as_ref().map_or(1.0, |w| w[pt_idx]);
            ir.add_residual_block(ResidualBlock {
                params: vec![intrinsics_id, distortion_id, sensor_id, pose_id],
                loss: opts.calib_loss,
                factor: FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 {
                    pw: [pt_3d.x, pt_3d.y, pt_3d.z],
                    uv: [pt_2d.x, pt_2d.y],
                    w,
                },
                residual_dim: 2,
            });
        }

        // Laser plane residuals (one per plane, typically just one)
        for (laser_idx, laser_pixel) in view.meta.laser_pixels.iter().enumerate() {
            let w = view.meta.laser_weight(laser_idx);

            // Select factor type based on options
            let factor = match opts.laser_residual_type {
                LaserlineResidualType::PointToPlane => FactorKind::LaserPlanePixel {
                    laser_pixel: [laser_pixel.x, laser_pixel.y],
                    w,
                },
                LaserlineResidualType::LineDistNormalized => FactorKind::LaserLineDist2D {
                    laser_pixel: [laser_pixel.x, laser_pixel.y],
                    w,
                },
            };

            ir.add_residual_block(ResidualBlock {
                params: vec![
                    intrinsics_id,
                    distortion_id,
                    sensor_id,
                    pose_id,
                    normal_id,
                    distance_id,
                ],
                loss: opts.laser_loss,
                factor,
                residual_dim: 1,
            });
        }
    }

    Ok((ir, initial_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{BrownConrady5, CorrespondenceView, FxFyCxCySkew, Pt3, ScheimpflugParams};

    #[test]
    fn ir_validation_catches_missing_param() {
        let dataset = vec![View::new(
            CorrespondenceView::new(
                vec![Pt3::new(0.0, 0.0, 0.0); 4],
                vec![Pt2::new(100.0, 100.0); 4],
            )
            .unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(200.0, 200.0); 5],
                laser_weights: None,
            },
        )];

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
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);
        let initial =
            LaserlineParams::new(intrinsics, distortion, ScheimpflugParams::default(), poses, plane)
                .unwrap();

        let opts = LaserlineSolveOptions::default();
        let (ir, mut initial_map) = build_laserline_ir(&dataset, &initial, &opts).unwrap();

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
    #[ignore = "TODO: Fix after laserline pipeline integration"]
    fn ir_uses_s2_for_plane_normal() {
        let dataset = vec![View::new(
            CorrespondenceView::new(
                vec![Pt3::new(0.0, 0.0, 0.0); 4],
                vec![Pt2::new(100.0, 100.0); 4],
            )
            .unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(200.0, 200.0); 5],
                laser_weights: None,
            },
        )];

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
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);
        let initial =
            LaserlineParams::new(intrinsics, distortion, ScheimpflugParams::default(), poses, plane)
                .unwrap();

        let opts = LaserlineSolveOptions::default();
        let (ir, _) = build_laserline_ir(&dataset, &initial, &opts).unwrap();

        let normal_id = ir.param_by_name("plane_normal").unwrap();
        let distance_id = ir.param_by_name("plane_distance").unwrap();

        let normal_param = &ir.params[normal_id.0];
        let distance_param = &ir.params[distance_id.0];

        assert_eq!(normal_param.dim, 3);
        assert_eq!(normal_param.manifold, ManifoldKind::S2);
        assert_eq!(distance_param.dim, 1);
        assert_eq!(distance_param.manifold, ManifoldKind::Euclidean);
    }
}
