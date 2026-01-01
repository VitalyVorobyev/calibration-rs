use calib_core::{
    BrownConrady5, Camera, CameraConfig, DistortionConfig, FxFyCxCySkew, IdentitySensor,
    IntrinsicsConfig, Iso3, Pinhole, ProjectionConfig, Pt3, Real, SensorConfig, Vec2,
};
use calib_optim::backend_lm::LmBackend;
use calib_optim::planar_intrinsics::{
    pack_initial_params, refine_planar_intrinsics, PinholeCamera, PlanarIntrinsicsProblem,
    PlanarViewObservations,
};
use calib_optim::problem::SolveOptions;
use calib_optim::robust::RobustKernel;
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsConfig {
    /// Robust kernel to use for residuals.
    pub robust_kernel: Option<RobustKernelConfig>,
    /// Maximum LM iterations (if `None`, use backend default).
    pub max_iters: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustKernelConfig {
    None,
    Huber { delta: Real },
    Cauchy { c: Real },
}

impl Default for PlanarIntrinsicsConfig {
    fn default() -> Self {
        Self {
            robust_kernel: Some(RobustKernelConfig::None),
            max_iters: None,
        }
    }
}

impl RobustKernelConfig {
    pub fn to_kernel(&self) -> RobustKernel {
        match *self {
            RobustKernelConfig::None => RobustKernel::None,
            RobustKernelConfig::Huber { delta } => RobustKernel::Huber { delta },
            RobustKernelConfig::Cauchy { c } => RobustKernel::Cauchy { c },
        }
    }
}

impl PlanarIntrinsicsConfig {
    pub fn kernel(&self) -> RobustKernel {
        self.robust_kernel
            .as_ref()
            .map(RobustKernelConfig::to_kernel)
            .unwrap_or(RobustKernel::None)
    }

    pub fn max_iters_or_default(&self) -> usize {
        self.max_iters.unwrap_or(50)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarViewData {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInput {
    pub views: Vec<PlanarViewData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsReport {
    pub camera: CameraConfig,
    pub final_cost: Real,
    pub iterations: usize,
    pub converged: bool,
}

/// Errors returned by the planar intrinsics pipeline.
#[derive(Debug, Clone)]
pub enum PlanarIntrinsicsError {
    /// No views were provided.
    EmptyViews,
    /// A view has mismatched 3D/2D point counts.
    MismatchedPointCounts {
        view: usize,
        points_3d: usize,
        points_2d: usize,
    },
    /// A view has too few correspondences to be usable.
    NotEnoughPoints { view: usize, points: usize },
}

impl fmt::Display for PlanarIntrinsicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanarIntrinsicsError::EmptyViews => {
                write!(f, "need at least one view for planar intrinsics")
            }
            PlanarIntrinsicsError::MismatchedPointCounts {
                view,
                points_3d,
                points_2d,
            } => write!(
                f,
                "view {} has mismatched 3D/2D points ({} vs {})",
                view, points_3d, points_2d
            ),
            PlanarIntrinsicsError::NotEnoughPoints { view, points } => {
                write!(f, "view {} needs at least 4 points (got {})", view, points)
            }
        }
    }
}

impl Error for PlanarIntrinsicsError {}

fn make_pinhole_camera(k: FxFyCxCySkew<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

fn pinhole_camera_config(camera: &PinholeCamera) -> CameraConfig {
    CameraConfig {
        projection: ProjectionConfig::Pinhole,
        distortion: DistortionConfig::BrownConrady5 {
            k1: camera.dist.k1,
            k2: camera.dist.k2,
            k3: camera.dist.k3,
            p1: camera.dist.p1,
            p2: camera.dist.p2,
            iters: Some(camera.dist.iters),
        },
        sensor: SensorConfig::Identity,
        intrinsics: IntrinsicsConfig::FxFyCxCySkew {
            fx: camera.k.fx,
            fy: camera.k.fy,
            cx: camera.k.cx,
            cy: camera.k.cy,
            skew: camera.k.skew,
        },
    }
}

pub fn run_planar_intrinsics(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
) -> Result<PlanarIntrinsicsReport, PlanarIntrinsicsError> {
    if input.views.is_empty() {
        return Err(PlanarIntrinsicsError::EmptyViews);
    }

    let mut observations = Vec::new();
    for (idx, view) in input.views.iter().enumerate() {
        if view.points_3d.len() != view.points_2d.len() {
            return Err(PlanarIntrinsicsError::MismatchedPointCounts {
                view: idx,
                points_3d: view.points_3d.len(),
                points_2d: view.points_2d.len(),
            });
        }
        if view.points_3d.len() < 4 {
            return Err(PlanarIntrinsicsError::NotEnoughPoints {
                view: idx,
                points: view.points_3d.len(),
            });
        }
        observations.push(PlanarViewObservations::new(
            view.points_3d.clone(),
            view.points_2d.clone(),
        ));
    }

    let problem = PlanarIntrinsicsProblem::new(observations).with_kernel(config.kernel());

    let initial_camera = make_pinhole_camera(
        FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
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

    let poses0: Vec<Iso3> = (0..problem.num_views())
        .map(|_| {
            Iso3::from_parts(
                Vector3::new(0.0, 0.0, 1.0).into(),
                UnitQuaternion::identity(),
            )
        })
        .collect();

    let x0 = pack_initial_params(&initial_camera, &poses0);

    let backend = LmBackend;
    let opts = SolveOptions {
        max_iters: config.max_iters_or_default(),
        ..Default::default()
    };

    let (camera, _poses, report) = refine_planar_intrinsics(&backend, &problem, x0, &opts);
    let camera_cfg = pinhole_camera_config(&camera);

    Ok(PlanarIntrinsicsReport {
        camera: camera_cfg,
        final_cost: report.final_cost,
        iterations: report.iterations,
        converged: report.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intrinsics_from_config(cfg: &CameraConfig) -> FxFyCxCySkew<Real> {
        match &cfg.intrinsics {
            IntrinsicsConfig::FxFyCxCySkew {
                fx,
                fy,
                cx,
                cy,
                skew,
            } => FxFyCxCySkew {
                fx: *fx,
                fy: *fy,
                cx: *cx,
                cy: *cy,
                skew: *skew,
            },
        }
    }

    #[test]
    fn planar_intrinsics_pipeline_synthetic_recovers_intrinsics() {
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let pose = Iso3::from_parts(translation.into(), rotation);

            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = pose.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
            });
        }

        let input = PlanarIntrinsicsInput { views };
        let config = PlanarIntrinsicsConfig::default();

        let report = run_planar_intrinsics(&input, &config).expect("pipeline should succeed");
        assert!(report.converged, "LM did not converge");
        assert!(
            report.final_cost < 1e-6,
            "final cost too high: {}",
            report.final_cost
        );

        let ki = intrinsics_from_config(&report.camera);
        assert!((ki.fx - k_gt.fx).abs() < 20.0);
        assert!((ki.fy - k_gt.fy).abs() < 20.0);
        assert!((ki.cx - k_gt.cx).abs() < 20.0);
        assert!((ki.cy - k_gt.cy).abs() < 20.0);
    }

    #[test]
    fn config_json_roundtrip() {
        let config = PlanarIntrinsicsConfig {
            robust_kernel: Some(RobustKernelConfig::Huber { delta: 2.5 }),
            max_iters: Some(80),
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(
            json.contains("Huber") && json.contains("2.5"),
            "json missing expected content: {}",
            json
        );

        let de: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.max_iters, config.max_iters);
        match (de.robust_kernel, config.robust_kernel) {
            (
                Some(RobustKernelConfig::Huber { delta: d1 }),
                Some(RobustKernelConfig::Huber { delta: d2 }),
            ) => assert!((d1 - d2).abs() < 1e-12),
            other => panic!("mismatch in kernels: {:?}", other),
        }
    }

    #[test]
    fn input_json_roundtrip() {
        let input = PlanarIntrinsicsInput {
            views: vec![PlanarViewData {
                points_3d: vec![
                    Pt3::new(0.0, 0.0, 0.0),
                    Pt3::new(1.0, 0.0, 0.0),
                    Pt3::new(1.0, 1.0, 0.0),
                    Pt3::new(0.0, 1.0, 0.0),
                ],
                points_2d: vec![
                    Vec2::new(100.0, 100.0),
                    Vec2::new(200.0, 100.0),
                    Vec2::new(200.0, 200.0),
                    Vec2::new(100.0, 200.0),
                ],
            }],
        };

        let json = serde_json::to_string_pretty(&input).unwrap();
        let de: PlanarIntrinsicsInput = serde_json::from_str(&json).unwrap();

        assert_eq!(de.views.len(), input.views.len());
        for (view_a, view_b) in de.views.iter().zip(input.views.iter()) {
            assert_eq!(view_a.points_3d.len(), view_b.points_3d.len());
            assert_eq!(view_a.points_2d.len(), view_b.points_2d.len());
            for (a, b) in view_a.points_3d.iter().zip(view_b.points_3d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
                assert!((a.z - b.z).abs() < 1e-12);
            }
            for (a, b) in view_a.points_2d.iter().zip(view_b.points_2d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn report_json_roundtrip() {
        let cam = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: -0.1,
                k2: 0.01,
                k3: 0.0,
                p1: 0.001,
                p2: -0.001,
                iters: 8,
            },
        );
        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_config(&cam),
            final_cost: 1e-8,
            iterations: 12,
            converged: true,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let de: PlanarIntrinsicsReport = serde_json::from_str(&json).unwrap();

        let ki_de = intrinsics_from_config(&de.camera);
        let ki_report = intrinsics_from_config(&report.camera);

        assert!((ki_de.fx - ki_report.fx).abs() < 1e-12);
        assert!((ki_de.fy - ki_report.fy).abs() < 1e-12);
        assert!((ki_de.cx - ki_report.cx).abs() < 1e-12);
        assert!((ki_de.cy - ki_report.cy).abs() < 1e-12);

        match (&de.camera.distortion, &report.camera.distortion) {
            (
                DistortionConfig::BrownConrady5 {
                    k1: k1a,
                    k2: k2a,
                    k3: k3a,
                    p1: p1a,
                    p2: p2a,
                    ..
                },
                DistortionConfig::BrownConrady5 {
                    k1: k1b,
                    k2: k2b,
                    k3: k3b,
                    p1: p1b,
                    p2: p2b,
                    ..
                },
            ) => {
                assert!((k1a - k1b).abs() < 1e-12);
                assert!((k2a - k2b).abs() < 1e-12);
                assert!((p1a - p1b).abs() < 1e-12);
                assert!((p2a - p2b).abs() < 1e-12);
                assert!((k3a - k3b).abs() < 1e-12);
            }
            other => panic!("distortion mismatch: {:?}", other),
        }

        assert!((de.final_cost - report.final_cost).abs() < 1e-12);
        assert_eq!(de.iterations, report.iterations);
        assert_eq!(de.converged, report.converged);
    }
}
