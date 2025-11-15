use calib_core::{CameraIntrinsics, Iso3, PinholeCamera, Pt3, RadialTangential, Real, Vec2};
use calib_optim::backend_lm::LmBackend;
use calib_optim::planar_intrinsics::{
    pack_initial_params, refine_planar_intrinsics, PlanarIntrinsicsProblem,
    PlanarViewObservations,
};
use calib_optim::problem::SolveOptions;
use calib_optim::robust::RobustKernel;
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

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

impl From<&RobustKernelConfig> for RobustKernel {
    fn from(cfg: &RobustKernelConfig) -> Self {
        match cfg {
            RobustKernelConfig::None => RobustKernel::None,
            RobustKernelConfig::Huber { delta } => RobustKernel::Huber { delta: *delta },
            RobustKernelConfig::Cauchy { c } => RobustKernel::Cauchy { c: *c },
        }
    }
}

fn kernel_from_option(opt: &Option<RobustKernelConfig>) -> RobustKernel {
    opt.as_ref()
        .map(|cfg| RobustKernel::from(cfg))
        .unwrap_or(RobustKernel::None)
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
    pub camera: PinholeCamera,
    pub final_cost: Real,
    pub iterations: usize,
    pub converged: bool,
}

pub fn run_planar_intrinsics(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
) -> PlanarIntrinsicsReport {
    // 1. Build PlanarViewObservations from input.
    // 2. Build PlanarIntrinsicsProblem with appropriate robust kernel.
    // 3. Generate a simple initial camera guess and per-view poses (e.g. identity poses, or something basic).
    // 4. Pack initial params.
    // 5. Run `refine_planar_intrinsics` using `LmBackend` and SolveOptions derived from config.
    // 6. Return a PlanarIntrinsicsReport.
    unimplemented!("Implement in the next TDD step.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::Pt2;

    #[ignore]
    #[test]
    fn planar_intrinsics_pipeline_smoke_test() {
        // simple synthetic board with 2x2 points
        let board_points = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.1, 0.0, 0.0),
            Pt3::new(0.1, 0.1, 0.0),
            Pt3::new(0.0, 0.1, 0.0),
        ];

        let camera = PinholeCamera {
            intrinsics: CameraIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            distortion: Some(RadialTangential::BrownConrady {
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            }),
        };
        let translation = Vector3::new(0.0, 0.0, 1.0);
        let rotation = UnitQuaternion::identity();
        let pose = Iso3::from_parts(translation.into(), rotation);

        let mut points_2d = Vec::new();
        for pw in &board_points {
            let pc = pose.transform_point(pw);
            let proj = camera.project(&pc);
            points_2d.push(Pt2::new(proj.x, proj.y).coords);
        }

        let input = PlanarIntrinsicsInput {
            views: vec![PlanarViewData {
                points_3d: board_points,
                points_2d,
            }],
        };

        let config = PlanarIntrinsicsConfig::default();

        // for now expect panic due to unimplemented!
        let _ = std::panic::catch_unwind(|| run_planar_intrinsics(&input, &config));
    }
}
