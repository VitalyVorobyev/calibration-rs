//! Camera model building blocks and composable pipelines.

mod camera;
mod config;
mod distortion;
mod intrinsics;
mod projection;
mod sensor;

pub use camera::*;
pub use config::*;
pub use distortion::*;
pub use intrinsics::*;
pub use projection::*;
pub use sensor::*;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn roundtrip_backproject_project_no_tilt_no_dist() {
        let cfg = CameraConfig {
            projection: ProjectionConfig::Pinhole,
            distortion: DistortionConfig::None,
            sensor: SensorConfig::Identity,
            intrinsics: IntrinsicsConfig::FxFyCxCySkew {
                fx: 800.0,
                fy: 810.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
        };
        let cam = cfg.build();

        let px = Vector2::new(1000.0, 200.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.dir * 2.5;
        let px2 = cam.project_point_c(&p).unwrap();

        let err = (px2 - px).norm();
        assert!(err < 1e-9, "err={err}");
    }

    #[test]
    fn scheimpflug_compiles_and_inverts() {
        let cfg = CameraConfig {
            projection: ProjectionConfig::Pinhole,
            distortion: DistortionConfig::None,
            sensor: SensorConfig::Scheimpflug(ScheimpflugParams {
                tilt_x: 0.02,
                tilt_y: -0.01,
            }),
            intrinsics: IntrinsicsConfig::FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
        };
        let cam = cfg.build();

        let px = Vector2::new(900.0, 500.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.dir * 3.0;
        let px2 = cam.project_point_c(&p).unwrap();

        let err = (px2 - px).norm();
        assert!(err < 1e-6, "err={err}");
    }
}
