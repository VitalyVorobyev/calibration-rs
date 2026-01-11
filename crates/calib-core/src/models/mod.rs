//! Camera model building blocks and composable pipelines.
//!
//! This module defines a composable camera pipeline with four stages:
//!
//! 1. `ProjectionModel`: map a 3D ray to normalized coordinates (e.g. pinhole).
//! 2. `DistortionModel`: apply radial/tangential distortion in normalized space.
//! 3. `SensorModel`: apply a sensor-plane homography (identity or tilt).
//! 4. `IntrinsicsModel`: map sensor-plane coordinates to pixels (K matrix).
//!
//! The combined mapping is:
//! `pixel = intrinsics(sensor(distortion(projection(dir))))`
//!
//! Parameter structs are provided for JSON serialization and for constructing
//! concrete camera models with f64 precision.

mod camera;
mod distortion;
mod intrinsics;
mod params;
mod projection;
mod sensor;

pub use camera::*;
pub use distortion::*;
pub use intrinsics::*;
pub use params::*;
pub use projection::*;
pub use sensor::*;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn roundtrip_backproject_project_no_tilt_no_dist() {
        let params = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::None,
            sensor: SensorParams::Identity,
            intrinsics: IntrinsicsParams::FxFyCxCySkew {
                params: FxFyCxCySkew {
                    fx: 800.0,
                    fy: 810.0,
                    cx: 640.0,
                    cy: 360.0,
                    skew: 0.0,
                },
            },
        };
        let cam = params.build();

        let px = Vector2::new(1000.0, 200.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.point * 2.5;
        let px2 = cam.project_point_c(&p).unwrap();

        let err = (px2 - px).norm();
        assert!(err < 1e-9, "err={err}");
    }

    #[test]
    fn scheimpflug_compiles_and_inverts() {
        let params = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::None,
            sensor: SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.02,
                    tilt_y: -0.01,
                },
            },
            intrinsics: IntrinsicsParams::FxFyCxCySkew {
                params: FxFyCxCySkew {
                    fx: 800.0,
                    fy: 800.0,
                    cx: 640.0,
                    cy: 360.0,
                    skew: 0.0,
                },
            },
        };
        let cam = params.build();

        let px = Vector2::new(900.0, 500.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.point * 3.0;
        let px2 = cam.project_point_c(&p).unwrap();

        let err = (px2 - px).norm();
        assert!(err < 1e-6, "err={err}");
    }
}
