//! Camera model building blocks and composable pipelines.
//!
//! This module defines a composable camera pipeline with four stages:
//!
//! 1. `ProjectionModel`: map a 3D ray to normalized coordinates (e.g. pinhole).
//! 2. `DistortionModel`: apply radial/tangential distortion in normalized space.
//! 3. `SensorModel`: apply a sensor-plane homography (identity or tilt).
//! 4. `IntrinsicsModel`: map sensor-plane coordinates to pixels (K matrix).
//! 5. `FieldModel`: an optional smooth displacement in pixels (identity by
//!    default).
//!
//! The combined mapping is:
//! `pixel = field(intrinsics(sensor(distortion(projection(dir)))))`
//!
//! Parameter structs are provided for JSON serialization and for constructing
//! concrete camera models with f64 precision.

mod camera;
mod distortion;
mod field;
mod fix;
mod intrinsics;
mod kinds;
mod params;
mod projection;
mod sensor;

pub use camera::*;
pub use distortion::*;
pub use field::{
    BSplineField, FIELD_MAX_CELLS, FIELD_MAX_LIPSCHITZ, FieldModel, FieldParams, NoField,
};
pub use fix::*;
pub use intrinsics::*;
pub use kinds::*;
pub use params::*;
pub use projection::*;
pub use sensor::*;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    #[test]
    fn roundtrip_backproject_project_no_tilt_no_dist() {
        let params = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::None,
            sensor: SensorParams::Identity,
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 810.0,
                cx: 640.0,
                cy: 360.0,
            },
            field: FieldParams::None,
        };
        let cam = params.build().unwrap();

        let px = Point2::new(1000.0, 200.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.point_at_unit_depth().unwrap() * 2.5;
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
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 640.0,
                cy: 360.0,
            },
            field: FieldParams::None,
        };
        let cam = params.build().unwrap();

        let px = Point2::new(900.0, 500.0);
        let ray = cam.backproject_pixel(&px);
        let p = ray.point_at_unit_depth().unwrap() * 3.0;
        let px2 = cam.project_point_c(&p).unwrap();

        let err = (px2 - px).norm();
        assert!(err < 1e-6, "err={err}");
    }

    /// A tilted, distorted camera over a 1280 × 1024 image.
    fn tilted_camera() -> CameraParams {
        CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.2,
                    k2: 0.05,
                    k3: 0.0,
                    p1: 0.0005,
                    p2: -0.0003,
                    iters: 20,
                },
            },
            SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.03,
                    tilt_y: -0.02,
                },
            },
            FxFyCxCy {
                fx: 1200.0,
                fy: 1190.0,
                cx: 640.0,
                cy: 512.0,
            },
        )
    }

    /// A smooth field of a few pixels: `du = 2 sin`, `dv = 1.5 cos` over the
    /// control grid.
    fn smooth_field(cells: [u32; 2]) -> FieldParams {
        let mut f = BSplineField::zeros(cells, [1280, 1024]).unwrap();
        let nx = cells[0] as usize + 3;
        for (k, (du, dv)) in f.du.iter_mut().zip(f.dv.iter_mut()).enumerate() {
            let (i, j) = ((k % nx) as f64, (k / nx) as f64);
            *du = 2.0 * (0.7 * i + 0.3 * j).sin();
            *dv = 1.5 * (0.4 * i - 0.6 * j).cos();
        }
        FieldParams::BSpline { params: f }
    }

    #[test]
    fn a_zero_field_is_the_plain_camera_bit_for_bit() {
        let plain = tilted_camera();
        let zero = tilted_camera().with_field(FieldParams::BSpline {
            params: BSplineField::zeros([8, 6], [1280, 1024]).unwrap(),
        });
        let (a, b) = (plain.build().unwrap(), zero.build().unwrap());
        for &(x, y, z) in &[(0.0, 0.0, 1.0), (0.3, -0.2, 1.1), (-0.5, 0.4, 0.9)] {
            let p = nalgebra::Vector3::new(x, y, z);
            let (pa, pb) = (
                a.project_point_c(&p).unwrap(),
                b.project_point_c(&p).unwrap(),
            );
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
            let (ra, rb) = (a.backproject_pixel(&pa), b.backproject_pixel(&pa));
            assert_eq!(
                ra.direction.map(f64::to_bits),
                rb.direction.map(f64::to_bits)
            );
        }
    }

    #[test]
    fn a_field_camera_round_trips_inside_and_beyond_the_image() {
        let camera = tilted_camera().with_field(smooth_field([8, 6]));
        assert_eq!(
            camera.spec().unwrap().field,
            FieldKind::BSpline {
                cells: [8, 6],
                image_size: [1280, 1024]
            }
        );
        let model = camera.build().unwrap();
        let plain = tilted_camera().build().unwrap();
        // Inside, on the border and beyond it (the field is clamped there).
        for &(u, v) in &[
            (640.0, 512.0),
            (10.0, 20.0),
            (1280.0, 1024.0),
            (0.0, 700.0),
            (-35.0, 512.0),
            (1320.0, -12.0),
            (700.0, 1100.0),
        ] {
            let px = Point2::new(u, v);
            let ray = model.backproject_pixel(&px);
            let back = model
                .project_point_c(&(ray.point_at_unit_depth().unwrap() * 2.0))
                .unwrap();
            assert!(
                (back - px).norm() < 1e-9,
                "({u}, {v}): {}",
                (back - px).norm()
            );
            // The field moves the projection.
            let moved = plain
                .project_point_c(&ray.point_at_unit_depth().unwrap())
                .unwrap();
            assert!(
                (moved - px).norm() > 1e-3,
                "({u}, {v}): the field is ignored"
            );
        }
    }

    #[test]
    fn a_folding_field_does_not_build() {
        let FieldParams::BSpline { params: mut f } = smooth_field([8, 6]) else {
            unreachable!()
        };
        // A 120 px jump over a 160 px knot spacing.
        f.du[20] += 120.0;
        let err = tilted_camera()
            .with_field(FieldParams::BSpline { params: f })
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("Lipschitz"), "{err}");
    }

    #[test]
    fn camera_json_omits_no_field_and_round_trips_one() {
        let plain = serde_json::to_value(tilted_camera()).unwrap();
        assert!(plain.get("field").is_none());
        let back: CameraParams = serde_json::from_value(plain).unwrap();
        assert!(back.field.is_none());

        let camera = tilted_camera().with_field(smooth_field([3, 2]));
        let json = serde_json::to_value(&camera).unwrap();
        assert_eq!(json["field"]["type"], "b_spline");
        let back: CameraParams = serde_json::from_value(json).unwrap();
        assert_eq!(back.field, camera.field);
        assert_eq!(back.spec(), camera.spec());

        // The spec omits an absent field too, and keeps a present one.
        let spec = serde_json::to_value(tilted_camera().spec().unwrap()).unwrap();
        assert!(spec.get("field").is_none());
        let spec = camera.spec().unwrap();
        let back: CameraModelSpec =
            serde_json::from_value(serde_json::to_value(spec).unwrap()).unwrap();
        assert_eq!(back, spec);
    }
}
