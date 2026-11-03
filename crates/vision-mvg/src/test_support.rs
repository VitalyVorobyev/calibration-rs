//! Cameras and rigs shared by the crate's unit tests.

use crate::PixelMatch;
use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    PointCovariance, ProjectionParams, Pt2, SensorParams, Uncertainty, Vec3, make_pinhole_camera,
    pinhole_camera_params,
};

/// A pose from Euler angles and a translation.
pub(crate) fn iso(rx: f64, ry: f64, rz: f64, t: Vec3) -> Iso3 {
    Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz)),
    )
}

/// An 800 px pinhole camera with radial distortion `k1` (undistortion
/// iterated to full precision).
pub(crate) fn pinhole_params(k1: f64) -> CameraParams {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 790.0,
        cx: 640.0,
        cy: 480.0,
    };
    let distortion = BrownConrady5 {
        k1,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 40,
    };
    pinhole_camera_params(&make_pinhole_camera(k, distortion))
}

/// A rig of one pinhole camera with radial distortion `k1`.
pub(crate) fn pinhole_rig(k1: f64) -> CalibratedRig {
    CalibratedRig::single(pinhole_params(k1), Uncertainty::default()).unwrap()
}

/// A 300 px/rad equidistant fisheye with Kannala–Brandt distortion.
pub(crate) fn fisheye_params() -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.02,
                k2: -0.002,
                k3: 0.0,
                k4: 0.0,
                iters: 0,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 300.0,
            fy: 300.0,
            cx: 640.0,
            cy: 480.0,
        },
    }
}

/// The pixel where camera `camera` of `rig` sees the camera-frame point `p`.
pub(crate) fn pixel_of(rig: &CalibratedRig, camera: usize, p: &Vec3) -> Pt2 {
    rig.model(camera)
        .expect("camera in rig")
        .project_point_c(p)
        .expect("point projects")
}

/// A deterministic LCG in `[-1, 1)`.
pub(crate) struct Lcg(pub u64);

impl Lcg {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A match of two pixels with an isotropic noise of 0.5 px each.
pub(crate) fn match_of(pt1: Pt2, pt2: Pt2) -> PixelMatch {
    let noise = PointCovariance::isotropic(0.5).expect("0.5 px is a valid noise");
    PixelMatch::new(pt1, noise, pt2, noise)
}
