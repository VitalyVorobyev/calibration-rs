//! The cameras a calibration export reports, for consumers that treat every
//! export alike.

use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, PinholeCamera,
    ProjectionParams, ScheimpflugParams, SensorParams, Uncertainty, pinhole_camera_params,
};
use vision_calibration_optim::{
    Error as OptimError, ProjectionUncertainty, projection_uncertainty,
};

use crate::Error;
use crate::laserline_device::LaserlineDeviceExport;
use crate::planar_intrinsics::PlanarIntrinsicsExport;
use crate::rig_extrinsics::RigExtrinsicsExport;
use crate::rig_handeye::RigHandeyeExport;
use crate::rig_handeye_laserline::RigHandeyeLaserlineExport;
use crate::rig_laserline_device::RigLaserlineDeviceExport;
use crate::scheimpflug_intrinsics::ScheimpflugIntrinsicsExport;
use crate::single_cam_handeye::SingleCamHandeyeExport;

/// A calibration export's cameras and the uncertainty of its parameters.
pub trait CalibratedCameras {
    /// Each camera's model, in camera order.
    fn camera_params(&self) -> Vec<CameraParams>;

    /// The uncertainty of the estimated parameters.
    fn parameter_uncertainty(&self) -> &Uncertainty;

    /// The projection uncertainty of camera `camera` over an image of
    /// `image_size` pixels: the major-axis σ of where a camera-frame ray
    /// lands, on a grid across the image (see [`ProjectionUncertainty`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `camera` is out of range, the result has
    /// no covariance, its covariance holds none of the camera's parameters
    /// (the final solve held them fixed), or the image or camera model is
    /// one [`projection_uncertainty`] rejects; [`Error::Optim`] when no ray
    /// reaches the image centre.
    fn projection_uncertainty(
        &self,
        camera: usize,
        image_size: [u32; 2],
    ) -> Result<ProjectionUncertainty, Error> {
        let cameras = self.camera_params();
        let params = cameras.get(camera).ok_or_else(|| {
            Error::invalid_input(format!(
                "camera {camera} out of range: the result has {} cameras",
                cameras.len()
            ))
        })?;
        let covariance = match self.parameter_uncertainty() {
            Uncertainty::Available(covariance) => covariance,
            Uncertainty::Unavailable { reason } => {
                return Err(Error::invalid_input(format!(
                    "the result has no parameter covariance: {reason:?}"
                )));
            }
        };
        projection_uncertainty(params, covariance, camera, image_size).map_err(|err| match err {
            OptimError::InvalidInput { reason } => Error::InvalidInput { reason },
            err => err.into(),
        })
    }
}

/// A pinhole camera with Brown-Conrady distortion and an optional
/// Scheimpflug sensor.
fn brown_conrady_camera_params(
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: Option<ScheimpflugParams>,
) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: distortion },
        sensor: sensor.map_or(SensorParams::Identity, |params| SensorParams::Scheimpflug {
            params,
        }),
        intrinsics: IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
    }
}

/// Rig cameras, each with its Scheimpflug sensor when the rig has them.
fn rig_camera_params(
    cameras: &[PinholeCamera],
    sensors: Option<&[ScheimpflugParams]>,
) -> Vec<CameraParams> {
    cameras
        .iter()
        .enumerate()
        .map(|(i, camera)| {
            let sensor = sensors.and_then(|s| s.get(i)).copied();
            brown_conrady_camera_params(camera.k, camera.dist, sensor)
        })
        .collect()
}

impl CalibratedCameras for PlanarIntrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.params.camera.clone()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for ScheimpflugIntrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.params.camera.clone()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for SingleCamHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![pinhole_camera_params(&self.camera)]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for LaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        let p = &self.estimate.params;
        vec![brown_conrady_camera_params(
            p.intrinsics,
            p.distortion,
            Some(p.sensor),
        )]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.estimate.uncertainty
    }
}

impl CalibratedCameras for RigExtrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        rig_camera_params(&self.cameras, self.sensors.as_deref())
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for RigHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        rig_camera_params(&self.cameras, self.sensors.as_deref())
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for RigLaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        rig_camera_params(&self.cameras, self.sensors.as_deref())
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

impl CalibratedCameras for RigHandeyeLaserlineExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        rig_camera_params(&self.cameras, Some(&self.sensors))
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}
