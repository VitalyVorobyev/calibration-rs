//! The cameras a calibration export reports, for consumers that treat every
//! export alike.

use serde::de::DeserializeOwned;

use nalgebra::Matrix2;
use vision_calibration_core::{
    CameraParams, ImageManifest, ParameterCovariance, PerFeatureResiduals, Uncertainty,
};
use vision_calibration_optim::{
    Error as OptimError, ProjectionUncertainty, camera_parameters, projection_covariance_at,
    projection_uncertainty,
};

use crate::Error;
use crate::common::ExportKind;
use crate::laserline_device::LaserlineDeviceExport;
use crate::planar_intrinsics::PlanarIntrinsicsExport;
use crate::rig_extrinsics::RigExtrinsicsExport;
use crate::rig_handeye::RigHandeyeExport;
use crate::rig_handeye_laserline::RigHandeyeLaserlineExport;
use crate::rig_laserline_device::RigLaserlineDeviceExport;
use crate::single_cam_handeye::SingleCamHandeyeExport;

/// A calibration export's cameras, the uncertainty of its parameters and the
/// observations behind them.
pub trait CalibratedCameras {
    /// Each camera's model, in camera order.
    fn camera_params(&self) -> Vec<CameraParams>;

    /// The uncertainty of the estimated parameters.
    fn parameter_uncertainty(&self) -> &Uncertainty;

    /// The residual of every observed feature at the solution.
    fn feature_residuals(&self) -> &PerFeatureResiduals;

    /// The images behind the observations, when the export records them.
    fn images(&self) -> Option<&ImageManifest>;

    /// The value of the camera parameter a covariance label names (`fx`,
    /// `cam1.k1`, `tilt_x`); `None` for any other label (a pose, a plane) or
    /// a camera the export does not have.
    fn camera_parameter(&self, label: &str) -> Option<f64> {
        let (camera, name) = split_camera_label(label);
        let params = self.camera_params().into_iter().nth(camera)?;
        camera_parameters(&params)
            .ok()?
            .into_iter()
            .find_map(|(l, v)| (l == name).then_some(v))
    }

    /// The projection uncertainty of camera `camera` over an image of
    /// `image_size` pixels: the major-axis σ of where a camera-frame ray
    /// lands, on a grid across the image (see [`ProjectionUncertainty`]).
    ///
    /// # Errors
    ///
    /// [`Error::CovarianceUnavailable`] when the result has no covariance;
    /// [`Error::InvalidInput`] when `camera` is out of range, its covariance
    /// holds none of the camera's parameters (the final solve held them
    /// fixed), or the image or camera model is one [`projection_uncertainty`]
    /// rejects; [`Error::Optim`] when no ray reaches the image centre.
    fn projection_uncertainty(
        &self,
        camera: usize,
        image_size: [u32; 2],
    ) -> Result<ProjectionUncertainty, Error> {
        let (params, covariance) = camera_with_covariance(self, camera)?;
        projection_uncertainty(&params, covariance, camera, image_size).map_err(optim_error)
    }

    /// The 2×2 covariance, in px², of where the camera-frame ray through each
    /// of `pixels` lands in camera `camera`, from the uncertainty of its
    /// parameters; `None` where no ray reaches a pixel (see
    /// [`projection_covariance_at`]). Its major-axis σ is what
    /// [`projection_uncertainty`](Self::projection_uncertainty) reports.
    ///
    /// # Errors
    ///
    /// [`Error::CovarianceUnavailable`] when the result has no covariance;
    /// [`Error::InvalidInput`] when `camera` is out of range, its covariance
    /// holds none of the camera's parameters, or the camera model is one
    /// [`projection_covariance_at`] rejects.
    fn projection_covariance(
        &self,
        camera: usize,
        pixels: &[[f64; 2]],
    ) -> Result<Vec<Option<Matrix2<f64>>>, Error> {
        let (params, covariance) = camera_with_covariance(self, camera)?;
        projection_covariance_at(&params, covariance, camera, pixels).map_err(optim_error)
    }
}

/// Camera `camera`'s model and the result's parameter covariance.
fn camera_with_covariance<E: CalibratedCameras + ?Sized>(
    export: &E,
    camera: usize,
) -> Result<(CameraParams, &ParameterCovariance), Error> {
    let cameras = export.camera_params();
    let count = cameras.len();
    let params = cameras.into_iter().nth(camera).ok_or_else(|| {
        Error::invalid_input(format!(
            "camera {camera} out of range: the result has {count} cameras"
        ))
    })?;
    match export.parameter_uncertainty() {
        Uncertainty::Available(covariance) => Ok((params, covariance)),
        Uncertainty::Unavailable { reason } => Err(Error::CovarianceUnavailable {
            reason: reason.clone(),
        }),
    }
}

/// An input the projection functions reject is the caller's invalid input.
fn optim_error(err: OptimError) -> Error {
    match err {
        OptimError::InvalidInput { reason } => Error::InvalidInput { reason },
        err => err.into(),
    }
}

/// The camera a covariance label belongs to and the label without its
/// camera prefix: `cam1.k1` is `(1, "k1")`. A label without a prefix belongs
/// to camera 0, as in a single-camera result.
pub(crate) fn split_camera_label(label: &str) -> (usize, &str) {
    label
        .strip_prefix("cam")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, name)| Some((index.parse().ok()?, name)))
        .unwrap_or((0, label))
}

/// The export a JSON value holds, by its `kind` field, as its cameras.
///
/// # Errors
///
/// [`Error::Serde`] when the value has no known `kind` or does not
/// deserialize as that export.
pub fn calibrated_cameras(export: serde_json::Value) -> Result<Box<dyn CalibratedCameras>, Error> {
    fn typed<T: CalibratedCameras + DeserializeOwned + 'static>(
        export: serde_json::Value,
    ) -> Result<Box<dyn CalibratedCameras>, Error> {
        Ok(Box::new(serde_json::from_value::<T>(export)?))
    }
    let kind: ExportKind = serde_json::from_value(
        export
            .get("kind")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    )?;
    match kind {
        ExportKind::PlanarIntrinsics => typed::<PlanarIntrinsicsExport>(export),
        ExportKind::SingleCamHandeye => typed::<SingleCamHandeyeExport>(export),
        ExportKind::LaserlineDevice => typed::<LaserlineDeviceExport>(export),
        ExportKind::RigExtrinsics => typed::<RigExtrinsicsExport>(export),
        ExportKind::RigHandeye => typed::<RigHandeyeExport>(export),
        ExportKind::RigLaserlineDevice => typed::<RigLaserlineDeviceExport>(export),
        ExportKind::RigHandeyeLaserline => typed::<RigHandeyeLaserlineExport>(export),
    }
}

impl CalibratedCameras for PlanarIntrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.params.camera.clone()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for SingleCamHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.camera.clone()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for LaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.estimate.params.camera.clone()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.estimate.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for RigExtrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for RigHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for RigLaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}

impl CalibratedCameras for RigHandeyeLaserlineExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn feature_residuals(&self) -> &PerFeatureResiduals {
        &self.per_feature_residuals
    }

    fn images(&self) -> Option<&ImageManifest> {
        self.image_manifest.as_ref()
    }
}
