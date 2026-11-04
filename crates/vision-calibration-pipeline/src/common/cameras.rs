//! The cameras a calibration export reports, for consumers that treat every
//! export alike.

use serde::de::DeserializeOwned;

use nalgebra::Matrix2;
use vision_calibration_core::{
    CalibratedRig, CameraParams, Handeye, Iso3, LaserPlane, ParameterCovariance, Uncertainty,
};
use vision_calibration_optim::{
    Error as OptimError, ProjectionUncertainty, camera_parameters, projection_covariance_at,
    projection_uncertainty,
};

use crate::Error;
use crate::common::{ExportDiagnostics, ExportKind};
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

    /// Each camera's pose in the rig (`cam_se3_rig`), in camera order; the
    /// identity for a single-camera export.
    fn cam_se3_rig(&self) -> Vec<Iso3>;

    /// The uncertainty of the estimated parameters.
    fn parameter_uncertainty(&self) -> &Uncertainty;

    /// The residual of every observed feature at the solution, and the
    /// images behind the observations when the export records them.
    fn diagnostics(&self) -> &ExportDiagnostics;

    /// Each camera's laser plane in its own frame, in camera order; empty
    /// for an export without a laser.
    fn laser_planes_cam(&self) -> Vec<LaserPlane> {
        Vec::new()
    }

    /// The rig's hand-eye, for an export calibrated on a robot; `None` for
    /// the others.
    fn handeye(&self) -> Option<Handeye> {
        None
    }

    /// The calibration as a value a measurement takes: the cameras, their
    /// poses in the rig, their laser planes (for a laser export), the
    /// hand-eye (for a hand-eye export) and the uncertainty of the
    /// estimated parameters (labelled as in
    /// [`parameter_uncertainty`](Self::parameter_uncertainty)).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when a camera's parameters do not build a
    /// model or the covariance lists a laser plane or the hand-eye in part.
    fn calibrated_rig(&self) -> Result<CalibratedRig, Error> {
        let invalid = |e: vision_calibration_core::Error| Error::invalid_input(e.to_string());
        let mut rig = CalibratedRig::new(
            self.camera_params(),
            self.cam_se3_rig(),
            self.parameter_uncertainty().clone(),
        )
        .map_err(invalid)?;
        let planes = self.laser_planes_cam();
        if !planes.is_empty() {
            rig = rig
                .with_laser_planes(planes.into_iter().map(Some).collect())
                .map_err(invalid)?;
        }
        match self.handeye() {
            Some(handeye) => rig.with_handeye(handeye).map_err(invalid),
            None => Ok(rig),
        }
    }

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

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        vec![Iso3::identity()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for SingleCamHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.camera.clone()]
    }

    fn handeye(&self) -> Option<Handeye> {
        Some(self.handeye.handeye())
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        vec![Iso3::identity()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for LaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        vec![self.params.camera.clone()]
    }

    fn laser_planes_cam(&self) -> Vec<LaserPlane> {
        vec![self.params.plane.clone()]
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        vec![Iso3::identity()]
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for RigExtrinsicsExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cam_se3_rig.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for RigHandeyeExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn handeye(&self) -> Option<Handeye> {
        Some(self.handeye.handeye())
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cam_se3_rig.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for RigLaserlineDeviceExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn laser_planes_cam(&self) -> Vec<LaserPlane> {
        self.laser_planes_cam.clone()
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cam_se3_rig.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

impl CalibratedCameras for RigHandeyeLaserlineExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras.clone()
    }

    fn handeye(&self) -> Option<Handeye> {
        Some(self.handeye.handeye())
    }

    fn laser_planes_cam(&self) -> Vec<LaserPlane> {
        self.laser_planes_cam.clone()
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cam_se3_rig.clone()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCy, ParameterCovariance, make_pinhole_camera, pinhole_camera_params,
    };

    fn camera() -> serde_json::Value {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        serde_json::to_value(pinhole_camera_params(&make_pinhole_camera(
            k,
            BrownConrady5::default(),
        )))
        .unwrap()
    }

    fn pose() -> serde_json::Value {
        serde_json::to_value(Iso3::identity()).unwrap()
    }

    fn plane(d: f64) -> serde_json::Value {
        json!({"normal": [0.0, 0.6, 0.8], "distance": d})
    }

    fn covariance(labels: &[&str]) -> Uncertainty {
        let n = labels.len();
        Uncertainty::Available(ParameterCovariance {
            labels: labels.iter().map(|l| l.to_string()).collect(),
            covariance: (0..n)
                .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
                .collect(),
            views: vec![],
            fixed: vec![],
            noise: vec![],
            num_residuals: 10,
            dof: 5,
            view_scatter: None,
        })
    }

    const PLANES: [&str; 8] = [
        "cam0.plane.nx",
        "cam0.plane.ny",
        "cam0.plane.nz",
        "cam0.plane.d",
        "cam1.plane.nx",
        "cam1.plane.ny",
        "cam1.plane.nz",
        "cam1.plane.d",
    ];

    /// Each laser export's rig carries its planes (in the camera frames); a
    /// planar export's carries none.
    #[test]
    fn laser_exports_attach_their_planes_to_the_rig() {
        let uncertainty = serde_json::to_value(covariance(&PLANES)).unwrap();

        let rig_laser: RigLaserlineDeviceExport = serde_json::from_value(json!({
            "kind": "rig_laserline_device",
            "laser_planes_rig": [plane(-0.3), plane(-0.4)],
            "laser_planes_cam": [plane(-0.3), plane(-0.4)],
            "per_camera_stats": [],
            "cameras": [camera(), camera()],
            "cam_se3_rig": [pose(), pose()],
            "rig_se3_target": [],
            "mean_reproj_error": 0.1,
            "per_cam_reproj_errors": [0.1, 0.1],
            "uncertainty": uncertainty,
        }))
        .unwrap();
        let rig = rig_laser.calibrated_rig().unwrap();
        assert_eq!(rig.laser_plane(0).unwrap().distance(), -0.3);
        assert_eq!(rig.laser_plane(1).unwrap().distance(), -0.4);

        let handeye: RigHandeyeLaserlineExport = serde_json::from_value(json!({
            "kind": "rig_handeye_laserline",
            "laser_planes_rig": [plane(-0.3), plane(-0.4)],
            "laser_planes_cam": [plane(-0.3), plane(-0.4)],
            "per_camera_stats": [],
            "cameras": [camera(), camera()],
            "cam_se3_rig": [pose(), pose()],
            "rig_se3_target": [],
            "handeye": {
                "mode": "EyeInHand",
                "gripper_se3_rig": pose(),
                "base_se3_target": pose(),
            },
            "mean_reproj_error": 0.1,
            "per_cam_reproj_errors": [0.1, 0.1],
            "uncertainty": serde_json::to_value(covariance(&PLANES)).unwrap(),
        }))
        .unwrap();
        let rig = handeye.calibrated_rig().unwrap();
        assert_eq!(rig.laser_plane(1).unwrap().distance(), -0.4);
        assert!(matches!(rig.handeye(), Some(Handeye::EyeInHand { .. })));
        assert_eq!(
            rig.handeye_labels().unwrap()[5],
            "gripper_se3_rig.tz",
            "the labels the hand-eye exports name it by"
        );

        // A plane the covariance lists in part is not guessed around; one it
        // does not list was held fixed and stays exact.
        let mut partial = rig_laser.clone();
        partial.uncertainty = covariance(&PLANES[..6]);
        let err = partial.calibrated_rig().unwrap_err();
        assert!(err.to_string().contains("whole or held"), "{err}");
        let mut held = rig_laser.clone();
        held.uncertainty = covariance(&PLANES[..4]);
        assert!(held.calibrated_rig().unwrap().laser_plane(1).is_some());
    }
}
