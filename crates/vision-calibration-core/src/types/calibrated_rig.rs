//! A calibration as a value: what a measurement needs from it.

use crate::models::{CameraModel, CameraParams};
use crate::types::Uncertainty;
use crate::{Error, Iso3};

/// What a measurement needs from a calibration: the cameras, where each sits
/// in the rig, and the uncertainty of their parameters.
///
/// The rig frame is whatever [`cam_se3_rig`](Self::cam_se3_rig) defines: the
/// reference camera has the identity (it need not be camera 0), and every
/// other camera's pose is relative to it. A single camera is a rig of one, with
/// the identity.
///
/// [`uncertainty`](Self::uncertainty) is the calibration's, joint over the
/// cameras' parameters and the extrinsics, with the labels of the export it
/// came from: `fx` for one camera; `cam1.k1` and `cam1.cam_se3_rig.rx` in a
/// rig.
#[derive(Debug, Clone)]
pub struct CalibratedRig {
    cameras: Vec<CameraParams>,
    models: Vec<CameraModel>,
    cam_se3_rig: Vec<Iso3>,
    uncertainty: Uncertainty,
}

impl CalibratedRig {
    /// A rig of `cameras`, camera `i` sitting at `cam_se3_rig[i]`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for no cameras, for different numbers of
    /// cameras and poses, or for a camera whose parameters do not build a
    /// model (the message names its index).
    pub fn new(
        cameras: Vec<CameraParams>,
        cam_se3_rig: Vec<Iso3>,
        uncertainty: Uncertainty,
    ) -> Result<Self, Error> {
        if cameras.is_empty() {
            return Err(Error::invalid_input("a rig needs at least one camera"));
        }
        if cameras.len() != cam_se3_rig.len() {
            return Err(Error::invalid_input(format!(
                "{} cameras but {} cam_se3_rig poses",
                cameras.len(),
                cam_se3_rig.len()
            )));
        }
        let models = cameras
            .iter()
            .enumerate()
            .map(|(i, c)| {
                c.build()
                    .map_err(|e| Error::invalid_input(format!("camera {i} does not build: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            cameras,
            models,
            cam_se3_rig,
            uncertainty,
        })
    }

    /// A rig of one camera, which defines the rig frame (identity pose).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if the camera does not build a model.
    pub fn single(camera: CameraParams, uncertainty: Uncertainty) -> Result<Self, Error> {
        Self::new(vec![camera], vec![Iso3::identity()], uncertainty)
    }

    /// Number of cameras.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// The cameras' parameters, in order.
    pub fn cameras(&self) -> &[CameraParams] {
        &self.cameras
    }

    /// The runtime model of camera `camera`, or `None` when out of range.
    pub fn model(&self, camera: usize) -> Option<&CameraModel> {
        self.models.get(camera)
    }

    /// Each camera's pose in the rig (`cam_se3_rig`: rig frame → camera
    /// frame), in camera order.
    pub fn cam_se3_rig(&self) -> &[Iso3] {
        &self.cam_se3_rig
    }

    /// The calibration's parameter uncertainty.
    pub fn uncertainty(&self) -> &Uncertainty {
        &self.uncertainty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistortionParams, FxFyCxCy, ProjectionParams, SensorParams};
    use nalgebra::{Translation3, UnitQuaternion};

    fn pinhole(fx: f64) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::None,
            sensor: SensorParams::Identity,
            intrinsics: FxFyCxCy {
                fx,
                fy: fx,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    #[test]
    fn single_is_a_rig_of_one_at_identity() {
        let rig = CalibratedRig::single(pinhole(800.0), Uncertainty::default()).unwrap();
        assert_eq!(rig.num_cameras(), 1);
        assert_eq!(rig.cam_se3_rig(), &[Iso3::identity()]);
        assert_eq!(rig.cameras()[0].intrinsics.fx, 800.0);
        assert!(rig.model(0).is_some());
        assert!(rig.model(1).is_none());
        assert_eq!(rig.uncertainty(), &Uncertainty::default());
    }

    #[test]
    fn new_keeps_order_and_poses() {
        let pose = Iso3::from_parts(
            Translation3::new(-0.1, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        let rig = CalibratedRig::new(
            vec![pinhole(800.0), pinhole(900.0)],
            vec![Iso3::identity(), pose],
            Uncertainty::default(),
        )
        .unwrap();
        assert_eq!(rig.num_cameras(), 2);
        assert_eq!(rig.cameras()[1].intrinsics.fx, 900.0);
        assert_eq!(rig.cam_se3_rig()[1], pose);
    }

    #[test]
    fn new_rejects_empty_and_mismatched() {
        let u = Uncertainty::default();
        assert!(matches!(
            CalibratedRig::new(vec![], vec![], u.clone()),
            Err(Error::InvalidInput { .. })
        ));
        assert!(matches!(
            CalibratedRig::new(vec![pinhole(800.0)], vec![], u),
            Err(Error::InvalidInput { .. })
        ));
    }

    #[test]
    fn new_names_the_camera_that_does_not_build() {
        let mut bad = pinhole(800.0);
        bad.sensor = SensorParams::Homography { h: [[0.0; 3]; 3] };
        let err = CalibratedRig::new(
            vec![pinhole(800.0), bad],
            vec![Iso3::identity(), Iso3::identity()],
            Uncertainty::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("camera 1"), "{err}");
    }
}
