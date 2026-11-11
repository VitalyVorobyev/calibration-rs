//! A calibration as a value: what a measurement needs from it.

use crate::models::{CameraModel, CameraParams};
use crate::types::{
    Handeye, LaserPlane, NoiseLevel, PointCovariance, ResidualKind, RobotFrame, Uncertainty,
};
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
///
/// A rig from a laser calibration also carries the laser plane of each
/// camera ([`with_laser_planes`](Self::with_laser_planes)), in that camera's
/// own frame; their labels in the uncertainty are `plane.nx`, `plane.ny`,
/// `plane.nz` and `plane.d` (`cam1.plane.nx` in a rig).
///
/// A rig from a hand-eye calibration also carries its hand-eye
/// ([`with_handeye`](Self::with_handeye)), so a measurement can be stated in
/// the robot's gripper or base frame; its labels are those of the export's
/// pose: `gripper_se3_rig.rx … tz` eye-in-hand and `rig_se3_base.rx … tz`
/// eye-to-hand (`gripper_se3_cam`, `cam_se3_base` for one camera).
#[derive(Debug, Clone)]
pub struct CalibratedRig {
    cameras: Vec<CameraParams>,
    models: Vec<CameraModel>,
    cam_se3_rig: Vec<Iso3>,
    laser_planes: Vec<Option<LaserPlane>>,
    handeye: Option<Handeye>,
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
            laser_planes: Vec::new(),
            handeye: None,
            uncertainty,
        })
    }

    /// The same rig with a laser plane per camera: `planes[i]` is camera
    /// `i`'s plane in its own frame, or `None` for a camera without a laser.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a number of planes other than the number
    /// of cameras, and for a plane the calibration's covariance (when it has
    /// one) lists in part: its four labels are `plane.nx`, `plane.ny`,
    /// `plane.nz`, `plane.d` (with the `cam<i>.` prefix in a rig of two or
    /// more cameras, as the camera parameters have), and a plane is estimated
    /// whole or held fixed. A plane the covariance does not list at all was
    /// held fixed and counts as exact.
    pub fn with_laser_planes(mut self, planes: Vec<Option<LaserPlane>>) -> Result<Self, Error> {
        if planes.len() != self.num_cameras() {
            return Err(Error::invalid_input(format!(
                "{} laser planes for {} cameras",
                planes.len(),
                self.num_cameras()
            )));
        }
        if let Some(covariance) = self.uncertainty.covariance() {
            for (i, plane) in planes.iter().enumerate() {
                if plane.is_none() {
                    continue;
                }
                let labels = Self::plane_labels(i, self.num_cameras() > 1);
                let listed = labels
                    .iter()
                    .filter(|l| covariance.index(l).is_some())
                    .count();
                if listed != 0 && listed != labels.len() {
                    return Err(Error::invalid_input(format!(
                        "camera {i}'s laser plane is in the calibration's covariance in part \
                         (the labels {}); a plane is estimated whole or held fixed",
                        labels.join(", ")
                    )));
                }
            }
        }
        self.laser_planes = planes;
        Ok(self)
    }

    /// The covariance labels of camera `camera`'s laser plane:
    /// `plane.nx`, `plane.ny`, `plane.nz`, `plane.d`, prefixed `cam<camera>.`
    /// in a rig of several cameras.
    #[doc(hidden)]
    pub fn plane_labels(camera: usize, multi_camera: bool) -> [String; 4] {
        ["nx", "ny", "nz", "d"].map(|c| {
            if multi_camera {
                format!("cam{camera}.plane.{c}")
            } else {
                format!("plane.{c}")
            }
        })
    }

    /// The same rig on a robot, fixed by `handeye`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a hand-eye the calibration's covariance
    /// (when it has one) lists in part: its six labels are
    /// [`handeye_labels`](Self::handeye_labels), and a pose is estimated
    /// whole or held fixed. A hand-eye the covariance does not list at all
    /// was held fixed and counts as exact.
    pub fn with_handeye(mut self, handeye: Handeye) -> Result<Self, Error> {
        if let Some(covariance) = self.uncertainty.covariance() {
            let labels = Self::handeye_labels_of(&handeye, self.num_cameras() > 1);
            let listed = labels
                .iter()
                .filter(|l| covariance.index(l).is_some())
                .count();
            if listed != 0 && listed != labels.len() {
                return Err(Error::invalid_input(format!(
                    "the hand-eye is in the calibration's covariance in part (the labels {}); \
                     a pose is estimated whole or held",
                    labels.join(", ")
                )));
            }
        }
        self.handeye = Some(handeye);
        Ok(self)
    }

    /// The rig's hand-eye; `None` for a rig not calibrated on a robot.
    pub fn handeye(&self) -> Option<&Handeye> {
        self.handeye.as_ref()
    }

    /// The covariance labels of the rig's hand-eye, the tangent `[rx, ry,
    /// rz, tx, ty, tz]` of its right perturbation: `gripper_se3_rig.*`
    /// eye-in-hand and `rig_se3_base.*` eye-to-hand (`gripper_se3_cam.*`
    /// and `cam_se3_base.*` for one camera, as its export names them);
    /// `None` for a rig without a hand-eye.
    pub fn handeye_labels(&self) -> Option<[String; 6]> {
        let handeye = self.handeye.as_ref()?;
        Some(Self::handeye_labels_of(handeye, self.num_cameras() > 1))
    }

    fn handeye_labels_of(handeye: &Handeye, multi_camera: bool) -> [String; 6] {
        let rig = if multi_camera { "rig" } else { "cam" };
        let name = match handeye.mount_frame() {
            RobotFrame::Gripper => format!("gripper_se3_{rig}"),
            RobotFrame::Base => format!("{rig}_se3_base"),
        };
        ["rx", "ry", "rz", "tx", "ty", "tz"].map(|c| format!("{name}.{c}"))
    }

    /// Camera `camera`'s laser plane, in its own frame; `None` for a camera
    /// out of range or without a laser plane.
    pub fn laser_plane(&self, camera: usize) -> Option<&LaserPlane> {
        self.laser_planes.get(camera)?.as_ref()
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

    /// The pixel noise of camera `camera`'s reprojections, as estimated by the
    /// calibration: an isotropic [`PointCovariance`] of the calibration's
    /// reprojection `σ̂` for that camera (or the one level shared by every
    /// camera).
    ///
    /// It states the noise of observations made with this calibration's
    /// detector and optics; pass it to the functions that weigh pixels by
    /// their covariance.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a camera out of range; for a calibration
    /// without a covariance (it has no noise level; state the pixel noise with
    /// [`PointCovariance::isotropic`] or [`PointCovariance::new`]); for a
    /// calibration with no reprojection noise level; and for one whose
    /// reprojections were whitened by stated per-point covariances, whose
    /// level `σ̂₀` is dimensionless and says nothing about a pixel `σ`.
    pub fn reprojection_noise(&self, camera: usize) -> Result<PointCovariance, Error> {
        if camera >= self.num_cameras() {
            return Err(Error::invalid_input(format!(
                "camera {camera} out of range: the rig has {} cameras",
                self.num_cameras()
            )));
        }
        let covariance = match &self.uncertainty {
            Uncertainty::Available(c) => c,
            Uncertainty::Unavailable { reason } => {
                return Err(Error::invalid_input(format!(
                    "the calibration has no covariance ({reason:?}), so it has no noise \
                     level; state the pixel noise instead"
                )));
            }
        };
        let reprojection = |c: Option<usize>| {
            covariance
                .noise
                .iter()
                .find(|n| n.kind == ResidualKind::Reprojection && n.camera == c)
        };
        let entry = reprojection(Some(camera))
            .or_else(|| reprojection(None))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "the calibration has no reprojection noise level for camera {camera}; \
                     state the pixel noise instead"
                ))
            })?;
        match entry.level {
            NoiseLevel::Sigma(sigma) => PointCovariance::isotropic(sigma),
            NoiseLevel::Sigma0(s0) => Err(Error::invalid_input(format!(
                "the calibration's reprojections were whitened by stated covariances \
                 (σ̂₀ = {s0}), so it has no pixel σ to reuse; state the pixel noise instead"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ParameterCovariance, ResidualNoise};
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

    fn plane() -> LaserPlane {
        LaserPlane::new(nalgebra::Vector3::new(0.0, 0.6, 0.8), -0.4).unwrap()
    }

    fn covariance_with(labels: Vec<String>) -> Uncertainty {
        let n = labels.len();
        Uncertainty::Available(ParameterCovariance {
            covariance: (0..n)
                .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
                .collect(),
            labels,
            views: vec![],
            fixed: vec![],
            noise: vec![],
            num_residuals: 10,
            dof: 5,
            view_scatter: None,
            second_order: None,
        })
    }

    #[test]
    fn laser_planes_attach_per_camera_and_are_checked() {
        let cameras = vec![pinhole(800.0), pinhole(800.0)];
        let poses = vec![Iso3::identity(), Iso3::identity()];
        let rig =
            CalibratedRig::new(cameras.clone(), poses.clone(), Uncertainty::default()).unwrap();
        assert!(rig.laser_plane(0).is_none());
        let rig = rig.with_laser_planes(vec![None, Some(plane())]).unwrap();
        assert!(rig.laser_plane(0).is_none());
        assert_eq!(rig.laser_plane(1).unwrap().distance(), -0.4);
        assert!(rig.laser_plane(2).is_none());

        // Wrong number of planes.
        let bare =
            CalibratedRig::new(cameras.clone(), poses.clone(), Uncertainty::default()).unwrap();
        assert!(bare.with_laser_planes(vec![Some(plane())]).is_err());

        // A covariance lists the plane of camera 1 in full or not at all.
        let labels = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let full = covariance_with(labels(&[
            "cam1.plane.nx",
            "cam1.plane.ny",
            "cam1.plane.nz",
            "cam1.plane.d",
        ]));
        let ok = CalibratedRig::new(cameras.clone(), poses.clone(), full).unwrap();
        assert!(ok.with_laser_planes(vec![None, Some(plane())]).is_ok());
        let partial = covariance_with(labels(&["cam1.plane.nx", "cam1.plane.d"]));
        let rig = CalibratedRig::new(cameras.clone(), poses.clone(), partial).unwrap();
        let err = rig
            .with_laser_planes(vec![None, Some(plane())])
            .unwrap_err();
        assert!(err.to_string().contains("whole or held"), "{err}");
        // A plane held fixed is not listed at all.
        let held = covariance_with(labels(&["cam1.fx"]));
        let rig = CalibratedRig::new(cameras, poses, held).unwrap();
        assert!(rig.with_laser_planes(vec![None, Some(plane())]).is_ok());

        // One camera: no prefix.
        let single = CalibratedRig::single(
            pinhole(800.0),
            covariance_with(labels(&["plane.nx", "plane.ny", "plane.nz", "plane.d"])),
        )
        .unwrap();
        assert!(single.with_laser_planes(vec![Some(plane())]).is_ok());
    }

    #[test]
    fn handeye_attaches_and_is_checked() {
        let pose = Iso3::from_parts(Translation3::new(0.1, 0.0, 0.2), UnitQuaternion::identity());
        let labels = |name: &str| {
            ["rx", "ry", "rz", "tx", "ty", "tz"]
                .map(|c| format!("{name}.{c}"))
                .to_vec()
        };
        let eih = Handeye::EyeInHand {
            gripper_se3_rig: pose,
        };
        let eth = Handeye::EyeToHand { rig_se3_base: pose };

        let bare = CalibratedRig::single(pinhole(800.0), Uncertainty::default()).unwrap();
        assert!(bare.handeye().is_none() && bare.handeye_labels().is_none());
        let rig = bare.with_handeye(eih).unwrap();
        assert_eq!(rig.handeye(), Some(&eih));
        assert_eq!(
            rig.handeye_labels().unwrap().to_vec(),
            labels("gripper_se3_cam")
        );

        let pair = || {
            CalibratedRig::new(
                vec![pinhole(800.0), pinhole(800.0)],
                vec![Iso3::identity(), Iso3::identity()],
                covariance_with(labels("rig_se3_base")),
            )
            .unwrap()
        };
        let rig = pair().with_handeye(eth).unwrap();
        assert_eq!(
            rig.handeye_labels().unwrap().to_vec(),
            labels("rig_se3_base")
        );
        // Held fixed: the covariance does not list it.
        assert!(pair().with_handeye(eih).is_ok());
        // Listed in part.
        let mut part = labels("gripper_se3_rig");
        part.truncate(3);
        let rig = CalibratedRig::new(
            vec![pinhole(800.0), pinhole(800.0)],
            vec![Iso3::identity(), Iso3::identity()],
            covariance_with(part),
        )
        .unwrap();
        let err = rig.with_handeye(eih).unwrap_err();
        assert!(err.to_string().contains("whole or held"), "{err}");
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

    fn rig_with(noise: Vec<ResidualNoise>) -> CalibratedRig {
        let cov = ParameterCovariance {
            labels: vec![],
            covariance: vec![],
            views: vec![],
            fixed: vec![],
            noise,
            num_residuals: 10,
            dof: 5,
            view_scatter: None,
            second_order: None,
        };
        CalibratedRig::new(
            vec![pinhole(800.0), pinhole(800.0)],
            vec![Iso3::identity(), Iso3::identity()],
            Uncertainty::Available(cov),
        )
        .unwrap()
    }

    fn level(level: NoiseLevel, camera: Option<usize>) -> ResidualNoise {
        ResidualNoise {
            kind: ResidualKind::Reprojection,
            level,
            count: 10,
            camera,
        }
    }

    #[test]
    fn reprojection_noise_uses_the_camera_entry_then_the_shared_one() {
        let rig = rig_with(vec![
            level(NoiseLevel::Sigma(0.5), None),
            level(NoiseLevel::Sigma(0.2), Some(1)),
        ]);
        let shared = rig.reprojection_noise(0).unwrap();
        assert_eq!(shared, PointCovariance::isotropic(0.5).unwrap());
        let own = rig.reprojection_noise(1).unwrap();
        assert_eq!(own, PointCovariance::isotropic(0.2).unwrap());
    }

    #[test]
    fn reprojection_noise_errors() {
        let rig = rig_with(vec![level(NoiseLevel::Sigma(0.5), None)]);
        let err = rig.reprojection_noise(2).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");

        let none = CalibratedRig::single(pinhole(800.0), Uncertainty::default()).unwrap();
        let err = none.reprojection_noise(0).unwrap_err();
        assert!(err.to_string().contains("NotComputed"), "{err}");

        let missing = rig_with(vec![ResidualNoise {
            kind: ResidualKind::LaserPlane,
            level: NoiseLevel::Sigma(1e-4),
            count: 3,
            camera: None,
        }]);
        let err = missing.reprojection_noise(0).unwrap_err();
        assert!(err.to_string().contains("no reprojection noise"), "{err}");

        let whitened = rig_with(vec![level(NoiseLevel::Sigma0(1.1), None)]);
        let err = whitened.reprojection_noise(0).unwrap_err();
        assert!(err.to_string().contains("whitened"), "{err}");
    }
}
