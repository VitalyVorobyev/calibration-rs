//! A device seed resolved for one dataset run: checked against the
//! manifest's cameras, their images' sizes and the config before any image
//! is read, then turned into the workflow's manual-init seeds in the
//! input's camera order and the config's rig frame.

use vision_calibration_core::{CameraModelSpec, SensorKind};
use vision_calibration_dataset::{DatasetSpec, ProblemKind};
use vision_calibration_optim::HandeyeMode;

use crate::common::IntrinsicsManualInit;
use crate::device_seed::{DeviceSeed, DeviceSeedError, SeedCamera, SeedLens};
use crate::laserline_device::{DeviceManualInit, LaserlineDeviceConfig};
use crate::planar_intrinsics::PlanarIntrinsicsConfig;
use crate::rig_extrinsics::RigExtrinsicsConfig;
use crate::rig_family::RigSeed;
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeSeed};
use crate::rig_handeye_laserline::{JointManualInit, RigHandeyeLaserlineConfig};
use crate::rig_laserline_device::{PlaneManualInit, RigLaserlineDeviceConfig};
use crate::single_cam_handeye::{SingleCamHandeyeConfig, SingleCamSeed};

/// A seed whose cameras are the manifest's, taken in the manifest's order
/// (the input's camera order).
pub(crate) struct OrderedSeed<'a> {
    seed: &'a DeviceSeed,
    /// The manifest's camera ids, in its order.
    ids: Vec<&'a str>,
}

impl<'a> OrderedSeed<'a> {
    /// `seed` for the dataset `spec`.
    ///
    /// # Errors
    ///
    /// Any [`DeviceSeed::validate`] error;
    /// [`DeviceSeedError::CameraIdsDiffer`] when the seed's camera ids are
    /// not the manifest's.
    pub(crate) fn new(
        seed: &'a DeviceSeed,
        spec: &'a DatasetSpec,
    ) -> Result<Self, DeviceSeedError> {
        Self::with_ids(seed, spec.cameras.iter().map(|c| c.id.as_str()).collect())
    }

    /// `seed` for the cameras `ids`, in the input's camera order.
    ///
    /// # Errors
    ///
    /// Those of [`new`](Self::new).
    fn with_ids(seed: &'a DeviceSeed, ids: Vec<&'a str>) -> Result<Self, DeviceSeedError> {
        seed.validate()?;
        let missing: Vec<String> = ids
            .iter()
            .filter(|id| seed.camera(id).is_none())
            .map(|id| (*id).to_string())
            .collect();
        let unexpected: Vec<String> = seed
            .cameras
            .iter()
            .filter(|camera| !ids.contains(&camera.id.as_str()))
            .map(|camera| camera.id.clone())
            .collect();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(DeviceSeedError::CameraIdsDiffer {
                missing,
                unexpected,
            });
        }
        Ok(Self { seed, ids })
    }

    /// Check each camera's image size, `[width, height]` in the manifest's
    /// camera order, against the seed's.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::ImageSizeMismatch`] for the first camera whose
    /// seed is for images of another size.
    pub(crate) fn check_image_sizes(
        &self,
        image_sizes: &[[u32; 2]],
    ) -> Result<(), DeviceSeedError> {
        for (index, dataset) in image_sizes.iter().enumerate() {
            let camera = self.camera(index)?;
            if camera.image_size != *dataset {
                return Err(DeviceSeedError::ImageSizeMismatch {
                    camera: camera.id.clone(),
                    seed: camera.image_size,
                    dataset: *dataset,
                });
            }
        }
        Ok(())
    }

    /// The seed camera of the input's camera `index`.
    fn camera(&self, index: usize) -> Result<&'a SeedCamera, DeviceSeedError> {
        let id = self.ids.get(index).copied().unwrap_or_default();
        self.seed
            .camera(id)
            .ok_or_else(|| DeviceSeedError::UnknownCameraId { id: id.to_string() })
    }

    /// The id of the one camera of a one-camera workflow.
    fn single_id(&self) -> Result<&'a str, DeviceSeedError> {
        self.ids.first().copied().ok_or(DeviceSeedError::NoCameras)
    }

    /// Check each camera's lens against the model `model_of(index)` the
    /// config names for it: a calibrated lens must be of that model, and a
    /// nominal lens's tilt needs a Scheimpflug sensor.
    fn check_models(
        &self,
        model_of: impl Fn(usize) -> CameraModelSpec,
    ) -> Result<(), DeviceSeedError> {
        for index in 0..self.ids.len() {
            let camera = self.camera(index)?;
            let config = model_of(index);
            match &camera.lens {
                SeedLens::Calibrated { camera: params } => {
                    let Some(seed) = params.spec() else {
                        return Err(DeviceSeedError::UnseedableSensor {
                            camera: camera.id.clone(),
                        });
                    };
                    if seed != config {
                        return Err(DeviceSeedError::ModelMismatch {
                            camera: camera.id.clone(),
                            seed,
                            config,
                        });
                    }
                }
                SeedLens::Nominal { sensor, .. } => {
                    let tilted = sensor.tilt_x != 0.0 || sensor.tilt_y != 0.0;
                    if tilted && config.sensor != SensorKind::Scheimpflug {
                        return Err(DeviceSeedError::TiltForUntiltedModel {
                            camera: camera.id.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// The seed of a planar intrinsics run.
    pub(crate) fn planar(
        &self,
        config: &PlanarIntrinsicsConfig,
    ) -> Result<IntrinsicsManualInit, DeviceSeedError> {
        self.check_models(|_| config.camera.model)?;
        self.seed.intrinsics_init(self.single_id()?)
    }

    /// The seed of a single-camera hand-eye run: the camera, and the
    /// hand-eye when the seed has one.
    pub(crate) fn single_cam_handeye(
        &self,
        config: &SingleCamHandeyeConfig,
    ) -> Result<SingleCamSeed, DeviceSeedError> {
        self.check_models(|_| config.camera.model)?;
        let id = self.single_id()?;
        let mode = config.handeye_init.handeye_mode;
        Ok(SingleCamSeed {
            intrinsics: self.seed.intrinsics_init(id)?,
            handeye: match self.seed.handeye {
                Some(_) => Some(self.seed.single_cam_handeye_init(id, mode)?),
                None => None,
            },
        })
    }

    /// The seed of a laserline device run: the lens and the laser plane.
    pub(crate) fn laserline_device(
        &self,
        config: &LaserlineDeviceConfig,
    ) -> Result<DeviceManualInit, DeviceSeedError> {
        self.check_models(|_| config.camera.model)?;
        self.seed.device_init(self.single_id()?)
    }

    /// The seed of the rig stages: every camera's lens, and the mounts in
    /// the frame of the camera `reference` when the seed has them.
    fn rig_stages(
        &self,
        model_of: impl Fn(usize) -> CameraModelSpec,
        reference: usize,
    ) -> Result<RigSeed, DeviceSeedError> {
        self.check_models(model_of)?;
        let mounted = self.seed.cameras.iter().all(|c| c.cam_se3_rig.is_some());
        Ok(RigSeed {
            intrinsics: self.seed.rig_intrinsics_init(&self.ids)?,
            cam_se3_rig: if mounted {
                Some(self.seed.cam_se3_rig(&self.ids, reference)?)
            } else {
                None
            },
        })
    }

    /// The seed of a rig extrinsics run.
    pub(crate) fn rig_extrinsics(
        &self,
        config: &RigExtrinsicsConfig,
    ) -> Result<RigSeed, DeviceSeedError> {
        self.rig_stages(
            |index| config.camera_for(index).model,
            config.rig.reference_camera_idx,
        )
    }

    /// The seed of a rig hand-eye run: the rig stages', and the hand-eye in
    /// the frame of the reference camera when the seed has one.
    pub(crate) fn rig_handeye(
        &self,
        config: &RigHandeyeConfig,
    ) -> Result<RigHandeyeSeed, DeviceSeedError> {
        let reference = config.rig.reference_camera_idx;
        let rig = self.rig_stages(|index| config.camera_for(index).model, reference)?;
        Ok(RigHandeyeSeed {
            rig,
            handeye: self.rig_handeye_init(reference, config.handeye_init.handeye_mode)?,
        })
    }

    fn rig_handeye_init(
        &self,
        reference: usize,
        mode: HandeyeMode,
    ) -> Result<Option<crate::rig_handeye::HandeyeManualInit>, DeviceSeedError> {
        match self.seed.handeye {
            Some(_) => Ok(Some(self.seed.handeye_init(&self.ids, reference, mode)?)),
            None => Ok(None),
        }
    }

    /// The seed of a rig laserline device run: the laser planes. The
    /// cameras and the rig are the input's upstream calibration, which the
    /// seed does not replace.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::NothingToSeed`] when the seed has no laser plane.
    pub(crate) fn rig_laserline_device(
        &self,
        _config: &RigLaserlineDeviceConfig,
    ) -> Result<PlaneManualInit, DeviceSeedError> {
        let planes = self.seed.plane_init(&self.ids)?;
        if planes.laser_planes_cam.is_none() {
            return Err(DeviceSeedError::NothingToSeed {
                topology: ProblemKind::RigLaserlineDevice,
            });
        }
        Ok(planes)
    }

    /// The seed of a joint rig hand-eye laserline run: the rig hand-eye
    /// warm start's, and the laser planes the plane stage starts from.
    pub(crate) fn rig_handeye_laserline(
        &self,
        config: &RigHandeyeLaserlineConfig,
    ) -> Result<(JointManualInit, RigHandeyeSeed), DeviceSeedError> {
        let rig = self.rig_handeye(&config.handeye)?;
        let mut manual = JointManualInit::new();
        manual.laser_planes_cam = self.seed.plane_init(&self.ids)?.laser_planes_cam;
        Ok((manual, rig))
    }
}

#[cfg(test)]
mod tests;
