//! A calibration in a problem: [`Problem::add_calibrated_rig`], how its
//! uncertainty enters ([`Use`]), and the prior that holds it.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Matrix6, RealField, UnitQuaternion, Vector3};
use vision_calibration_core::{CalibratedRig, Handeye, Iso3, RobotFrame, RobotPose};

use super::residual::{Bridge, Residual, se3};
use super::{BlockId, CAMERA_SLOTS, Camera, CameraId, Manifold, Owner, PoseChain, Problem, Term};
use crate::Error;
use crate::ir::{CustomNoise, CustomResidual, FactorKind, FixedMask, RobustLoss};
use crate::params::camera::{camera_block_labels, camera_spec};
use crate::params::pose_se3::iso3_to_se3_dvec;
use crate::uncertainty::pose_labels;

/// How a calibration's uncertainty enters a problem, as
/// [`Problem::add_calibrated_rig`] takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Use {
    /// The calibration's parameters stay fixed at their values, and their
    /// covariance `Σ_c` is carried into the solution's: with `S = ∂x̂/∂c`
    /// how the solution moves with them,
    ///
    /// ```text
    /// Σ_x = Σ_x,observations + S Σ_c Sᵀ
    /// ```
    ///
    /// The estimate is the one a fixed camera gives; only its covariance
    /// changes. Every quantity solved in the problem shares `Σ_c`, so their
    /// cross-covariance carries it too. This is the use for a measurement.
    Consider,
    /// The calibration's parameters are free, held by a Gaussian prior with
    /// the calibration's covariance (a whitened residual `L⁻¹(c − c₀)` with
    /// `Σ_c = L Lᵀ`), so the problem's observations refine them. This is
    /// the use for bundle adjustment that refines the cameras.
    ///
    /// The prior weighs in units of its covariance, so every residual's
    /// noise is stated too ([`PixelNoise::Stated`](super::PixelNoise::Stated),
    /// [`Noise::Known`](super::Noise::Known)): a residual of estimated noise
    /// would weigh as if its σ were 1 in its unit. It runs on the
    /// tiny-solver backend only.
    Prior,
}

/// A calibrated rig's cameras and extrinsics in a [`Problem`], as
/// [`Problem::add_calibrated_rig`] returns them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigBlocks {
    cameras: Vec<CameraId>,
    /// Each camera's `cam_se3_rig` block; none for a rig of one camera.
    cam_se3_rig: Vec<Option<BlockId>>,
}

impl RigBlocks {
    /// The number of cameras.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// Camera `camera` of the rig; `None` out of range.
    pub fn camera(&self, camera: usize) -> Option<CameraId> {
        self.cameras.get(camera).copied()
    }

    /// The SE3 block holding camera `camera`'s `cam_se3_rig`; `None` out of
    /// range and for a rig of one camera, whose camera is the rig frame.
    pub fn cam_se3_rig(&self, camera: usize) -> Option<BlockId> {
        self.cam_se3_rig.get(camera).copied().flatten()
    }

    /// The chain from a target posed by `rig_se3_target` to camera
    /// `camera`: [`PoseChain::Rig`], or [`PoseChain::Pose`] for a rig of
    /// one camera. `None` for a camera out of range.
    pub fn chain(&self, camera: usize, rig_se3_target: BlockId) -> Option<PoseChain> {
        let _ = self.cameras.get(camera)?;
        Some(match self.cam_se3_rig(camera) {
            Some(cam_se3_rig) => PoseChain::Rig {
                cam_se3_rig,
                rig_se3_target,
            },
            None => PoseChain::Pose(rig_se3_target),
        })
    }
}

/// The calibration a problem holds.
#[derive(Debug, Clone)]
pub(super) struct Calibration {
    pub(super) usage: Use,
    /// The coordinates the calibration estimated, `(block index, tangent
    /// index)`, in the order of `sigma`.
    pub(super) coordinates: Vec<(usize, usize)>,
    /// Their covariance `Σ_c`; `None` for a calibration without one (which
    /// only [`Use::Consider`] adds).
    pub(super) sigma: Option<DMatrix<f64>>,
    /// The rig's hand-eye, for [`Problem::add_rig_pose`]; `None` for a rig
    /// without one.
    pub(super) handeye: Option<HandeyeSource>,
}

/// A calibration's hand-eye and its covariance with the calibration's
/// coordinates.
#[derive(Debug, Clone)]
pub(super) struct HandeyeSource {
    handeye: Handeye,
    /// `Σ_hh` and `Cov(h, c)` over the calibration's coordinates (6 ×
    /// their number); `None` when the covariance does not list the hand-eye
    /// (it was held, exact) or there is no covariance.
    sigma: Option<(Matrix6<f64>, DMatrix<f64>)>,
}

impl Problem {
    /// Add a calibrated rig's cameras and extrinsics, with its uncertainty
    /// entering as `usage` says.
    ///
    /// The cameras are added in the rig's order and labelled as
    /// [`add_camera`](Self::add_camera) labels them. A rig of two or more
    /// cameras has an SE3 block per camera holding its `cam_se3_rig`,
    /// labelled `cam<k>.cam_se3_rig`; reproject through it with
    /// [`RigBlocks::chain`]. The parameters the calibration estimated are
    /// those its covariance labels (`fx`, `cam1.k1`,
    /// `cam1.cam_se3_rig.rx`, …); the others, which the calibration held
    /// fixed, stay fixed in either use.
    ///
    /// A calibration without a covariance is added fixed with
    /// [`Use::Consider`], and the solution's uncertainty is then
    /// unavailable for that reason
    /// ([`UncertaintyUnavailable::CalibrationWithoutCovariance`](vision_calibration_core::UncertaintyUnavailable::CalibrationWithoutCovariance)):
    /// a σ from the measurement's own observations alone would leave out
    /// the calibration's share. To ask for that σ anyway, add the cameras
    /// fixed with [`add_camera`](Self::add_camera).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a second calibrated rig in the problem,
    /// [`Use::Prior`] with a calibration that has no covariance, a
    /// covariance that is not positive definite over the estimated
    /// parameters, an extrinsic pose estimated in part, a block name
    /// `cam<k>.cam_se3_rig` already used, or a camera
    /// [`add_camera`](Self::add_camera) rejects.
    pub fn add_calibrated_rig(
        &mut self,
        rig: &CalibratedRig,
        usage: Use,
    ) -> Result<RigBlocks, Error> {
        if self.calibration.is_some() {
            return Err(Error::invalid_input("a problem holds one calibrated rig"));
        }
        let covariance = rig.uncertainty().covariance();
        if usage == Use::Prior && covariance.is_none() {
            return Err(Error::invalid_input(
                "a calibration without a covariance cannot be a prior; add it with \
                 Use::Consider",
            ));
        }
        let position = |label: &str| covariance.and_then(|c| c.index(label));
        let multi = rig.num_cameras() > 1;
        let prefix = |k: usize, s: &str| {
            if multi {
                format!("cam{k}.{s}")
            } else {
                s.to_string()
            }
        };
        if multi {
            for k in 0..rig.num_cameras() {
                self.check_name(&format!("cam{k}.cam_se3_rig"))?;
            }
        }

        // (block index, tangent index, position in the covariance)
        let mut estimated: Vec<(usize, usize, usize)> = Vec::new();
        let mut cameras = Vec::with_capacity(rig.num_cameras());
        for (k, camera) in rig.cameras().iter().enumerate() {
            let spec = camera_spec(camera)?;
            let values = camera.blocks();
            if values.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error::invalid_input(format!(
                    "camera {k}: parameters must be finite"
                )));
            }
            let index = self.cameras.len();
            let mut blocks = Vec::new();
            for (slot, values) in CAMERA_SLOTS.into_iter().zip(values) {
                if values.is_empty() {
                    continue;
                }
                let labels = camera_block_labels(slot, spec, |s| prefix(k, s))
                    .expect("a camera slot has labels")
                    .coordinates;
                let at: Vec<Option<usize>> = labels.iter().map(|l| position(l)).collect();
                let held: Vec<usize> = (0..values.len()).filter(|&i| at[i].is_none()).collect();
                let fixed = match usage {
                    Use::Consider => FixedMask::all_fixed(values.len()),
                    Use::Prior => FixedMask::fix_indices(&held),
                };
                let id = self.push_block(super::Block {
                    name: format!("camera{index}.{slot}"),
                    values: DVector::from_vec(values),
                    manifold: Manifold::Euclidean,
                    fixed,
                    bounds: Vec::new(),
                    labels: None,
                    owner: Owner::Camera {
                        camera: index,
                        slot,
                    },
                });
                estimated.extend(
                    at.iter()
                        .enumerate()
                        .filter_map(|(i, p)| p.map(|p| (id.index, i, p))),
                );
                blocks.push(id);
            }
            self.cameras.push(Camera { spec, blocks });
            cameras.push(CameraId {
                problem: self.id,
                index,
            });
        }

        let mut cam_se3_rig = vec![None; rig.num_cameras()];
        if multi {
            for (k, pose) in rig.cam_se3_rig().iter().enumerate() {
                let name = format!("cam{k}.cam_se3_rig");
                let at: Vec<Option<usize>> =
                    pose_labels(&name).iter().map(|l| position(l)).collect();
                let free = at.iter().all(Option::is_some);
                if !free && at.iter().any(Option::is_some) {
                    return Err(Error::invalid_input(format!(
                        "{name}: the calibration estimated part of the pose; a pose is \
                         estimated whole or held"
                    )));
                }
                let fixed = if free && usage == Use::Prior {
                    FixedMask::all_free()
                } else {
                    FixedMask::all_fixed(7)
                };
                let id = self.push_block(super::Block {
                    name,
                    values: iso3_to_se3_dvec(pose),
                    manifold: Manifold::Se3,
                    fixed,
                    bounds: Vec::new(),
                    labels: None,
                    owner: Owner::User,
                });
                if free {
                    estimated.extend(
                        at.iter()
                            .enumerate()
                            .map(|(i, p)| (id.index, i, p.unwrap())),
                    );
                }
                cam_se3_rig[k] = Some(id);
            }
        }

        let coordinates: Vec<(usize, usize)> = estimated.iter().map(|&(b, i, _)| (b, i)).collect();
        let sigma = covariance.map(|c| {
            DMatrix::from_fn(estimated.len(), estimated.len(), |a, b| {
                c.covariance[estimated[a].2][estimated[b].2]
            })
        });
        if usage == Use::Prior && !coordinates.is_empty() {
            let sigma = sigma.as_ref().expect("a prior has a covariance");
            self.add_prior(&coordinates, sigma)?;
        }
        let handeye = rig.handeye().map(|&handeye| {
            let at: Option<Vec<usize>> = rig
                .handeye_labels()
                .expect("a rig with a hand-eye labels it")
                .iter()
                .map(|l| position(l))
                .collect();
            let sigma = covariance.zip(at).map(|(c, at)| {
                (
                    Matrix6::from_fn(|a, b| c.covariance[at[a]][at[b]]),
                    DMatrix::from_fn(6, estimated.len(), |a, b| {
                        c.covariance[at[a]][estimated[b].2]
                    }),
                )
            });
            HandeyeSource { handeye, sigma }
        });
        self.calibration = Some(Calibration {
            usage,
            coordinates,
            sigma,
            handeye,
        });
        Ok(RigBlocks {
            cameras,
            cam_se3_rig,
        })
    }

    /// Add the calibrated rig's pose in the robot frame `frame`,
    /// `rig_se3_<frame>`, as an SE3 block named `name`, held at its value,
    /// whose uncertainty is considered with the calibration's (see
    /// [`Use::Consider`]).
    ///
    /// The pose follows from the rig's hand-eye
    /// ([`CalibratedRig::with_handeye`]) and, in the frame that moves with
    /// the robot (the base eye-in-hand, the gripper eye-to-hand), the
    /// robot's pose `robot` at the capture. Its uncertainty is the
    /// hand-eye's, correlated with the cameras' through the calibration's
    /// covariance, plus the robot pose's stated covariance, independent of
    /// the calibration's and of every other robot pose's. A hand-eye the
    /// covariance does not list counts as exact, as does a robot pose
    /// without a covariance. Reproject through it as the target pose of a
    /// [`RigBlocks::chain`]: the points then live in `frame`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] before a calibrated rig is added, for one
    /// added with [`Use::Prior`], a rig without a hand-eye, a robot pose
    /// given in the frame the rig is fixed in (the gripper eye-in-hand, the
    /// base eye-to-hand: it does not enter) or missing in the other, and a
    /// name [`add_block`](Self::add_block) rejects.
    pub fn add_rig_pose(
        &mut self,
        name: impl Into<String>,
        frame: RobotFrame,
        robot: Option<&RobotPose>,
    ) -> Result<BlockId, Error> {
        let name = name.into();
        self.check_name(&name)?;
        let calibration = self.calibration.as_ref().ok_or_else(|| {
            Error::invalid_input("add the calibrated rig before its poses on the robot")
        })?;
        if calibration.usage != Use::Consider {
            return Err(Error::invalid_input(
                "a rig pose on the robot is considered: add the calibrated rig with \
                 Use::Consider",
            ));
        }
        let source = calibration
            .handeye
            .as_ref()
            .ok_or_else(|| Error::invalid_input("the calibration has no hand-eye"))?;
        let fixed_in = source.handeye.mount_frame();
        let base_se3_gripper = match (frame == fixed_in, robot) {
            (true, None) => Iso3::identity(),
            (true, Some(_)) => {
                return Err(Error::invalid_input(format!(
                    "the rig is fixed in the {} frame: the robot's pose does not enter",
                    frame.name()
                )));
            }
            (false, Some(robot)) => *robot.base_se3_gripper(),
            (false, None) => {
                return Err(Error::invalid_input(format!(
                    "the rig moves in the {} frame with the robot: state the robot's pose",
                    frame.name()
                )));
            }
        };
        let at = source.handeye.rig_in_frame(frame, &base_se3_gripper);
        let robot_share = at
            .d_robot
            .zip(robot.and_then(RobotPose::covariance))
            .map(|(b, sigma_r)| b * sigma_r * b.transpose());
        let handeye_share = source.sigma.as_ref().map(|(sigma_hh, sigma_hc)| {
            let a = at.d_handeye;
            (
                a * sigma_hc,
                a * sigma_hh * a.transpose(),
                sigma_hh * a.transpose(),
            )
        });

        let id = self.push_block(super::Block {
            name,
            values: iso3_to_se3_dvec(&at.rig_se3_frame),
            manifold: Manifold::Se3,
            fixed: FixedMask::all_fixed(7),
            bounds: Vec::new(),
            labels: None,
            owner: Owner::User,
        });
        let calibration = self
            .calibration
            .as_mut()
            .expect("the calibration was checked above");
        let Some(sigma) = calibration.sigma.as_mut() else {
            // Without the calibration's covariance the solution has none.
            return Ok(id);
        };
        if handeye_share.is_none() && robot_share.is_none() {
            return Ok(id);
        }
        // The pose's tangent ξ = A ε_h + B ε_r joins the coordinates:
        // Cov(ξ, c) = A Cov(h, c), Cov(ξ, ξ) = A Σ_hh Aᵀ + B Σ_r Bᵀ.
        let m = sigma.nrows();
        let mut grown = DMatrix::zeros(m + 6, m + 6);
        grown.view_mut((0, 0), (m, m)).copy_from(sigma);
        let mut own = robot_share.unwrap_or_else(Matrix6::zeros);
        if let Some((cross, sigma_xi, _)) = &handeye_share {
            grown.view_mut((m, 0), (6, m)).copy_from(cross);
            grown.view_mut((0, m), (m, 6)).copy_from(&cross.transpose());
            own += sigma_xi;
        }
        grown.view_mut((m, m), (6, 6)).copy_from(&own);
        *sigma = grown;
        calibration
            .coordinates
            .extend((0..6).map(|i| (id.index, i)));
        // Cov(h, ξ) = Σ_hh Aᵀ, for the next pose.
        if let Some(HandeyeSource {
            sigma: Some((_, sigma_hc)),
            ..
        }) = calibration.handeye.as_mut()
        {
            let new_columns = handeye_share.map_or_else(Matrix6::zeros, |(_, _, c)| c);
            let mut grown = DMatrix::zeros(6, m + 6);
            grown.view_mut((0, 0), (6, m)).copy_from(sigma_hc);
            grown.view_mut((0, m), (6, 6)).copy_from(&new_columns);
            *sigma_hc = grown;
        }
        Ok(id)
    }

    /// The whitened prior `L⁻¹(c − c₀)` over the calibration's estimated
    /// `coordinates`, at the blocks' current values.
    fn add_prior(
        &mut self,
        coordinates: &[(usize, usize)],
        sigma: &DMatrix<f64>,
    ) -> Result<(), Error> {
        let l = sigma.clone().cholesky().ok_or_else(|| {
            Error::invalid_input(
                "the calibration's covariance is not positive definite over its estimated \
                 parameters",
            )
        })?;
        let whiten = l
            .l()
            .try_inverse()
            .ok_or_else(|| Error::invalid_input("the calibration's covariance is singular"))?;
        let mut parts: Vec<Part> = Vec::new();
        let mut blocks: Vec<usize> = Vec::new();
        for &(block, index) in coordinates {
            if blocks.last() != Some(&block) {
                let b = &self.blocks[block];
                blocks.push(block);
                parts.push(match b.manifold {
                    Manifold::Se3 => Part::Pose {
                        reference: se3(&b.values),
                    },
                    _ => Part::Vector {
                        reference: b.values.as_slice().to_vec(),
                        indices: Vec::new(),
                    },
                });
            }
            if let Some(Part::Vector { indices, .. }) = parts.last_mut() {
                indices.push(index);
            }
        }
        let prior = CalibrationPrior { parts, whiten };
        self.residuals.push(Term {
            factor: FactorKind::Custom(CustomResidual {
                factor: Arc::new(Bridge::new(prior, 1.0)),
                noise: CustomNoise::Whitened,
            }),
            blocks,
            loss: RobustLoss::None,
        });
        Ok(())
    }
}

/// One block a [`CalibrationPrior`] reads.
#[derive(Debug, Clone)]
enum Part {
    /// A Euclidean block: its reference values and the components the
    /// prior holds.
    Vector {
        reference: Vec<f64>,
        indices: Vec<usize>,
    },
    /// An SE3 block, held over its whole tangent.
    Pose { reference: Iso3 },
}

/// The calibration's Gaussian prior: `L⁻¹ δ`, with `δ` each held
/// coordinate's step from its reference, in the coordinates the covariance
/// is over (an SE3 block's right-perturbation tangent).
struct CalibrationPrior {
    parts: Vec<Part>,
    /// `L⁻¹` for `Σ_c = L Lᵀ`.
    whiten: DMatrix<f64>,
}

impl Residual for CalibrationPrior {
    fn dim(&self) -> usize {
        self.whiten.nrows()
    }

    fn block_dims(&self) -> Vec<usize> {
        self.parts
            .iter()
            .map(|p| match p {
                Part::Vector { reference, .. } => reference.len(),
                Part::Pose { .. } => 7,
            })
            .collect()
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let lift = |v: f64| T::from_f64(v).unwrap();
        let mut delta = Vec::with_capacity(self.dim());
        for (part, block) in self.parts.iter().zip(blocks) {
            match part {
                Part::Vector { reference, indices } => {
                    delta.extend(
                        indices
                            .iter()
                            .map(|&i| block[i].clone() - lift(reference[i])),
                    );
                }
                Part::Pose { reference } => {
                    // The step `X = X₀ · (Exp ω, v)` of the SE3 manifold:
                    // `ω = Log(R₀ᵀ R)`, `v = R₀ᵀ (t − t₀)`.
                    let pose = se3(block);
                    let r0 = reference.rotation.cast::<T>();
                    let t0 = reference.translation.vector.cast::<T>();
                    let omega = so3_log(&(r0.inverse() * pose.rotation));
                    let v = r0.inverse_transform_vector(&(pose.translation.vector - t0));
                    delta.extend(omega.iter().cloned());
                    delta.extend(v.iter().cloned());
                }
            }
        }
        let delta = DVector::from_vec(delta);
        self.whiten.map(lift) * delta
    }
}

/// The rotation vector of `q`, differentiable at the identity.
fn so3_log<T: RealField>(q: &UnitQuaternion<T>) -> Vector3<T> {
    let (mut w, mut v) = (q.scalar(), q.imag());
    if w < T::zero() {
        w = -w;
        v = -v;
    }
    let n2 = v.norm_squared();
    let two = T::from_f64(2.0).unwrap();
    // `2·atan2(n, w)/n`, by its series below an angle of about 2e-6 rad.
    let scale = if n2 < T::from_f64(1e-12).unwrap() {
        let three = T::from_f64(3.0).unwrap();
        two.clone() / w.clone() * (T::one() - n2 / (three * w.clone() * w))
    } else {
        let n = n2.sqrt();
        two * n.clone().atan2(w) / n
    };
    v * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn so3_log_inverts_the_exponential() {
        for axis in [Vector3::x(), Vector3::new(0.3, -0.5, 0.8).normalize()] {
            for angle in [0.0, 1e-9, 1e-4, 0.3, 2.5] {
                let w = axis * angle;
                let back = so3_log(&UnitQuaternion::from_scaled_axis(w));
                assert!((back - w).norm() < 1e-14, "{w:?}: {back:?}");
            }
        }
    }
}
