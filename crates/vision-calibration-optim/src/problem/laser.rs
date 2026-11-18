//! Laser planes in a problem and the laser residual that reads them:
//! [`Problem::add_plane`], [`Problem::add_laser`], [`Solution::plane`].

use nalgebra::{DVector, Point3};
use vision_calibration_core::{LaserPlane, Pt2};

use super::residual::se3;
use super::{Block, CameraId, Manifold, Owner, PoseChain, Problem, Solution, Term};
use crate::Error;
use crate::LaserlineResidualType;
use crate::factors::camera_blocks::CameraBlocks;
use crate::factors::laserline::{laser_line_distance_residual, laser_point_to_plane_residual};
use crate::ir::{FactorKind, FixedMask, LaserChain, RobustLoss};
use crate::params::laser_plane::plane_from_split_dvec;
use crate::uncertainty::BlockLabels;

/// A laser plane of a [`Problem`], as [`Problem::add_plane`] returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneId {
    problem: u64,
    index: usize,
}

/// The noise of a laser residual, as [`Problem::add_laser`] takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LaserNoise {
    /// Shared by the problem's laser residuals of the same kind, and
    /// estimated from their scatter at the solution: metres for
    /// [`LaserlineResidualType::PointToPlane`], pixels for
    /// [`LaserlineResidualType::LineDistNormalized`].
    Estimated,
}

/// A plane's two blocks: its unit normal and its distance.
#[derive(Debug, Clone)]
pub(super) struct Plane {
    pub(super) name: String,
    pub(super) normal: usize,
    pub(super) distance: usize,
}

impl Plane {
    /// The covariance labels of the plane's normal (`<name>.nx`, `.ny`,
    /// `.nz`) or distance (`<name>.d`), as the workflows label a camera's
    /// plane (`cam1.plane.nx`).
    pub(super) fn block_labels(&self, distance: bool) -> BlockLabels {
        let name = &self.name;
        if distance {
            BlockLabels::shared_owned(format!("{name}.d"), vec![format!("{name}.d")])
        } else {
            BlockLabels::shared_owned(
                format!("{name}.normal"),
                ["nx", "ny", "nz"]
                    .iter()
                    .map(|c| format!("{name}.{c}"))
                    .collect(),
            )
        }
    }
}

impl Problem {
    /// Add a laser plane named `name` with initial value `plane`, in the
    /// frame of the cameras whose laser residuals read it.
    ///
    /// The plane is estimated as a unit normal and a signed distance
    /// (`n · p + d = 0`), labelled in the covariance `<name>.nx`,
    /// `<name>.ny`, `<name>.nz` and `<name>.d`. Naming camera `i`'s plane
    /// `cam<i>.plane` (`plane` for one camera) gives the labels the
    /// workflows' results use, so the solution's covariance builds a
    /// [`CalibratedRig`](vision_calibration_core::CalibratedRig) with
    /// laser planes.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for an empty or reserved name (`cam`
    /// followed by digits), or one a block or plane already uses.
    pub fn add_plane(
        &mut self,
        name: impl Into<String>,
        plane: &LaserPlane,
    ) -> Result<PlaneId, Error> {
        let name = name.into();
        let normal_name = format!("{name}.normal");
        let distance_name = format!("{name}.d");
        for n in [&name, &normal_name, &distance_name] {
            self.check_name(n)?;
        }
        let index = self.planes.len();
        let n = plane.normal();
        let mut push = |name: String, values: Vec<f64>, manifold, distance| {
            self.push_block(Block {
                name,
                values: DVector::from_vec(values),
                manifold,
                fixed: FixedMask::all_free(),
                bounds: Vec::new(),
                labels: None,
                owner: Owner::Plane {
                    plane: index,
                    distance,
                },
            })
            .index
        };
        let normal = push(
            normal_name,
            vec![n.x, n.y, n.z],
            Manifold::UnitVector,
            false,
        );
        let distance = push(
            distance_name,
            vec![plane.distance()],
            Manifold::Euclidean,
            true,
        );
        self.planes.push(Plane {
            name,
            normal,
            distance,
        });
        Ok(PlaneId {
            problem: self.id,
            index,
        })
    }

    /// Hold the plane at its initial value.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a plane of another problem.
    pub fn fix_plane(&mut self, plane: PlaneId) -> Result<(), Error> {
        let p = self.plane_index(plane)?;
        let (normal, distance) = (self.planes[p].normal, self.planes[p].distance);
        for b in [normal, distance] {
            let block = &mut self.blocks[b];
            block.fixed = FixedMask::all_fixed(block.values.len());
        }
        Ok(())
    }

    /// Add the library's laser residual: the laser pixel `pixel`, seen by
    /// `camera` on a planar target posed by `chain`, against the laser
    /// plane `plane` (in that camera's frame), with its noise and a robust
    /// loss.
    ///
    /// The target is the plane `z = 0` of its frame. `residual` picks the
    /// measure: [`LaserlineResidualType::PointToPlane`] back-projects the
    /// pixel, intersects its ray with the target and takes the signed
    /// distance of that point from the laser plane, in metres;
    /// [`LaserlineResidualType::LineDistNormalized`] takes the distance of
    /// the pixel from the image of the line where the laser plane meets the
    /// target, in pixels. The residuals of each kind form one noise family
    /// ([`LaserNoise::Estimated`]), reported as
    /// [`ResidualKind::LaserPlane`](vision_calibration_core::ResidualKind::LaserPlane)
    /// or [`ResidualKind::LaserLine`](vision_calibration_core::ResidualKind::LaserLine).
    ///
    /// Its noise is estimated, so a problem with a laser residual cannot
    /// take a calibration added with [`Use::Prior`](super::Use::Prior).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a camera, block or plane of another
    /// problem, a chain block that is not SE3 or a rig chain through one
    /// block twice, a non-finite pixel, a loss with a scale that is not
    /// positive, [`LaserlineResidualType::LineDistNormalized`] with a
    /// projection that does not image lines as lines (use point-to-plane),
    /// or a pixel whose ray does not meet the target in front of the camera
    /// at the initial values.
    // Each argument is a distinct required input, in the order of
    // `add_reprojection`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn add_laser(
        &mut self,
        camera: CameraId,
        chain: PoseChain,
        plane: PlaneId,
        pixel: Pt2,
        residual: LaserlineResidualType,
        noise: LaserNoise,
        loss: RobustLoss,
    ) -> Result<(), Error> {
        let index = self.residuals.len();
        let fail = |what: String| Error::invalid_input(format!("residual {index}: {what}"));
        let cam = self.camera(camera).map_err(|e| fail(e.to_string()))?;
        let p = self.plane_index(plane).map_err(|e| fail(e.to_string()))?;
        let (ir_chain, poses) = match chain {
            PoseChain::Pose(pose) => (LaserChain::SinglePose, vec![pose]),
            PoseChain::Rig {
                cam_se3_rig,
                rig_se3_target,
            } => (LaserChain::Rig, vec![cam_se3_rig, rig_se3_target]),
        };
        let poses = poses
            .into_iter()
            .map(|b| self.index(b))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| fail(e.to_string()))?;
        if let Some(&b) = poses
            .iter()
            .find(|&&b| self.blocks[b].manifold != Manifold::Se3)
        {
            return Err(fail(format!(
                "the pose block {} is not SE3",
                self.blocks[b].name
            )));
        }
        if poses.len() == 2 && poses[0] == poses[1] {
            return Err(fail(format!(
                "the rig chain reads block {} twice",
                self.blocks[poses[0]].name
            )));
        }
        let uv = [pixel.x, pixel.y];
        if !uv.iter().all(|v| v.is_finite()) {
            return Err(fail("the pixel must be finite".into()));
        }
        loss.validate().map_err(|e| fail(e.to_string()))?;
        let spec = self.cameras[cam].spec;
        if residual == LaserlineResidualType::LineDistNormalized
            && !spec.projection.preserves_lines()
        {
            return Err(fail(format!(
                "the laser line-distance residual needs a projection that images lines as \
                 lines; {:?} does not (use point-to-plane)",
                spec.projection
            )));
        }
        let LaserNoise::Estimated = noise;
        let w = 1.0;

        let mut blocks: Vec<usize> = self.cameras[cam].blocks.iter().map(|b| b.index).collect();
        blocks.extend(&poses);
        blocks.extend([self.planes[p].normal, self.planes[p].distance]);
        let values: Vec<DVector<f64>> = blocks
            .iter()
            .map(|&i| self.blocks[i].values.clone())
            .collect();

        // The kernel answers a ray that misses the target with a constant
        // that has no gradient, so such a pixel is rejected here.
        let pose = |i: usize| se3(&self.blocks[poses[i]].values);
        let cam_se3_target = match ir_chain {
            LaserChain::Rig => pose(0) * pose(1),
            _ => pose(0),
        };
        let (camera_blocks, _) = CameraBlocks::split(spec, &values);
        let ray = camera_blocks.pixel_ray(uv);
        let origin = cam_se3_target.inverse_transform_point(&Point3::from(ray.origin));
        let direction = cam_se3_target.inverse_transform_vector(&ray.direction);
        let t = -origin.z / direction.z;
        let hits = direction.z.abs() > 1e-9
            && t.is_finite()
            && t > 0.0
            && (cam_se3_target * (origin + direction * t)).z > 0.0;
        if !hits {
            return Err(fail(format!(
                "the ray of pixel {uv:?} does not meet the target in front of camera {cam} at \
                 the initial values"
            )));
        }
        let factor = match residual {
            LaserlineResidualType::PointToPlane => FactorKind::LaserPointToPlane {
                model: spec,
                chain: ir_chain,
                laser_pixel: uv,
                w,
            },
            LaserlineResidualType::LineDistNormalized => FactorKind::LaserLineDistance {
                model: spec,
                chain: ir_chain,
                laser_pixel: uv,
                w,
            },
        };
        let at_initial = match residual {
            LaserlineResidualType::PointToPlane => {
                laser_point_to_plane_residual(spec, &ir_chain, &values, uv, w)
            }
            LaserlineResidualType::LineDistNormalized => {
                laser_line_distance_residual(spec, &ir_chain, &values, uv, w)
            }
        };
        if !at_initial[0].is_finite() {
            return Err(fail(format!(
                "the laser residual of pixel {uv:?} is not finite at the initial values"
            )));
        }
        self.residuals.push(Term {
            factor,
            blocks,
            loss,
        });
        Ok(())
    }

    /// The plane's index, for a plane of this problem.
    pub(super) fn plane_index(&self, plane: PlaneId) -> Result<usize, Error> {
        if plane.problem == self.id && plane.index < self.planes.len() {
            Ok(plane.index)
        } else {
            Err(Error::invalid_input("the plane belongs to another problem"))
        }
    }
}

impl Solution {
    /// The laser plane at the solution; `None` for a plane of another
    /// problem.
    pub fn plane(&self, plane: PlaneId) -> Option<LaserPlane> {
        if plane.problem != self.problem {
            return None;
        }
        let p = self.planes.get(plane.index)?;
        plane_from_split_dvec(
            self.values[p.normal].as_view(),
            self.values[p.distance].as_view(),
        )
        .ok()
    }
}
