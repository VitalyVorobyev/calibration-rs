//! A seed for a device type: [`DeviceSeed::aggregate`], the median of the
//! seeds of several units, with the spread of each value across them.

use std::collections::BTreeSet;

use nalgebra::{
    Matrix4, Quaternion, SymmetricEigen, Translation3, UnitQuaternion, Vector3, Vector4,
};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraModelSpec, CameraParams, DistortionParams, FxFyCxCy, INTRINSICS_PARAM_NAMES, ImageGrid,
    Iso3, LaserPlane, ProjectionParams, SensorParams, distortion_map,
};

use super::{DeviceSeed, DeviceSeedError, SeedCamera, SeedHandeye, SeedLens};

/// `1 / Φ⁻¹(3/4)`: the median absolute deviation of a normal sample times
/// this estimates its standard deviation.
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

/// Grid columns of the fold check: 32 nodes across, about one every 40 px of
/// a 1280 px image, enough to see a fold of the outer field.
const FOLD_CHECK_COLUMNS: u32 = 32;

/// A seed for a device type, from the seeds of several of its units, with
/// the spread of every value across them ([`DeviceSeed::aggregate`]).
///
/// The seed is a starting point for another unit of the type. It is not a
/// calibration of any unit (never deploy it), not an estimate of a "true"
/// camera of the type, and its spread is descriptive: it is neither the
/// seed's uncertainty nor a prior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SeedAggregate {
    /// The aggregated seed: the median of every value, its cameras in id
    /// order, its rig frame the camera with the smallest id.
    pub seed: DeviceSeed,
    /// How many seeds were aggregated.
    pub units: usize,
    /// Each aggregated value's spread across the seeds, in the seed's order:
    /// each camera's parameters, its mount and its laser plane, then the
    /// hand-eye.
    pub spread: Vec<SeedSpread>,
    /// What some seeds have and others do not, left out of the aggregate
    /// rather than filled in: `<camera>.cam_se3_rig`, `<camera>.laser_plane`
    /// or `handeye`.
    pub omitted: Vec<String>,
}

/// The values one quantity takes across the aggregated seeds, and their
/// robust summary.
///
/// With the values `x₁ … xₙ`: `median` is the middle one in sorted order
/// (the mean of the two middle ones for an even `n`); `mad_sigma` is
/// `1.4826 · median(|xᵢ − median|)`, the median absolute deviation scaled
/// to estimate a standard deviation for normally spread values (0 for one
/// value); `min` and `max` the extremes. One outlying unit moves neither the
/// median nor `mad_sigma` much; with two units the median is their mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SeedSpread {
    /// The quantity: `<camera>.<parameter>` for a camera parameter by its
    /// covariance name (`cam0.fx`, `cam0.k1`, `cam0.tilt_x`);
    /// `<camera>.cam_se3_rig.<axis>` for a mount; `<camera>.laser_plane.<nx,
    /// ny, nz, distance>` for a laser plane; `handeye.<axis>` for the
    /// hand-eye. A pose's axes are `rx`, `ry`, `rz`, the rotation vector
    /// about the seeds' chordal mean rotation, radians, and `tx`, `ty`,
    /// `tz`, the translation, metres.
    pub label: String,
    /// Each seed's value, in the order the seeds were given.
    pub values: Vec<f64>,
    /// The median of the values: the aggregate's value.
    pub median: f64,
    /// `1.4826 · median(|xᵢ − median|)`, in the values' unit.
    pub mad_sigma: f64,
    /// The smallest value.
    pub min: f64,
    /// The largest value.
    pub max: f64,
}

impl DeviceSeed {
    /// A starting point for a device type from the seeds of several of its
    /// units: each value the median across them.
    ///
    /// Give one seed per unit (several of one unit would weigh it more),
    /// each made from a calibration ([`from_calibration`](Self::from_calibration),
    /// [`from_run`](Self::from_run)). The seeds must have the same camera
    /// ids, each camera the same image size and camera model in every seed,
    /// and the same hand-eye mode where they have one.
    ///
    /// - Every seed is first re-expressed in the frame of the camera with
    ///   the smallest id (its mount the identity), so the result does not
    ///   depend on which camera each unit's run took as the reference; the
    ///   hand-eye moves with the rig frame.
    /// - Each camera parameter is the median of its values.
    /// - A rotation (a mount's, the hand-eye's) is the chordal mean `R̄` of
    ///   the seeds' rotations (the top eigenvector of `Σ qᵢqᵢᵀ`, quaternions
    ///   `qᵢ`), corrected by the componentwise median `δ̃` of the rotation
    ///   vectors `δᵢ = Log(R̄ᵀRᵢ)`: `R̄·Exp(δ̃)`. A translation is the
    ///   componentwise median.
    /// - A laser plane, its sign made `distance > 0` (a laser plane does not
    ///   pass through its camera's centre), is the componentwise median of
    ///   the normals, renormalized, with the median distance.
    /// - A mount, laser plane or hand-eye that only some seeds have is left
    ///   out and listed in [`SeedAggregate::omitted`]; without mounts, a
    ///   seed of several cameras also leaves out the hand-eye.
    ///
    /// The result is the same, bit for bit, for the seeds in any order; its
    /// cameras are in id order.
    ///
    /// # Errors
    ///
    /// Any [`validate`](Self::validate) error of a seed;
    /// [`DeviceSeedError::AggregateMismatch`] for no seed, other camera ids,
    /// image sizes, models or hand-eye modes;
    /// [`DeviceSeedError::NominalNotAggregated`] for a datasheet lens;
    /// [`DeviceSeedError::AggregateFolds`] when an aggregated lens folds
    /// inside the image where every seed's lens of that camera is valid
    /// (checked on a grid of 32 columns over the image).
    pub fn aggregate(units: &[DeviceSeed]) -> Result<SeedAggregate, DeviceSeedError> {
        let first = units.first().ok_or_else(|| mismatch("no seed was given"))?;
        for unit in units {
            unit.validate()?;
        }
        let ids: BTreeSet<&str> = first.cameras.iter().map(|c| c.id.as_str()).collect();
        for (index, unit) in units.iter().enumerate() {
            let other: BTreeSet<&str> = unit.cameras.iter().map(|c| c.id.as_str()).collect();
            if other != ids {
                return Err(mismatch(format!(
                    "seed {index} has the cameras {other:?}, seed 0 {ids:?}"
                )));
            }
        }
        // `validate` refuses a seed without cameras.
        let reference = ids.first().copied().unwrap_or_default();
        let several = ids.len() > 1;
        // Mounts in every seed's every camera, or none in the aggregate.
        let mounted = units
            .iter()
            .all(|unit| unit.cameras.iter().all(|c| c.cam_se3_rig.is_some()));
        let gauges: Vec<Iso3> = units
            .iter()
            .map(|unit| unit.mount_or_identity(reference))
            .collect();

        let mut spread = Vec::new();
        let mut omitted = Vec::new();
        let mut cameras = Vec::with_capacity(ids.len());
        for id in &ids {
            let of_units: Vec<&SeedCamera> = units
                .iter()
                .map(|unit| {
                    unit.camera(id)
                        .expect("every seed has every id, checked above")
                })
                .collect();
            let image_size = of_units[0].image_size;
            if let Some(other) = of_units.iter().find(|c| c.image_size != image_size) {
                return Err(mismatch(format!(
                    "camera `{id}` is for {image_size:?} px images in one seed and {:?} px in \
                     another",
                    other.image_size
                )));
            }
            let lenses = of_units
                .iter()
                .map(|c| match &c.lens {
                    SeedLens::Calibrated { camera } => Ok(camera),
                    SeedLens::Nominal { .. } => Err(DeviceSeedError::NominalNotAggregated {
                        camera: (*id).to_string(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let camera = median_camera(id, &lenses, image_size, &mut spread)?;
            let mut seed_camera = SeedCamera::new(*id, image_size, SeedLens::Calibrated { camera });

            if several && mounted {
                seed_camera.cam_se3_rig = Some(if *id == reference {
                    Iso3::identity()
                } else {
                    let mounts: Vec<Iso3> = of_units
                        .iter()
                        .zip(&gauges)
                        .map(|(c, ref_se3_rig)| {
                            c.cam_se3_rig
                                .expect("every camera is mounted, checked above")
                                * ref_se3_rig.inverse()
                        })
                        .collect();
                    median_pose(&format!("{id}.cam_se3_rig"), &mounts, &mut spread)
                });
            } else if of_units.iter().any(|c| c.cam_se3_rig.is_some()) {
                omitted.push(format!("{id}.cam_se3_rig"));
            }

            let planes: Option<Vec<&LaserPlane>> =
                of_units.iter().map(|c| c.laser_plane.as_ref()).collect();
            match planes {
                Some(planes) => {
                    seed_camera.laser_plane = Some(median_plane(id, &planes, &mut spread)?);
                }
                None if of_units.iter().any(|c| c.laser_plane.is_some()) => {
                    omitted.push(format!("{id}.laser_plane"));
                }
                None => {}
            }
            cameras.push(seed_camera);
        }

        let mut seed = DeviceSeed::new(cameras);
        let handeyes: Option<Vec<SeedHandeye>> = units.iter().map(|unit| unit.handeye).collect();
        match handeyes {
            // Without mounts, a seed of several cameras has no rig frame to
            // state a hand-eye in.
            Some(_) if several && !mounted => omitted.push("handeye".to_string()),
            Some(handeyes) => {
                seed.handeye = Some(median_handeye(&handeyes, &gauges, &mut spread)?);
            }
            None if units.iter().any(|unit| unit.handeye.is_some()) => {
                omitted.push("handeye".to_string());
            }
            None => {}
        }
        seed.description = Some(format!(
            "the median of {} seeds of one device type",
            units.len()
        ));
        seed.validate()?;
        Ok(SeedAggregate {
            seed,
            units: units.len(),
            spread,
            omitted,
        })
    }

    /// The mount of camera `id` in the seed's rig frame, or the identity
    /// when the seed has no mounts (a seed of one camera, whose frame is its
    /// rig frame).
    fn mount_or_identity(&self, id: &str) -> Iso3 {
        self.camera(id)
            .and_then(|camera| camera.cam_se3_rig)
            .unwrap_or_else(Iso3::identity)
    }
}

fn mismatch(what: impl Into<String>) -> DeviceSeedError {
    DeviceSeedError::AggregateMismatch { what: what.into() }
}

/// The median of `values`, with its spread pushed to `spread` as `label`.
fn median_of(label: String, values: Vec<f64>, spread: &mut Vec<SeedSpread>) -> f64 {
    let summary = summarize(label, values);
    let median = summary.median;
    spread.push(summary);
    median
}

/// The robust summary of `values` (not empty), as [`SeedSpread`] defines it.
fn summarize(label: String, values: Vec<f64>) -> SeedSpread {
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    let median = middle(&sorted);
    let mut deviations: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    SeedSpread {
        label,
        median,
        mad_sigma: MAD_TO_SIGMA * middle(&deviations),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        values,
    }
}

/// The median of sorted, non-empty `sorted`.
fn middle(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// The camera whose every parameter is the median of `lenses`' (one model),
/// checked not to fold where every lens is valid.
fn median_camera(
    id: &str,
    lenses: &[&CameraParams],
    image_size: [u32; 2],
    spread: &mut Vec<SeedSpread>,
) -> Result<CameraParams, DeviceSeedError> {
    // `validate` refuses a homography sensor, the one without a spec.
    let spec_of = |lens: &CameraParams| lens.spec().expect("a validated lens has a model");
    let spec: CameraModelSpec = spec_of(lenses[0]);
    if let Some(other) = lenses.iter().map(|l| spec_of(l)).find(|s| *s != spec) {
        return Err(mismatch(format!(
            "camera `{id}` is of the model {spec:?} in one seed and {other:?} in another"
        )));
    }
    let labels: Vec<&str> = INTRINSICS_PARAM_NAMES
        .iter()
        .chain(spec.projection.param_names())
        .chain(spec.distortion.param_names())
        .chain(spec.sensor.param_names())
        .copied()
        .collect();
    let values: Vec<Vec<f64>> = lenses.iter().map(|lens| lens.blocks().concat()).collect();
    let medians: Vec<f64> = labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            median_of(
                format!("{id}.{label}"),
                values.iter().map(|v| v[index]).collect(),
                spread,
            )
        })
        .collect();
    let (k, rest) = medians.split_at(INTRINSICS_PARAM_NAMES.len());
    let (projection, rest) = rest.split_at(spec.projection.dim());
    let (distortion, sensor) = rest.split_at(spec.distortion.dim());
    let built = || -> Result<CameraParams, vision_calibration_core::Error> {
        Ok(CameraParams::new(
            ProjectionParams::from_values(spec.projection, projection)?,
            DistortionParams::from_values(spec.distortion, distortion)?,
            SensorParams::from_values(spec.sensor, sensor)?,
            FxFyCxCy {
                fx: k[0],
                fy: k[1],
                cx: k[2],
                cy: k[3],
            },
        ))
    };
    let camera = built().map_err(|e| mismatch(format!("camera `{id}`: {e}")))?;
    check_unfolded(id, &camera, lenses, image_size)?;
    Ok(camera)
}

/// Every node of a grid over the image where each of `lenses` has a value
/// must have one for `camera`.
fn check_unfolded(
    id: &str,
    camera: &CameraParams,
    lenses: &[&CameraParams],
    image_size: [u32; 2],
) -> Result<(), DeviceSeedError> {
    let folds = || DeviceSeedError::AggregateFolds {
        camera: id.to_string(),
    };
    let grid = ImageGrid::with_columns(image_size, FOLD_CHECK_COLUMNS)
        .map_err(|e| mismatch(format!("camera `{id}`: {e}")))?;
    let aggregate = distortion_map(camera, &grid).map_err(|_| folds())?;
    let mut everywhere = vec![true; grid.len()];
    for lens in lenses {
        // A lens that does not build was refused by `validate`; one that
        // fails here constrains no node.
        let Ok(map) = distortion_map(lens, &grid) else {
            continue;
        };
        for (valid, dx) in everywhere.iter_mut().zip(&map.total.dx) {
            *valid &= dx.is_finite();
        }
    }
    let lost = everywhere
        .iter()
        .zip(&aggregate.total.dx)
        .any(|(valid, dx)| *valid && !dx.is_finite());
    if lost { Err(folds()) } else { Ok(()) }
}

/// The median pose of `poses`, its spread pushed as `<label>.<axis>`.
fn median_pose(label: &str, poses: &[Iso3], spread: &mut Vec<SeedSpread>) -> Iso3 {
    let rotations: Vec<UnitQuaternion<f64>> = poses.iter().map(|p| p.rotation).collect();
    let mean = chordal_mean(&rotations);
    let deltas: Vec<Vector3<f64>> = rotations
        .iter()
        .map(|r| (mean.inverse() * r).scaled_axis())
        .collect();
    let mut delta = Vector3::zeros();
    for (axis, name) in ["rx", "ry", "rz"].iter().enumerate() {
        delta[axis] = median_of(
            format!("{label}.{name}"),
            deltas.iter().map(|d| d[axis]).collect(),
            spread,
        );
    }
    let mut t = [0.0; 3];
    for (axis, name) in ["tx", "ty", "tz"].iter().enumerate() {
        t[axis] = median_of(
            format!("{label}.{name}"),
            poses.iter().map(|p| p.translation.vector[axis]).collect(),
            spread,
        );
    }
    Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        mean * UnitQuaternion::from_scaled_axis(delta),
    )
}

/// The chordal (Markley) mean of `rotations`: the unit quaternion
/// maximizing `Σ (qᵀqᵢ)²`, the top eigenvector of `Σ qᵢqᵢᵀ`. The sum runs
/// over the quaternions in a canonical order, so it does not depend on the
/// order given.
fn chordal_mean(rotations: &[UnitQuaternion<f64>]) -> UnitQuaternion<f64> {
    let mut quaternions: Vec<Vector4<f64>> =
        rotations.iter().map(|r| canonical(r.coords)).collect();
    quaternions.sort_by(|a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.total_cmp(y))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut scatter = Matrix4::<f64>::zeros();
    for q in &quaternions {
        scatter += q * q.transpose();
    }
    let eigen = SymmetricEigen::new(scatter);
    let top = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index);
    let q = canonical(eigen.eigenvectors.column(top).into_owned());
    UnitQuaternion::new_normalize(Quaternion::from(q))
}

/// The quaternion `[i, j, k, w]` or its negative, the same rotation, with
/// its first non-zero coordinate in `w, i, j, k` order positive.
fn canonical(q: Vector4<f64>) -> Vector4<f64> {
    let leading = [q[3], q[0], q[1], q[2]]
        .into_iter()
        .find(|v| *v != 0.0)
        .unwrap_or(0.0);
    if leading < 0.0 { -q } else { q }
}

/// The median of camera `id`'s laser planes, each first given the sign
/// `distance > 0`.
fn median_plane(
    id: &str,
    planes: &[&LaserPlane],
    spread: &mut Vec<SeedSpread>,
) -> Result<LaserPlane, DeviceSeedError> {
    let signed: Vec<(Vector3<f64>, f64)> = planes
        .iter()
        .map(|p| {
            let (n, d) = (p.normal().into_inner(), p.distance());
            if d < 0.0 { (-n, -d) } else { (n, d) }
        })
        .collect();
    let mut normal = Vector3::zeros();
    for (axis, name) in ["nx", "ny", "nz"].iter().enumerate() {
        normal[axis] = median_of(
            format!("{id}.laser_plane.{name}"),
            signed.iter().map(|(n, _)| n[axis]).collect(),
            spread,
        );
    }
    let distance = median_of(
        format!("{id}.laser_plane.distance"),
        signed.iter().map(|(_, d)| *d).collect(),
        spread,
    );
    let length = normal.norm();
    // NaN fails the test too.
    if length.is_nan() || length <= 1e-12 {
        return Err(mismatch(format!(
            "the laser planes of camera `{id}` have no common orientation"
        )));
    }
    LaserPlane::new(normal / length, distance).map_err(|e| mismatch(format!("camera `{id}`: {e}")))
}

/// The median hand-eye of `handeyes` (one mode), each re-expressed in the
/// frame whose mount in its seed's rig frame is `gauges[i]`.
fn median_handeye(
    handeyes: &[SeedHandeye],
    gauges: &[Iso3],
    spread: &mut Vec<SeedSpread>,
) -> Result<SeedHandeye, DeviceSeedError> {
    let eye_in_hand = matches!(handeyes[0], SeedHandeye::EyeInHand { .. });
    let poses = handeyes
        .iter()
        .zip(gauges)
        .map(|(handeye, ref_se3_rig)| match (handeye, eye_in_hand) {
            (SeedHandeye::EyeInHand { gripper_se3_rig }, true) => {
                Ok(gripper_se3_rig * ref_se3_rig.inverse())
            }
            (SeedHandeye::EyeToHand { rig_se3_base }, false) => Ok(ref_se3_rig * rig_se3_base),
            _ => Err(mismatch("the seeds' hand-eyes are of different modes")),
        })
        .collect::<Result<Vec<Iso3>, _>>()?;
    let pose = median_pose("handeye", &poses, spread);
    Ok(if eye_in_hand {
        SeedHandeye::EyeInHand {
            gripper_se3_rig: pose,
        }
    } else {
        SeedHandeye::EyeToHand { rig_se3_base: pose }
    })
}

#[cfg(test)]
mod tests;
