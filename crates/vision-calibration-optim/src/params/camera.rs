//! The parameter blocks of a camera, for every problem builder.
//!
//! A camera is its [`CameraParams`]; its model ([`CameraModelSpec`]) decides
//! which blocks it has. [`add_camera_blocks`] packs cameras into an IR with
//! their resolved [`FixedParams`] and optional [`CameraBounds`],
//! [`unpack_camera`] reads a camera back from a solution, and
//! [`camera_block_labels`] names a camera block's coordinates for the
//! covariance. The block names are part of the covariance and regression
//! contract: `cam`, `proj`, `dist` and `sensor`, with a `/<camera>` suffix in
//! a multi-camera problem.

use std::collections::HashMap;

use nalgebra::{DVector, DVectorView};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraModelSpec, CameraParams, FixedParams, INTRINSICS_PARAM_NAMES, ProjectionKind,
    ProjectionParams, ScheimpflugParams, SensorKind, SensorParams,
};

use crate::Error;
use crate::ir::{Bound, FixedMask, ManifoldKind, ParamId, ProblemIR};
use crate::params::distortion::{pack_distortion_params, unpack_distortion_params};
use crate::params::intrinsics::{INTRINSICS_DIM, pack_intrinsics, unpack_intrinsics};
use crate::uncertainty::BlockLabels;

/// Optional per-parameter box bounds on a camera.
///
/// Each `Some((lower, upper))` constrains that parameter; `None` leaves it
/// unbounded. On a tilted (Scheimpflug) sensor the bounds are the safety net
/// that turns the cold-start runaway (`fx → 0`, `cx → -1e19`) into a finite,
/// recoverable solve. Tilt bounds need a Scheimpflug sensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct CameraBounds {
    /// Bounds on `fx` (pixels).
    pub fx: Option<(f64, f64)>,
    /// Bounds on `fy` (pixels).
    pub fy: Option<(f64, f64)>,
    /// Bounds on `cx` (pixels).
    pub cx: Option<(f64, f64)>,
    /// Bounds on `cy` (pixels).
    pub cy: Option<(f64, f64)>,
    /// Bounds on `tilt_x` (radians); Scheimpflug sensor only.
    pub tilt_x: Option<(f64, f64)>,
    /// Bounds on `tilt_y` (radians); Scheimpflug sensor only.
    pub tilt_y: Option<(f64, f64)>,
}

impl CameraBounds {
    /// Per-index bounds for the intrinsics block (`[fx, fy, cx, cy]`).
    fn intrinsics_bounds(&self) -> Option<Vec<Bound>> {
        index_bounds(&[self.fx, self.fy, self.cx, self.cy])
    }

    /// Per-index bounds for the sensor block (`[tilt_x, tilt_y]`).
    fn sensor_bounds(&self) -> Option<Vec<Bound>> {
        index_bounds(&[self.tilt_x, self.tilt_y])
    }
}

fn index_bounds(bounds: &[Option<(f64, f64)>]) -> Option<Vec<Bound>> {
    let v: Vec<Bound> = bounds
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| b.map(|(lower, upper)| Bound { idx, lower, upper }))
        .collect();
    (!v.is_empty()).then_some(v)
}

/// The model of `camera`; a general homography sensor has none, so it
/// cannot be calibrated.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a homography sensor.
pub(crate) fn camera_spec(camera: &CameraParams) -> Result<CameraModelSpec, Error> {
    camera.spec().ok_or_else(|| {
        Error::invalid_input(
            "a calibrated camera needs a named sensor kind; a general homography sensor \
             cannot be calibrated",
        )
    })
}

/// One camera to pack: its values, fixed parameters and bounds.
pub(crate) struct CameraBlockInput<'a> {
    /// The camera's initial values; their model ([`camera_spec`]) decides
    /// the blocks.
    pub params: &'a CameraParams,
    /// Its fixed parameters, resolved against its model.
    pub fixed: &'a FixedParams,
    /// Optional box bounds on intrinsics and tilt.
    pub bounds: Option<&'a CameraBounds>,
}

/// The IR blocks of one camera.
#[derive(Debug, Clone)]
pub(crate) struct CameraBlockIds {
    intrinsics: ParamId,
    projection: Option<ParamId>,
    distortion: Option<ParamId>,
    sensor: Option<ParamId>,
}

impl CameraBlockIds {
    /// The camera's blocks in factor order: `[intrinsics, projection?,
    /// distortion?, sensor?]`, each optional block present exactly when its
    /// kind has parameters.
    pub(crate) fn ids(&self) -> impl Iterator<Item = ParamId> + '_ {
        std::iter::once(self.intrinsics)
            .chain(self.projection)
            .chain(self.distortion)
            .chain(self.sensor)
    }

    /// A factor's parameter list: the camera blocks followed by `chain`.
    pub(crate) fn with(&self, chain: &[ParamId]) -> Vec<ParamId> {
        self.ids().chain(chain.iter().copied()).collect()
    }
}

/// The name of slot `slot`'s block of camera `index` (`None` in a
/// single-camera problem).
fn block_name(slot: &str, index: Option<usize>) -> String {
    match index {
        Some(i) => format!("{slot}/{i}"),
        None => slot.to_string(),
    }
}

/// Add the blocks of `cameras` to `ir`, slot by slot: every camera's
/// intrinsics (`cam`), then projection (`proj`), distortion (`dist`) and
/// sensor (`sensor`) blocks for the kinds that have parameters. Blocks are
/// named with a `/<camera>` suffix when `indexed`.
///
/// # Errors
///
/// [`Error::InvalidInput`] when a camera has a general homography sensor,
/// or tilt bounds are given for a camera without a Scheimpflug sensor.
pub(crate) fn add_camera_blocks(
    ir: &mut ProblemIR,
    initial: &mut HashMap<String, DVector<f64>>,
    cameras: &[CameraBlockInput<'_>],
    indexed: bool,
) -> Result<Vec<CameraBlockIds>, Error> {
    let index = |i: usize| indexed.then_some(i);
    let specs = cameras
        .iter()
        .map(|camera| camera_spec(camera.params))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, (camera, spec)) in cameras.iter().zip(&specs).enumerate() {
        if camera
            .bounds
            .and_then(CameraBounds::sensor_bounds)
            .is_some()
            && spec.sensor != SensorKind::Scheimpflug
        {
            return Err(Error::invalid_input(format!(
                "tilt bounds need a Scheimpflug sensor; camera {i}'s sensor is {:?}",
                spec.sensor
            )));
        }
    }

    let mut intrinsics = Vec::with_capacity(cameras.len());
    for (i, camera) in cameras.iter().enumerate() {
        let name = block_name("cam", index(i));
        intrinsics.push(ir.add_param_block(
            &name,
            INTRINSICS_DIM,
            ManifoldKind::Euclidean,
            FixedMask::fix_indices(&camera.fixed.intrinsics),
            camera.bounds.and_then(CameraBounds::intrinsics_bounds),
        ));
        initial.insert(name, pack_intrinsics(&camera.params.intrinsics));
    }

    let mut slot = |slot: &str,
                    dim: fn(CameraModelSpec) -> usize,
                    fixed: fn(&FixedParams) -> &[usize],
                    bounds: fn(&CameraBounds) -> Option<Vec<Bound>>,
                    pack: fn(&CameraParams) -> DVector<f64>|
     -> Vec<Option<ParamId>> {
        cameras
            .iter()
            .zip(&specs)
            .enumerate()
            .map(|(i, (camera, spec))| {
                let dim = dim(*spec);
                (dim > 0).then(|| {
                    let name = block_name(slot, index(i));
                    let id = ir.add_param_block(
                        &name,
                        dim,
                        ManifoldKind::Euclidean,
                        FixedMask::fix_indices(fixed(camera.fixed)),
                        camera.bounds.and_then(bounds),
                    );
                    initial.insert(name, pack(camera.params));
                    id
                })
            })
            .collect()
    };
    let projection = slot(
        "proj",
        |s| s.projection.dim(),
        |f| f.projection.as_slice(),
        |_| None,
        |c| pack_projection(&c.projection),
    );
    let distortion = slot(
        "dist",
        |s| s.distortion.dim(),
        |f| f.distortion.as_slice(),
        |_| None,
        |c| pack_distortion_params(&c.distortion),
    );
    let sensor = slot(
        "sensor",
        |s| s.sensor.dim(),
        |f| f.sensor.as_slice(),
        CameraBounds::sensor_bounds,
        |c| pack_sensor(&c.sensor),
    );

    Ok(intrinsics
        .into_iter()
        .zip(projection)
        .zip(distortion)
        .zip(sensor)
        .map(
            |(((intrinsics, projection), distortion), sensor)| CameraBlockIds {
                intrinsics,
                projection,
                distortion,
                sensor,
            },
        )
        .collect())
}

/// The camera of model `spec` whose blocks a solution holds (camera `index`
/// of a multi-camera problem, `None` in a single-camera one).
///
/// # Errors
///
/// [`Error::Numerical`] when a block is missing from the solution;
/// [`Error::InvalidInput`] when a block has the wrong length.
pub(crate) fn unpack_camera(
    solution: &HashMap<String, DVector<f64>>,
    spec: CameraModelSpec,
    index: Option<usize>,
) -> Result<CameraParams, Error> {
    let block = |slot: &str| {
        let name = block_name(slot, index);
        solution
            .get(&name)
            .map(DVector::as_view)
            .ok_or_else(|| Error::numerical(format!("missing {name} solution block")))
    };
    let intrinsics = unpack_intrinsics(block("cam")?)?;
    let projection = match spec.projection {
        ProjectionKind::Pinhole => ProjectionParams::Pinhole,
    };
    let distortion = if spec.distortion.dim() > 0 {
        unpack_distortion_params(spec.distortion, block("dist")?)?
    } else {
        unpack_distortion_params(spec.distortion, DVector::zeros(0).as_view())?
    };
    let sensor = unpack_sensor(spec.sensor, || block("sensor"))?;
    Ok(CameraParams {
        projection,
        distortion,
        sensor,
        intrinsics,
    })
}

/// The values of a camera's blocks in factor order (`[intrinsics,
/// projection?, distortion?, sensor?]`), for evaluating a factor outside a
/// solve.
pub(crate) fn camera_block_values(camera: &CameraParams) -> Vec<DVector<f64>> {
    let mut blocks = vec![pack_intrinsics(&camera.intrinsics)];
    blocks.extend(
        [
            pack_projection(&camera.projection),
            pack_distortion_params(&camera.distortion),
            pack_sensor(&camera.sensor),
        ]
        .into_iter()
        .filter(|b| !b.is_empty()),
    );
    blocks
}

/// The projection block of `projection` (empty for a pinhole).
fn pack_projection(projection: &ProjectionParams) -> DVector<f64> {
    match projection {
        ProjectionParams::Pinhole => DVector::zeros(0),
    }
}

/// The sensor block of `sensor`: `[tilt_x, tilt_y]` for a Scheimpflug sensor,
/// empty otherwise.
fn pack_sensor(sensor: &SensorParams) -> DVector<f64> {
    match sensor {
        SensorParams::Scheimpflug { params } => {
            DVector::from_row_slice(&[params.tilt_x, params.tilt_y])
        }
        SensorParams::Identity | SensorParams::Homography { .. } => DVector::zeros(0),
    }
}

fn unpack_sensor<'a>(
    kind: SensorKind,
    block: impl FnOnce() -> Result<DVectorView<'a, f64>, Error>,
) -> Result<SensorParams, Error> {
    match kind {
        SensorKind::Identity => Ok(SensorParams::Identity),
        SensorKind::Scheimpflug => {
            let s = block()?;
            if s.len() != kind.dim() {
                return Err(Error::invalid_input(format!(
                    "expected a sensor block of length {}, got {}",
                    kind.dim(),
                    s.len()
                )));
            }
            Ok(SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: s[0],
                    tilt_y: s[1],
                },
            })
        }
    }
}

/// Labels of a camera block's coordinates, for a block whose slot name is
/// `slot` (`cam`, `proj`, `dist` or `sensor`); `None` for any other block.
/// `prefix` turns a parameter name into its label (adding `cam<i>.` in a
/// multi-camera problem).
pub(crate) fn camera_block_labels(
    slot: &str,
    spec: CameraModelSpec,
    prefix: impl Fn(&str) -> String,
) -> Option<BlockLabels> {
    let (name, coordinates): (&str, &[&str]) = match slot {
        "cam" => ("intrinsics", &INTRINSICS_PARAM_NAMES),
        "proj" => ("projection", spec.projection.param_names()),
        "dist" => ("distortion", spec.distortion.param_names()),
        "sensor" => ("sensor", spec.sensor.param_names()),
        _ => return None,
    };
    let coordinates: Vec<String> = coordinates.iter().map(|c| prefix(c)).collect();
    Some(BlockLabels::shared_owned(prefix(name), coordinates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, CameraFix, DistortionKind, DistortionParams, FxFyCxCy,
    };

    fn camera(sensor: SensorParams) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.1,
                    k2: 0.02,
                    k3: 0.0,
                    p1: 0.001,
                    p2: -0.001,
                    iters: 8,
                },
            },
            sensor,
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 790.0,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    #[test]
    fn blocks_are_grouped_by_slot_and_round_trip() {
        let pinhole = camera(SensorParams::Identity);
        let tilted = camera(SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.02,
                tilt_y: -0.01,
            },
        });
        let specs = [
            camera_spec(&pinhole).unwrap(),
            camera_spec(&tilted).unwrap(),
        ];
        let fixed: Vec<FixedParams> = specs
            .iter()
            .map(|s| CameraFix::default().resolve(*s).unwrap())
            .collect();
        let inputs = [
            CameraBlockInput {
                params: &pinhole,
                fixed: &fixed[0],
                bounds: None,
            },
            CameraBlockInput {
                params: &tilted,
                fixed: &fixed[1],
                bounds: None,
            },
        ];
        let mut ir = ProblemIR::new();
        let mut initial = HashMap::new();
        let ids = add_camera_blocks(&mut ir, &mut initial, &inputs, true).unwrap();
        let names: Vec<&str> = ir.params.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["cam/0", "cam/1", "dist/0", "dist/1", "sensor/1"]);
        assert_eq!(ids[0].ids().count(), 2);
        assert_eq!(ids[1].ids().count(), 3);
        assert!(ir.params[2].fixed.contains(2), "k3 fixed by default");

        let back = unpack_camera(&initial, specs[1], Some(1)).unwrap();
        assert_eq!(back.spec(), Some(specs[1]));
        let SensorParams::Scheimpflug { params } = back.sensor else {
            panic!("tilted camera expected");
        };
        assert_eq!((params.tilt_x, params.tilt_y), (0.02, -0.01));
    }

    #[test]
    fn tilt_bounds_need_a_tilted_sensor() {
        let pinhole = camera(SensorParams::Identity);
        let spec = camera_spec(&pinhole).unwrap();
        let fixed = CameraFix::default().resolve(spec).unwrap();
        let bounds = CameraBounds {
            tilt_x: Some((-0.1, 0.1)),
            ..CameraBounds::default()
        };
        let input = CameraBlockInput {
            params: &pinhole,
            fixed: &fixed,
            bounds: Some(&bounds),
        };
        let err = add_camera_blocks(&mut ProblemIR::new(), &mut HashMap::new(), &[input], false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Scheimpflug"), "{err}");
    }

    #[test]
    fn labels_name_each_slot() {
        let spec = CameraModelSpec::pinhole(DistortionKind::Division1, SensorKind::Scheimpflug);
        let dist = camera_block_labels("dist", spec, |s| format!("cam1.{s}")).unwrap();
        assert_eq!(dist.coordinates, ["cam1.lambda"]);
        let sensor = camera_block_labels("sensor", spec, str::to_string).unwrap();
        assert_eq!(sensor.coordinates, ["tilt_x", "tilt_y"]);
        assert!(camera_block_labels("extr", spec, str::to_string).is_none());
    }
}
