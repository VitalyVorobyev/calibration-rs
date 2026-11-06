//! Multiple-view geometry: measurements from a calibration result.
//!
//! Every function takes a calibration result (an export payload, as
//! `projection_uncertainty` does) and builds the `CalibratedRig` from it, so a
//! rig that carries laser planes serves `triangulate_laser` too. Observations
//! cross as serde payloads, poses as `(K, 4, 4)` arrays (a robot pose as a
//! row-major `(4, 4)` list with its `(6, 6)` covariance), point sets as
//! `(N, 3)` and `(3N, 3N)` arrays. Results are the serde payloads of the Rust
//! result types; the stereo rectification is an opaque class.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use vision_calibration::common::calibrated_cameras;
use vision_calibration::core::{
    CalibratedRig, CameraParams, FxFyCxCy, Iso3, PointCovariance, Pt2, Pt3, RobotFrame, RobotPose,
    Uncertainty,
};
use vision_calibration::mvg::bundle_adjust::{BundleAdjustOptions, Scale, bundle_adjust};
use vision_calibration::mvg::dense::{BlockMatchOptions, DisparityMap, GrayImage, match_block};
use vision_calibration::mvg::laser::{LaserObservation, triangulate_laser};
use vision_calibration::mvg::metrology::{self, PointSet};
use vision_calibration::mvg::pose::{PoseObservation, PoseOptions, estimate_pose};
use vision_calibration::mvg::pose_recovery::recover_relative_pose;
use vision_calibration::mvg::rectification::{
    DenseCloud, RectifiedMatch, RectifyOptions, StereoRectification, rectify,
};
use vision_calibration::mvg::robust::{
    estimate_essential, estimate_homography, recover_relative_pose_robust,
};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::PixelMatch;
use vision_calibration::mvg::types::{Captures, Correspondence2D};
use vision_calibration::mvg::{MvgError, RansacOptions};
use vision_calibration::nalgebra::{
    DMatrix, Matrix3, Matrix6, Rotation3, Translation3, UnitQuaternion,
};
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::Use;

use crate::control::PyRunControl;
use crate::session::{parse_checked, parse_optional_checked, to_py};
use crate::validation::value_err;
use crate::{parse_payload, runtime_err};

// ---------------------------------------------------------------------------
// Boundary helpers
// ---------------------------------------------------------------------------

/// An error of the MVG layer as a Python exception: an input the caller can
/// fix (a count, an index, a value) is a `ValueError`; a degenerate geometry
/// or a numerical failure a `RuntimeError`.
fn mvg_err(err: MvgError) -> PyErr {
    use vision_calibration::core::Error as CoreError;
    match &err {
        MvgError::InvalidInput { .. }
        | MvgError::InsufficientData { .. }
        | MvgError::CountMismatch { .. }
        | MvgError::Core(CoreError::InvalidInput { .. } | CoreError::InsufficientData { .. }) => {
            value_err(err.to_string())
        }
        _ => runtime_err(err.to_string()),
    }
}

/// The calibrated rig a calibration result (an export payload) describes.
fn rig_of(result: &Bound<'_, PyAny>) -> PyResult<CalibratedRig> {
    let export: serde_json::Value = parse_payload(result, "result")?;
    let cameras = calibrated_cameras(export)
        .map_err(|err| PyTypeError::new_err(format!("invalid calibration result: {err}")))?;
    cameras
        .calibrated_rig()
        .map_err(|err| value_err(err.to_string()))
}

/// A rotation-and-translation from a 4×4 matrix row slice.
fn iso_from_rows(rows: &[f64], index: usize) -> PyResult<Iso3> {
    let at = |r: usize, c: usize| rows[4 * r + c];
    let bottom = [at(3, 0), at(3, 1), at(3, 2), at(3, 3)];
    if bottom
        .iter()
        .zip([0.0, 0.0, 0.0, 1.0])
        .any(|(a, b): (&f64, f64)| (a - b).abs() > 1e-9)
    {
        return Err(value_err(format!(
            "pose {index}: the last row of a pose matrix must be [0, 0, 0, 1]"
        )));
    }
    let r = Matrix3::from_fn(&at);
    let defect = (r * r.transpose() - Matrix3::identity()).abs().max();
    if !(defect < 1e-6 && r.determinant() > 0.0) {
        return Err(value_err(format!(
            "pose {index}: the upper-left 3x3 block is not a rotation"
        )));
    }
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Ok(Iso3::from_parts(
        Translation3::new(at(0, 3), at(1, 3), at(2, 3)),
        q,
    ))
}

/// Poses from a `(K, 4, 4)` array of `rig_se3_world`-style matrices.
fn poses_of(poses: &PyReadonlyArray3<'_, f64>, name: &str) -> PyResult<Vec<Iso3>> {
    let shape = poses.shape();
    if shape[1] != 4 || shape[2] != 4 {
        return Err(value_err(format!(
            "{name} must have shape (K, 4, 4), got ({}, {}, {})",
            shape[0], shape[1], shape[2]
        )));
    }
    let data = poses.as_slice()?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(value_err(format!("{name}: non-finite value")));
    }
    data.as_chunks::<16>()
        .0
        .iter()
        .enumerate()
        .map(|(k, rows)| iso_from_rows(rows, k))
        .collect()
}

/// Captures on a robot as they cross from Python: the frame, and the robot's
/// pose at each capture (none for the frame the rig is fixed in).
#[derive(Deserialize)]
struct RobotCapturesPayload {
    frame: RobotFrame,
    robot: Vec<RobotPosePayload>,
}

/// A robot pose as it crosses from Python: a row-major `(4, 4)` matrix and
/// the `(6, 6)` covariance of its right perturbation, or none.
#[derive(Deserialize)]
struct RobotPosePayload {
    base_se3_gripper: Vec<f64>,
    covariance: Option<[[f64; 6]; 6]>,
}

/// The [`Captures`] of a call, owned.
enum OwnedCaptures {
    Rig,
    World(Vec<Iso3>),
    Handeye(RobotFrame),
    Robot(RobotFrame, Vec<RobotPose>),
}

impl OwnedCaptures {
    fn captures(&self) -> Captures<'_> {
        match self {
            OwnedCaptures::Rig => Captures::Rig,
            OwnedCaptures::World(poses) => Captures::World(poses),
            OwnedCaptures::Handeye(frame) => Captures::Handeye(*frame),
            OwnedCaptures::Robot(frame, robot) => Captures::Robot {
                frame: *frame,
                robot,
            },
        }
    }
}

/// The captures of a call: the world poses, the robot captures, or neither
/// (one capture in the rig frame).
fn captures_of(
    rig_se3_world: Option<PyReadonlyArray3<'_, f64>>,
    robot: Option<&Bound<'_, PyAny>>,
) -> PyResult<OwnedCaptures> {
    let robot = robot.filter(|r| !r.is_none());
    match (rig_se3_world, robot) {
        (None, None) => Ok(OwnedCaptures::Rig),
        (Some(poses), None) => Ok(OwnedCaptures::World(poses_of(&poses, "rig_se3_world")?)),
        (None, Some(robot)) => {
            let payload: RobotCapturesPayload = parse_checked(robot, "captures")?;
            if payload.robot.is_empty() {
                return Ok(OwnedCaptures::Handeye(payload.frame));
            }
            let poses = payload
                .robot
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    if p.base_se3_gripper.len() != 16 {
                        return Err(value_err(format!(
                            "robot pose {k}: base_se3_gripper must be a (4, 4) matrix"
                        )));
                    }
                    let pose = iso_from_rows(&p.base_se3_gripper, k)?;
                    match p.covariance {
                        None => Ok(RobotPose::exact(pose)),
                        Some(c) => RobotPose::new(pose, Matrix6::from_fn(|r, s| c[r][s]))
                            .map_err(|e| value_err(format!("robot pose {k}: {e}"))),
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(OwnedCaptures::Robot(payload.frame, poses))
        }
        (Some(_), Some(_)) => Err(value_err(
            "give the world poses or the robot captures, not both",
        )),
    }
}

fn array_2d<'a>(
    array: &'a PyReadonlyArray2<'_, f64>,
    cols: usize,
    name: &str,
) -> PyResult<&'a [f64]> {
    let shape = array.shape();
    if shape[1] != cols {
        return Err(value_err(format!(
            "{name} must have shape (N, {cols}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    let data = array.as_slice()?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(value_err(format!("{name}: non-finite value")));
    }
    Ok(data)
}

fn points_of(points: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<Pt3>> {
    Ok(array_2d(points, 3, name)?
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| Pt3::new(p[0], p[1], p[2]))
        .collect())
}

/// RANSAC settings as Python states them: every field.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RansacPayload {
    thresh: f64,
    max_iters: usize,
    min_inliers: usize,
    confidence: f64,
    seed: u64,
    refit_on_inliers: bool,
}

impl From<RansacPayload> for RansacOptions {
    fn from(p: RansacPayload) -> Self {
        Self {
            max_iters: p.max_iters,
            thresh: p.thresh,
            min_inliers: p.min_inliers,
            confidence: p.confidence,
            seed: p.seed,
            refit_on_inliers: p.refit_on_inliers,
        }
    }
}

fn ransac_of(ransac: &Bound<'_, PyAny>) -> PyResult<RansacOptions> {
    Ok(parse_checked::<RansacPayload>(ransac, "ransac")?.into())
}

// ---------------------------------------------------------------------------
// Observations and two-view geometry
// ---------------------------------------------------------------------------

/// The pixel noise a calibration states for a camera, as a covariance.
#[pyfunction]
pub(crate) fn mvg_reprojection_noise(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    camera: usize,
) -> PyResult<Py<PyAny>> {
    let noise = rig_of(result)?
        .reprojection_noise(camera)
        .map_err(|err| value_err(err.to_string()))?;
    to_py(py, &noise)
}

#[pyfunction(signature = (result, cameras, matches, ransac, progress=None))]
pub(crate) fn mvg_estimate_essential(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    cameras: [usize; 2],
    matches: &Bound<'_, PyAny>,
    ransac: &Bound<'_, PyAny>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let matches: Vec<PixelMatch> = parse_checked(matches, "matches")?;
    let ransac = ransac_of(ransac)?;
    let control = PyRunControl::new(progress);
    let estimate =
        py.detach(|| estimate_essential(&rig, cameras, &matches, &ransac, control.control()));
    let estimate = control.finish(estimate, mvg_err)?;
    to_py(py, &estimate)
}

#[pyfunction]
pub(crate) fn mvg_recover_relative_pose(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    cameras: [usize; 2],
    matches: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let matches: Vec<PixelMatch> = parse_checked(matches, "matches")?;
    let pose = recover_relative_pose(&rig, cameras, &matches).map_err(mvg_err)?;
    to_py(py, &pose)
}

#[pyfunction(signature = (result, cameras, matches, ransac, progress=None))]
pub(crate) fn mvg_recover_relative_pose_robust(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    cameras: [usize; 2],
    matches: &Bound<'_, PyAny>,
    ransac: &Bound<'_, PyAny>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let matches: Vec<PixelMatch> = parse_checked(matches, "matches")?;
    let ransac = ransac_of(ransac)?;
    let control = PyRunControl::new(progress);
    let pose = py.detach(|| {
        recover_relative_pose_robust(&rig, cameras, &matches, &ransac, control.control())
    });
    let pose = control.finish(pose, mvg_err)?;
    to_py(py, &pose)
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[pyfunction(signature = (result, rig_se3_world, robot, tracks, progress=None))]
pub(crate) fn mvg_triangulate(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    rig_se3_world: Option<PyReadonlyArray3<'_, f64>>,
    robot: Option<&Bound<'_, PyAny>>,
    tracks: &Bound<'_, PyAny>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let captures = captures_of(rig_se3_world, robot)?;
    let tracks: Vec<Vec<RigObservation>> = parse_checked(tracks, "tracks")?;
    let control = PyRunControl::new(progress);
    let found = py.detach(|| triangulate(&rig, captures.captures(), &tracks, control.control()));
    let found = control.finish(found, mvg_err)?;
    to_py(py, &found)
}

#[pyfunction(signature = (result, observations, ransac=None, loss=None, progress=None))]
pub(crate) fn mvg_estimate_pose(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    observations: &Bound<'_, PyAny>,
    ransac: Option<&Bound<'_, PyAny>>,
    loss: Option<&Bound<'_, PyAny>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let observations: Vec<PoseObservation> = parse_checked(observations, "observations")?;
    let mut options = PoseOptions::default();
    options.ransac = ransac.filter(|r| !r.is_none()).map(ransac_of).transpose()?;
    options.loss = parse_optional_checked::<RobustLoss>(loss, "loss")?.unwrap_or_default();
    let control = PyRunControl::new(progress);
    options.control = control.control().clone();
    let pose = py.detach(|| estimate_pose(&rig, &observations, &options));
    let pose = control.finish(pose, mvg_err)?;
    to_py(py, &pose)
}

/// A bundle adjustment as it crosses to Python: the refined captures and
/// points, the cameras as they ended (refined under a prior), the joint
/// uncertainty and the solve.
#[derive(Serialize)]
struct BundleAdjustmentPayload<'a> {
    rig_se3_world: &'a [Iso3],
    points: &'a [Pt3],
    cameras: &'a [CameraParams],
    cam_se3_rig: &'a [Iso3],
    uncertainty: &'a Uncertainty,
    final_cost: f64,
    num_iters: usize,
    reprojection_rms_px: f64,
}

#[pyfunction(signature = (result, rig_se3_world, points, tracks, scale, calibration="consider", loss=None, max_iters=100, progress=None))]
#[allow(clippy::too_many_arguments)] // the Rust options, flattened
pub(crate) fn mvg_bundle_adjust(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    rig_se3_world: PyReadonlyArray3<'_, f64>,
    points: PyReadonlyArray2<'_, f64>,
    tracks: &Bound<'_, PyAny>,
    scale: &Bound<'_, PyAny>,
    calibration: &str,
    loss: Option<&Bound<'_, PyAny>>,
    max_iters: usize,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let poses = poses_of(&rig_se3_world, "rig_se3_world")?;
    let points = points_of(&points, "points")?;
    let tracks: Vec<Vec<RigObservation>> = parse_checked(tracks, "tracks")?;
    let scale: Scale = parse_checked(scale, "scale")?;
    let mut options = BundleAdjustOptions::new(scale);
    options.calibration = match calibration {
        "consider" => Use::Consider,
        "prior" => Use::Prior,
        other => {
            return Err(value_err(format!(
                "calibration must be \"consider\" or \"prior\", got {other:?}"
            )));
        }
    };
    options.loss = parse_optional_checked::<RobustLoss>(loss, "loss")?.unwrap_or_default();
    options.max_iters = max_iters;
    let control = PyRunControl::new(progress);
    options.control = control.control().clone();
    let adjusted = py.detach(|| bundle_adjust(&rig, &poses, &points, &tracks, &options));
    let adjusted = control.finish(adjusted, mvg_err)?;
    to_py(
        py,
        &BundleAdjustmentPayload {
            rig_se3_world: &adjusted.rig_se3_world,
            points: &adjusted.points,
            cameras: adjusted.rig.cameras(),
            cam_se3_rig: adjusted.rig.cam_se3_rig(),
            uncertainty: &adjusted.uncertainty,
            final_cost: adjusted.report.final_cost,
            num_iters: adjusted.report.num_iters,
            reprojection_rms_px: adjusted.reprojection_rms_px,
        },
    )
}

#[pyfunction(signature = (result, observations, rig_se3_world=None, robot=None))]
pub(crate) fn mvg_triangulate_laser(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    observations: &Bound<'_, PyAny>,
    rig_se3_world: Option<PyReadonlyArray3<'_, f64>>,
    robot: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rig = rig_of(result)?;
    let captures = captures_of(rig_se3_world, robot)?;
    let observations: Vec<LaserObservation> = parse_checked(observations, "observations")?;
    let found = triangulate_laser(&rig, captures.captures(), &observations).map_err(mvg_err)?;
    to_py(py, &found)
}

// ---------------------------------------------------------------------------
// Metrology
// ---------------------------------------------------------------------------

/// The point set of `(N, 3)` points and their `(3N, 3N)` joint covariance.
fn point_set(
    points: &PyReadonlyArray2<'_, f64>,
    covariance: &PyReadonlyArray2<'_, f64>,
) -> PyResult<PointSet> {
    let pts = points_of(points, "points")?;
    let shape = covariance.shape();
    let data = covariance.as_slice()?;
    PointSet::new(pts, DMatrix::from_row_slice(shape[0], shape[1], data)).map_err(mvg_err)
}

#[pyfunction]
pub(crate) fn mvg_distance(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    a: usize,
    b: usize,
) -> PyResult<Py<PyAny>> {
    let set = point_set(&points, &covariance)?;
    to_py(py, &metrology::distance(&set, a, b).map_err(mvg_err)?)
}

#[pyfunction]
pub(crate) fn mvg_angle(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    a: usize,
    vertex: usize,
    b: usize,
) -> PyResult<Py<PyAny>> {
    let set = point_set(&points, &covariance)?;
    to_py(py, &metrology::angle(&set, a, vertex, b).map_err(mvg_err)?)
}

#[pyfunction]
pub(crate) fn mvg_fit_plane(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    indices: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    let set = point_set(&points, &covariance)?;
    to_py(py, &metrology::fit_plane(&set, &indices).map_err(mvg_err)?)
}

#[pyfunction]
pub(crate) fn mvg_point_to_plane(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    plane: Vec<usize>,
    p: usize,
) -> PyResult<Py<PyAny>> {
    let set = point_set(&points, &covariance)?;
    to_py(
        py,
        &metrology::point_to_plane(&set, &plane, p).map_err(mvg_err)?,
    )
}

// ---------------------------------------------------------------------------
// Stereo
// ---------------------------------------------------------------------------

/// A stereo rectification of two rig cameras (see `vision_calibration.mvg`).
#[pyclass(
    module = "vision_calibration._vision_calibration",
    name = "StereoRectification",
    frozen
)]
pub(crate) struct PyStereoRectification {
    inner: StereoRectification,
}

fn side_of(side: usize) -> PyResult<usize> {
    if side < 2 {
        Ok(side)
    } else {
        Err(value_err(format!(
            "side must be 0 (left) or 1 (right), got {side}"
        )))
    }
}

fn matrix_array<'py, const R: usize, const C: usize>(
    py: Python<'py>,
    rows: impl Fn(usize, usize) -> f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data: Vec<f64> = (0..R)
        .flat_map(|r| (0..C).map(move |c| (r, c)))
        .map(|(r, c)| rows(r, c))
        .collect();
    PyArray1::from_vec(py, data).reshape([R, C])
}

fn gray_image(image: &PyReadonlyArray2<'_, u8>, name: &str) -> PyResult<GrayImage> {
    let shape = image.shape();
    if shape[0] == 0 || shape[1] == 0 {
        return Err(value_err(format!("{name} is empty")));
    }
    Ok(GrayImage::new(
        shape[1],
        shape[0],
        image.as_slice()?.to_vec(),
    ))
}

#[pymethods]
impl PyStereoRectification {
    /// The rig camera indices `[left, right]`.
    fn cameras(&self) -> [usize; 2] {
        self.inner.cameras()
    }

    /// The rectified pair's baseline, metres.
    fn baseline(&self) -> f64 {
        self.inner.baseline()
    }

    /// The rectified intrinsics `(fx, fy, cx, cy)`.
    fn k_rect(&self) -> (f64, f64, f64, f64) {
        let k = self.inner.k_rect();
        (k.fx, k.fy, k.cx, k.cy)
    }

    /// The rotation `(3, 3)` from a side's camera frame to the rectified frame.
    fn rect_r_cam<'py>(&self, py: Python<'py>, side: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let r = self.inner.rect_r_cam(side_of(side)?);
        matrix_array::<3, 3>(py, |i, j| r[(i, j)])
    }

    /// The rectified pixel of a raw pixel, or `None` when it has no ray ahead.
    fn rectified_pixel(&self, side: usize, pixel: [f64; 2]) -> PyResult<Option<(f64, f64)>> {
        let side = side_of(side)?;
        Ok(self
            .inner
            .rectified_pixel(side, &Pt2::new(pixel[0], pixel[1]))
            .map(|p| (p.x, p.y)))
    }

    /// The raw pixel of a rectified pixel, or `None` when the camera does not
    /// project it.
    fn raw_pixel(&self, side: usize, pixel: [f64; 2]) -> PyResult<Option<(f64, f64)>> {
        let side = side_of(side)?;
        Ok(self
            .inner
            .raw_pixel(side, &Pt2::new(pixel[0], pixel[1]))
            .map(|p| (p.x, p.y)))
    }

    /// Resample a raw `(H, W)` uint8 image of a side onto the rectified grid of
    /// `width` × `height` pixels; pixels outside the raw image are 0.
    fn rectify_image<'py>(
        &self,
        py: Python<'py>,
        side: usize,
        image: PyReadonlyArray2<'py, u8>,
        width: usize,
        height: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let side = side_of(side)?;
        if width == 0 || height == 0 {
            return Err(value_err(
                "the rectified image needs positive width and height",
            ));
        }
        let src = gray_image(&image, "image")?;
        let out = py.detach(|| self.inner.rectify_image(side, &src, width, height));
        PyArray1::from_vec(py, out.data).reshape([height, width])
    }

    /// The 4×4 matrix mapping `(x, y, disparity, 1)` of the left rectified image
    /// to homogeneous camera coordinates.
    fn disparity_to_depth<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = self.inner.disparity_to_depth();
        matrix_array::<4, 4>(py, |i, j| q[(i, j)])
    }

    /// The point, in the left camera's frame, of a rectified left pixel and its
    /// disparity; `None` for a non-positive or non-finite disparity.
    fn point(&self, left: [f64; 2], disparity: f64) -> Option<[f64; 3]> {
        self.inner
            .point(&Pt2::new(left[0], left[1]), disparity)
            .map(|p| [p.x, p.y, p.z])
    }

    /// The `(3, 3)` covariance of that point from a disparity σ, m².
    fn disparity_covariance<'py>(
        &self,
        py: Python<'py>,
        left: [f64; 2],
        disparity: f64,
        sigma_d: f64,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.inner
            .disparity_covariance(&Pt2::new(left[0], left[1]), disparity, sigma_d)
            .map(|c| matrix_array::<3, 3>(py, |i, j| c[(i, j)]))
            .transpose()
    }

    /// The point cloud of a `(H, W)` float32 disparity map (`NaN` invalid)
    /// with a disparity σ, the points' covariance on demand.
    fn dense_cloud(
        &self,
        py: Python<'_>,
        result: &Bound<'_, PyAny>,
        disparity: PyReadonlyArray2<'_, f32>,
        sigma_d: f64,
    ) -> PyResult<PyDenseCloud> {
        let rig = rig_of(result)?;
        let shape = disparity.shape();
        let map = DisparityMap::new(shape[1], shape[0], disparity.as_slice()?.to_vec());
        let cloud = py
            .detach(|| self.inner.dense_cloud(&rig, &map, sigma_d))
            .map_err(mvg_err)?;
        Ok(PyDenseCloud { inner: cloud })
    }

    /// Triangulate rectified matches with the calibration's covariance.
    #[pyo3(signature = (result, matches, pixel_noise, progress=None))]
    fn triangulate(
        &self,
        py: Python<'_>,
        result: &Bound<'_, PyAny>,
        matches: &Bound<'_, PyAny>,
        pixel_noise: &Bound<'_, PyAny>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let rig = rig_of(result)?;
        let matches: Vec<RectifiedMatch> = parse_checked(matches, "matches")?;
        let noise: PointCovariance = parse_checked(pixel_noise, "pixel_noise")?;
        let control = PyRunControl::new(progress);
        let found = py.detach(|| {
            self.inner
                .triangulate(&rig, &matches, noise, control.control())
        });
        let found = control.finish(found, mvg_err)?;
        to_py(py, &found)
    }
}

/// The points of a disparity map, with their covariance on demand (see
/// `vision_calibration.mvg`).
#[pyclass(
    module = "vision_calibration._vision_calibration",
    name = "DenseCloud",
    frozen
)]
pub(crate) struct PyDenseCloud {
    inner: DenseCloud,
}

#[pymethods]
impl PyDenseCloud {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The `(N, 3)` points, rig frame, metres.
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data: Vec<f64> = self
            .inner
            .points()
            .iter()
            .flat_map(|p| [p.point.x, p.point.y, p.point.z])
            .collect();
        PyArray1::from_vec(py, data).reshape([self.inner.len(), 3])
    }

    /// The `(N, 2)` rectified left pixels `(x, y)` of the points.
    fn pixels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let data: Vec<i64> = self
            .inner
            .points()
            .iter()
            .flat_map(|p| p.pixel.map(|v| v as i64))
            .collect();
        PyArray1::from_vec(py, data).reshape([self.inner.len(), 2])
    }

    /// The `(N,)` disparities, pixels.
    fn disparities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let data: Vec<f64> = self.inner.points().iter().map(|p| p.disparity).collect();
        PyArray1::from_vec(py, data)
    }

    /// The disparity σ the cloud was built with, pixels.
    fn sigma_d(&self) -> f64 {
        self.inner.sigma_d()
    }

    /// Why the points have no covariance (the reason's kind), or `None`.
    fn covariance_unavailable(&self) -> Option<String> {
        self.inner.covariance_unavailable().map(|reason| {
            serde_json::to_value(reason)
                .ok()
                .and_then(|v| v["kind"].as_str().map(str::to_owned))
                .unwrap_or_else(|| format!("{reason:?}"))
        })
    }

    /// Point `i`'s `(3, 3)` covariance, m², or `None`.
    fn point_covariance<'py>(
        &self,
        py: Python<'py>,
        i: usize,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.inner
            .point_covariance(i)
            .map(|c| matrix_array::<3, 3>(py, |r, s| c[(r, s)]))
            .transpose()
    }

    /// The `(3k, 3k)` joint covariance of the points `indices`, m².
    fn covariance<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let c = py
            .detach(|| self.inner.covariance(&indices))
            .map_err(mvg_err)?;
        let n = c.nrows();
        // nalgebra is column-major; the covariance is symmetric, but copy
        // row by row so the layout never depends on it.
        let data: Vec<f64> = (0..n)
            .flat_map(|r| (0..n).map(move |s| (r, s)))
            .map(|(r, s)| c[(r, s)])
            .collect();
        PyArray1::from_vec(py, data).reshape([n, n])
    }
}

/// Rectify two cameras of a calibration.
#[pyfunction(signature = (result, cameras, k_rect=None, scale=1.0))]
pub(crate) fn mvg_rectify(
    result: &Bound<'_, PyAny>,
    cameras: [usize; 2],
    k_rect: Option<&Bound<'_, PyAny>>,
    scale: f64,
) -> PyResult<PyStereoRectification> {
    let rig = rig_of(result)?;
    let mut options = RectifyOptions::default();
    options.k_rect = parse_optional_checked::<FxFyCxCy<f64>>(k_rect, "k_rect")?;
    options.scale = scale;
    Ok(PyStereoRectification {
        inner: rectify(&rig, cameras, &options).map_err(mvg_err)?,
    })
}

/// Block-matching settings as Python states them: every field.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockMatchPayload {
    min_disparity: i32,
    num_disparities: i32,
    block_size: usize,
    min_correlation: f32,
    uniqueness_ratio: f32,
    lr_consistency: bool,
    lr_max_diff: f32,
    subpixel: bool,
    semi_global: bool,
    sgm_p1: f32,
    sgm_p2: f32,
}

/// Dense disparity of a rectified pair; `NaN` marks an invalid pixel.
#[pyfunction]
pub(crate) fn mvg_match_block<'py>(
    py: Python<'py>,
    left: PyReadonlyArray2<'py, u8>,
    right: PyReadonlyArray2<'py, u8>,
    options: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let p: BlockMatchPayload = parse_checked(options, "options")?;
    let opts = BlockMatchOptions {
        min_disparity: p.min_disparity,
        num_disparities: p.num_disparities,
        block_size: p.block_size,
        min_correlation: p.min_correlation,
        uniqueness_ratio: p.uniqueness_ratio,
        lr_consistency: p.lr_consistency,
        lr_max_diff: p.lr_max_diff,
        subpixel: p.subpixel,
        semi_global: p.semi_global,
        sgm_p1: p.sgm_p1,
        sgm_p2: p.sgm_p2,
    };
    let (l, r) = (gray_image(&left, "left")?, gray_image(&right, "right")?);
    let map = py.detach(|| match_block(&l, &r, &opts)).map_err(mvg_err)?;
    PyArray1::from_vec(py, map.data).reshape([map.height, map.width])
}

/// Robust homography of plain image-point correspondences `(N, 2, 2)`:
/// `[match][view][x, y]`.
#[pyfunction]
pub(crate) fn mvg_estimate_homography(
    py: Python<'_>,
    correspondences: PyReadonlyArray3<'_, f64>,
    ransac: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let shape = correspondences.shape();
    if shape[1] != 2 || shape[2] != 2 {
        return Err(value_err(format!(
            "correspondences must have shape (N, 2, 2), got ({}, {}, {})",
            shape[0], shape[1], shape[2]
        )));
    }
    let data = correspondences.as_slice()?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(value_err("correspondences: non-finite value"));
    }
    let corrs: Vec<Correspondence2D> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| Correspondence2D::new(Pt2::new(c[0], c[1]), Pt2::new(c[2], c[3])))
        .collect();
    let estimate = estimate_homography(&corrs, &ransac_of(ransac)?).map_err(mvg_err)?;
    to_py(
        py,
        &HomographyPayload {
            homography: estimate.homography,
            inliers: estimate.inliers,
            inlier_rms: estimate.inlier_rms,
        },
    )
}

#[derive(Serialize)]
struct HomographyPayload {
    homography: vision_calibration::core::Mat3,
    inliers: Vec<usize>,
    inlier_rms: f64,
}

/// The Rust defaults the Python option dataclasses mirror.
#[pyfunction]
pub(crate) fn mvg_defaults(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let r = RansacOptions::new(1.0);
    let b = BlockMatchOptions::default();
    to_py(
        py,
        &serde_json::json!({
            "ransac": {
                "max_iters": r.max_iters, "min_inliers": r.min_inliers,
                "confidence": r.confidence, "seed": r.seed,
                "refit_on_inliers": r.refit_on_inliers,
            },
            "block_match": {
                "min_disparity": b.min_disparity, "num_disparities": b.num_disparities,
                "block_size": b.block_size, "min_correlation": b.min_correlation,
                "uniqueness_ratio": b.uniqueness_ratio, "lr_consistency": b.lr_consistency,
                "lr_max_diff": b.lr_max_diff, "subpixel": b.subpixel,
                "semi_global": b.semi_global, "sgm_p1": b.sgm_p1, "sgm_p2": b.sgm_p2,
            },
        }),
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mvg_reprojection_noise, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_estimate_essential, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_recover_relative_pose, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_recover_relative_pose_robust, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_triangulate, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_estimate_pose, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_bundle_adjust, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_triangulate_laser, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_distance, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_angle, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_fit_plane, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_point_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_rectify, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_match_block, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_estimate_homography, m)?)?;
    m.add_function(wrap_pyfunction!(mvg_defaults, m)?)?;
    m.add_class::<PyStereoRectification>()?;
    m.add_class::<PyDenseCloud>()?;
    Ok(())
}
