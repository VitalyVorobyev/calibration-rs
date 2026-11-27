//! [`projection_difference`]: where two calibrations of one camera project
//! the same ray, after the best compensating rotation.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::distortion_map::{backproject_node, project_ray};
use super::{DisplacementPlane, ImageGrid};
use crate::models::{CameraModel, CameraParams, ProjectionModel};
use crate::{Error, Real};

/// The central-difference step of the rotation fit's Jacobian, radians.
/// The pixel error of the step is `O(h²)` of the projection's third
/// derivative (about `1e-12` relative) and the rounding error about
/// `1e-16 · |q| / h` (below `1e-9` px/rad for pixels under `1e4`): both far
/// below what moves the fitted rotation.
const JACOBIAN_STEP_RAD: f64 = 1e-6;

/// The fit stops once a Gauss-Newton step is shorter than this, radians.
const STEP_TOLERANCE_RAD: f64 = 1e-12;

/// The most Gauss-Newton iterations of the rotation fit. A step that does
/// not lower the cost is halved, so the cost never rises; the fit converges
/// in a handful of iterations for the small rotations two calibrations of
/// one camera differ by.
const MAX_ITERATIONS: usize = 50;

/// The most halvings of a Gauss-Newton step that does not lower the cost:
/// past them the fit sits at the minimum within rounding and stops.
const MAX_HALVINGS: usize = 40;

/// The fewest fitted nodes: two non-parallel rays fix a rotation, a third
/// leaves a residual.
const MIN_FIT_NODES: usize = 3;

/// How the second camera's frame is aligned with the first's before their
/// projections are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Compensation {
    /// Compare the rays as the two calibrations place them, without
    /// re-orienting either camera: for a camera frame used as calibrated,
    /// such as one run's intrinsics combined with another run's extrinsics.
    None,
    /// Fit the rotation of the second camera's frame that brings its
    /// projections closest to the first's (the default). The orientation of
    /// a camera frame is not separately observable from what uses the
    /// camera, so the difference left is the part no re-orientation removes.
    #[default]
    Rotation,
}

/// The nodes the compensating rotation is fitted on, and the fit
/// statistics of a [`ProjectionDifference`] are taken over.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum FitRegion {
    /// Every node where both cameras have a value (the default). Corners
    /// far from any observation are extrapolations of the fitted models and
    /// pull the rotation; prefer [`Covered`](Self::Covered) when the
    /// observations are at hand.
    #[default]
    WholeImage,
    /// The nodes whose `stride × stride` block holds at least one of these
    /// pixels (of the first camera's image; pixel centres at integer
    /// coordinates): the region the observations covered, where the data
    /// constrained the calibrations. Pixels off the image, and non-finite
    /// ones, are ignored. Each covered block counts once, however many
    /// pixels fall in it.
    Covered(Vec<[f64; 2]>),
}

/// The options of [`projection_difference`].
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct ProjectionDifferenceOptions {
    /// Whether the second camera's frame is rotated onto the first's.
    /// Default: [`Compensation::Rotation`].
    pub compensation: Compensation,
    /// The nodes the rotation is fitted on and the fit statistics are taken
    /// over. Default: [`FitRegion::WholeImage`].
    pub fit_region: FitRegion,
}

impl ProjectionDifferenceOptions {
    /// Options of the given compensation and fit region.
    pub fn new(compensation: Compensation, fit_region: FitRegion) -> Self {
        Self {
            compensation,
            fit_region,
        }
    }
}

/// Where two calibrations of one camera project the same ray, over the
/// first camera's image (see [`projection_difference`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProjectionDifference {
    /// The nodes the plane is sampled at, in the first camera's image.
    pub grid: ImageGrid,
    /// `b(R·ray) − u` at each node `u`, `ray` the first camera's ray through
    /// it, pixels: the second camera's pixel minus the first's. `NaN` where
    /// either camera has no value.
    pub difference: DisplacementPlane,
    /// The fitted rotation `R`, which takes the first camera's ray
    /// directions into the second camera's frame; the identity with
    /// [`Compensation::None`].
    pub camb_so3_cama: UnitQuaternion<Real>,
    /// Whether each node took part in the rotation fit and the fit
    /// statistics: in the fit region, with a value. Row-major, one entry
    /// per node.
    pub fitted: Vec<bool>,
    /// The number of nodes with a value.
    pub valid_nodes: usize,
    /// The number of fitted nodes (`true` entries of `fitted`).
    pub fit_nodes: usize,
    /// The root mean square of `|difference|` over the fitted nodes, pixels:
    /// the minimum the rotation reaches.
    pub rms_fit_px: f64,
    /// The largest `|difference|` over the fitted nodes, pixels.
    pub max_fit_px: f64,
    /// The root mean square of `|difference|` over every node with a value,
    /// pixels.
    pub rms_valid_px: f64,
    /// The largest `|difference|` over every node with a value, pixels.
    pub max_valid_px: f64,
}

impl ProjectionDifference {
    /// The angle of the fitted rotation, radians, in `[0, π]`.
    pub fn rotation_angle(&self) -> f64 {
        self.camb_so3_cama.angle()
    }
}

/// Where the camera `b` projects the rays the camera `a` sees at each node
/// of `grid`, after the rotation of `b`'s frame that brings the two closest
/// (see [`ProjectionDifference`]).
///
/// Two calibrations of one camera, of the same model or of different
/// models, compared through what they do rather than through their
/// parameters: at a node `u` of `a`'s image,
///
/// ```text
/// d(u) = a⁻¹(u)                                 a's ray through u
/// Δ(u) = b(R·d(u)) − u                          pixels, b − a
/// R    = argmin_R Σ_{u ∈ F} |b(R·d(u)) − u|²    over rotations, F the fitted nodes
/// ```
///
/// The rotation is the gauge between a camera's intrinsics and its
/// orientation: the poses of a calibration absorb any rotation of its
/// camera frame, and a principal-point shift trades with a small rotation.
/// Without it two equally good calibrations can differ by a near-uniform
/// shift that no use of the camera with re-estimated poses would see. The
/// models are central, so rays are compared by direction (at infinity) and
/// no translation enters. The fit is in `b`'s pixels, so `rms_fit_px` is
/// the smallest RMS difference any rotation reaches over the fit region
/// (the image-space mapping error of
/// [Hagemann et al. (2022)](https://doi.org/10.1007/s11263-021-01528-x)).
///
/// `a` is the reference: the grid is its image, the plane is indexed by its
/// pixels, and the sign is `b − a`. A node has a value where `a`
/// backprojects it (the rules of [`distortion_map`](super::distortion_map):
/// a finite inverse sensor homography, a round trip within `1e-6` px, the
/// projection's domain, no fold of the distortion out to the ray) and `b`
/// projects the rotated ray: in `b`'s projection domain, with `b`'s
/// distortion unfolded on the segment from the axis to it, and in front of
/// `b`'s sensor homography. A node beyond `b`'s fold is `NaN`, never a value
/// from the outer branch, and takes no part in the fit. A pixel `b` puts
/// off the image is a value: `b` says that ray misses the sensor.
///
/// # The fit
///
/// Gauss-Newton on `R ← Exp(δ)·R` from the identity, the `2 × 3` Jacobian
/// of each node by central differences (step `1e-6` rad), each node
/// weighted equally; a step that does not lower the cost is halved. It
/// stops when a step is below `1e-12` rad, no halving lowers the cost, or
/// after 50 iterations. The nodes are visited in grid order, so the result
/// is deterministic. A node `b` cannot project at the current rotation
/// leaves that iteration's fit. The fit samples the region at the grid's
/// nodes: compare differences computed on one grid.
///
/// # Reading the difference
///
/// - The **fit region** decides what the rotation absorbs. Over the whole
///   image the corners, far from any observation and extrapolated by both
///   models, pull the rotation; [`FitRegion::Covered`] with the observed
///   features fits where the data constrained the cameras, and the values
///   outside it are extrapolations (`max_valid_px`).
/// - For cameras of different fields of view (a fisheye and a pinhole)
///   the grid is `a`'s image: put the camera whose rays should be compared
///   first. A wide-angle `a` has nodes whose rays a pinhole `b` sends far
///   off the image, or cannot project at all (`NaN`).
/// - Distortion layers of two models differ even when their projections
///   agree ([`distortion_map`](super::distortion_map)); this difference does
///   not depend on how either model splits its projection between the focal
///   length and the distortion.
/// - The difference carries no σ. How precisely each calibration places a
///   ray is its own projection uncertainty, evaluated at the same nodes.
///
/// # Errors
///
/// - [`Error::Singular`] for a sensor homography that is not invertible (as
///   [`CameraParams::build`]), or a rotation fit whose normal equations are
///   singular (the fitted rays all on one line).
/// - [`Error::InsufficientData`] when fewer than 3 nodes of the fit region
///   have a value.
/// - [`Error::InvalidInput`] for a camera whose projection is not central.
///
/// # Example
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, ImageGrid,
///     ProjectionDifferenceOptions, ProjectionParams, SensorParams, projection_difference,
/// };
///
/// let camera = |cx: f64, k1: f64| {
///     CameraParams::new(
///         ProjectionParams::Pinhole,
///         DistortionParams::BrownConrady5 {
///             params: BrownConrady5 { k1, ..BrownConrady5::default() },
///         },
///         SensorParams::Identity,
///         FxFyCxCy { fx: 800.0, fy: 800.0, cx, cy: 479.5 },
///     )
/// };
/// let grid = ImageGrid::with_columns([1280, 960], 64)?;
/// let options = ProjectionDifferenceOptions::default();
/// // A principal point 2 px to the right: a rotation of the frame absorbs
/// // most of the shift.
/// let shifted = projection_difference(&camera(639.5, -0.1), &camera(641.5, -0.1), &grid, &options)?;
/// assert!(shifted.rms_fit_px < 0.5 && shifted.rotation_angle() > 2.0e-3);
/// // A k1 smaller by 0.02 is radial: no rotation absorbs it.
/// let radial = projection_difference(&camera(639.5, -0.1), &camera(639.5, -0.12), &grid, &options)?;
/// assert!(radial.max_fit_px > 10.0 && radial.rotation_angle() < 1e-12);
/// # Ok::<(), vision_calibration_core::Error>(())
/// ```
pub fn projection_difference(
    a: &CameraParams,
    b: &CameraParams,
    grid: &ImageGrid,
    options: &ProjectionDifferenceOptions,
) -> Result<ProjectionDifference, Error> {
    let model_a = a.build()?;
    let model_b = b.build()?;
    if !model_a.proj.is_central() || !model_b.proj.is_central() {
        return Err(Error::invalid_input(
            "projection_difference compares central cameras only",
        ));
    }
    let len = grid.len();
    let in_region = region_mask(grid, &options.fit_region);
    // a's ray through each node, unit length; None where a has no ray.
    let nodes: Vec<[f64; 2]> = grid.nodes().collect();
    let rays: Vec<Option<Vector3<Real>>> = nodes
        .iter()
        .map(|&u| backproject_node(&model_a, u).and_then(|p| p.try_normalize(0.0)))
        .collect();
    let fit_set: Vec<usize> = (0..len)
        .filter(|&i| in_region[i] && rays[i].is_some())
        .collect();

    let rotation = match options.compensation {
        Compensation::None => UnitQuaternion::identity(),
        Compensation::Rotation => fit_rotation(&model_b, &rays, &nodes, &fit_set)?,
    };

    let mut difference = DisplacementPlane::nan(len);
    let mut fitted = vec![false; len];
    let (mut valid_nodes, mut sum_valid, mut max_valid) = (0usize, 0.0, 0.0f64);
    let (mut fit_nodes, mut sum_fit, mut max_fit) = (0usize, 0.0, 0.0f64);
    for i in 0..len {
        let Some(delta) = rays[i].and_then(|d| node_difference(&model_b, &rotation, &d, nodes[i]))
        else {
            continue;
        };
        difference.dx[i] = delta[0];
        difference.dy[i] = delta[1];
        let squared = delta[0] * delta[0] + delta[1] * delta[1];
        valid_nodes += 1;
        sum_valid += squared;
        max_valid = max_valid.max(squared.sqrt());
        if in_region[i] {
            fitted[i] = true;
            fit_nodes += 1;
            sum_fit += squared;
            max_fit = max_fit.max(squared.sqrt());
        }
    }
    if fit_nodes < MIN_FIT_NODES {
        return Err(Error::InsufficientData {
            need: MIN_FIT_NODES,
            got: fit_nodes,
        });
    }
    Ok(ProjectionDifference {
        grid: *grid,
        difference,
        camb_so3_cama: rotation,
        fitted,
        valid_nodes,
        fit_nodes,
        rms_fit_px: (sum_fit / fit_nodes as f64).sqrt(),
        max_fit_px: max_fit,
        rms_valid_px: (sum_valid / valid_nodes as f64).sqrt(),
        max_valid_px: max_valid,
    })
}

/// Whether each node of `grid` is in `region`.
fn region_mask(grid: &ImageGrid, region: &FitRegion) -> Vec<bool> {
    match region {
        FitRegion::WholeImage => vec![true; grid.len()],
        FitRegion::Covered(pixels) => {
            let mut mask = vec![false; grid.len()];
            for &pixel in pixels {
                if let Some(i) = block_index(grid, pixel) {
                    mask[i] = true;
                }
            }
            mask
        }
    }
}

/// The row-major index of the block holding the image point `pixel`, or
/// `None` off the image (pixel `(0, 0)` covers `[-0.5, 0.5)²`).
fn block_index(grid: &ImageGrid, pixel: [f64; 2]) -> Option<usize> {
    let stride = f64::from(grid.stride());
    let [width, height] = grid.image_size();
    let cell = |v: f64, size: u32| {
        // NaN fails the range test.
        (v >= -0.5 && v < f64::from(size) - 0.5).then(|| ((v + 0.5) / stride).floor() as usize)
    };
    let column = cell(pixel[0], width)?;
    let row = cell(pixel[1], height)?;
    Some(row * grid.columns() + column)
}

/// `b(R·d) − u`, pixels, or `None` where `b` cannot project `R·d`.
fn node_difference(
    b: &CameraModel,
    rotation: &UnitQuaternion<Real>,
    d: &Vector3<Real>,
    u: [f64; 2],
) -> Option<[f64; 2]> {
    let q = project_ray(b, &(rotation * d))?.pixel;
    Some([q.x - u[0], q.y - u[1]])
}

/// The rotation `R` that minimizes `Σ |b(R·d_i) − u_i|²` over the nodes
/// `fit_set` (see [`projection_difference`]).
fn fit_rotation(
    b: &CameraModel,
    rays: &[Option<Vector3<Real>>],
    nodes: &[[f64; 2]],
    fit_set: &[usize],
) -> Result<UnitQuaternion<Real>, Error> {
    let residual = |rotation: &UnitQuaternion<Real>, i: usize| {
        let d = rays[i]?;
        node_difference(b, rotation, &d, nodes[i])
    };
    let mut rotation = UnitQuaternion::identity();
    for _ in 0..MAX_ITERATIONS {
        // The nodes b projects at the current rotation, with their residual
        // and Jacobian; the normal equations JᵀJ δ = −Jᵀr.
        let mut active = Vec::with_capacity(fit_set.len());
        let mut jtj = Matrix3::<Real>::zeros();
        let mut jtr = Vector3::<Real>::zeros();
        let mut cost = 0.0;
        for &i in fit_set {
            let Some(r) = residual(&rotation, i) else {
                continue;
            };
            let Some(jacobian) = node_jacobian(&rotation, |rot| residual(rot, i)) else {
                continue;
            };
            let [jx, jy] = jacobian;
            jtj += jx * jx.transpose() + jy * jy.transpose();
            jtr += jx * r[0] + jy * r[1];
            cost += r[0] * r[0] + r[1] * r[1];
            active.push(i);
        }
        if active.len() < MIN_FIT_NODES {
            return Err(Error::InsufficientData {
                need: MIN_FIT_NODES,
                got: active.len(),
            });
        }
        let step = jtj.cholesky().ok_or(Error::Singular)?.solve(&(-jtr));
        if !step.iter().all(|v| v.is_finite()) {
            return Err(Error::Singular);
        }
        // Halve a step that does not lower the cost over the same nodes.
        let mut step = step;
        let mut accepted = None;
        for _ in 0..MAX_HALVINGS {
            let candidate = UnitQuaternion::from_scaled_axis(step) * rotation;
            let new_cost = active.iter().try_fold(0.0, |sum, &i| {
                residual(&candidate, i).map(|r| sum + r[0] * r[0] + r[1] * r[1])
            });
            if new_cost.is_some_and(|c| c < cost) {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        let Some(next) = accepted else {
            // No step lowers the cost: the minimum, within rounding.
            break;
        };
        rotation = next;
        if step.norm() < STEP_TOLERANCE_RAD {
            break;
        }
    }
    Ok(rotation)
}

/// The two rows of `∂r/∂δ` at `δ = 0` for `r(Exp(δ)·R)`, as columns of
/// `δ`-space vectors (`[∂r_x/∂δ, ∂r_y/∂δ]`), by central differences; `None`
/// where a perturbed rotation leaves `b`'s domain.
fn node_jacobian(
    rotation: &UnitQuaternion<Real>,
    residual: impl Fn(&UnitQuaternion<Real>) -> Option<[f64; 2]>,
) -> Option<[Vector3<Real>; 2]> {
    let mut jx = Vector3::zeros();
    let mut jy = Vector3::zeros();
    for k in 0..3 {
        let mut delta = Vector3::zeros();
        delta[k] = JACOBIAN_STEP_RAD;
        let plus = residual(&(UnitQuaternion::from_scaled_axis(delta) * rotation))?;
        let minus = residual(&(UnitQuaternion::from_scaled_axis(-delta) * rotation))?;
        jx[k] = (plus[0] - minus[0]) / (2.0 * JACOBIAN_STEP_RAD);
        jy[k] = (plus[1] - minus[1]) / (2.0 * JACOBIAN_STEP_RAD);
    }
    Some([jx, jy])
}
