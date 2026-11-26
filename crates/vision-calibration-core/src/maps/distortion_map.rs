//! [`distortion_map`]: a camera's lens and sensor displacement over its image.

use nalgebra::{Point2, Vector3};

use super::ImageGrid;
use crate::models::{
    AnySensor, CameraModel, CameraParams, DistortionModel, IntrinsicsModel, ProjectionModel,
    SensorModel,
};
use crate::{Error, Real};

/// The largest distance, pixels, between a grid node and the forward
/// projection of the ray backprojected through it: a node whose inverse
/// does not reproduce it (Newton did not converge, or the inverse left the
/// model's domain) has no value.
const ROUND_TRIP_PX: f64 = 1e-6;

/// The smallest determinant of the distortion's Jacobian `∂D/∂n` on the
/// segment from the optical axis to a node's ray: below it the distortion
/// folds the image over (or nearly so) there, and the inverse found is not
/// on the sheet of the image the axis lies on.
const MIN_JACOBIAN_DET: f64 = 1e-6;

/// Points of the segment from the axis to a node's ray (the ray included)
/// at which [`MIN_JACOBIAN_DET`] is checked.
const FOLD_SAMPLES: u32 = 16;

/// The smallest `|z|` of the inverse sensor homography's output: below it
/// the node maps to infinity on the normalized plane.
const MIN_HOMOGRAPHY_DEPTH: f64 = 1e-12;

/// A displacement field on an [`ImageGrid`], pixels: one `(dx, dy)` per
/// node, row-major, `NaN` where the map has no value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DisplacementPlane {
    /// The `x` component at each node, pixels.
    pub dx: Vec<f64>,
    /// The `y` component at each node, pixels.
    pub dy: Vec<f64>,
}

impl DisplacementPlane {
    /// A plane of `len` nodes, all `NaN`.
    fn nan(len: usize) -> Self {
        Self {
            dx: vec![f64::NAN; len],
            dy: vec![f64::NAN; len],
        }
    }

    /// The length of the displacement at each node, pixels; `NaN` where the
    /// plane has no value.
    pub fn magnitude(&self) -> Vec<f64> {
        self.dx
            .iter()
            .zip(&self.dy)
            .map(|(x, y)| x.hypot(*y))
            .collect()
    }

    /// The largest displacement length over the nodes with a value, pixels;
    /// `None` when no node has one.
    pub fn max_magnitude(&self) -> Option<f64> {
        self.dx
            .iter()
            .zip(&self.dy)
            .map(|(x, y)| x.hypot(*y))
            .filter(|m| m.is_finite())
            .reduce(f64::max)
    }
}

/// The lens and sensor displacement of a camera over its image, in three
/// layers that sum exactly: `total = distortion + sensor`.
///
/// Every layer is `actual − reference` at the same camera ray, in pixels,
/// indexed by the pixel where the ray actually lands (the grid node): for
/// the camera `q = K(S(D(P(ray))))` and the node `u = q`,
///
/// - `distortion = K(D(n)) − K(n)`, `n = P(ray)`: the lens distortion as
///   the camera would show it without a sensor tilt;
/// - `sensor = K(S(D(n))) − K(D(n))`: what the sensor model adds (zero for
///   an identity sensor);
/// - `total = distortion + sensor = u − K(n)`, so `u − total` is the pixel
///   the ray would reach through the reference camera `K∘P`.
///
/// Barrel distortion therefore points inward at the corners. The reference
/// keeps the projection: for an equidistant (fisheye) camera it is the
/// `r = f·θ` image, defined over the whole domain, and the distortion layer
/// is the Kannala-Brandt polynomial's departure from it.
///
/// See [`distortion_map`] for the nodes without a value and the caveats.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DistortionMap {
    /// The nodes the planes are sampled at.
    pub grid: ImageGrid,
    /// `K∘S∘D∘P − K∘P` at the same ray: the whole displacement.
    pub total: DisplacementPlane,
    /// `K∘D∘P − K∘P`: the distortion as the camera would show it without a
    /// sensor tilt.
    pub distortion: DisplacementPlane,
    /// `K∘S∘D∘P − K∘D∘P`: what the sensor model adds; zero for an identity
    /// sensor.
    pub sensor: DisplacementPlane,
    /// The number of nodes with a value; the others are `NaN` in every
    /// plane.
    pub valid_nodes: usize,
}

/// The lens and sensor displacement of `camera` at every node of `grid`
/// (see [`DistortionMap`] for the layers and the sign).
///
/// The grid is in the image's own pixels: for a camera with a region of
/// interest, the region's size, with pixel `(0, 0)` its top-left.
///
/// A node has a value when the ray backprojected through it
/// (`K⁻¹`, `S⁻¹`, Newton's `D⁻¹`, `P⁻¹`) exists and is the lens's: the
/// inverse sensor homography is finite there, the ray projects back to the
/// node within `1e-6` px, the ray lies in the projection's domain, and the
/// distortion is orientation-preserving all the way out to it: the
/// determinant of `∂D/∂n` is above `1e-6` at 16 evenly spaced points of the
/// segment from the axis to the ray's normalized point, the last the point
/// itself (central differences with step `1e-6·max(1, |n|)`). The last
/// check refuses a Newton solution past the fold of a radial polynomial:
/// on its decreasing branch, on a branch where the polynomial rises again,
/// or on the branch that maps a ray through the axis to the opposite side;
/// each of these can project back to the node exactly. Nodes beyond the
/// fold, outside a fisheye polynomial's invertible range, or where the
/// sensor homography maps to infinity are `NaN` in every layer, as is a
/// node inside the fold whose Newton inverse settles on another branch.
///
/// # Reading the map
///
/// - The model has no image circle: a Kannala-Brandt polynomial that stays
///   monotone past the lens's real field of view extrapolates there. Far
///   from the observed features any value is an extrapolation of the fitted
///   polynomial.
/// - The distortion layer's shape depends on how the calibration split the
///   projection between `K` and the distortion coefficients, which the data
///   constrain only jointly (`fx` against `k1`). Two calibrations that
///   project alike can show different distortion layers: compare their
///   projections, not their layers.
/// - A general [`SensorParams::Homography`](crate::SensorParams::Homography)
///   can carry a scale or shift that duplicates `K`; its sensor layer then
///   depends on that split too. A Scheimpflug tilt maps the optical axis to
///   itself, so its layer is zero at the principal point.
///
/// # Errors
///
/// [`Error::Singular`] for a sensor homography that is not invertible (as
/// [`CameraParams::build`]).
///
/// # Example
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, ImageGrid, ProjectionParams,
///     SensorParams, distortion_map,
/// };
///
/// let camera = CameraParams::new(
///     ProjectionParams::Pinhole,
///     DistortionParams::BrownConrady5 {
///         params: BrownConrady5 { k1: -0.1, ..BrownConrady5::default() },
///     },
///     SensorParams::Identity,
///     FxFyCxCy { fx: 800.0, fy: 800.0, cx: 639.5, cy: 479.5 },
/// );
/// let grid = ImageGrid::with_columns([1280, 960], 64)?;
/// let map = distortion_map(&camera, &grid)?;
/// assert_eq!(map.valid_nodes, grid.len());
/// // Barrel distortion: the top-left corner moves inward (down and right).
/// assert!(map.total.dx[0] > 0.0 && map.total.dy[0] > 0.0);
/// # Ok::<(), vision_calibration_core::Error>(())
/// ```
pub fn distortion_map(camera: &CameraParams, grid: &ImageGrid) -> Result<DistortionMap, Error> {
    let model = camera.build()?;
    let len = grid.len();
    let mut total = DisplacementPlane::nan(len);
    let mut distortion = DisplacementPlane::nan(len);
    let mut sensor = DisplacementPlane::nan(len);
    let mut valid_nodes = 0;
    for (i, node) in grid.nodes().enumerate() {
        if let Some(layers) = node_layers(&model, node) {
            let [dist, sens] = layers;
            distortion.dx[i] = dist[0];
            distortion.dy[i] = dist[1];
            sensor.dx[i] = sens[0];
            sensor.dy[i] = sens[1];
            total.dx[i] = dist[0] + sens[0];
            total.dy[i] = dist[1] + sens[1];
            valid_nodes += 1;
        }
    }
    Ok(DistortionMap {
        grid: *grid,
        total,
        distortion,
        sensor,
        valid_nodes,
    })
}

/// The `[distortion, sensor]` displacement at the pixel `node`, or `None`
/// where the node has no value (see [`distortion_map`]).
fn node_layers(model: &CameraModel, node: [f64; 2]) -> Option<[[f64; 2]; 2]> {
    let u = Point2::new(node[0], node[1]);
    let s = model.k.pixel_to_sensor(&u);
    let n_d = match &model.sensor {
        AnySensor::Identity(_) => s,
        AnySensor::Homography(h) => {
            let q = h.h_inv * Vector3::new(s.x, s.y, 1.0);
            // NaN fails the first test, so a non-finite depth is refused too.
            let deep_enough = q.z.abs() > MIN_HOMOGRAPHY_DEPTH;
            if !deep_enough || !q.iter().all(|v| v.is_finite()) {
                return None;
            }
            Point2::new(q.x / q.z, q.y / q.z)
        }
    };
    let ray = model.proj.unproject(&model.dist.undistort(&n_d));
    let point = ray.origin + ray.direction;
    if !model.proj.in_domain(&point) {
        return None;
    }
    // Forward again from the ray: the layers are the forward model's.
    let n = model.proj.project(&point);
    let nd = model.dist.distort(&n);
    let full = model
        .k
        .sensor_to_pixel(&model.sensor.normalized_to_sensor(&nd));
    // A NaN round trip compares false and is refused.
    let round_trips = (full - u).norm() <= ROUND_TRIP_PX;
    if !round_trips {
        return None;
    }
    if !unfolded_to(&model.dist, &n) {
        return None;
    }
    let reference = model.k.sensor_to_pixel(&n);
    let lens = model.k.sensor_to_pixel(&nd);
    let layers = [
        [lens.x - reference.x, lens.y - reference.y],
        [full.x - lens.x, full.y - lens.y],
    ];
    layers
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(layers)
}

/// Whether `det ∂D/∂n > MIN_JACOBIAN_DET` at [`FOLD_SAMPLES`] evenly
/// spaced points of the segment from the axis to `n`, `n` included.
fn unfolded_to<D: DistortionModel<Real>>(dist: &D, n: &Point2<Real>) -> bool {
    (1..=FOLD_SAMPLES).all(|i| {
        let t = f64::from(i) / f64::from(FOLD_SAMPLES);
        distortion_jacobian_det(dist, &Point2::from(n.coords * t)) > MIN_JACOBIAN_DET
    })
}

/// `det ∂D/∂n` at `n` by central differences with step
/// `h = 1e-6·max(1, |n|)`: only its sign and size against
/// [`MIN_JACOBIAN_DET`] matter, so the `O(h²)` truncation is immaterial.
pub(super) fn distortion_jacobian_det<D: DistortionModel<Real>>(dist: &D, n: &Point2<Real>) -> f64 {
    let h = 1e-6 * n.coords.norm().max(1.0);
    let column = |dx: f64, dy: f64| {
        let plus = dist.distort(&Point2::new(n.x + dx, n.y + dy));
        let minus = dist.distort(&Point2::new(n.x - dx, n.y - dy));
        (plus - minus) / (2.0 * h)
    };
    let jx = column(h, 0.0);
    let jy = column(0.0, h);
    jx.x * jy.y - jy.x * jx.y
}
