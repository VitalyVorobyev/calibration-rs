//! Projection uncertainty: how precisely a camera's estimated parameters
//! place a camera-frame direction in the image.
//!
//! For the ray through pixel `q`, the pixel it projects to moves with the
//! camera parameters `θ` as `J_q = ∂q/∂θ`, so its covariance is
//! `J_q Σ_θθ J_qᵀ`, a 2×2 ellipse. The reported figure is the ellipse's
//! major-axis standard deviation in pixels, on a grid of nodes across the
//! image. `J_q` is exact: it is the forward-mode derivative of the camera
//! kernels the solver projects with.
//!
//! This is the camera-frame variant. The ray is fixed in the camera's own
//! frame and only the camera's parameters (intrinsics, distortion, sensor
//! tilt) vary, so the figure does not depend on range: a central camera
//! maps every point of a ray to one pixel. A principal-point error and a
//! small camera rotation look alike, and a downstream fit that re-estimates
//! the camera pose absorbs part of it, so the figure is conservative for
//! such uses; it is exact for a camera whose pose is held at its estimate.

use nalgebra::{DVector, Dyn, Matrix2, OMatrix, SVector, U2, Vector2, Vector3};
use num_dual::{DualDVec64, jacobian};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, ParameterCovariance, ProjectionParams, Pt2, SensorParams,
};

use crate::Error;
use crate::factors::camera_blocks::CameraBlocks;
use crate::ir::{CameraModelSpec, ProjectionKind, SensorKind};
use crate::params::distortion::pack_distortion_params;
use crate::params::intrinsics::pack_intrinsics;
use crate::uncertainty::{INTRINSICS_LABELS, SENSOR_LABELS};

/// Grid nodes across the image, `[columns, rows]`.
const GRID: [usize; 2] = [16, 12];
/// Newton steps that place a node's ray on its pixel.
const MAX_NEWTON_STEPS: usize = 20;
/// A node's ray projects within this distance of its pixel, in pixels.
const PIXEL_TOLERANCE: f64 = 1e-9;

/// The projection uncertainty of one camera over its image.
///
/// Grid nodes are evenly spaced from pixel `(0, 0)` to
/// `(width − 1, height − 1)`, corners included, and listed row by row from
/// the top-left. Each value is the major-axis standard deviation, in pixels,
/// of where the ray through the node projects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ProjectionUncertainty {
    /// `[width, height]` of the image, pixels.
    pub image_size: [u32; 2],
    /// Nodes per grid row.
    pub columns: usize,
    /// Grid rows.
    pub rows: usize,
    /// Major-axis σ at each node, pixels. `None` where no ray reaches the
    /// node, as where a strong distortion folds the image over.
    pub sigma: Vec<Option<f64>>,
    /// σ at the image centre, pixels.
    pub center: f64,
    /// The largest σ of the four corners, pixels; `None` when no ray reaches
    /// any of them.
    pub corner: Option<f64>,
    /// The largest σ of the grid and the centre, pixels.
    pub max: f64,
}

/// The projection uncertainty of `camera` over an image of `image_size`
/// pixels, from the covariance of its estimated parameters.
///
/// `camera_index` names the camera in a multi-camera covariance, whose labels
/// carry a `cam<i>.` prefix; a single-camera covariance has none, and its
/// camera is index 0. Parameters the covariance does not list were held
/// fixed and count as exact.
///
/// # Errors
///
/// [`Error::InvalidInput`] when the image is smaller than 2×2 pixels, the
/// camera model has a general sensor homography, or the covariance
/// lists none of the camera's parameters (they were all held fixed);
/// [`Error::Numerical`] when no ray reaches the image centre.
pub fn projection_uncertainty(
    camera: &CameraParams,
    covariance: &ParameterCovariance,
    camera_index: usize,
    image_size: [u32; 2],
) -> Result<ProjectionUncertainty, Error> {
    let [width, height] = image_size;
    if width < 2 || height < 2 {
        return Err(Error::invalid_input(format!(
            "image size {width}×{height} is smaller than 2×2 pixels"
        )));
    }
    let model = CameraVector::new(camera)?;
    let theta_covariance = model.covariance(covariance, camera_index)?;
    let start = camera.build()?;
    let sigma_at = |pixel: [f64; 2]| -> Option<f64> {
        let guess = start
            .backproject_pixel(&Pt2::new(pixel[0], pixel[1]))
            .point_at_unit_depth()?;
        let j = model.jacobian(
            Vector2::new(pixel[0], pixel[1]),
            Vector2::new(guess.x, guess.y),
        )?;
        Some(major_axis_sigma(&(&j * &theta_covariance * j.transpose())))
    };

    let [columns, rows] = GRID;
    let node = |i: usize, n: usize, size: u32| i as f64 * f64::from(size - 1) / (n - 1) as f64;
    let sigma: Vec<Option<f64>> = (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (c, r)))
        .map(|(c, r)| sigma_at([node(c, columns, width), node(r, rows, height)]))
        .collect();
    let center = sigma_at([f64::from(width - 1) / 2.0, f64::from(height - 1) / 2.0])
        .ok_or_else(|| Error::numerical("no ray reaches the image centre"))?;
    let corners = [0, columns - 1, (rows - 1) * columns, rows * columns - 1];
    let corner = corners.iter().filter_map(|&i| sigma[i]).reduce(f64::max);
    let max = sigma.iter().flatten().copied().fold(center, f64::max);
    Ok(ProjectionUncertainty {
        image_size,
        columns,
        rows,
        sigma,
        center,
        corner,
        max,
    })
}

/// A camera's parameters by their covariance label (`fx` … `cy`, the
/// distortion coefficients by name, `tilt_x` and `tilt_y` for a Scheimpflug
/// sensor), in the solver's order.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a camera model the solver does not estimate:
/// a general sensor homography.
pub fn camera_parameters(camera: &CameraParams) -> Result<Vec<(&'static str, f64)>, Error> {
    let model = CameraVector::new(camera)?;
    Ok(model
        .labels
        .into_iter()
        .zip(model.theta.iter().copied())
        .collect())
}

/// `sqrt` of the larger eigenvalue of a symmetric 2×2 covariance.
fn major_axis_sigma(c: &Matrix2<f64>) -> f64 {
    let mean = 0.5 * (c[(0, 0)] + c[(1, 1)]);
    let half_diff = 0.5 * (c[(0, 0)] - c[(1, 1)]);
    let off = 0.5 * (c[(0, 1)] + c[(1, 0)]);
    (mean + half_diff.hypot(off)).max(0.0).sqrt()
}

/// A camera's parameters as one vector `θ`, in the solver's block order:
/// intrinsics, distortion, sensor.
pub(crate) struct CameraVector {
    pub(crate) model: CameraModelSpec,
    pub(crate) theta: DVector<f64>,
    pub(crate) labels: Vec<&'static str>,
}

impl CameraVector {
    pub(crate) fn new(camera: &CameraParams) -> Result<Self, Error> {
        let ProjectionParams::Pinhole = camera.projection;
        let k = camera.intrinsics;
        let (sensor, tilt) = match &camera.sensor {
            SensorParams::Identity => (SensorKind::Identity, Vec::new()),
            SensorParams::Scheimpflug { params } => {
                (SensorKind::Scheimpflug, vec![params.tilt_x, params.tilt_y])
            }
            SensorParams::Homography { .. } => {
                return Err(Error::invalid_input(
                    "projection uncertainty needs a Scheimpflug or identity sensor, \
                     not a general homography",
                ));
            }
        };
        // The sensor is identity or Scheimpflug here, so the camera has a spec.
        let distortion = camera
            .spec()
            .expect("a non-homography camera has a spec")
            .distortion;
        let intrinsics = pack_intrinsics(&k);
        let dist = pack_distortion_params(&camera.distortion);
        let theta = DVector::from_iterator(
            intrinsics.len() + dist.len() + tilt.len(),
            intrinsics
                .iter()
                .chain(dist.iter())
                .chain(tilt.iter())
                .copied(),
        );
        let sensor_labels: &[&'static str] = if tilt.is_empty() { &[] } else { &SENSOR_LABELS };
        let labels = INTRINSICS_LABELS
            .iter()
            .chain(distortion.param_names())
            .chain(sensor_labels)
            .copied()
            .collect();
        Ok(Self {
            model: CameraModelSpec {
                projection: ProjectionKind::Pinhole,
                distortion,
                sensor,
            },
            theta,
            labels,
        })
    }

    /// The covariance of `θ`, zero for the parameters `covariance` does not
    /// list.
    fn covariance(
        &self,
        covariance: &ParameterCovariance,
        camera_index: usize,
    ) -> Result<OMatrix<f64, Dyn, Dyn>, Error> {
        let prefixed = format!("cam{camera_index}.");
        let prefix = if covariance.labels.iter().any(|l| l.starts_with(&prefixed)) {
            prefixed
        } else if camera_index == 0 {
            String::new()
        } else {
            return Err(Error::invalid_input(format!(
                "the covariance has no parameters of camera {camera_index}"
            )));
        };
        let index: Vec<Option<usize>> = self
            .labels
            .iter()
            .map(|label| covariance.index(&format!("{prefix}{label}")))
            .collect();
        if index.iter().all(Option::is_none) {
            return Err(Error::invalid_input(format!(
                "the covariance has no intrinsics, distortion or sensor parameters of camera \
                 {camera_index}: the solve held them fixed"
            )));
        }
        let n = self.theta.len();
        Ok(OMatrix::<f64, Dyn, Dyn>::from_fn(n, n, |i, j| {
            match (index[i], index[j]) {
                (Some(a), Some(b)) => covariance.covariance[a][b],
                _ => 0.0,
            }
        }))
    }

    /// `∂q/∂θ` at the ray through `pixel`, found by Newton steps from
    /// `guess` on the `z = 1` plane; `None` when the steps do not reach it.
    fn jacobian(&self, pixel: Vector2<f64>, guess: Vector2<f64>) -> Option<OMatrix<f64, U2, Dyn>> {
        node_jacobian(self.model, &self.theta, pixel, guess)
    }
}

fn node_jacobian(
    model: CameraModelSpec,
    theta: &DVector<f64>,
    pixel: Vector2<f64>,
    guess: Vector2<f64>,
) -> Option<OMatrix<f64, U2, Dyn>> {
    let n = theta.len();
    // Variables: θ, then the ray's point (x, y) on the z = 1 plane.
    let mut x = theta.clone().resize_vertically(n + 2, 0.0);
    x[n] = guess.x;
    x[n + 1] = guess.y;
    for _ in 0..MAX_NEWTON_STEPS {
        let (q, j) = jacobian(|v: DVector<DualDVec64>| project(model, &v, n), &x);
        let miss = pixel - q;
        if !miss.iter().all(|v| v.is_finite()) {
            return None;
        }
        if miss.norm() < PIXEL_TOLERANCE {
            return Some(j.columns(0, n).into_owned());
        }
        let step = j.fixed_columns::<2>(n).into_owned().lu().solve(&miss)?;
        x[n] += step.x;
        x[n + 1] += step.y;
    }
    None
}

/// The pixel of the ray `(v[n], v[n + 1], 1)` for camera parameters `v[..n]`
/// (`[intrinsics, projection, distortion, sensor]`, concatenated).
fn project(model: CameraModelSpec, v: &DVector<DualDVec64>, n: usize) -> SVector<DualDVec64, 2> {
    let all = v.as_slice();
    let (intrinsics, rest) = all.split_at(4);
    let (projection, rest) = rest.split_at(model.projection.dim());
    let (distortion, rest) = rest.split_at(model.distortion.dim());
    let sensor = &rest[..model.sensor.dim()];
    let camera = CameraBlocks::new(model, intrinsics, projection, distortion, sensor);
    let one = DualDVec64::from_re(1.0);
    let ray = Vector3::new(v[n].clone(), v[n + 1].clone(), one);
    let (u, w) = camera.project(&ray);
    SVector::<DualDVec64, 2>::new(u, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, DistortionParams, FxFyCxCy, ResidualKind, ResidualNoise, ScheimpflugParams,
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
                    p2: -0.0005,
                    iters: 8,
                },
            },
            sensor,
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 810.0,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    fn covariance(labels: &[&str], variances: &[f64]) -> ParameterCovariance {
        let n = labels.len();
        ParameterCovariance {
            labels: labels.iter().map(|l| l.to_string()).collect(),
            covariance: (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { variances[i] } else { 0.0 })
                        .collect()
                })
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: vec![ResidualNoise {
                kind: ResidualKind::Reprojection,
                level: vision_calibration_core::NoiseLevel::Sigma(0.1),
                count: 100,
                camera: None,
            }],
            num_residuals: 100,
            dof: 90,
        }
    }

    /// `params` with its parameters replaced by `theta` (Brown-Conrady,
    /// optional tilt).
    fn with_theta(params: &CameraParams, theta: &DVector<f64>) -> CameraParams {
        let t = |i: usize| theta[i];
        let sensor = match params.sensor {
            SensorParams::Scheimpflug { .. } => SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: t(9),
                    tilt_y: t(10),
                },
            },
            _ => SensorParams::Identity,
        };
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: t(4),
                    k2: t(5),
                    k3: t(6),
                    p1: t(7),
                    p2: t(8),
                    iters: 8,
                },
            },
            sensor,
            intrinsics: FxFyCxCy {
                fx: t(0),
                fy: t(1),
                cx: t(2),
                cy: t(3),
            },
        }
    }

    /// Central differences of the core camera model's projection of `ray`.
    fn numeric_jacobian(params: &CameraParams, ray: Vector3<f64>) -> OMatrix<f64, U2, Dyn> {
        let theta = CameraVector::new(params).unwrap().theta;
        let pixel = |theta: &DVector<f64>| {
            let q = with_theta(params, theta)
                .build()
                .unwrap()
                .project_point_c(&ray)
                .unwrap();
            Vector2::new(q.x, q.y)
        };
        let mut j = OMatrix::<f64, U2, Dyn>::zeros(theta.len());
        for k in 0..theta.len() {
            // Step 1e-6 relative (absolute below 1): central differences are
            // then accurate to about 1e-10 relative on these smooth models.
            let h = 1e-6 * theta[k].abs().max(1.0);
            let mut plus = theta.clone();
            plus[k] += h;
            let mut minus = theta.clone();
            minus[k] -= h;
            j.set_column(k, &((pixel(&plus) - pixel(&minus)) / (2.0 * h)));
        }
        j
    }

    #[test]
    fn exact_jacobian_matches_central_differences() {
        for sensor in [
            SensorParams::Identity,
            SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.03,
                    tilt_y: -0.02,
                },
            },
        ] {
            let params = camera(sensor);
            let model = CameraVector::new(&params).unwrap();
            let built = params.build().unwrap();
            let pixel = Vector2::new(600.0, 40.0);
            let guess = built
                .backproject_pixel(&Pt2::new(pixel.x, pixel.y))
                .point_at_unit_depth()
                .unwrap();
            let exact = model
                .jacobian(pixel, Vector2::new(guess.x, guess.y))
                .unwrap();
            // Recover the ray Newton converged to, then difference there.
            let ray = built
                .backproject_pixel(&Pt2::new(pixel.x, pixel.y))
                .point_at_unit_depth()
                .unwrap();
            let numeric = numeric_jacobian(&params, ray);
            let scale = exact.abs().max();
            assert!(
                (&exact - &numeric).abs().max() < 1e-4 * scale,
                "exact {exact} numeric {numeric}"
            );
        }
    }

    #[test]
    fn camera_parameters_are_labelled_like_the_covariance() {
        let params = camera(SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.03,
                tilt_y: -0.02,
            },
        });
        let values = camera_parameters(&params).unwrap();
        let labels: Vec<_> = values.iter().map(|(l, _)| *l).collect();
        assert_eq!(
            labels,
            [
                "fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "tilt_x", "tilt_y"
            ]
        );
        assert_eq!(values[0].1, 800.0);
        assert_eq!(values[10].1, -0.02);
    }

    #[test]
    fn principal_point_alone_moves_every_pixel_by_its_sigma() {
        let params = camera(SensorParams::Identity);
        let cov = covariance(&["cx", "cy"], &[0.25, 0.04]);
        let map = projection_uncertainty(&params, &cov, 0, [640, 480]).unwrap();
        assert_eq!((map.columns, map.rows), (16, 12));
        assert_eq!(map.sigma.len(), 16 * 12);
        for s in map.sigma.iter().chain([&Some(map.center)]) {
            assert!((s.unwrap() - 0.5).abs() < 1e-9, "{s:?}");
        }
        assert!((map.max - 0.5).abs() < 1e-9);
    }

    #[test]
    fn focal_uncertainty_grows_away_from_the_centre() {
        let params = camera(SensorParams::Identity);
        let cov = covariance(&["fx", "fy"], &[1.0, 1.0]);
        let map = projection_uncertainty(&params, &cov, 0, [640, 480]).unwrap();
        // A focal error moves a pixel in proportion to its sensor-plane
        // coordinate: at (0, 0) that is x = −cx/fx = −0.4 with σ(fx) = 1.
        assert!(map.center < 1e-3, "{}", map.center);
        let corner = map.corner.unwrap();
        assert!((corner - 0.4).abs() < 1e-9, "{corner}");
        assert!((map.max - corner).abs() < 1e-9, "{}", map.max);
    }

    #[test]
    fn multi_camera_labels_select_the_camera() {
        let params = camera(SensorParams::Identity);
        let cov = covariance(&["cam0.cx", "cam1.cx"], &[0.01, 1.0]);
        let one = projection_uncertainty(&params, &cov, 1, [640, 480]).unwrap();
        assert!((one.center - 1.0).abs() < 1e-9);
        let zero = projection_uncertainty(&params, &cov, 0, [640, 480]).unwrap();
        assert!((zero.center - 0.1).abs() < 1e-9);
        assert!(projection_uncertainty(&params, &cov, 2, [640, 480]).is_err());
    }

    #[test]
    fn a_camera_held_fixed_has_no_projection_uncertainty() {
        let params = camera(SensorParams::Identity);
        let cov = covariance(&["cam_se3_rig.tx"], &[1e-6]);
        let err = projection_uncertainty(&params, &cov, 0, [640, 480]).unwrap_err();
        assert!(err.to_string().contains("held them fixed"), "{err}");
        assert!(projection_uncertainty(&params, &cov, 0, [1, 480]).is_err());
    }
}
