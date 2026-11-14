//! Synthetic cameras and view designs for the study.
//!
//! A camera is specified physically: image size, horizontal field of view,
//! the radial distortion at the image corner and the sensor tilt. The
//! optical axis (the library's principal point, the distortion centre)
//! meets the sensor near its centre, as on a Scheimpflug mount whose lens is
//! centred over a sensor tilted about its centre.
//!
//! A view design places a planar grid board by image location, depth, board
//! tilt and in-plane roll; the coverage region also crops the detections, so
//! "central" and "one-sided" mean the corners are only seen there.

use anyhow::{Result, bail};
use nalgebra::{Point2, Point3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3, PlanarDataset,
    ProjectionParams, ScheimpflugParams, SensorParams, ThinPrism, View,
};

use crate::model::Cam;

/// Deterministic generator (SplitMix64) with uniform and Gaussian draws.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[a, b)`.
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
    /// Uniform in `[-a, a)`.
    pub fn sym(&mut self, a: f64) -> f64 {
        self.range(-a, a)
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A camera, specified physically.
#[derive(Clone, Copy, Debug)]
pub struct CamSpec {
    /// Image width and height in pixels.
    pub size: (f64, f64),
    /// Horizontal field of view of the equivalent untilted camera, degrees.
    pub hfov_deg: f64,
    /// Radial distortion at the image corner, `k1·r_c²` (negative: barrel);
    /// `r_c` is the half-diagonal over the focal length.
    pub corner_distortion: f64,
    /// Sensor tilt `(tilt_x, tilt_y)`, degrees.
    pub tilt_deg: (f64, f64),
    /// True tangential terms `(p1, p2)` (zero unless a cell says otherwise).
    pub p: (f64, f64),
}

impl CamSpec {
    pub fn baseline() -> Self {
        Self {
            size: (1280.0, 960.0),
            hfov_deg: 50.0,
            corner_distortion: -0.05,
            tilt_deg: (5.0, 0.0),
            p: (0.0, 0.0),
        }
    }

    pub fn focal(&self) -> f64 {
        0.5 * self.size.0 / (0.5 * self.hfov_deg.to_radians()).tan()
    }

    /// Normalized corner radius `r_c`.
    pub fn corner_radius(&self) -> f64 {
        0.5 * (self.size.0.powi(2) + self.size.1.powi(2)).sqrt() / self.focal()
    }

    pub fn k1(&self) -> f64 {
        self.corner_distortion / self.corner_radius().powi(2)
    }

    /// The true camera in the library's parameters. `fy` is 0.2 % above
    /// `fx` and the principal point sits a few pixels off the image centre,
    /// so nothing is exactly symmetric.
    pub fn camera(&self) -> Cam {
        let f = self.focal();
        let mut d = [0.0; 9];
        d[0] = self.k1();
        d[3] = self.p.0;
        d[4] = self.p.1;
        Cam {
            k: [
                f,
                1.002 * f,
                0.5 * self.size.0 + 7.0,
                0.5 * self.size.1 - 5.0,
            ],
            d,
            tilt: [self.tilt_deg.0.to_radians(), self.tilt_deg.1.to_radians()],
        }
    }
}

/// Where the corners are detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// The whole image.
    Full,
    /// The central rectangle of half the image area.
    Central,
    /// The left half of the image.
    OneSided,
}

impl Coverage {
    pub fn name(self) -> &'static str {
        match self {
            Coverage::Full => "full",
            Coverage::Central => "central 50 %",
            Coverage::OneSided => "left half",
        }
    }

    /// `(u0, v0, u1, v1)` of the region.
    fn region(self, w: f64, h: f64) -> (f64, f64, f64, f64) {
        match self {
            Coverage::Full => (0.0, 0.0, w, h),
            Coverage::Central => {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                let (dw, dh) = (0.5 * w * (1.0 - s), 0.5 * h * (1.0 - s));
                (dw, dh, w - dw, h - dh)
            }
            Coverage::OneSided => (0.0, 0.0, 0.5 * w, h),
        }
    }
}

/// How the board is presented.
#[derive(Clone, Copy, Debug)]
pub struct ViewDesign {
    pub views: usize,
    /// Board tilt about its two in-plane axes, uniform in `±` this, degrees.
    pub board_tilt_deg: f64,
    /// Farthest over nearest board distance (1 = one depth).
    pub depth_ratio: f64,
    pub coverage: Coverage,
    /// In-plane board rotation, uniform in `±` this, degrees.
    pub roll_deg: f64,
    /// Board corners `(nx, ny)`.
    pub grid: (usize, usize),
    pub seed: u64,
}

impl ViewDesign {
    pub fn baseline() -> Self {
        Self {
            views: 20,
            board_tilt_deg: 30.0,
            depth_ratio: 1.5,
            coverage: Coverage::Full,
            roll_deg: 15.0,
            grid: (12, 9),
            seed: 7,
        }
    }
}

/// One view: board points (target frame) and their noise-free pixels.
#[derive(Clone, Debug)]
pub struct ViewObs {
    pub board: Vec<[f64; 3]>,
    pub pixels: Vec<[f64; 2]>,
}

/// A camera, its true board poses and the noise-free observations.
#[derive(Clone, Debug)]
pub struct Scene {
    pub cam: Cam,
    pub poses: Vec<Iso3>,
    pub views: Vec<ViewObs>,
}

impl Scene {
    pub fn num_points(&self) -> usize {
        self.views.iter().map(|v| v.pixels.len()).sum()
    }

    /// The observations with Gaussian pixel noise of `sigma` per coordinate.
    pub fn dataset(&self, sigma: f64, seed: u64) -> PlanarDataset {
        self.dataset_with_view_error(sigma, 0.0, seed)
    }

    /// The observations with Gaussian pixel noise of `sigma` per coordinate
    /// plus, per view, an error shared by its corners: a quadratic field over
    /// the board (`x̃² − ⅓`, `x̃ỹ`, `ỹ² − ⅓` per pixel axis, board coordinates
    /// scaled to `[−1, 1]`) with Gaussian coefficients of `view_error` px (a
    /// board that is not flat, a detector bias that changes with the view).
    /// The independent noise draws are those of [`Scene::dataset`] whatever
    /// `view_error` is.
    pub fn dataset_with_view_error(&self, sigma: f64, view_error: f64, seed: u64) -> PlanarDataset {
        let mut rng = Rng::new(seed);
        let mut field_rng = Rng::new(seed ^ 0x5EED_F1E1D);
        let views = self
            .views
            .iter()
            .map(|v| {
                let p3 = v
                    .board
                    .iter()
                    .map(|b| Point3::new(b[0], b[1], b[2]))
                    .collect();
                let coeffs: [f64; 6] = std::array::from_fn(|_| view_error * field_rng.normal());
                let (lo, hi) = v.board.iter().fold(
                    ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
                    |(lo, hi), b| {
                        (
                            [lo[0].min(b[0]), lo[1].min(b[1])],
                            [hi[0].max(b[0]), hi[1].max(b[1])],
                        )
                    },
                );
                let scaled = |x: f64, a: f64, b: f64| {
                    if b > a {
                        2.0 * (x - a) / (b - a) - 1.0
                    } else {
                        0.0
                    }
                };
                let p2 = v
                    .board
                    .iter()
                    .zip(&v.pixels)
                    .map(|(b, p)| {
                        let (x, y) = (scaled(b[0], lo[0], hi[0]), scaled(b[1], lo[1], hi[1]));
                        let basis = [x * x - 1.0 / 3.0, x * y, y * y - 1.0 / 3.0];
                        let du: f64 = (0..3).map(|k| coeffs[k] * basis[k]).sum();
                        let dv: f64 = (0..3).map(|k| coeffs[3 + k] * basis[k]).sum();
                        Point2::new(
                            p[0] + du + sigma * rng.normal(),
                            p[1] + dv + sigma * rng.normal(),
                        )
                    })
                    .collect();
                View::without_meta(
                    CorrespondenceView::new(p3, p2).expect("a view keeps at least 12 points"),
                )
            })
            .collect();
        PlanarDataset::new(views).expect("a scene has views")
    }
}

/// The library camera (thin-prism distortion, a superset of Brown-Conrady
/// with the same coefficients) for a [`Cam`].
pub fn camera_params(cam: &Cam) -> CameraParams {
    let d = cam.d;
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::ThinPrism {
            params: ThinPrism {
                k1: d[0],
                k2: d[1],
                k3: d[2],
                p1: d[3],
                p2: d[4],
                s1: d[5],
                s2: d[6],
                s3: d[7],
                s4: d[8],
                iters: 20,
            },
        },
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: cam.tilt[0],
                tilt_y: cam.tilt[1],
            },
        },
        FxFyCxCy {
            fx: cam.k[0],
            fy: cam.k[1],
            cx: cam.k[2],
            cy: cam.k[3],
        },
    )
}

/// Build the scene: board poses drawn per `design`, corners projected
/// through the true camera and kept inside the coverage region.
pub fn build_scene(spec: CamSpec, design: &ViewDesign) -> Result<Scene> {
    let cam = spec.camera();
    let model = camera_params(&cam).build()?;
    let (w, h) = spec.size;
    let (u0, v0, u1, v1) = design.coverage.region(w, h);
    let f = cam.k[0];
    let z0 = 1.0;
    let (nx, ny) = design.grid;
    // The board spans half the coverage region's width at the reference
    // depth, whatever the field of view.
    let spacing = 0.5 * (u1 - u0) / f * z0 / (nx - 1) as f64;
    let local_centre = Vector3::new(
        0.5 * spacing * (nx - 1) as f64,
        0.5 * spacing * (ny - 1) as f64,
        0.0,
    );
    let board: Vec<[f64; 3]> = (0..ny)
        .flat_map(|j| (0..nx).map(move |i| [i as f64 * spacing, j as f64 * spacing, 0.0]))
        .collect();
    let mut rng = Rng::new(design.seed);
    let half_log = 0.5 * design.depth_ratio.ln();
    let (mut poses, mut views) = (Vec::new(), Vec::new());
    let min_points = (board.len() as f64 * 0.5).ceil() as usize;
    for _ in 0..design.views {
        let mut accepted = None;
        for attempt in 0..200 {
            let mu = 0.12;
            let u = rng.range(u0 + mu * (u1 - u0), u1 - mu * (u1 - u0));
            let v = rng.range(v0 + mu * (v1 - v0), v1 - mu * (v1 - v0));
            let z = z0 * rng.sym(half_log).exp();
            let ray = model.backproject_pixel(&Point2::new(u, v)).direction;
            if ray.z <= 0.0 {
                continue;
            }
            let centre = ray / ray.z * z;
            // Face the viewing ray, then tilt about the board's axes and roll.
            let face = UnitQuaternion::rotation_between(&Vector3::z(), &centre.normalize())
                .unwrap_or_else(UnitQuaternion::identity);
            let tilt = design.board_tilt_deg.to_radians();
            let rot = face
                * UnitQuaternion::from_euler_angles(rng.sym(tilt), rng.sym(tilt), 0.0)
                * UnitQuaternion::from_euler_angles(
                    0.0,
                    0.0,
                    rng.sym(design.roll_deg.to_radians()),
                );
            let t = centre - rot * local_centre;
            let pose = Iso3::from_parts(Translation3::from(t), rot);
            let mut obs = ViewObs {
                board: Vec::new(),
                pixels: Vec::new(),
            };
            for b in &board {
                let pc = pose.transform_point(&Point3::new(b[0], b[1], b[2]));
                if pc.z <= 0.0 {
                    continue;
                }
                let Some(px) = model.project_point(&pc) else {
                    continue;
                };
                if px.x >= u0 && px.x <= u1 && px.y >= v0 && px.y <= v1 {
                    obs.board.push(*b);
                    obs.pixels.push([px.x, px.y]);
                }
            }
            let need = if attempt < 150 { min_points } else { 12 };
            if obs.pixels.len() >= need {
                accepted = Some((pose, obs));
                break;
            }
        }
        let Some((pose, obs)) = accepted else {
            bail!("could not place a board view for {design:?}");
        };
        poses.push(pose);
        views.push(obs);
    }
    Ok(Scene { cam, poses, views })
}
