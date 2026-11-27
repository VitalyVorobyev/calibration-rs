//! The quality report's conditioning check on tilted (Scheimpflug) cameras.
//!
//! A sensor tilt is determined only through the lens distortion it
//! decentres: with weak distortion it trades off against the principal
//! point, and free tangential or thin-prism terms absorb it to first order.
//! Each cell here builds a synthetic camera (1280×960, 20 board views over
//! the whole image, 0.2 px noise), calibrates it through the seeded
//! Scheimpflug route on both solver backends, and reads the
//! `parameter_conditioning` check: a well-determined tilt passes, each of
//! the three ways to lose it warns with its own cause.
//!
//! The default run holds the four cells the check is designed around and a
//! two-solve check of the gain it states for holding the tangential terms;
//! the wider sweep is release-mode and `#[ignore]`d:
//!
//! ```text
//! cargo test --release -p vision-calibration --test quality_tilt_conditioning -- --ignored
//! ```

#![allow(missing_docs)]

use nalgebra::{Point2, Translation3, UnitQuaternion, Vector3};
use vision_calibration::analysis::{
    CheckKind, CheckStatus, QualityCheck, QualityThresholds, assess,
};
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::core::{
    BrownConrady5, CameraParams, CorrespondenceView, DistortionKind, DistortionParams, FxFyCxCy,
    Iso3, PlanarDataset, ProjectionParams, Pt2, ScheimpflugParams, SensorKind, SensorParams,
    Uncertainty, UncertaintyUnavailable, Vec2, View,
};
use vision_calibration::optim::SolverBackend;
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init_with_seed,
    step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar};

const IMAGE: [u32; 2] = [1280, 960];
const VIEWS: usize = 20;
/// Per-axis noise σ of the corners, pixels.
const NOISE_PX: f64 = 0.2;
const BACKENDS: [SolverBackend; 2] = [SolverBackend::TinySolver, SolverBackend::Factrs];

/// Which camera parameters the solve estimates besides the default set
/// (`fx, fy, cx, cy, k1, k2, tilt_x, tilt_y`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Free {
    Default,
    /// `p1, p2` free too.
    Tangential,
    /// Thin-prism distortion with `s1..s4` free too.
    Prism,
}

/// One synthetic camera and the free set it is solved with.
#[derive(Debug, Clone, Copy)]
struct Cell {
    /// Horizontal field of view, degrees.
    hfov_deg: f64,
    /// Radial distortion at the image corner, `k1·r_c²` (`r_c` the
    /// half-diagonal over the focal length; `k1` is the only non-zero true
    /// distortion term).
    corner_distortion: f64,
    /// Sensor tilt `(tilt_x, tilt_y)`, degrees.
    tilt_deg: (f64, f64),
    free: Free,
}

impl Cell {
    /// 50° field, −5 % distortion at the image corner (`k1 = −0.147`), a 5°
    /// tilt.
    const BASELINE: Self = Self {
        hfov_deg: 50.0,
        corner_distortion: -0.05,
        tilt_deg: (5.0, 0.0),
        free: Free::Default,
    };

    fn focal(&self) -> f64 {
        0.5 * f64::from(IMAGE[0]) / (0.5 * self.hfov_deg.to_radians()).tan()
    }

    fn k1(&self) -> f64 {
        let half_diagonal = 0.5 * f64::from(IMAGE[0]).hypot(f64::from(IMAGE[1]));
        self.corner_distortion / (half_diagonal / self.focal()).powi(2)
    }

    fn truth(&self) -> CameraParams {
        let f = self.focal();
        CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: self.k1(),
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 20,
                },
            },
            SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: self.tilt_deg.0.to_radians(),
                    tilt_y: self.tilt_deg.1.to_radians(),
                },
            },
            FxFyCxCy {
                fx: f,
                fy: 1.002 * f,
                cx: 0.5 * f64::from(IMAGE[0]) + 7.0,
                cy: 0.5 * f64::from(IMAGE[1]) - 5.0,
            },
        )
    }
}

/// SplitMix64, uniform in `[-a, a)`.
struct Rng(u64);

impl Rng {
    fn sym(&mut self, a: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        a * (2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0)
    }
}

/// Twenty views of a 12 × 9 corner board spanning half the image width at
/// unit depth: centred over the whole image, depth within 1.5 : 1, tilted
/// up to 30° about each board axis and rolled up to 15°. Corners outside
/// the image are dropped; a view keeps at least half of them.
fn dataset(cell: &Cell) -> PlanarDataset {
    let truth = cell.truth();
    let camera = truth.build().expect("a valid camera");
    let (w, h) = (f64::from(IMAGE[0]), f64::from(IMAGE[1]));
    let f = cell.focal();
    let (nx, ny) = (12, 9);
    let spacing = 0.5 * w / f / (nx - 1) as f64;
    let board = planar::grid_points(nx, ny, spacing);
    let centre = Vector3::new(
        0.5 * spacing * (nx - 1) as f64,
        0.5 * spacing * (ny - 1) as f64,
        0.0,
    );
    let noise = UniformPixelNoise {
        seed: 20_261_122,
        max_abs_px: NOISE_PX * 3.0_f64.sqrt(),
    };
    let mut rng = Rng(7);
    let mut views = Vec::with_capacity(VIEWS);
    while views.len() < VIEWS {
        let u = 0.5 * w + rng.sym(0.38 * w);
        let v = 0.5 * h + rng.sym(0.38 * h);
        let z = (rng.sym(0.5 * 1.5_f64.ln())).exp();
        let ray = Vector3::new(
            (u - truth.intrinsics.cx) / f,
            (v - truth.intrinsics.cy) / f,
            1.0,
        );
        let at = ray * z;
        let face = UnitQuaternion::rotation_between(&Vector3::z(), &at.normalize())
            .unwrap_or_else(UnitQuaternion::identity);
        let tilt = 30.0_f64.to_radians();
        let rotation = face
            * UnitQuaternion::from_euler_angles(rng.sym(tilt), rng.sym(tilt), 0.0)
            * UnitQuaternion::from_euler_angles(0.0, 0.0, rng.sym(15.0_f64.to_radians()));
        let pose = Iso3::from_parts(Translation3::from(at - rotation * centre), rotation);
        let index = views.len();
        let (mut p3, mut p2) = (Vec::new(), Vec::new());
        for (k, point) in board.iter().enumerate() {
            let Some(px) = camera.project_point(&pose.transform_point(point)) else {
                continue;
            };
            let px: Point2<f64> = px;
            if (0.0..=w).contains(&px.x) && (0.0..=h).contains(&px.y) {
                let noisy = noise.apply(index, k, Vec2::new(px.x, px.y));
                p3.push(*point);
                p2.push(Pt2::new(noisy.x, noisy.y));
            }
        }
        if 2 * p2.len() >= board.len() {
            views.push(View::without_meta(
                CorrespondenceView::new(p3, p2).expect("a view keeps half the board"),
            ));
        }
    }
    PlanarDataset::new(views).expect("twenty views")
}

/// The seeded Scheimpflug route on `backend`: a focal 3 % low, the principal
/// point at the image centre and the tilt 0.5° off, with `free`'s
/// distortion terms estimated.
fn calibrate(cell: &Cell, free: Free, backend: SolverBackend) -> PlanarIntrinsicsExport {
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model.sensor = SensorKind::Scheimpflug;
    match free {
        Free::Default => {}
        Free::Tangential => config.camera.fix.distortion = Some(vec!["k3".into()]),
        Free::Prism => {
            config.camera.model.distortion = DistortionKind::ThinPrism9;
            config.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
        }
    }
    config.solver.backend = backend;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset(cell)).expect("input");
    session.set_config(config).expect("config");
    let f = 0.97 * cell.focal();
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: f,
        fy: f,
        cx: 0.5 * f64::from(IMAGE[0]),
        cy: 0.5 * f64::from(IMAGE[1]),
    });
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: (cell.tilt_deg.0 + 0.5).to_radians(),
        tilt_y: (cell.tilt_deg.1 - 0.5).to_radians(),
    });
    step_init_with_seed(&mut session, seed, None).expect("seeded init");
    step_optimize(&mut session, None).expect("optimize");
    session.export().expect("export")
}

/// The conditioning check of a calibration, and the inflation of `tilt_x`
/// its covariance gives (the check's own number for that parameter).
fn conditioning(export: &PlanarIntrinsicsExport) -> QualityCheck {
    let report = assess(export, &[IMAGE], &QualityThresholds::default()).expect("assess");
    report
        .checks
        .into_iter()
        .find(|c| c.kind == CheckKind::ParameterConditioning)
        .expect("a conditioning check")
}

/// Labels and covariance of the shared parameters.
fn covariance(export: &PlanarIntrinsicsExport) -> (&[String], &[Vec<f64>]) {
    let Uncertainty::Available(cov) = &export.uncertainty else {
        panic!("no covariance: {:?}", export.uncertainty);
    };
    (&cov.labels, &cov.covariance)
}

fn index(labels: &[String], name: &str) -> usize {
    labels
        .iter()
        .position(|l| l == name)
        .unwrap_or_else(|| panic!("no {name} in {labels:?}"))
}

/// The σ inflation of `name` by its correlations with every other shared
/// parameter, `√(S_ii (S⁻¹)_ii)`, by a direct inverse.
fn inflation(export: &PlanarIntrinsicsExport, name: &str) -> f64 {
    let (labels, s) = covariance(export);
    let n = labels.len();
    let m = nalgebra::DMatrix::from_fn(n, n, |r, c| s[r][c]);
    let i = index(labels, name);
    let p = m.try_inverse().expect("an invertible covariance");
    (s[i][i] * p[(i, i)]).sqrt()
}

/// The stated σ of `name`.
fn sigma(export: &PlanarIntrinsicsExport, name: &str) -> f64 {
    let (labels, s) = covariance(export);
    let i = index(labels, name);
    s[i][i].sqrt()
}

/// One cell on one backend: the check, and the inflation of `tilt_x`.
fn run(cell: &Cell, backend: SolverBackend) -> (QualityCheck, f64) {
    let export = calibrate(cell, cell.free, backend);
    let check = conditioning(&export);
    let tilt = inflation(&export, "tilt_x");
    println!(
        "{cell:?} {backend:?}: {:?}, worst ×{:.2}, tilt_x ×{tilt:.2}, σ(tilt_x) {:.4}° — {}",
        check.status,
        check.measurement.map_or(f64::NAN, |m| m.value),
        sigma(&export, "tilt_x").to_degrees(),
        check.message
    );
    (check, tilt)
}

#[test]
fn a_well_determined_tilt_passes() {
    for backend in BACKENDS {
        let (check, tilt) = run(&Cell::BASELINE, backend);
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.message);
        assert!((2.0..5.0).contains(&tilt), "tilt_x ×{tilt:.2}");
    }
}

#[test]
fn weak_distortion_names_the_principal_point_trade_off() {
    let cell = Cell {
        corner_distortion: -0.005,
        ..Cell::BASELINE
    };
    for backend in BACKENDS {
        let (check, tilt) = run(&cell, backend);
        assert_eq!(check.status, CheckStatus::Warn, "{}", check.message);
        assert!(tilt > 7.0, "tilt_x ×{tilt:.2}");
        assert!(check.subjects.contains(&"tilt_x".to_string()));
        assert!(
            check
                .message
                .contains("a Scheimpflug tilt is told apart from the principal point only by"),
            "{}",
            check.message
        );
    }
}

#[test]
fn free_tangential_terms_name_the_tangential_trade_off() {
    let cell = Cell {
        free: Free::Tangential,
        ..Cell::BASELINE
    };
    for backend in BACKENDS {
        let (check, tilt) = run(&cell, backend);
        assert_eq!(check.status, CheckStatus::Warn, "{}", check.message);
        assert!(tilt > 7.0, "tilt_x ×{tilt:.2}");
        assert!(
            check.message.contains(
                "free tangential terms absorb a tilt to first order (holding p1, p2 cuts \
                 σ(tilt_x)"
            ),
            "{}",
            check.message
        );
    }
}

#[test]
fn free_thin_prism_terms_name_the_prism_trade_off() {
    let cell = Cell {
        free: Free::Prism,
        ..Cell::BASELINE
    };
    for backend in BACKENDS {
        let (check, tilt) = run(&cell, backend);
        assert_eq!(check.status, CheckStatus::Warn, "{}", check.message);
        assert!(tilt > 7.0, "tilt_x ×{tilt:.2}");
        assert!(
            check.message.contains(
                "free thin-prism terms absorb a tilt to first order (holding s1, s2, s3, s4"
            ),
            "{}",
            check.message
        );
    }
}

/// The gain the check states for holding `p1, p2`, against two solves: the
/// stated σ(tilt_x) with them free over the one with them held.
#[test]
fn the_stated_gain_of_holding_the_tangential_terms_matches_two_solves() {
    for backend in BACKENDS {
        let free = calibrate(&Cell::BASELINE, Free::Tangential, backend);
        let held = calibrate(&Cell::BASELINE, Free::Default, backend);
        let ratio = sigma(&free, "tilt_x") / sigma(&held, "tilt_x");

        // σ(tilt_x) with p1, p2 held, from the free solve's covariance:
        // S_tt − S_tG S_GG⁻¹ S_Gt.
        let (labels, s) = covariance(&free);
        let (t, g) = (
            index(labels, "tilt_x"),
            [index(labels, "p1"), index(labels, "p2")],
        );
        let s_gg =
            nalgebra::Matrix2::new(s[g[0]][g[0]], s[g[0]][g[1]], s[g[1]][g[0]], s[g[1]][g[1]]);
        let s_tg = nalgebra::Vector2::new(s[t][g[0]], s[t][g[1]]);
        let conditional = s[t][t] - s_tg.dot(&(s_gg.try_inverse().expect("invertible") * s_tg));
        let gain = (s[t][t] / conditional).sqrt();

        // The gain the message states.
        let check = conditioning(&free);
        let stated: f64 = check
            .message
            .split("cuts σ(tilt_x) ")
            .nth(1)
            .and_then(|rest| rest.split('×').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no stated gain in {}", check.message));
        println!(
            "{backend:?}: σ(tilt_x) free/held {ratio:.3}, gain from the covariance {gain:.3}, \
             stated {stated}"
        );
        assert!(
            (stated - gain).abs() <= 0.5_f64.max(0.05 * gain),
            "stated {stated} vs {gain:.3}"
        );
        assert!(
            (gain / ratio - 1.0).abs() < 0.10,
            "gain {gain:.3} vs two-solve ratio {ratio:.3}"
        );
    }
}

#[test]
fn the_check_is_deterministic() {
    let a = calibrate(&Cell::BASELINE, Free::Default, SolverBackend::TinySolver);
    let b = calibrate(&Cell::BASELINE, Free::Default, SolverBackend::TinySolver);
    let thresholds = QualityThresholds::default();
    assert_eq!(
        assess(&a, &[IMAGE], &thresholds).unwrap(),
        assess(&b, &[IMAGE], &thresholds).unwrap()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The wider sweep (release mode)
// ─────────────────────────────────────────────────────────────────────────────

/// Tilts up to (20°, 20°), a 90° field with and without `p1, p2` free, and a
/// 20° field: the inflation of `tilt_x` per cell and backend, and a pass
/// where the tilt is determined.
#[test]
#[ignore = "release-mode sweep; run with --release -- --ignored"]
fn sweep_reports_the_tilt_inflation() {
    let cells = [
        (
            "tilt 20°",
            Cell {
                tilt_deg: (20.0, 0.0),
                ..Cell::BASELINE
            },
            Some(CheckStatus::Pass),
        ),
        (
            "tilt (20°, 20°)",
            Cell {
                tilt_deg: (20.0, 20.0),
                ..Cell::BASELINE
            },
            None,
        ),
        (
            "90° field, corner −0.15",
            Cell {
                hfov_deg: 90.0,
                corner_distortion: -0.15,
                ..Cell::BASELINE
            },
            None,
        ),
        (
            "90° field, corner −0.15, p free",
            Cell {
                hfov_deg: 90.0,
                corner_distortion: -0.15,
                free: Free::Tangential,
                ..Cell::BASELINE
            },
            None,
        ),
        (
            "20° field, corner −0.02",
            Cell {
                hfov_deg: 20.0,
                corner_distortion: -0.02,
                ..Cell::BASELINE
            },
            Some(CheckStatus::Pass),
        ),
    ];
    for (name, cell, expected) in cells {
        for backend in BACKENDS {
            let (check, tilt) = run(&cell, backend);
            println!(
                "SWEEP {name} {backend:?}: tilt_x ×{tilt:.2}, {:?}",
                check.status
            );
            if let Some(expected) = expected {
                assert_eq!(check.status, expected, "{name}: {}", check.message);
            }
        }
    }
}

#[test]
fn zero_distortion_leaves_scheimpflug_tilt_unobservable() {
    let mut rejected = PlanarIntrinsicsConfig::default();
    rejected.camera.model.distortion = DistortionKind::None;
    rejected.camera.model.sensor = SensorKind::Scheimpflug;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    let err = session
        .set_config(rejected)
        .expect_err("distortion-free Scheimpflug must be refused");
    assert!(
        err.to_string().contains("needs a distortion model"),
        "unexpected error: {err}"
    );

    let cell = Cell {
        corner_distortion: 0.0,
        ..Cell::BASELINE
    };
    // Without distortion nothing observes the tilt: it trades against the
    // principal point along a long, curved valley whose floor ends on the
    // principal point's bound. The solve crawls along it and does not reach
    // its minimum within the budget, so the result states no covariance
    // (and claims no determined tilt); it once stopped early in the valley
    // and reported one, with the tilt's σ inflated sevenfold.
    for backend in BACKENDS {
        let export = calibrate(&cell, cell.free, backend);
        let check = conditioning(&export);
        assert_eq!(
            check.status,
            CheckStatus::NotApplicable,
            "{}",
            check.message
        );
        assert!(
            matches!(
                &export.uncertainty,
                Uncertainty::Unavailable {
                    reason: UncertaintyUnavailable::NotConverged
                }
            ),
            "{backend:?}: {:?}",
            export.uncertainty
        );
    }
}
