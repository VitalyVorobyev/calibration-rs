//! Closed-form ellipse-center (conic-center) bias for coded ring-grid markers.
//!
//! A coded ring marker is a pair of concentric circles on the board plane.
//! Under perspective — and Scheimpflug tilt, which is a homography acting on
//! normalized coordinates — a circle projects to a conic whose **ellipse
//! center** is *not* the projection of the circle's 3D center. The offset is
//! the ellipse-center bias this module quantifies.
//!
//! # Mechanism (why the closed form is exact for the no-distortion map)
//!
//! Board plane → pixel factors, when lens distortion is set aside, into a
//! single homography
//!
//! ```text
//! H = K · H_tilt · [r1 r2 t]
//! ```
//!
//! (intrinsics `K`, Scheimpflug tilt `H_tilt`, and the planar-pose homography
//! `[r1 r2 t]` from `camera_se3_target`). A circle with conic matrix `C` on
//! the plane images to the conic `C' = H⁻ᵀ C H⁻¹`; the bias is
//! `center(C') − H·center`. Distortion is *not* a homography, so it is
//! excluded from `H` here — its contribution is treated separately in the
//! math note (`docs/notes/ringgrid-bias.md`).
//!
//! # This module is a diagnostic, not a correction
//!
//! The `ringgrid` detector (≥ 0.7) already removes the *projective* part of
//! this bias at the observation level, before calibration ever sees a center,
//! via an intrinsics-free two-conic pencil
//! (`CircleRefinementMethod::ProjectiveCenter`, on by default). The functions
//! here recover the magnitude of the effect the detector cancels and let the
//! diagnostic in [`diagnose`] check whether any of it survives into the
//! calibration residual field. See `docs/notes/ringgrid-bias.md` for the
//! evidence and the decision not to add a redundant pipeline-side correction.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use vision_calibration::core::{IntrinsicsParams, SensorParams};
use vision_calibration::scheimpflug_intrinsics::ScheimpflugIntrinsicsExport;

/// Symmetric conic matrix of a circle of radius `r` centered at `(cx, cy)` on
/// a plane. A homogeneous plane point `p = (x, y, 1)` lies on the circle iff
/// `pᵀ C p = 0`.
pub fn circle_conic(cx: f64, cy: f64, r: f64) -> Matrix3<f64> {
    Matrix3::new(
        1.0,
        0.0,
        -cx,
        0.0,
        1.0,
        -cy,
        -cx,
        -cy,
        cx * cx + cy * cy - r * r,
    )
}

/// Push a plane conic `c` through the plane→image homography `h`:
/// `C' = H⁻ᵀ C H⁻¹`. Returns `None` if `h` is singular.
pub fn project_conic(c: &Matrix3<f64>, h: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let h_inv = h.try_inverse()?;
    Some(h_inv.transpose() * c * h_inv)
}

/// Inhomogeneous center `(u, v)` of a non-degenerate central conic `C`.
///
/// The center is the stationary point of the quadratic form, i.e. the
/// solution of `[[C₀₀, C₀₁], [C₀₁, C₁₁]] · [u, v]ᵀ = −[C₀₂, C₁₂]ᵀ`. Returns
/// `None` when the top-left 2×2 block is singular (parabolic / degenerate).
pub fn conic_center(c: &Matrix3<f64>) -> Option<[f64; 2]> {
    let a = Matrix2::new(c[(0, 0)], c[(0, 1)], c[(0, 1)], c[(1, 1)]);
    let b = Vector2::new(-c[(0, 2)], -c[(1, 2)]);
    let x = a.try_inverse()? * b;
    Some([x.x, x.y])
}

/// Project the plane point `(x, y, 1)` through `h`, dehomogenizing. `None`
/// when the image point is at infinity.
pub fn project_point(h: &Matrix3<f64>, x: f64, y: f64) -> Option<[f64; 2]> {
    let p = h * Vector3::new(x, y, 1.0);
    if p.z.abs() < 1e-12 {
        return None;
    }
    Some([p.x / p.z, p.y / p.z])
}

/// Predicted single-conic ellipse-center bias, in pixels, for a circle of
/// radius `r` centered at `(cx, cy)` on the plane, imaged through the
/// no-distortion homography `h`: `center(H⁻ᵀ C H⁻¹) − H·(cx, cy)`.
pub fn predicted_center_bias(h: &Matrix3<f64>, cx: f64, cy: f64, r: f64) -> Option<[f64; 2]> {
    let conic = project_conic(&circle_conic(cx, cy, r), h)?;
    let ellipse_c = conic_center(&conic)?;
    let true_c = project_point(h, cx, cy)?;
    Some([ellipse_c[0] - true_c[0], ellipse_c[1] - true_c[1]])
}

/// Build the `3×3` intrinsics matrix `K` from serializable intrinsics.
fn k_matrix(intr: &IntrinsicsParams) -> Matrix3<f64> {
    let IntrinsicsParams::FxFyCxCySkew { params } = intr;
    Matrix3::new(
        params.fx,
        params.skew,
        params.cx,
        0.0,
        params.fy,
        params.cy,
        0.0,
        0.0,
        1.0,
    )
}

/// The Scheimpflug tilt homography (normalized → sensor), or identity for a
/// non-tilted sensor.
fn tilt_homography(sensor: &SensorParams) -> Matrix3<f64> {
    match sensor {
        SensorParams::Scheimpflug { params } => params.compile().h,
        SensorParams::Homography { h } => Matrix3::from_row_slice(&[
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        ]),
        SensorParams::Identity => Matrix3::identity(),
    }
}

/// Summary statistics for a scalar sample (all in the sample's own units).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Number of samples.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// Maximum.
    pub max: f64,
}

impl Stats {
    /// Order statistics of `values` (consumed and sorted in place).
    pub fn from_values(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let pct = |q: f64| values[((q * (n - 1) as f64).round() as usize).min(n - 1)];
        Self {
            n,
            mean,
            median: pct(0.5),
            p95: pct(0.95),
            max: values[n - 1],
        }
    }
}

/// Result of [`diagnose`]: the closed-form bias distribution and how much of
/// it survives into the calibration residual field.
#[derive(Debug, Clone)]
pub struct RinggridBiasReport {
    /// Number of (view, marker) observations analyzed.
    pub n_obs: usize,
    /// `|bias|` of the outer-ring conic center, px.
    pub bias_outer: Stats,
    /// `|bias|` of the inner-ring conic center, px.
    pub bias_inner: Stats,
    /// `|bias|` of the mean of the two single-conic centers — a proxy for a
    /// naive detector that averages the inner/outer ellipse centers, px.
    pub bias_mean_conic: Stats,
    /// `|observed − projected|`, the calibration residual actually seen, px.
    pub residual: Stats,
    /// `Σ(r·b) / Σ|b|²` with `b` the outer-conic bias vector and `r` the
    /// residual vector. ≈ 1 if the residual field *is* the naive bias
    /// (detector correction absent); ≈ 0 if the residual is orthogonal noise
    /// (detector already removed the projective bias).
    pub residual_bias_fraction: f64,
    /// Mean cosine between residual and outer-bias vectors (−1..1).
    pub mean_cosine: f64,
    /// Pearson correlation between `|outer bias|` and pixel radius from the
    /// principal point — positive if the bias grows toward the periphery.
    pub bias_vs_radius_corr: f64,
}

/// Compute the closed-form ellipse-center bias distribution for a solved
/// Scheimpflug ring-grid camera and compare it against the residual field.
///
/// `r_outer_m` / `r_inner_m` are the ring's outer / inner radii in metres
/// (board spec). Per (view, marker) the bias uses the no-distortion
/// homography `K · H_tilt · [r1 r2 t]` from the exported model and pose.
pub fn diagnose(
    export: &ScheimpflugIntrinsicsExport,
    r_outer_m: f64,
    r_inner_m: f64,
) -> RinggridBiasReport {
    let k = k_matrix(&export.params.camera.intrinsics);
    let cx0 = k[(0, 2)];
    let cy0 = k[(1, 2)];
    let h_tilt = tilt_homography(&export.params.camera.sensor);
    let poses = &export.params.camera_se3_target;

    // Cache the no-distortion homography per view.
    let h_nodistort: Vec<Matrix3<f64>> = poses
        .iter()
        .map(|iso| {
            let rot = iso.rotation.to_rotation_matrix();
            let r = rot.matrix();
            let t = iso.translation.vector;
            let h_pose = Matrix3::from_columns(&[r.column(0).into(), r.column(1).into(), t]);
            k * h_tilt * h_pose
        })
        .collect();

    let mut outer = Vec::new();
    let mut inner = Vec::new();
    let mut mean_conic = Vec::new();
    let mut resid = Vec::new();
    let mut radius = Vec::new();
    let mut sum_dot = 0.0;
    let mut sum_bb = 0.0;
    let mut cos_acc = 0.0;
    let mut cos_n = 0usize;

    for rec in &export.per_feature_residuals.target {
        let Some(projected) = rec.projected_px else {
            continue;
        };
        let h = &h_nodistort[rec.pose];
        let (cx, cy) = (rec.target_xyz_m[0], rec.target_xyz_m[1]);
        let (Some(bo), Some(bi)) = (
            predicted_center_bias(h, cx, cy, r_outer_m),
            predicted_center_bias(h, cx, cy, r_inner_m),
        ) else {
            continue;
        };

        let bo_n = (bo[0] * bo[0] + bo[1] * bo[1]).sqrt();
        let bi_n = (bi[0] * bi[0] + bi[1] * bi[1]).sqrt();
        let bm = [(bo[0] + bi[0]) * 0.5, (bo[1] + bi[1]) * 0.5];
        let bm_n = (bm[0] * bm[0] + bm[1] * bm[1]).sqrt();

        let r_vec = [
            rec.observed_px[0] - projected[0],
            rec.observed_px[1] - projected[1],
        ];
        let r_n = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1]).sqrt();

        outer.push(bo_n);
        inner.push(bi_n);
        mean_conic.push(bm_n);
        resid.push(r_n);
        radius
            .push(((rec.observed_px[0] - cx0).powi(2) + (rec.observed_px[1] - cy0).powi(2)).sqrt());

        sum_dot += r_vec[0] * bo[0] + r_vec[1] * bo[1];
        sum_bb += bo_n * bo_n;
        if r_n > 1e-9 && bo_n > 1e-9 {
            cos_acc += (r_vec[0] * bo[0] + r_vec[1] * bo[1]) / (r_n * bo_n);
            cos_n += 1;
        }
    }

    let bias_vs_radius_corr = pearson(&outer, &radius);

    RinggridBiasReport {
        n_obs: outer.len(),
        bias_outer: Stats::from_values(outer),
        bias_inner: Stats::from_values(inner),
        bias_mean_conic: Stats::from_values(mean_conic),
        residual: Stats::from_values(resid),
        residual_bias_fraction: if sum_bb > 0.0 { sum_dot / sum_bb } else { 0.0 },
        mean_cosine: if cos_n > 0 {
            cos_acc / cos_n as f64
        } else {
            0.0
        },
        bias_vs_radius_corr,
    }
}

/// Pearson correlation coefficient of two equal-length samples.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || n != b.len() {
        return 0.0;
    }
    let na = n as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / na;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, Matrix3};

    /// Fit a conic `[a, b, c, d, e, f]` (for `a u² + b u v + c v² + d u + e v +
    /// f = 0`) to sampled image points via the algebraic DLT null vector, and
    /// return it as a symmetric `3×3` conic matrix.
    fn fit_conic(points: &[[f64; 2]]) -> Matrix3<f64> {
        let mut m = DMatrix::<f64>::zeros(points.len(), 6);
        for (i, p) in points.iter().enumerate() {
            let (u, v) = (p[0], p[1]);
            m[(i, 0)] = u * u;
            m[(i, 1)] = u * v;
            m[(i, 2)] = v * v;
            m[(i, 3)] = u;
            m[(i, 4)] = v;
            m[(i, 5)] = 1.0;
        }
        let svd = m.svd(false, true);
        let v_t = svd.v_t.unwrap();
        // Smallest singular value is last (nalgebra sorts descending).
        let row = v_t.row(5);
        let (a, b, c, d, e, f) = (row[0], row[1], row[2], row[3], row[4], row[5]);
        Matrix3::new(
            a,
            b / 2.0,
            d / 2.0,
            b / 2.0,
            c,
            e / 2.0,
            d / 2.0,
            e / 2.0,
            f,
        )
    }

    fn synthetic_h() -> Matrix3<f64> {
        // A perspective+tilt-like homography (non-affine bottom row).
        Matrix3::new(
            1100.0, 15.0, 360.0, -12.0, 1080.0, 270.0, 8.0e-4, -6.0e-4, 1.0,
        )
    }

    #[test]
    fn conic_center_matches_sampled_ellipse_fit() {
        let h = synthetic_h();
        let (cx, cy, r) = (0.012_f64, -0.008, 0.0048);

        // Push the circle conic through H algebraically.
        let conic_algebraic = project_conic(&circle_conic(cx, cy, r), &h).unwrap();
        let center_algebraic = conic_center(&conic_algebraic).unwrap();

        // Independently: sample the circle, project each point, fit a conic.
        let pts: Vec<[f64; 2]> = (0..128)
            .map(|i| {
                let th = std::f64::consts::TAU * i as f64 / 128.0;
                project_point(&h, cx + r * th.cos(), cy + r * th.sin()).unwrap()
            })
            .collect();
        let center_fit = conic_center(&fit_conic(&pts)).unwrap();

        let err = ((center_algebraic[0] - center_fit[0]).powi(2)
            + (center_algebraic[1] - center_fit[1]).powi(2))
        .sqrt();
        assert!(
            err < 1e-6,
            "algebraic conic center vs sampled-ellipse-fit center disagree: {err:.3e} px"
        );
    }

    #[test]
    fn bias_is_nonzero_and_equals_center_minus_projection() {
        let h = synthetic_h();
        // Plane units chosen so the imaged marker has a realistic pixel extent
        // under a strongly projective (non-affine) homography — the regime
        // where the ellipse-center bias is clearly above numerical noise.
        let (cx, cy, r) = (0.4_f64, -0.3, 0.25);

        let bias = predicted_center_bias(&h, cx, cy, r).unwrap();
        let bias_norm = (bias[0] * bias[0] + bias[1] * bias[1]).sqrt();
        // Under a genuinely projective (non-affine) homography the conic
        // center must differ from the projected circle center.
        assert!(
            bias_norm > 1e-2,
            "expected a measurable ellipse-center bias, got {bias_norm:.3e} px"
        );

        // Cross-check the definition: bias = center(C') − H·c.
        let conic = project_conic(&circle_conic(cx, cy, r), &h).unwrap();
        let center = conic_center(&conic).unwrap();
        let proj = project_point(&h, cx, cy).unwrap();
        assert!((bias[0] - (center[0] - proj[0])).abs() < 1e-9);
        assert!((bias[1] - (center[1] - proj[1])).abs() < 1e-9);
    }

    /// End-to-end diagnosis on the private ring-grid cameras (Q3-RINGGRID-BIAS).
    ///
    /// `#[ignore]`d: needs `--features tier-b` and the private dataset. Run
    /// with `cargo test -p vision-calibration-bench --features tier-b
    /// --lib ringgrid_bias::tests::diagnose_private_ringgrid -- --ignored
    /// --nocapture`. Prints the closed-form bias distribution alongside the
    /// calibration residual field per camera. The recorded numbers live in
    /// `docs/notes/ringgrid-bias.md`.
    #[cfg(feature = "tier-b")]
    #[test]
    #[ignore = "needs privatedata/rtv3d_ringgrid + tier-b"]
    fn diagnose_private_ringgrid() {
        use crate::registry::load_registry;
        use crate::run::{detect_scheimpflug_seeded_input, solve_scheimpflug_seeded};
        use std::path::PathBuf;

        // Board spec (privatedata/rtv3d_ringgrid/board_ringgrid.json).
        const R_OUTER_M: f64 = 0.0048;
        const R_INNER_M: f64 = 0.0032;

        // Registry `data_root`s are workspace-root-relative; `cargo test` runs
        // with CWD = crate dir, so hop up to the workspace root first.
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        std::env::set_current_dir(&workspace_root).expect("chdir to workspace root");

        let registry = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("registry")
            .join("private.json");
        let reg = load_registry(&registry).expect("load private registry");

        println!(
            "\n{:<20} {:>6}  {:>26}  {:>26}  {:>10}  {:>8}  {:>7}",
            "camera",
            "n_obs",
            "outer-bias px (mean/med/p95/max)",
            "residual px (mean/med/p95/max)",
            "resid/bias",
            "cos",
            "r(rad)"
        );
        for id in [
            "rtv3d_ringgrid_cam0",
            "rtv3d_ringgrid_cam1",
            "rtv3d_ringgrid_cam2",
            "rtv3d_ringgrid_cam3",
            "rtv3d_ringgrid_cam4",
            "rtv3d_ringgrid_cam5",
        ] {
            let entry = reg.find(id).expect("entry present");
            let detected = detect_scheimpflug_seeded_input(entry).expect("detect");
            let solve =
                solve_scheimpflug_seeded(detected.dataset, detected.seed, id).expect("solve");
            let rep = diagnose(&solve.export, R_OUTER_M, R_INNER_M);
            println!(
                "{:<20} {:>6}  {:>6.3}/{:>6.3}/{:>6.3}/{:>6.3}  {:>6.3}/{:>6.3}/{:>6.3}/{:>6.3}  {:>10.3}  {:>8.3}  {:>7.3}",
                id,
                rep.n_obs,
                rep.bias_outer.mean,
                rep.bias_outer.median,
                rep.bias_outer.p95,
                rep.bias_outer.max,
                rep.residual.mean,
                rep.residual.median,
                rep.residual.p95,
                rep.residual.max,
                rep.residual_bias_fraction,
                rep.mean_cosine,
                rep.bias_vs_radius_corr,
            );
        }
    }

    #[test]
    fn affine_homography_has_no_bias() {
        // An affine map (bottom row [0,0,1]) sends a circle to an ellipse
        // whose center *is* the mapped circle center — zero bias.
        let h = Matrix3::new(900.0, 40.0, 320.0, -30.0, 950.0, 240.0, 0.0, 0.0, 1.0);
        let bias = predicted_center_bias(&h, 0.01, -0.02, 0.0048).unwrap();
        let bias_norm = (bias[0] * bias[0] + bias[1] * bias[1]).sqrt();
        assert!(
            bias_norm < 1e-9,
            "affine map must not bias the center: {bias_norm:.3e}"
        );
    }
}
