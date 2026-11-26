//! Tests of the residual map on synthetic records: recovery of an injected
//! field and the estimator choice, the position and sign rules, outliers,
//! subsets, support, determinism, and the Monte-Carlo calibration of the
//! view-clustered standard error.

use vision_calibration_core::{ImageGrid, TargetFeatureResidual};

use super::*;

// ─── Synthetic records ──────────────────────────────────────────────────────

/// SplitMix64: a deterministic generator independent of any dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal pair (Box-Muller).
    fn normal2(&mut self) -> [f64; 2] {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        let r = (-2.0 * u.ln()).sqrt();
        let t = std::f64::consts::TAU * v;
        [r * t.cos(), r * t.sin()]
    }
}

/// A synthetic capture: `per_view` projections uniform over the central
/// `coverage` fraction of each axis, each residual the field at the
/// projection, plus a per-view common offset (`view_sigma` per axis, a
/// stand-in for pose error) and iid noise (`noise` per axis).
struct Capture {
    size: [u32; 2],
    views: usize,
    per_view: usize,
    coverage: f64,
    noise: f64,
    view_sigma: f64,
}

impl Capture {
    fn new(size: [u32; 2], views: usize, per_view: usize) -> Self {
        Self {
            size,
            views,
            per_view,
            coverage: 1.0,
            noise: 0.1,
            view_sigma: 0.0,
        }
    }

    fn records(
        &self,
        seed: u64,
        field: impl Fn([f64; 2]) -> [f64; 2],
    ) -> Vec<TargetFeatureResidual> {
        let mut rng = Rng(seed);
        let [w, h] = [f64::from(self.size[0]), f64::from(self.size[1])];
        let margin = (1.0 - self.coverage) / 2.0;
        let mut out = Vec::with_capacity(self.views * self.per_view);
        for view in 0..self.views {
            let o = rng.normal2();
            let offset = [o[0] * self.view_sigma, o[1] * self.view_sigma];
            for feature in 0..self.per_view {
                let p = [
                    -0.5 + w * (margin + self.coverage * rng.uniform()),
                    -0.5 + h * (margin + self.coverage * rng.uniform()),
                ];
                let n = rng.normal2();
                let f = field(p);
                let e = [
                    f[0] + offset[0] + self.noise * n[0],
                    f[1] + offset[1] + self.noise * n[1],
                ];
                out.push(record(view, feature, p, e));
            }
        }
        out
    }
}

fn record(view: usize, feature: usize, p: [f64; 2], e: [f64; 2]) -> TargetFeatureResidual {
    let mut r = TargetFeatureResidual::default();
    r.view = view;
    r.feature = feature;
    r.projected_px = Some(p);
    r.observed_px = [p[0] + e[0], p[1] + e[1]];
    r.error_px = Some(e[0].hypot(e[1]));
    r
}

fn zero(_: [f64; 2]) -> [f64; 2] {
    [0.0, 0.0]
}

fn options(bandwidth: Bandwidth) -> ResidualMapOptions {
    let mut o = ResidualMapOptions::new();
    o.bandwidth = bandwidth;
    o
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The Nadaraya-Watson (kernel-average) estimate at `x0` with the same
/// truncated Gaussian kernel: the alternative the local-linear fit is
/// compared against.
fn nadaraya_watson(records: &[TargetFeatureResidual], x0: [f64; 2], h: f64) -> [f64; 2] {
    let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for r in records {
        let (Some(p), Some(e)) = (r.projected_px, r.residual_px()) else {
            continue;
        };
        let d2 = (p[0] - x0[0]).powi(2) + (p[1] - x0[1]).powi(2);
        if d2 > (4.0 * h).powi(2) {
            continue;
        }
        let w = (-d2 / (2.0 * h * h)).exp();
        sw += w;
        sx += w * e[0];
        sy += w * e[1];
    }
    [sx / sw, sy / sw]
}

/// The corner nodes of a grid, as plane indices.
fn corners(grid: &ImageGrid) -> [usize; 4] {
    let (c, r) = (grid.columns(), grid.rows());
    [0, c - 1, (r - 1) * c, r * c - 1]
}

// ─── B-T2: recovery and the estimator choice ────────────────────────────────

/// A cubic radial field `μ = a·r²·(x − c)` (0.3 px at r = 1000 px) plus a
/// linear rotation `b·(−y, x)` (0.1 px at r = 1000 px).
fn cubic_field(centre: [f64; 2]) -> impl Fn([f64; 2]) -> [f64; 2] {
    let (a, b) = (3e-10, 1e-4);
    move |p| {
        let (x, y) = (p[0] - centre[0], p[1] - centre[1]);
        let r2 = x * x + y * y;
        [a * r2 * x - b * y, a * r2 * y + b * x]
    }
}

/// The local-linear map recovers an injected field within its
/// smoothing bias `½h²|∇²μ|` plus three standard errors at every node with
/// a value; at the image corners, on noise-free records, its error is under
/// 0.3× the kernel average's (whose edge bias is O(h·|∇μ|)).
#[test]
fn residual_map_recovers_an_injected_field() {
    let size = [1600, 1200];
    let centre = [799.5, 599.5];
    let field = cubic_field(centre);
    let h = 100.0;
    let grid = ImageGrid::with_columns(size, 64).unwrap();
    let mut capture = Capture::new(size, 20, 600);
    let records = capture.records(7, &field);
    let map = residual_map(&records, 0, &grid, &options(Bandwidth::Pixels(h))).unwrap();
    assert_eq!(map.bandwidth_selection, BandwidthSelection::Given);
    let mut worst: f64 = 0.0;
    let mut valid = 0;
    for (i, x0) in grid.nodes().enumerate() {
        if map.mean_dx[i].is_nan() {
            continue;
        }
        valid += 1;
        let truth = field(x0);
        let err = (map.mean_dx[i] - truth[0]).hypot(map.mean_dy[i] - truth[1]);
        let r = (x0[0] - centre[0]).hypot(x0[1] - centre[1]);
        // |∇²μ| of the cubic term: 8·a·r per component pair.
        let bias = 0.5 * h * h * 8.0 * 3e-10 * r;
        let bound = bias + 3.0 * map.std_error[i];
        worst = worst.max(err / bound);
        assert!(
            err <= bound,
            "node {i} at {x0:?}: error {err:.4} px > bias {bias:.4} + 3 SE {:.4}",
            map.std_error[i]
        );
    }
    assert!(valid * 10 >= grid.len() * 9, "{valid} of {}", grid.len());
    eprintln!("recovery: worst error / (bias + 3 SE) = {worst:.3}");

    // The estimator choice, on noise-free records: pure smoothing bias.
    capture.noise = 0.0;
    let clean = capture.records(7, &field);
    let map = residual_map(&clean, 0, &grid, &options(Bandwidth::Pixels(h))).unwrap();
    let nodes: Vec<[f64; 2]> = grid.nodes().collect();
    for i in corners(&grid) {
        let truth = field(nodes[i]);
        let ll = (map.mean_dx[i] - truth[0]).hypot(map.mean_dy[i] - truth[1]);
        let nw = nadaraya_watson(&clean, nodes[i], h);
        let nw = (nw[0] - truth[0]).hypot(nw[1] - truth[1]);
        eprintln!(
            "corner {:?}: |μ| {:.4} px, local-linear error {ll:.4} px, kernel-average error \
             {nw:.4} px, ratio {:.3}",
            nodes[i],
            truth[0].hypot(truth[1]),
            ll / nw
        );
        assert!(ll < 0.3 * nw, "corner {i}: {ll} vs {nw}");
    }
}

// ─── B-T4, B-T5: position and sign ──────────────────────────────────────────

/// Records sit at their projection. Projections in the left half
/// carry a 300 px residual (their observations lie in the right half);
/// the right half's records carry none. The right half of the map stays
/// zero, which binning by the observation would not give.
#[test]
fn residual_map_places_records_at_their_projection() {
    let size = [1280, 960];
    let capture = Capture {
        noise: 0.0,
        ..Capture::new(size, 10, 2000)
    };
    let records = capture.records(3, |p| [if p[0] < 640.0 { 300.0 } else { 0.0 }, 0.0]);
    let grid = ImageGrid::with_columns(size, 64).unwrap();
    let mut opts = options(Bandwidth::Pixels(30.0));
    opts.outlier_factor = 1e9;
    let map = residual_map(&records, 0, &grid, &opts).unwrap();
    assert!(map.outliers.is_empty());
    let (mut left, mut right) = (0, 0);
    for (i, [x, _]) in grid.nodes().enumerate() {
        if map.mean_dx[i].is_nan() {
            continue;
        }
        if x < 640.0 - 130.0 {
            assert!((map.mean_dx[i] - 300.0).abs() < 1e-9, "node {i}");
            left += 1;
        } else if x > 640.0 + 130.0 {
            assert_eq!(map.mean_dx[i], 0.0, "node {i}");
            right += 1;
        }
    }
    assert!(left > 100 && right > 100);
}

/// `observed = projected + (0.5, 0)` everywhere maps to
/// `mean = (0.5, 0)`: the sign is `observed − projected`.
#[test]
fn residual_map_sign_is_observed_minus_projected() {
    let size = [1280, 960];
    let capture = Capture {
        noise: 0.0,
        ..Capture::new(size, 8, 500)
    };
    let records = capture.records(5, |_| [0.5, 0.0]);
    let grid = ImageGrid::with_columns(size, 32).unwrap();
    let map = residual_map(&records, 0, &grid, &options(Bandwidth::Auto)).unwrap();
    let mut valid = 0;
    for i in 0..grid.len() {
        if map.mean_dx[i].is_nan() {
            continue;
        }
        valid += 1;
        assert!((map.mean_dx[i] - 0.5).abs() < 1e-9, "node {i}");
        assert!(map.mean_dy[i].abs() < 1e-9, "node {i}");
        assert!((map.rms[i] - 0.5).abs() < 1e-9, "node {i}");
        // No scatter: zero standard error (or none, with few views), no
        // significance.
        let se = map.std_error[i];
        assert!(se.is_nan() || se.abs() < 1e-9, "{se}");
        assert!(map.significance[i].is_nan());
    }
    assert!(valid > grid.len() / 2);
}

// ─── B-T6: outliers ─────────────────────────────────────────────────────────

/// 1 % of the records carry a 30 px gross error. They are left out
/// and listed, and the map equals the clean records' map within one
/// standard error.
#[test]
fn residual_map_leaves_gross_errors_out() {
    let size = [1280, 960];
    let centre = [639.5, 479.5];
    let grid = ImageGrid::with_columns(size, 32).unwrap();
    let capture = Capture::new(size, 20, 300);
    let clean = capture.records(11, cubic_field(centre));
    let mut corrupted = clean.clone();
    let mut rng = Rng(99);
    let mut gross = Vec::new();
    for (i, r) in corrupted.iter_mut().enumerate() {
        if rng.uniform() < 0.01 {
            let t = std::f64::consts::TAU * rng.uniform();
            r.observed_px[0] += 30.0 * t.cos();
            r.observed_px[1] += 30.0 * t.sin();
            gross.push(i);
        }
    }
    assert!(gross.len() > 30);
    let opts = options(Bandwidth::Pixels(80.0));
    let reference = residual_map(&clean, 0, &grid, &opts).unwrap();
    let map = residual_map(&corrupted, 0, &grid, &opts).unwrap();
    assert!(reference.outliers.is_empty());
    assert_eq!(map.outliers, gross);
    assert_eq!(map.used, clean.len() - gross.len());
    let mut compared = 0;
    for i in 0..grid.len() {
        if map.mean_dx[i].is_nan() || reference.mean_dx[i].is_nan() {
            continue;
        }
        let d =
            (map.mean_dx[i] - reference.mean_dx[i]).hypot(map.mean_dy[i] - reference.mean_dy[i]);
        assert!(
            d <= reference.std_error[i],
            "node {i}: {d} > {}",
            reference.std_error[i]
        );
        compared += 1;
    }
    assert!(compared > grid.len() / 2);
}

// ─── B-T7: a subset of views ────────────────────────────────────────────────

/// One view: the mean and RMS are the local-linear fit of that view's
/// records alone, and nothing is significant (one view is one cluster).
#[test]
fn residual_map_of_one_view_has_no_significance() {
    let size = [1280, 960];
    let capture = Capture {
        view_sigma: 0.05,
        ..Capture::new(size, 10, 800)
    };
    let records = capture.records(21, cubic_field([639.5, 479.5]));
    let grid = ImageGrid::with_columns(size, 32).unwrap();
    let mut opts = options(Bandwidth::Pixels(90.0));
    opts.views = Some(vec![3]);
    let map = residual_map(&records, 0, &grid, &opts).unwrap();
    assert_eq!((map.views, map.used), (1, 800));
    assert!(map.significance.iter().all(|v| v.is_nan()));
    assert!(map.std_error.iter().all(|v| v.is_nan()));
    let alone: Vec<TargetFeatureResidual> =
        records.iter().filter(|r| r.view == 3).cloned().collect();
    let mut opts_alone = options(Bandwidth::Pixels(90.0));
    opts_alone.outlier_factor = opts.outlier_factor;
    let expected = residual_map(&alone, 0, &grid, &opts_alone).unwrap();
    assert_eq!(bits(&map.mean_dx), bits(&expected.mean_dx));
    assert_eq!(bits(&map.mean_dy), bits(&expected.mean_dy));
    assert_eq!(bits(&map.rms), bits(&expected.rms));
    assert!(map.mean_dx.iter().filter(|v| v.is_finite()).count() > grid.len() / 2);
    assert!(map.effective_views.iter().all(|&g| g <= 1.0 + 1e-12));
}

// ─── B-T8: support ──────────────────────────────────────────────────────────

/// A hole in the coverage is NaN inside, with its effective samples
/// below the floor (zero at its centre).
#[test]
fn residual_map_is_nan_in_a_coverage_hole() {
    let size = [1280, 960];
    let centre = [639.5, 479.5];
    let capture = Capture::new(size, 10, 1500);
    let records: Vec<TargetFeatureResidual> = capture
        .records(4, zero)
        .into_iter()
        .filter(|r| {
            let p = r.projected_px.unwrap_or_default();
            (p[0] - centre[0]).hypot(p[1] - centre[1]) > 250.0
        })
        .collect();
    let grid = ImageGrid::new(size, 20).unwrap();
    let opts = options(Bandwidth::Pixels(30.0));
    let map = residual_map(&records, 0, &grid, &opts).unwrap();
    let mut inside = 0;
    for (i, x) in grid.nodes().enumerate() {
        let r = (x[0] - centre[0]).hypot(x[1] - centre[1]);
        if r < 250.0 - 60.0 {
            assert!(map.mean_dx[i].is_nan() && map.rms[i].is_nan(), "node {i}");
            assert!(
                map.effective_samples[i] < opts.min_effective_samples,
                "node {i}"
            );
            inside += 1;
        }
        if r < 250.0 - 4.0 * 30.0 {
            assert_eq!(map.effective_samples[i], 0.0);
            assert_eq!(map.effective_views[i], 0.0);
        } else if r > 250.0 + 60.0 {
            assert!(map.mean_dx[i].is_finite(), "node {i} outside the hole");
        }
    }
    assert!(inside > 50);
}

// ─── B-T9: determinism, validation, selection ───────────────────────────────

/// The same records give bit-identical planes, cross-validation
/// included; other cameras' records are ignored and unprojected ones
/// counted.
#[test]
fn residual_map_is_deterministic() {
    let size = [1280, 960];
    let capture = Capture {
        view_sigma: 0.03,
        ..Capture::new(size, 12, 300)
    };
    let mut records = capture.records(8, cubic_field([639.5, 479.5]));
    // Another camera's records and an unprojected one.
    let mut other = capture.records(9, |_| [5.0, 5.0]);
    for r in &mut other {
        r.camera = 1;
    }
    records.extend(other);
    let mut failed = TargetFeatureResidual::default();
    failed.view = 2;
    records.push(failed);
    let grid = ImageGrid::with_columns(size, 64).unwrap();
    let opts = ResidualMapOptions::default();
    let a = residual_map(&records, 0, &grid, &opts).unwrap();
    let b = residual_map(&records, 0, &grid, &opts).unwrap();
    for (x, y) in [
        (&a.mean_dx, &b.mean_dx),
        (&a.mean_dy, &b.mean_dy),
        (&a.rms, &b.rms),
        (&a.std_error, &b.std_error),
        (&a.significance, &b.significance),
        (&a.effective_samples, &b.effective_samples),
        (&a.effective_views, &b.effective_views),
    ] {
        assert_eq!(bits(x), bits(y));
    }
    assert_eq!(a.bandwidth_selection, b.bandwidth_selection);
    assert_eq!(a.bandwidth_px.to_bits(), b.bandwidth_px.to_bits());
    assert_eq!((a.used, a.unprojected, a.views), (12 * 300, 1, 12));
    assert!(matches!(
        a.bandwidth_selection,
        BandwidthSelection::CrossValidated { folds: 5, .. }
    ));
    // Camera 1 alone sees its own constant field.
    let one = residual_map(&records, 1, &grid, &options(Bandwidth::Pixels(100.0))).unwrap();
    let i = grid.len() / 2;
    assert!((one.mean_dx[i] - 5.0).abs() < 0.05, "{}", one.mean_dx[i]);
}

/// Options out of range, a camera without records and a view
/// without records are input errors; too few views take the fallback
/// bandwidth.
#[test]
fn residual_map_validates_its_input() {
    let size = [640, 480];
    let records = Capture::new(size, 3, 200).records(1, zero);
    let grid = ImageGrid::with_columns(size, 16).unwrap();
    let bad = |f: &dyn Fn(&mut ResidualMapOptions)| {
        let mut o = ResidualMapOptions::default();
        f(&mut o);
        matches!(
            residual_map(&records, 0, &grid, &o),
            Err(Error::InvalidInput { .. })
        )
    };
    assert!(bad(&|o| o.bandwidth = Bandwidth::Pixels(0.0)));
    assert!(bad(&|o| o.bandwidth = Bandwidth::Pixels(f64::NAN)));
    assert!(bad(&|o| o.outlier_factor = 0.0));
    assert!(bad(&|o| o.min_effective_samples = 0.5));
    assert!(bad(&|o| o.min_effective_views = 1.0));
    assert!(bad(&|o| o.min_effective_views = f64::INFINITY));
    assert!(bad(&|o| o.views = Some(vec![])));
    assert!(bad(&|o| o.views = Some(vec![0, 7])));
    assert!(matches!(
        residual_map(&records, 2, &grid, &ResidualMapOptions::default()),
        Err(Error::InvalidInput { .. })
    ));
    let map = residual_map(&records, 0, &grid, &ResidualMapOptions::default()).unwrap();
    assert_eq!(map.bandwidth_selection, BandwidthSelection::FewViews);
    assert!((map.bandwidth_px - 0.1 * 800.0).abs() < 1e-12);
}

#[test]
fn residual_map_options_round_trip_through_json() {
    let mut o = ResidualMapOptions::new();
    o.bandwidth = Bandwidth::Pixels(42.5);
    o.views = Some(vec![1, 4]);
    o.min_effective_views = 3.0;
    let json = serde_json::to_string(&o).unwrap();
    assert_eq!(
        serde_json::from_str::<ResidualMapOptions>(&json).unwrap(),
        o
    );
    let defaults: ResidualMapOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(defaults, ResidualMapOptions::default());
    assert_eq!(
        serde_json::from_str::<ResidualMapOptions>(r#"{"bandwidth":"auto"}"#).unwrap(),
        ResidualMapOptions::default()
    );
    assert!(serde_json::from_str::<ResidualMapOptions>(r#"{"bandwith":"auto"}"#).is_err());
    let selection = BandwidthSelection::CrossValidated {
        folds: 5,
        ladder_px: vec![48.0, 72.0],
        scores_px2: vec![0.021, 0.02],
        score_std_errors_px2: vec![0.001, 0.0012],
    };
    let json = serde_json::to_string(&selection).unwrap();
    assert!(json.contains(r#""kind":"cross_validated""#), "{json}");
    assert_eq!(
        serde_json::from_str::<BandwidthSelection>(&json).unwrap(),
        selection
    );
}

// ─── The significance's reference distribution ──────────────────────────────

/// The `χ` of `hotelling_chi` has `exp(−χ²/2)` equal to the upper tail of
/// `F(2, ν − 1)` at `T²·(ν − 1)/(2ν)`, here integrated numerically from the
/// density `(1 + 2x/m)^(−(m+2)/2)` (an independent path to the closed form);
/// it tends to `√T²` with many views and is undefined at `ν ≤ 1`.
#[test]
fn significance_has_the_hotelling_tail() {
    for nu in [2.5, 4.0, 5.0, 19.0, 99.0] {
        let m = nu - 1.0;
        let density = |x: f64| (1.0 + 2.0 * x / m).powf(-(m + 2.0) / 2.0);
        for t2 in [0.5, 4.0, 9.18, 25.0] {
            let f0 = t2 * (nu - 1.0) / (2.0 * nu);
            // ∫_{f0}^{∞} density, by x = f0 + u/(1 − u) on u ∈ [0, 1).
            let n = 200_000;
            let tail: f64 = (0..n)
                .map(|i| {
                    let u = (i as f64 + 0.5) / n as f64;
                    let x = f0 + u / (1.0 - u);
                    density(x) / (1.0 - u).powi(2) / n as f64
                })
                .sum();
            let chi = hotelling_chi(t2, nu);
            let p = (-0.5 * chi * chi).exp();
            assert!(
                (p - tail).abs() <= 1e-4 * tail.max(1e-3),
                "ν = {nu}, T² = {t2}: closed form {p}, integral {tail}"
            );
            assert!(
                chi <= t2.sqrt() + 1e-12,
                "the small-sample χ is never larger"
            );
        }
    }
    assert!((hotelling_chi(9.0, 1e9) - 3.0).abs() < 1e-6);
    assert!(hotelling_chi(9.0, 1.0).is_nan());
}

// ─── B-T3: no gain from no field ────────────────────────────────────────────

/// The share of nodes with a significance above `3.03` (pointwise
/// `p < 0.01`) and whether cross-validation chose one of the two largest
/// scored bandwidths, for a zero field.
fn null_run(capture: &Capture, seed: u64, columns: u32) -> (f64, bool) {
    let records = capture.records(seed, zero);
    let grid = ImageGrid::with_columns(capture.size, columns).unwrap();
    let map = residual_map(&records, 0, &grid, &ResidualMapOptions::default()).unwrap();
    let smooth = match &map.bandwidth_selection {
        BandwidthSelection::CrossValidated { ladder_px, .. } => {
            let n = ladder_px.len();
            n > 0 && ladder_px[n.saturating_sub(2)..].contains(&map.bandwidth_px)
        }
        _ => false,
    };
    (share_above(&map.significance), smooth)
}

/// The share of the nodes with a significance that exceed `3.03`.
fn share_above(significance: &[f64]) -> f64 {
    let valid = significance.iter().filter(|v| v.is_finite()).count();
    let high = significance.iter().filter(|&&v| v > 3.03).count();
    if valid == 0 {
        f64::NAN
    } else {
        high as f64 / valid as f64
    }
}

/// The mean of `values` and its Monte-Carlo standard error, ignoring NaN.
fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// `f` over `seeds` on scoped threads, in seed order.
fn par_seeds<T: Send>(seeds: std::ops::Range<u64>, f: impl Fn(u64) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let seeds: Vec<u64> = seeds.collect();
    let chunk = seeds.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = seeds
            .chunks(chunk)
            .map(|c| {
                let f = &f;
                scope.spawn(move || c.iter().map(|&s| f(s)).collect::<Vec<T>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("seed thread"))
            .collect()
    })
}

/// The pointwise rate the significance promises, and how many Monte-Carlo
/// standard errors of the measured share it may exceed it by.
const NOMINAL_RATE: f64 = 0.01;
const MC_SIGMAS: f64 = 3.0;

/// With no field, the share of nodes with `χ > 3.03` is the nominal
/// 1 % within three Monte-Carlo standard errors (400 seeds; the nodes of one
/// map are correlated, so the error comes from the per-seed shares), and
/// cross-validation picks one of the two smoothest bandwidths.
#[test]
#[ignore = "release-mode Monte-Carlo; run by scripts/uncertainty_gates.py"]
fn residual_map_finds_no_field_where_there_is_none() {
    let capture = Capture {
        view_sigma: 0.05,
        ..Capture::new([1280, 960], 20, 300)
    };
    let seeds = 400;
    let runs = par_seeds(0..seeds, |seed| null_run(&capture, 1000 + seed, 64));
    let shares: Vec<f64> = runs.iter().map(|r| r.0).collect();
    let smooth = runs.iter().filter(|r| r.1).count();
    let (share, error) = mean_and_error(&shares);
    eprintln!(
        "no field, 20 views × 300, cross-validated: χ > 3.03 at {:.2} ± {:.2} % of nodes \
         ({seeds} seeds); CV chose one of the two smoothest in {smooth} of {seeds} runs",
        100.0 * share,
        100.0 * error
    );
    assert!(
        share <= NOMINAL_RATE + MC_SIGMAS * error,
        "{share} ± {error}"
    );
    assert!(smooth as u64 * 10 >= seeds * 9, "{smooth} of {seeds}");
}

// ─── B-T1: the standard error against Monte-Carlo scatter ───────────────────

/// Monte-Carlo of the map at fixed nodes under a zero field: the scatter of
/// `μ̂` across seeds against the stated standard error (RMS over seeds), for
/// the view-clustered error and the per-record one; and, when asked, the
/// whole map's mean `χ²` and per-seed share of nodes with `χ > 3.03`.
struct MonteCarlo {
    /// Per node: scatter / RMS stated SE, view-clustered; NaN without support.
    clustered: Vec<f64>,
    /// Per node: scatter / RMS stated SE, records as independent.
    per_record: Vec<f64>,
    /// Per node: the mean effective views.
    effective_views: Vec<f64>,
    /// The mean `χ²` over valid nodes and seeds; NaN when not asked.
    mean_chi2: f64,
    /// Per seed: the share of the map's nodes with `χ > 3.03`.
    shares: Vec<f64>,
}

/// One seed of [`monte_carlo`]: per node `(mean, clustered SE², per-record
/// SE², effective views)` where supported, the map's `Σχ²` and node count,
/// and its share above 3.03.
type SeedRun = (Vec<Option<([f64; 2], f64, f64, f64)>>, f64, usize, f64);

fn monte_carlo(
    capture: &Capture,
    h: f64,
    seeds: u64,
    nodes: &[[f64; 2]],
    whole_map: bool,
) -> MonteCarlo {
    let grid = ImageGrid::with_columns(capture.size, 32).unwrap();
    let opts = options(Bandwidth::Pixels(h));
    let runs: Vec<SeedRun> = par_seeds(0..seeds, |seed| {
        let records = capture.records(5000 + seed, zero);
        let (mut chi2, mut count, mut share) = (0.0, 0, f64::NAN);
        if whole_map {
            let map = residual_map(&records, 0, &grid, &opts).unwrap();
            for &s in &map.significance {
                if s.is_finite() {
                    chi2 += s * s;
                    count += 1;
                }
            }
            share = share_above(&map.significance);
        }
        // The fixed nodes, with both clusterings.
        let selected = select(&records, 0, None).unwrap();
        let mut samples = selected.samples;
        remove_outliers(&mut samples, opts.outlier_factor);
        let views = renumber_views(&mut samples).len();
        let smoother = Smoother::new(&samples, h);
        let mut scratch = Scratch::new(views);
        let fits = nodes
            .iter()
            .map(|&x0| {
                let by_view = smoother.fit(x0, &mut scratch, Clusters::Views);
                let by_record = smoother.fit(x0, &mut scratch, Clusters::Records);
                let usable =
                    by_view.supported(opts.min_effective_samples) && by_view.std_error.is_finite();
                usable.then(|| {
                    (
                        by_view.mean,
                        by_view.std_error.powi(2),
                        by_record.std_error.powi(2),
                        by_view.effective_views,
                    )
                })
            })
            .collect();
        (fits, chi2, count, share)
    });
    let mut clustered = Vec::new();
    let mut per_record = Vec::new();
    let mut effective_views = Vec::new();
    for j in 0..nodes.len() {
        let fits: Vec<_> = runs.iter().filter_map(|r| r.0[j]).collect();
        let k = fits.len() as f64;
        if fits.len() < seeds as usize / 2 {
            clustered.push(f64::NAN);
            per_record.push(f64::NAN);
            effective_views.push(f64::NAN);
            continue;
        }
        let mx = fits.iter().map(|f| f.0[0]).sum::<f64>() / k;
        let my = fits.iter().map(|f| f.0[1]).sum::<f64>() / k;
        let (mut cxx, mut cxy, mut cyy) = (0.0, 0.0, 0.0);
        for f in &fits {
            cxx += (f.0[0] - mx).powi(2);
            cxy += (f.0[0] - mx) * (f.0[1] - my);
            cyy += (f.0[1] - my).powi(2);
        }
        let (cxx, cxy, cyy) = (cxx / (k - 1.0), cxy / (k - 1.0), cyy / (k - 1.0));
        let major = (0.5 * (cxx + cyy) + (0.5 * (cxx - cyy)).hypot(cxy)).sqrt();
        let se_view = (fits.iter().map(|f| f.1).sum::<f64>() / k).sqrt();
        let se_record = (fits.iter().map(|f| f.2).sum::<f64>() / k).sqrt();
        clustered.push(major / se_view);
        per_record.push(major / se_record);
        effective_views.push(fits.iter().map(|f| f.3).sum::<f64>() / k);
    }
    let chi2: f64 = runs.iter().map(|r| r.1).sum();
    let count: usize = runs.iter().map(|r| r.2).sum();
    MonteCarlo {
        clustered,
        per_record,
        effective_views,
        mean_chi2: if whole_map {
            chi2 / count as f64
        } else {
            f64::NAN
        },
        shares: runs.iter().map(|r| r.3).collect(),
    }
}

/// The view-clustered standard error matches the Monte-Carlo scatter
/// of the mean (ratio in [0.8, 1.25]) at the centre, an edge and a corner,
/// with per-view common offsets (σ 0.05 px) standing in for pose error and
/// iid noise (σ 0.1 px), at 20 views and at about 5 effective views; the
/// mean `χ²` is near its 2 degrees of freedom at 20 views; at the centre,
/// where many records of every view support the node, treating the records
/// as independent understates the scatter by more than 1.5×.
#[test]
#[ignore = "release-mode Monte-Carlo; run by scripts/uncertainty_gates.py"]
fn residual_map_standard_error_matches_monte_carlo() {
    let size = [1280, 960];
    let nodes = [[639.5, 479.5], [639.5, 15.0], [15.0, 15.0]];
    let mut failures = Vec::new();
    for (views, per_view) in [(20, 300), (5, 1200)] {
        let capture = Capture {
            view_sigma: 0.05,
            ..Capture::new(size, views, per_view)
        };
        let mc = monte_carlo(&capture, 100.0, 200, &nodes, views == 20);
        eprintln!(
            "{views} views × {per_view}: scatter / SE (centre, edge, corner) clustered \
             {:.3?}, per-record {:.3?}; effective views {:.2?}; mean χ² {:.3}",
            mc.clustered, mc.per_record, mc.effective_views, mc.mean_chi2
        );
        for (j, r) in mc.clustered.iter().enumerate() {
            if !(0.8..=1.25).contains(r) {
                failures.push(format!("{views} views, node {j}: clustered ratio {r:.3}"));
            }
        }
        if mc.per_record[0].is_nan() || mc.per_record[0] < 1.5 {
            failures.push(format!(
                "{views} views: per-record ratio at the centre {:.3}",
                mc.per_record[0]
            ));
        }
        if views == 20 && !(1.6..=2.5).contains(&mc.mean_chi2) {
            failures.push(format!("{views} views: mean χ² {:.3}", mc.mean_chi2));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// One capture of the sweep: its report line, its no-field share and that
/// share's Monte-Carlo error, and its failures.
fn sweep_capture(
    size: [u32; 2],
    coverage: f64,
    views: usize,
    per_view: usize,
) -> (String, f64, f64, Vec<String>) {
    let capture = Capture {
        view_sigma: 0.05,
        coverage,
        ..Capture::new(size, views, per_view)
    };
    let [w, hgt] = [f64::from(size[0]), f64::from(size[1])];
    let (x0, y0) = (w * (1.0 - coverage) / 2.0, hgt * (1.0 - coverage) / 2.0);
    let nodes = [
        [w / 2.0 - 0.5, hgt / 2.0 - 0.5],
        [w / 2.0 - 0.5, y0 + 15.0],
        [x0 + 15.0, y0 + 15.0],
    ];
    let mc = monte_carlo(&capture, 0.1 * w.hypot(hgt), 200, &nodes, true);
    let (share, error) = mean_and_error(&mc.shares);
    let seeds = 10;
    let smooth = (0..seeds)
        .filter(|&seed| null_run(&capture, 2000 + seed, 32).1)
        .count();
    let label = format!("{size:?} coverage {coverage}, {views} views × {per_view}");
    let line = format!(
        "sweep {label}: clustered {:.3?} (effective views {:.1?}); χ > 3.03 at \
         {:.2} ± {:.2} % of nodes (200 seeds); CV smoothest {smooth}/{seeds}",
        mc.clustered,
        mc.effective_views,
        100.0 * share,
        100.0 * error
    );
    let mut failures = Vec::new();
    for (j, r) in mc.clustered.iter().enumerate() {
        if mc.effective_views[j] >= 5.0 && !(0.8..=1.25).contains(r) {
            failures.push(format!("{label}: node {j} clustered ratio {r:.3}"));
        }
    }
    if share > NOMINAL_RATE + MC_SIGMAS * error {
        failures.push(format!(
            "{label}: χ > 3.03 at {:.2} ± {:.2} %",
            100.0 * share,
            100.0 * error
        ));
    }
    if smooth * 10 < seeds as usize * 9 {
        failures.push(format!("{label}: CV smoothest {smooth}/{seeds}"));
    }
    (line, share, error, failures)
}

/// The standard-error and no-field tests swept over captures: features per
/// view {50, 300, 1000} × views {6, 20, 100} × coverage {full, central 60 %}
/// × aspect {4:3, 3:1}. At `h = 0.1·diagonal` and 200 seeds: the
/// standard error at the coverage's centre, top edge and top-left corner
/// (each 15 px inside it, where at least 5 effective views support one), and
/// the share of the map's nodes with `χ > 3.03` against the nominal 1 %
/// within three Monte-Carlo standard errors. Cross-validation's choice over
/// 10 seeds. The captures run on their own threads; the lines print in sweep
/// order, then the share per number of views.
#[test]
#[ignore = "release-mode Monte-Carlo; run by scripts/uncertainty_gates.py"]
fn residual_map_sweep_over_captures() {
    let mut captures = Vec::new();
    for size in [[1280u32, 960u32], [1800, 600]] {
        for coverage in [1.0, 0.6] {
            for views in [6, 20, 100] {
                for per_view in [50, 300, 1000] {
                    captures.push((size, coverage, views, per_view));
                }
            }
        }
    }
    let results: Vec<(String, f64, f64, Vec<String>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = captures
            .iter()
            .map(|&(size, coverage, views, per_view)| {
                scope.spawn(move || sweep_capture(size, coverage, views, per_view))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("sweep thread"))
            .collect()
    });
    let mut failures = Vec::new();
    for (line, _, _, f) in &results {
        eprintln!("{line}");
        failures.extend(f.iter().cloned());
    }
    for views in [6, 20, 100] {
        let class: Vec<(f64, f64)> = captures
            .iter()
            .zip(&results)
            .filter(|(c, _)| c.2 == views)
            .map(|(_, r)| (r.1, r.2))
            .collect();
        let n = class.len() as f64;
        let share = class.iter().map(|c| c.0).sum::<f64>() / n;
        let error = class.iter().map(|c| c.1 * c.1).sum::<f64>().sqrt() / n;
        eprintln!(
            "sweep {views} views: χ > 3.03 at {:.2} ± {:.2} % of nodes over {} captures",
            100.0 * share,
            100.0 * error,
            class.len()
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
