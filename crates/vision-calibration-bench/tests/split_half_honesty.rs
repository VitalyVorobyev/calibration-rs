//! Split-half honesty of the stated projection uncertainty on synthetic
//! truth.
//!
//! Two calibrations from disjoint halves of the views disagree at a pixel by
//! `d`: the pixel's ray cast with camera A and projected with camera B, minus
//! the pixel. With each half's 2 × 2 projection covariance `Σ_A`, `Σ_B`,
//! `z² = dᵀ (Σ_A + Σ_B)⁻¹ d` follows χ² with 2 degrees of freedom when the
//! stated uncertainty is honest. Pooled over the even/odd split plus 20
//! seeded random half-splits, the 16 × 12 grid nodes inside the hull of the
//! detected corners, and many scenes: `r = Σ z² / (2 N)` is about 1, and
//! about 68.3 % and 95.4 % of the `(split, node)` pairs fall below 2.30 and
//! 6.18.
//!
//! These are the synthetic truth gates of the real-data feature-covariance
//! study and of the view-scatter scaling: on real cameras the same measure
//! is what showed the stated σ short before the view scatter was applied.

use nalgebra::{Matrix2, Vector2};
use vision_calibration::common::{CalibratedCameras, IntrinsicsManualInit};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init_with_seed,
    step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_core::synthetic::{planar as synth, poses};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCy, NoMeta, PlanarDataset, PointCovariance, Pt2, View,
    make_pinhole_camera,
};

/// Seeded random half-splits besides the even/odd split.
const RANDOM_SPLITS: usize = 20;
/// Grid nodes across the image, `[columns, rows]`.
const GRID: [usize; 2] = [16, 12];
/// The image of the synthetic camera, `[width, height]`.
const IMAGE: [u32; 2] = [640, 480];
/// χ²(2) quantiles of the 1σ and 2σ coverage figures.
const CHI2_68: f64 = 2.30;
/// See [`CHI2_68`].
const CHI2_95: f64 = 6.18;

/// Deterministic splitmix64 generator.
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

    /// Standard normal (Box–Muller).
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

/// FNV-1a hash of a label: a stable seed.
fn seed_of(label: &str) -> u64 {
    label.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The half-splits of `n` views: the even/odd split, then the seeded random
/// halves. `true` puts a view in half A.
fn splits(n: usize) -> Vec<Vec<bool>> {
    let mut rng = Rng(seed_of("synthetic/cam0/splits"));
    let mut out = vec![(0..n).map(|i| i % 2 == 0).collect::<Vec<_>>()];
    for _ in 0..RANDOM_SPLITS {
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        let mut in_a = vec![false; n];
        for &i in &order[..n / 2] {
            in_a[i] = true;
        }
        out.push(in_a);
    }
    out
}

/// The grid nodes over the image, row by row from the top-left, from pixel
/// `(0, 0)` to `(width − 1, height − 1)`.
fn grid() -> Vec<[f64; 2]> {
    let [columns, rows] = GRID;
    let node = |i: usize, n: usize, size: u32| i as f64 * f64::from(size - 1) / (n - 1) as f64;
    (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (c, r)))
        .map(|(c, r)| [node(c, columns, IMAGE[0]), node(r, rows, IMAGE[1])])
        .collect()
}

/// The convex hull of `points`, counter-clockwise (Andrew's monotone chain).
fn convex_hull(points: &[Pt2]) -> Vec<Pt2> {
    let mut p: Vec<Pt2> = points.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: &Pt2, a: &Pt2, b: &Pt2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut hull: Vec<Pt2> = Vec::with_capacity(2 * p.len());
    for pass in [false, true] {
        let start = hull.len();
        let ordered: Box<dyn Iterator<Item = &Pt2>> = if pass {
            Box::new(p.iter().rev())
        } else {
            Box::new(p.iter())
        };
        for q in ordered {
            while hull.len() >= start + 2
                && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], q) <= 0.0
            {
                hull.pop();
            }
            hull.push(*q);
        }
        hull.pop();
    }
    hull
}

/// Whether `q` lies inside or on the counter-clockwise convex polygon `hull`.
fn inside(hull: &[Pt2], q: &Pt2) -> bool {
    hull.len() >= 3
        && (0..hull.len()).all(|i| {
            let (a, b) = (&hull[i], &hull[(i + 1) % hull.len()]);
            (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x) >= 0.0
        })
}

/// Honesty sums over `(split, node)` pairs.
#[derive(Debug, Clone, Copy, Default)]
struct ZSums {
    sum_z2: f64,
    pairs: usize,
    below_68: usize,
    below_95: usize,
}

impl ZSums {
    fn push(&mut self, z2: f64) {
        self.sum_z2 += z2;
        self.pairs += 1;
        self.below_68 += usize::from(z2 < CHI2_68);
        self.below_95 += usize::from(z2 < CHI2_95);
    }

    fn add(&mut self, other: &ZSums) {
        self.sum_z2 += other.sum_z2;
        self.pairs += other.pairs;
        self.below_68 += other.below_68;
        self.below_95 += other.below_95;
    }

    /// `r = Σ z² / (2 N)`: 1 when the stated uncertainty is honest.
    fn ratio(&self) -> f64 {
        self.sum_z2 / (2.0 * self.pairs as f64)
    }

    fn coverage_68(&self) -> f64 {
        self.below_68 as f64 / self.pairs as f64
    }

    fn coverage_95(&self) -> f64 {
        self.below_95 as f64 / self.pairs as f64
    }
}

/// A synthetic camera seen in `views` views with N(0, 0.1²) px corner noise,
/// each view also warped radially about the principal point by its own
/// `κ ~ N(0, view_warp²)` (px⁻²): an error every corner of a view shares, as
/// a target that bends differently in each view would make. Returns the
/// views without covariances and the same views stating the true isotropic
/// covariance.
fn scene(seed: u64, views: usize, view_warp: f64) -> [Vec<View<NoMeta>>; 2] {
    let camera = make_pinhole_camera(
        FxFyCxCy {
            fx: 800.0,
            fy: 790.0,
            cx: 320.0,
            cy: 240.0,
        },
        BrownConrady5 {
            k1: -0.1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    );
    let board = synth::grid_points(8, 6, 0.03);
    let specs = poses::seeded_board_pose_specs(views, 0.5, 11 + seed);
    let poses = poses::centered_board_poses(&specs, (0.105, 0.075), 0.5);
    let mut rng = Rng(5 + seed);
    let sigma = 0.1;
    let mut uniform = Vec::new();
    let mut stated = Vec::new();
    for pose in &poses {
        let view = synth::project_view_all(&camera, pose, &board).unwrap();
        let kappa = if view_warp == 0.0 {
            0.0
        } else {
            view_warp * rng.normal()
        };
        let px: Vec<_> = view
            .points_2d
            .iter()
            .map(|p| {
                let (dx, dy) = (p.x - 320.0, p.y - 240.0);
                let warp = 1.0 + kappa * (dx * dx + dy * dy);
                Pt2::new(
                    320.0 + dx * warp + sigma * rng.normal(),
                    240.0 + dy * warp + sigma * rng.normal(),
                )
            })
            .collect();
        let cov = vec![PointCovariance::new(sigma * sigma, 0.0, sigma * sigma).unwrap(); px.len()];
        uniform.push(View::without_meta(
            CorrespondenceView::new(board.clone(), px.clone()).unwrap(),
        ));
        stated.push(View::without_meta(
            CorrespondenceView::new_with_covariances(board.clone(), px, cov).unwrap(),
        ));
    }
    [uniform, stated]
}

/// Calibrate `views` with the default planar-intrinsics configuration.
fn calibrate(views: Vec<View<NoMeta>>) -> PlanarIntrinsicsExport {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session
        .set_config(PlanarIntrinsicsConfig::default())
        .unwrap();
    session
        .set_input(PlanarDataset::new(views).unwrap())
        .unwrap();
    step_init_with_seed(&mut session, IntrinsicsManualInit::default(), None).unwrap();
    step_optimize(&mut session, None).unwrap();
    session.export().unwrap()
}

/// The honesty sums of camera `b` against camera `a` at the nodes marked in
/// `in_hull`: nodes where a ray misses or the summed covariance is singular
/// are left out.
fn compare(
    a: &PlanarIntrinsicsExport,
    b: &PlanarIntrinsicsExport,
    nodes: &[[f64; 2]],
    in_hull: &[bool],
) -> ZSums {
    let cov_a = a.projection_covariance(0, nodes).unwrap();
    let cov_b = b.projection_covariance(0, nodes).unwrap();
    let model_a = a.camera_params()[0].build().unwrap();
    let model_b = b.camera_params()[0].build().unwrap();
    let mut sums = ZSums::default();
    for (i, &[u, v]) in nodes.iter().enumerate() {
        if !in_hull[i] {
            continue;
        }
        let Some(ray) = model_a
            .backproject_pixel(&Pt2::new(u, v))
            .point_at_unit_depth()
        else {
            continue;
        };
        let Some(q) = model_b.project_point_c(&ray) else {
            continue;
        };
        let d = Vector2::new(q.x - u, q.y - v);
        let sum: Option<Matrix2<f64>> = cov_a[i].zip(cov_b[i]).map(|(ca, cb)| ca + cb);
        let Some(inv) = sum.and_then(|s| s.try_inverse()) else {
            continue;
        };
        let z2 = (d.transpose() * inv * d)[(0, 0)];
        if z2.is_finite() && z2 >= 0.0 {
            sums.push(z2);
        }
    }
    sums
}

/// The split-half honesty of each arm's calibrations, pooled over `scenes`
/// independent scenes, every split and every node inside the corners' hull.
/// Pooling over many scenes matters: one scene's splits share its data, so
/// their `z²` move together, and one scene's `r` spans about 0.5–1.5 (5–95 %)
/// even when the stated uncertainty is exact.
fn pooled_honesty(
    scenes: u64,
    scene: impl Fn(u64) -> [Vec<View<NoMeta>>; 2] + Sync,
    arms: &[usize],
) -> Vec<ZSums> {
    let nodes = grid();
    let jobs: Vec<(u64, usize)> = (0..scenes)
        .flat_map(|s| arms.iter().map(move |&arm| (s, arm)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(vec![ZSums::default(); jobs.len()]);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(jobs.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(seed, arm)) = jobs.get(i) else {
                        break;
                    };
                    let views = scene(seed);
                    let corners: Vec<Pt2> = views[0]
                        .iter()
                        .flat_map(|v| v.obs.points_2d.iter().copied())
                        .collect();
                    let hull = convex_hull(&corners);
                    let in_hull: Vec<bool> = nodes
                        .iter()
                        .map(|n| inside(&hull, &Pt2::new(n[0], n[1])))
                        .collect();
                    let views = &views[arm];
                    let mut sums = ZSums::default();
                    for split in splits(views.len()) {
                        let half = |a: bool| {
                            views
                                .iter()
                                .zip(&split)
                                .filter(|(_, in_a)| **in_a == a)
                                .map(|(v, _)| v.clone())
                                .collect::<Vec<_>>()
                        };
                        let (a, b) = (calibrate(half(true)), calibrate(half(false)));
                        sums.add(&compare(&a, &b, &nodes, &in_hull));
                    }
                    results
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)[i] = sums;
                }
            });
        }
    });
    // Summed in job order, so the totals do not depend on the scheduling.
    let per_job = results
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut totals = vec![ZSums::default(); arms.len()];
    for (&(_, arm), sums) in jobs.iter().zip(&per_job) {
        let position = arms.iter().position(|&x| x == arm).unwrap();
        totals[position].add(sums);
    }
    totals
}

/// `r` within 0.85–1.15 and both coverages within ±5 points of nominal.
fn assert_honest(label: &str, total: &ZSums) {
    let (r, c68, c95) = (total.ratio(), total.coverage_68(), total.coverage_95());
    eprintln!(
        "{label}: r {r:.3}, ≤2.30 {:.1} %, ≤6.18 {:.1} %, {} pairs",
        100.0 * c68,
        100.0 * c95,
        total.pairs
    );
    assert!((0.85..=1.15).contains(&r), "{label}: r = {r}");
    assert!((c68 - 0.683).abs() <= 0.05, "{label}: coverage 68 = {c68}");
    assert!((c95 - 0.954).abs() <= 0.05, "{label}: coverage 95 = {c95}");
}

/// Arm 0: no covariance. Arm 1: every corner states its true covariance.
const UNIFORM: usize = 0;
const STATED: usize = 1;

/// With exactly true independent noise the stated uncertainty is honest,
/// with and without the per-corner covariance (24 scenes of 15 views).
#[test]
fn synthetic_truth_gate() {
    let totals = pooled_honesty(24, |seed| scene(seed, 15, 0.0), &[UNIFORM, STATED]);
    assert_honest("truth gate, no covariance", &totals[0]);
    assert_honest("truth gate, stated covariance", &totals[1]);
}

/// With an error each view's corners share (a per-view radial warp of about
/// 0.2 px at the image corners, besides the 0.1 px noise), the view scatter
/// keeps the stated uncertainty honest. Without it `r` is several times 1
/// (12 scenes of 24 views).
#[test]
fn synthetic_truth_gate_with_view_errors() {
    let totals = pooled_honesty(12, |seed| scene(seed, 24, 1.5e-8), &[UNIFORM]);
    assert_honest("truth gate with view errors", &totals[0]);
}
