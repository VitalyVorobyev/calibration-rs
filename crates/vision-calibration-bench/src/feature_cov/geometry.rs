//! The study's geometric measures: the homography residual of undistorted
//! corners, and the comparison of two cameras through rays.

use anyhow::{Result, anyhow};
use nalgebra::{Matrix2, Matrix3, SMatrix, SVector, Vector2, Vector3};
use serde::Serialize;
use vision_calibration::mvg::dlt_homography;
use vision_calibration_core::{CameraParams, Pt2};

/// Grid nodes across the image, `[columns, rows]`, as the projection
/// uncertainty grid lays them out.
pub const GRID: [usize; 2] = [16, 12];

/// χ²(2) quantiles of the 1σ and 2σ coverage figures.
pub const CHI2_68: f64 = 2.30;
/// See [`CHI2_68`].
pub const CHI2_95: f64 = 6.18;

/// The grid nodes over an image of `image_size`, row by row from the
/// top-left, from pixel `(0, 0)` to `(width − 1, height − 1)`.
pub fn grid(image_size: [u32; 2]) -> Vec<[f64; 2]> {
    let [columns, rows] = GRID;
    let node = |i: usize, n: usize, size: u32| i as f64 * f64::from(size - 1) / (n - 1) as f64;
    (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (c, r)))
        .map(|(c, r)| {
            [
                node(c, columns, image_size[0]),
                node(r, rows, image_size[1]),
            ]
        })
        .collect()
}

/// Ideal-pinhole pixels of `px` under `camera`: each pixel's ray, projected
/// with the camera's own `K` and no distortion or sensor tilt.
pub fn undistort(camera: &CameraParams, px: &[Pt2]) -> Result<Vec<Pt2>> {
    let model = camera.build()?;
    let k = camera.intrinsics;
    px.iter()
        .map(|p| {
            let d = model.backproject_pixel(p).direction;
            anyhow::ensure!(d.z > 0.0, "a corner's ray points away from the camera");
            Ok(Pt2::new(k.fx * d.x / d.z + k.cx, k.fy * d.y / d.z + k.cy))
        })
        .collect()
}

/// Similarity that maps points to centroid 0 and mean distance √2.
fn normalizer(p: &[Pt2]) -> Matrix3<f64> {
    let n = p.len() as f64;
    let (mx, my) = p
        .iter()
        .fold((0.0, 0.0), |a, q| (a.0 + q.x / n, a.1 + q.y / n));
    let md = p
        .iter()
        .map(|q| ((q.x - mx).powi(2) + (q.y - my).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = std::f64::consts::SQRT_2 / md.max(1e-12);
    Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0)
}

fn apply(h: &Matrix3<f64>, p: &Pt2) -> Pt2 {
    let v = h * Vector3::new(p.x, p.y, 1.0);
    Pt2::new(v.x / v.z, v.y / v.z)
}

/// Squared residuals (px²) of the maximum-likelihood homography from the
/// board points to `px`: a normalized DLT, then Gauss–Newton on the image
/// error (the board points are exact).
pub fn homography_squared_errors(board: &[Pt2], px: &[Pt2]) -> Result<Vec<f64>> {
    anyhow::ensure!(
        board.len() == px.len() && board.len() >= 4,
        "a homography needs at least 4 matched points"
    );
    let ts = normalizer(board);
    let td = normalizer(px);
    let bs: Vec<Pt2> = board.iter().map(|b| apply(&ts, b)).collect();
    let ds: Vec<Pt2> = px.iter().map(|p| apply(&td, p)).collect();
    let h0 = dlt_homography(&bs, &ds)?;
    anyhow::ensure!(h0[(2, 2)].abs() > 1e-12, "degenerate homography");
    let h0 = h0 / h0[(2, 2)];
    // Row-major parameters h11 h12 h13 h21 h22 h23 h31 h32 (h33 = 1).
    let mut h = SVector::<f64, 8>::from_fn(|i, _| h0[(i / 3, i % 3)]);
    let mat =
        |h: &SVector<f64, 8>| Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0);
    for _ in 0..20 {
        let m = mat(&h);
        let mut jtj = SMatrix::<f64, 8, 8>::zeros();
        let mut jtr = SVector::<f64, 8>::zeros();
        for (b, d) in bs.iter().zip(&ds) {
            let (x, y) = (b.x, b.y);
            let w = m[(2, 0)] * x + m[(2, 1)] * y + 1.0;
            let u = (m[(0, 0)] * x + m[(0, 1)] * y + m[(0, 2)]) / w;
            let v = (m[(1, 0)] * x + m[(1, 1)] * y + m[(1, 2)]) / w;
            let ju = SVector::<f64, 8>::from_column_slice(&[
                x / w,
                y / w,
                1.0 / w,
                0.0,
                0.0,
                0.0,
                -x * u / w,
                -y * u / w,
            ]);
            let jv = SVector::<f64, 8>::from_column_slice(&[
                0.0,
                0.0,
                0.0,
                x / w,
                y / w,
                1.0 / w,
                -x * v / w,
                -y * v / w,
            ]);
            jtj += ju * ju.transpose() + jv * jv.transpose();
            jtr += ju * (d.x - u) + jv * (d.y - v);
        }
        let Some(step) = jtj.cholesky().map(|c| c.solve(&jtr)) else {
            break;
        };
        h += step;
        if step.norm() < 1e-12 {
            break;
        }
    }
    let to_px = td
        .try_inverse()
        .ok_or_else(|| anyhow!("singular normalizer"))?
        * mat(&h)
        * ts;
    Ok(board
        .iter()
        .zip(px)
        .map(|(b, p)| (apply(&to_px, b) - p).norm_squared())
        .collect())
}

/// A node's projection covariance, px²; `None` where no ray reaches it.
pub type Ellipse = Option<Matrix2<f64>>;

/// The convex hull of `points`, counter-clockwise (Andrew's monotone chain).
pub fn convex_hull(points: &[Pt2]) -> Vec<Pt2> {
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
pub fn inside(hull: &[Pt2], q: &Pt2) -> bool {
    hull.len() >= 3
        && (0..hull.len()).all(|i| {
            let (a, b) = (&hull[i], &hull[(i + 1) % hull.len()]);
            (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x) >= 0.0
        })
}

/// Additive honesty sums over `(split, node)` pairs: pooling several splits
/// is adding their sums.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ZSums {
    /// `Σ z²`.
    pub sum_z2: f64,
    /// Pairs with a `z²`.
    pub pairs: usize,
    /// Pairs with `z² < 2.30`.
    pub below_68: usize,
    /// Pairs with `z² < 6.18`.
    pub below_95: usize,
}

impl ZSums {
    fn push(&mut self, z2: f64) {
        self.sum_z2 += z2;
        self.pairs += 1;
        self.below_68 += usize::from(z2 < CHI2_68);
        self.below_95 += usize::from(z2 < CHI2_95);
    }

    /// The sum of several.
    pub fn total<'a>(items: impl IntoIterator<Item = &'a ZSums>) -> ZSums {
        items.into_iter().fold(ZSums::default(), |a, b| ZSums {
            sum_z2: a.sum_z2 + b.sum_z2,
            pairs: a.pairs + b.pairs,
            below_68: a.below_68 + b.below_68,
            below_95: a.below_95 + b.below_95,
        })
    }

    /// `r = Σ z² / (2 N)`: 1 when the stated uncertainty is honest.
    pub fn ratio(&self) -> f64 {
        if self.pairs == 0 {
            f64::NAN
        } else {
            self.sum_z2 / (2.0 * self.pairs as f64)
        }
    }

    /// Fraction of pairs below `z² = 2.30` (nominal 0.683).
    pub fn coverage_68(&self) -> f64 {
        self.below_68 as f64 / self.pairs as f64
    }

    /// Fraction of pairs below `z² = 6.18` (nominal 0.954).
    pub fn coverage_95(&self) -> f64 {
        self.below_95 as f64 / self.pairs as f64
    }
}

/// One split's figures over a set of grid nodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct GridSums {
    /// `Σ |d|²`, px².
    pub sum_d2: f64,
    /// Nodes where camera A's ray reaches camera B.
    pub nodes: usize,
    /// The honesty sums, over the nodes where both cameras state a
    /// covariance.
    pub z: ZSums,
}

impl GridSums {
    /// RMS distance, px, between each node and where its camera-A ray lands
    /// in camera B.
    pub fn rms_px(&self) -> f64 {
        if self.nodes == 0 {
            f64::NAN
        } else {
            (self.sum_d2 / self.nodes as f64).sqrt()
        }
    }
}

/// How two cameras of one sensor agree: over the decisive grid (the nodes
/// inside the hull of the camera's detected corners) and over the full grid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SplitComparison {
    /// Nodes inside the corner hull (decisive).
    pub hull: GridSums,
    /// Every node (reported only).
    pub full: GridSums,
}

/// Compare camera `a` with camera `b` at `nodes`: cast each node's ray with
/// `a`, project it with `b` and take the pixel difference `d`. With both
/// cameras' projection covariances at the nodes (`Σ_A`, `Σ_B`), also
/// `z² = dᵀ (Σ_A + Σ_B)⁻¹ d`, which assumes the two cameras' errors are
/// independent (true for calibrations on disjoint views). `in_hull` marks
/// the decisive nodes. Nodes where a ray misses or the summed covariance is
/// singular are left out.
pub fn compare(
    a: &CameraParams,
    b: &CameraParams,
    nodes: &[[f64; 2]],
    in_hull: &[bool],
    covariances: Option<(&[Ellipse], &[Ellipse])>,
) -> Result<SplitComparison> {
    let (ma, mb) = (a.build()?, b.build()?);
    let mut hull = GridSums::default();
    let mut full = GridSums::default();
    for (i, &[u, v]) in nodes.iter().enumerate() {
        let Some(ray) = ma.backproject_pixel(&Pt2::new(u, v)).point_at_unit_depth() else {
            continue;
        };
        let Some(q) = mb.project_point_c(&ray) else {
            continue;
        };
        let d = Vector2::new(q.x - u, q.y - v);
        let z2 = covariances.and_then(|(ca, cb)| {
            let inv = (ca[i]? + cb[i]?).try_inverse()?;
            let z = (d.transpose() * inv * d)[(0, 0)];
            (z.is_finite() && z >= 0.0).then_some(z)
        });
        let add = |t: &mut GridSums| {
            t.sum_d2 += d.norm_squared();
            t.nodes += 1;
            if let Some(z) = z2 {
                t.z.push(z);
            }
        };
        add(&mut full);
        if in_hull[i] {
            add(&mut hull);
        }
    }
    anyhow::ensure!(full.nodes > 0, "no grid ray of camera A reaches camera B");
    Ok(SplitComparison { hull, full })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, DistortionParams, FxFyCxCy, ProjectionParams, SensorParams,
    };

    fn camera(fx: f64, k1: f64) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            },
            sensor: SensorParams::Identity,
            intrinsics: FxFyCxCy {
                fx,
                fy: fx,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    #[test]
    fn homography_fit_is_exact_on_a_projective_grid() {
        let h = Matrix3::new(900.0, 40.0, 300.0, -20.0, 880.0, 200.0, 0.1, 0.2, 1.0);
        let board: Vec<Pt2> = (0..7)
            .flat_map(|i| (0..5).map(move |j| Pt2::new(f64::from(i) * 0.03, f64::from(j) * 0.03)))
            .collect();
        let px: Vec<Pt2> = board.iter().map(|b| apply(&h, b)).collect();
        let e = homography_squared_errors(&board, &px).unwrap();
        assert!(e.iter().all(|e| *e < 1e-12), "{e:?}");
    }

    #[test]
    fn undistortion_removes_the_lens_from_a_projected_grid() {
        let cam = camera(800.0, -0.2);
        let model = cam.build().unwrap();
        let h = Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0);
        let board: Vec<Pt2> = (0..6)
            .flat_map(|i| {
                (0..4).map(move |j| Pt2::new(f64::from(i) * 0.1 - 0.25, f64::from(j) * 0.1 - 0.15))
            })
            .collect();
        // Board at unit depth, frontal: the ideal pixels are `h · board`.
        let px: Vec<Pt2> = board
            .iter()
            .map(|b| model.project_point_c(&Vector3::new(b.x, b.y, 1.0)).unwrap())
            .collect();
        let ideal = undistort(&cam, &px).unwrap();
        for (b, q) in board.iter().zip(&ideal) {
            assert!((apply(&h, b) - q).norm() < 1e-6);
        }
    }

    #[test]
    fn identical_cameras_agree_and_a_focal_change_shows() {
        let nodes = grid([640, 480]);
        assert_eq!(nodes.len(), GRID[0] * GRID[1]);
        let a = camera(800.0, -0.1);
        let all = vec![true; nodes.len()];
        let same = compare(&a, &a, &nodes, &all, None).unwrap();
        assert!(same.full.rms_px() < 1e-6, "{same:?}");
        let b = camera(801.0, -0.1);
        let diff = compare(&a, &b, &nodes, &all, None).unwrap();
        assert!(diff.full.rms_px() > 0.1, "{diff:?}");
        assert_eq!(diff.hull, diff.full);
    }

    #[test]
    fn z_statistics_follow_the_stated_covariance() {
        let nodes = grid([640, 480]);
        let all = vec![true; nodes.len()];
        let a = camera(800.0, 0.0);
        let b = camera(800.8, 0.0);
        // A covariance far larger than the difference: z² ≈ 0, all covered.
        let big = vec![Some(Matrix2::identity() * 1e6); nodes.len()];
        let z = compare(&a, &b, &nodes, &all, Some((&big, &big)))
            .unwrap()
            .full
            .z;
        assert_eq!(z.pairs, nodes.len());
        assert!(z.ratio() < 1e-3 && z.coverage_68() == 1.0 && z.coverage_95() == 1.0);
        // A tiny covariance: nothing covered except where d ≈ 0.
        let tiny = vec![Some(Matrix2::identity() * 1e-12); nodes.len()];
        let z = compare(&a, &b, &nodes, &all, Some((&tiny, &tiny)))
            .unwrap()
            .full
            .z;
        assert!(z.ratio() > 1e3);
        assert_eq!(ZSums::total([&z, &z]).pairs, 2 * z.pairs);
    }

    #[test]
    fn hull_keeps_the_nodes_the_corners_surround() {
        let corners: Vec<Pt2> = (0..5)
            .flat_map(|i| {
                (0..4).map(move |j| {
                    Pt2::new(100.0 + 50.0 * f64::from(i), 100.0 + 40.0 * f64::from(j))
                })
            })
            .collect();
        let hull = convex_hull(&corners);
        assert_eq!(hull.len(), 4, "{hull:?}");
        assert!(inside(&hull, &Pt2::new(200.0, 150.0)));
        assert!(inside(&hull, &Pt2::new(100.0, 100.0)), "a vertex is inside");
        assert!(!inside(&hull, &Pt2::new(99.0, 150.0)));
        assert!(!inside(&hull, &Pt2::new(200.0, 221.0)));
        // Split counts: hull nodes are a subset of the full grid.
        let nodes = grid([640, 480]);
        let in_hull: Vec<bool> = nodes
            .iter()
            .map(|n| inside(&hull, &Pt2::new(n[0], n[1])))
            .collect();
        let a = camera(800.0, 0.0);
        let c = compare(&a, &camera(801.0, 0.0), &nodes, &in_hull, None).unwrap();
        assert_eq!(c.hull.nodes, in_hull.iter().filter(|&&x| x).count());
        assert!(c.hull.nodes > 0 && c.hull.nodes < c.full.nodes);
    }
}
