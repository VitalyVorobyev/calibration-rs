//! A non-ideal target, fitted jointly over a camera set: terms that belong
//! to the board, not to a camera, so every camera that sees the board shares
//! them while each keeps its own intrinsics and poses.
//!
//! - **In-plane point offsets** (free target points): each board point moved
//!   in the board plane by `(dx, dy)`, either per point or separable
//!   (`dx` per column, `dy` per row, the form a display's pixel grid takes).
//!   Gauge: a planar similarity (translation, rotation, scale), removed by
//!   holding two points (per point) or two columns and two rows
//!   (separable, which also holds the aspect ratio at its nominal value).
//! - **Out-of-plane shape** `w(x, y)`, a polynomial of degree 2..=d in the
//!   normalized board coordinates (constant and linear terms are a pose).
//! - **Cover-layer refraction**: the pattern seen through a plane-parallel
//!   layer of thickness `t` and index `n` ([`N_COVER`]). A ray from the
//!   camera centre `C` to the ink point `P` appears, at the board plane, at
//!   `P + t·d_∥·(1/|d_z| − 1/sqrt(n²|d|² − |d_∥|²))` with `d = P − C` in the
//!   board frame (exact for a flat layer up to the second order of `t/|d|`,
//!   since the angle is taken from `P` rather than the refraction point). Its
//!   paraxial part, `t(1 − 1/n)`, is a shift of the plane toward the camera
//!   and goes into the pose; the rest grows as `tan³θ`.
//! - **A generic view-angle shift** `a₃·tan²θ·d_∥/|d_z|` (outward along the
//!   in-plane view direction): the cubic term of any shift that depends on
//!   the view angle only, refraction's included (`a₃ = t(n² − 1)/(2n³)` to
//!   lowest order), so its sign and size check the refraction model.
//!
//! The camera is the seeded route's Brown-Conrady + Scheimpflug projection
//! (as in `model.rs`) with an optional per-camera image field.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use nalgebra::{DVector, Point3, RealField};
use vision_calibration_core::{Iso3, NoMeta, Uncertainty, View};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    BlockId, Manifold, Noise, Problem, Residual, SolveOptions, se3, solve,
};

use crate::model::{apply_h, monomials, tilt_h};

/// Refractive index of the cover layer (glass and acrylic are 1.49–1.52).
pub const N_COVER: f64 = 1.5;
/// Board cell size, millimetres: the column and row of a separable offset.
pub const CELL_MM: f64 = 5.0;
/// A board point (column, row) needs this many observations for a free
/// offset.
const MIN_OFFSET_OBS: usize = 8;

/// A board point, at 10 µm.
pub type Key = (i64, i64);

pub fn key_of(board_mm: [f64; 2]) -> Key {
    (
        (board_mm[0] * 100.0).round() as i64,
        (board_mm[1] * 100.0).round() as i64,
    )
}

fn col_of(mm: f64) -> i64 {
    (mm / CELL_MM).round() as i64
}

/// The form of the in-plane offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offsets {
    None,
    PerPoint,
    Separable,
}

/// Whether a target term is absent, shared by every camera, or fitted per
/// camera (the consistency check of a shared term).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Off,
    Shared,
    PerCamera,
}

impl Scope {
    fn groups(self, cams: usize) -> usize {
        match self {
            Scope::Off => 0,
            Scope::Shared => 1,
            Scope::PerCamera => cams,
        }
    }
    fn group(self, cam: usize) -> usize {
        match self {
            Scope::PerCamera => cam,
            _ => 0,
        }
    }
}

/// Which terms a joint fit has.
#[derive(Debug, Clone)]
pub struct TCfg {
    pub name: &'static str,
    pub offsets: Offsets,
    /// Out-of-plane shape, degree 2..=`bow_deg`.
    pub bow_deg: usize,
    pub bow: Scope,
    pub cover: Scope,
    pub angle: Scope,
    /// Image field, degree 2..=`field_deg` (0: none), per camera or, with
    /// `field_shared`, one field for every camera (a lens-type field).
    pub field_deg: usize,
    pub field_shared: bool,
    /// Hold every camera's intrinsics, distortion and tilt at their initial
    /// values (the route's), so a field cannot trade with them.
    pub hold_camera: bool,
    /// Evaluate the field's basis at the observed pixel instead of at the
    /// model's own projection (errors in variables: see `model.rs`).
    pub field_at_obs: bool,
}

impl TCfg {
    pub const fn base(name: &'static str) -> Self {
        Self {
            name,
            offsets: Offsets::None,
            bow_deg: 0,
            bow: Scope::Off,
            cover: Scope::Off,
            angle: Scope::Off,
            field_deg: 0,
            field_shared: false,
            hold_camera: false,
            field_at_obs: false,
        }
    }
}

/// One camera's model values.
#[derive(Debug, Clone)]
pub struct CamModel {
    pub intr: [f64; 4],
    /// `k1 k2 k3 p1 p2`; only `k1 k2` are free.
    pub dist: [f64; 5],
    pub tilt: [f64; 2],
    pub field: Vec<f64>,
}

/// The target's values (millimetres).
#[derive(Debug, Clone, Default)]
pub struct TargetModel {
    pub bow: Vec<Vec<f64>>,
    pub cover: Vec<f64>,
    pub angle: Vec<f64>,
    pub points: BTreeMap<Key, [f64; 2]>,
    pub cols: BTreeMap<i64, f64>,
    pub rows: BTreeMap<i64, f64>,
}

/// The board's normalization for the shape polynomial: centre and
/// half-size, metres.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub centre: [f64; 2],
    pub half: f64,
}

/// One camera's share of a joint fit.
pub struct JCam<'a> {
    pub size: [u32; 2],
    pub ids: Vec<usize>,
    pub views: Vec<&'a View<NoMeta>>,
    pub model: CamModel,
    pub poses: Vec<Iso3>,
}

/// One residual with its geometry.
#[derive(Debug, Clone)]
pub struct TPoint {
    pub cam: usize,
    pub view: usize,
    pub board_mm: [f64; 2],
    pub observed_px: [f64; 2],
    pub residual_px: [f64; 2],
    /// The residual mapped into the board plane (inverse Jacobian of the
    /// projection), millimetres.
    pub board_res_mm: [f64; 2],
    /// Angle between the ray and the board normal, radians.
    pub theta: f64,
    /// Unit in-plane direction from the camera's foot to the point.
    pub dir: [f64; 2],
    /// Depth of the point in the camera frame, millimetres.
    pub depth_mm: f64,
}

/// The result of a joint fit.
#[derive(Debug, Clone)]
pub struct JFit {
    pub cams: Vec<CamModel>,
    pub target: TargetModel,
    #[allow(dead_code)] // kept for inspection in a debugger
    pub poses: Vec<Vec<Iso3>>,
    pub points: Vec<TPoint>,
    pub iters: usize,
    pub cost: f64,
    pub covariance: String,
    /// Stated σ of the target scalars and camera parameters, by label.
    pub sigma: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Free {
    All,
    PosesOnly,
}

#[derive(Clone)]
struct JResidual {
    board: [f64; 2],
    observed: [f64; 2],
    obs_norm: [f64; 2],
    half: [f64; 2],
    field_at_obs: bool,
    board_norm: [f64; 2],
    bow_terms: Vec<(i32, i32)>,
    field_terms: Vec<(i32, i32)>,
    offsets: Offsets,
    cover: bool,
    angle: bool,
    dims: Vec<usize>,
}

// Block order: intr(4), dist(5), tilt(2), field(2·nf|1), bow(nb|1),
// cover(1), angle(1), off_a(2|1), off_b(1), pose(7).

impl JResidual {
    fn project<T: RealField>(&self, b: &[DVector<T>]) -> (T, T) {
        let c = |v: f64| T::from_f64(v).unwrap();
        let mm = c(1e-3);
        let (mut bx, mut by) = (c(self.board[0]), c(self.board[1]));
        match self.offsets {
            Offsets::PerPoint => {
                bx += b[7][0].clone() * mm.clone();
                by += b[7][1].clone() * mm.clone();
            }
            Offsets::Separable => {
                bx += b[7][0].clone() * mm.clone();
                by += b[8][0].clone() * mm.clone();
            }
            Offsets::None => {}
        }
        let mut w = T::zero();
        for (k, &(i, j)) in self.bow_terms.iter().enumerate() {
            w +=
                b[4][k].clone() * c(1e-3 * self.board_norm[0].powi(i) * self.board_norm[1].powi(j));
        }
        let pose = se3(&b[9]);
        let mut p = Point3::new(bx, by, w);
        if self.cover || self.angle {
            let cb = pose.inverse_transform_point(&Point3::origin());
            let d = p.clone() - cb;
            let din2 = d.x.clone() * d.x.clone() + d.y.clone() * d.y.clone();
            let dz2 = d.z.clone() * d.z.clone();
            let dz = dz2.clone().sqrt();
            let mut s = T::zero();
            if self.cover {
                let n2 = c(N_COVER * N_COVER);
                let inner = (n2 * (din2.clone() + dz2.clone()) - din2.clone()).sqrt();
                s += b[5][0].clone() * mm.clone() * (T::one() / dz.clone() - T::one() / inner);
            }
            if self.angle {
                s += b[6][0].clone() * mm.clone() * din2 / (dz2 * dz);
            }
            p.x += d.x.clone() * s.clone();
            p.y += d.y.clone() * s;
        }
        let pc = pose * p;
        let (x, y) = (pc.x.clone() / pc.z.clone(), pc.y.clone() / pc.z.clone());
        let d = &b[1];
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r4 = r2.clone() * r2.clone();
        let radial = T::one()
            + d[0].clone() * r2.clone()
            + d[1].clone() * r4.clone()
            + d[2].clone() * r4 * r2.clone();
        let two = c(2.0);
        let xy = x.clone() * y.clone();
        let xd = x.clone() * radial.clone()
            + two.clone() * d[3].clone() * xy.clone()
            + d[4].clone() * (r2.clone() + two.clone() * x.clone() * x);
        let yd = y.clone() * radial
            + d[3].clone() * (r2 + two.clone() * y.clone() * y)
            + two * d[4].clone() * xy;
        let h = tilt_h(b[2][0].clone(), b[2][1].clone());
        let (xd, yd) = apply_h(&h, xd, yd);
        let mut u = b[0][0].clone() * xd + b[0][2].clone();
        let mut v = b[0][1].clone() * yd + b[0][3].clone();
        let nf = self.field_terms.len();
        if nf > 0 {
            let (xn, yn) = if self.field_at_obs {
                (c(self.obs_norm[0]), c(self.obs_norm[1]))
            } else {
                (
                    (u.clone() - c(self.half[0])) / c(self.half[0]),
                    (v.clone() - c(self.half[1])) / c(self.half[1]),
                )
            };
            let (mut du, mut dv) = (T::zero(), T::zero());
            for (k, &(i, j)) in self.field_terms.iter().enumerate() {
                let m = xn.clone().powi(i) * yn.clone().powi(j);
                du += b[3][k].clone() * m.clone();
                dv += b[3][nf + k].clone() * m;
            }
            u += du;
            v += dv;
        }
        (u, v)
    }
}

impl Residual for JResidual {
    fn dim(&self) -> usize {
        2
    }
    fn block_dims(&self) -> Vec<usize> {
        self.dims.clone()
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let (u, v) = self.project(blocks);
        DVector::from_vec(vec![
            u - T::from_f64(self.observed[0]).unwrap(),
            v - T::from_f64(self.observed[1]).unwrap(),
        ])
    }
}

/// Two well-observed, far-apart elements of `free` (by position).
fn gauge_pair<K: Copy>(free: &[(K, [f64; 2], usize)]) -> Option<(K, K)> {
    let mut by_count: Vec<&(K, [f64; 2], usize)> = free.iter().collect();
    by_count.sort_by_key(|a| std::cmp::Reverse(a.2));
    let top = &by_count[..by_count.len().div_ceil(2)];
    let mut best: Option<(f64, K, K)> = None;
    for (i, a) in top.iter().enumerate() {
        for b in &top[i + 1..] {
            let d = (a.1[0] - b.1[0]).hypot(a.1[1] - b.1[1]);
            if best.as_ref().is_none_or(|x| d > x.0) {
                best = Some((d, a.0, b.0));
            }
        }
    }
    best.map(|(_, a, b)| (a, b))
}

/// Fit `cfg` jointly over `cams` from `init`.
pub fn fit(
    cfg: &TCfg,
    cams: &[JCam<'_>],
    init: &TargetModel,
    frame: Frame,
    free: Free,
    max_iters: usize,
) -> Result<JFit> {
    let nc = cams.len();
    let bow_terms = if cfg.bow == Scope::Off {
        Vec::new()
    } else {
        monomials(2, cfg.bow_deg)
    };
    let field_terms = monomials(2, cfg.field_deg);
    let field_terms = if cfg.field_deg == 0 {
        Vec::new()
    } else {
        field_terms
    };
    let mut p = Problem::new();
    // A residual reads each block once, so every unused slot has its own
    // fixed placeholder: field, shape, cover, angle, offset a, offset b.
    let mut dummies = Vec::with_capacity(6);
    for i in 0..6 {
        let d = p.add_block(format!("unused{i}"), &[0.0], Manifold::Euclidean)?;
        p.fix(d)?;
        dummies.push(d);
    }
    let dummy = dummies[0];
    let pad = |v: &[f64], n: usize| {
        let mut out = v.to_vec();
        out.resize(n, 0.0);
        out
    };

    // Cameras.
    let shared_field = if cfg.field_shared && !field_terms.is_empty() {
        let b = p.add_block(
            "field",
            &pad(&cams[0].model.field, 2 * field_terms.len()),
            Manifold::Euclidean,
        )?;
        if free == Free::PosesOnly {
            p.fix(b)?;
        }
        Some(b)
    } else {
        None
    };
    let mut cam_blocks = Vec::with_capacity(nc);
    for (ci, cam) in cams.iter().enumerate() {
        let m = &cam.model;
        let bi = p.add_block(format!("intr{ci}"), &m.intr, Manifold::Euclidean)?;
        p.set_labels(bi, &["fx", "fy", "cx", "cy"])?;
        let bd = p.add_block(format!("dist{ci}"), &m.dist, Manifold::Euclidean)?;
        p.set_labels(bd, &["k1", "k2", "k3", "p1", "p2"])?;
        p.fix_components(bd, &[2, 3, 4])?;
        let bt = p.add_block(format!("tilt{ci}"), &m.tilt, Manifold::Euclidean)?;
        p.set_labels(bt, &["tilt_x", "tilt_y"])?;
        let bf = if field_terms.is_empty() {
            dummy
        } else if let Some(b) = shared_field {
            b
        } else {
            p.add_block(
                format!("field{ci}"),
                &pad(&m.field, 2 * field_terms.len()),
                Manifold::Euclidean,
            )?
        };
        if free == Free::All && cfg.hold_camera {
            for b in [bi, bd, bt] {
                p.fix(b)?;
            }
        }
        if free == Free::PosesOnly {
            for b in [bi, bd, bt, bf] {
                if b != dummy && Some(b) != shared_field {
                    p.fix(b)?;
                }
            }
        }
        cam_blocks.push([bi, bd, bt, bf]);
    }

    // Target scalars and shape, per group.
    let group_blocks = |name: &str,
                        scope: Scope,
                        init: &[Vec<f64>],
                        dim: usize,
                        p: &mut Problem|
     -> Result<Vec<BlockId>> {
        let mut out = Vec::new();
        for g in 0..scope.groups(nc) {
            let v = init.get(g).cloned().unwrap_or_default();
            let b = p.add_block(format!("{name}{g}"), &pad(&v, dim), Manifold::Euclidean)?;
            if free == Free::PosesOnly {
                p.fix(b)?;
            }
            out.push(b);
        }
        Ok(out)
    };
    let bow_blocks = group_blocks("bow", cfg.bow, &init.bow, bow_terms.len().max(1), &mut p)?;
    let scal = |v: &[f64]| v.iter().map(|x| vec![*x]).collect::<Vec<_>>();
    let cover_blocks = group_blocks("cover", cfg.cover, &scal(&init.cover), 1, &mut p)?;
    for &b in &cover_blocks {
        p.set_labels(b, &["t_mm"])?;
    }
    let angle_blocks = group_blocks("angle", cfg.angle, &scal(&init.angle), 1, &mut p)?;
    for &b in &angle_blocks {
        p.set_labels(b, &["a3_mm"])?;
    }

    // In-plane offsets.
    let mut point_count: BTreeMap<Key, (usize, [f64; 2])> = BTreeMap::new();
    let mut col_count: BTreeMap<i64, usize> = BTreeMap::new();
    let mut row_count: BTreeMap<i64, usize> = BTreeMap::new();
    for cam in cams {
        for v in &cam.views {
            for q in &v.obs.points_3d {
                let mm = [q.x * 1e3, q.y * 1e3];
                point_count.entry(key_of(mm)).or_insert((0, mm)).0 += 1;
                *col_count.entry(col_of(mm[0])).or_default() += 1;
                *row_count.entry(col_of(mm[1])).or_default() += 1;
            }
        }
    }
    let mut point_blocks: BTreeMap<Key, BlockId> = BTreeMap::new();
    let mut col_blocks: BTreeMap<i64, BlockId> = BTreeMap::new();
    let mut row_blocks: BTreeMap<i64, BlockId> = BTreeMap::new();
    match cfg.offsets {
        Offsets::PerPoint => {
            let mut free_pts = Vec::new();
            for (&k, &(n, mm)) in &point_count {
                let v = init.points.get(&k).copied().unwrap_or([0.0; 2]);
                let b = p.add_block(format!("pt{}_{}", k.0, k.1), &v, Manifold::Euclidean)?;
                if free == Free::PosesOnly || n < MIN_OFFSET_OBS {
                    p.fix(b)?;
                } else {
                    free_pts.push((k, mm, n));
                }
                point_blocks.insert(k, b);
            }
            if free == Free::All {
                let (a, b) = gauge_pair(&free_pts).context("no gauge points")?;
                p.fix(point_blocks[&a])?;
                p.fix(point_blocks[&b])?;
            }
        }
        Offsets::Separable => {
            for (counts, blocks, init_v, tag) in [
                (&col_count, &mut col_blocks, &init.cols, "col"),
                (&row_count, &mut row_blocks, &init.rows, "row"),
            ] {
                let mut free_l = Vec::new();
                for (&k, &n) in counts {
                    let v = init_v.get(&k).copied().unwrap_or(0.0);
                    let b = p.add_block(format!("{tag}{k}"), &[v], Manifold::Euclidean)?;
                    if free == Free::PosesOnly || n < MIN_OFFSET_OBS {
                        p.fix(b)?;
                    } else {
                        free_l.push((k, [k as f64, 0.0], n));
                    }
                    blocks.insert(k, b);
                }
                if free == Free::All {
                    let (a, b) = gauge_pair(&free_l).context("no gauge lines")?;
                    p.fix(blocks[&a])?;
                    p.fix(blocks[&b])?;
                }
            }
        }
        Offsets::None => {}
    }

    // Poses and residuals.
    let mut pose_blocks: Vec<Vec<BlockId>> = Vec::with_capacity(nc);
    let mut residuals: Vec<(usize, usize, usize, JResidual, Vec<BlockId>)> = Vec::new();
    for (ci, cam) in cams.iter().enumerate() {
        let (w, h) = (f64::from(cam.size[0]), f64::from(cam.size[1]));
        let mut pbs = Vec::with_capacity(cam.views.len());
        for (vi, (&id, view)) in cam.ids.iter().zip(&cam.views).enumerate() {
            let pb = p.add_pose(format!("pose{ci}_{id}"), &cam.poses[vi])?;
            pbs.push(pb);
            for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
                let mm = [p3.x * 1e3, p3.y * 1e3];
                let (off_a, off_b, da, db) = match cfg.offsets {
                    Offsets::PerPoint => (point_blocks[&key_of(mm)], dummies[5], 2, 1),
                    Offsets::Separable => {
                        (col_blocks[&col_of(mm[0])], row_blocks[&col_of(mm[1])], 1, 1)
                    }
                    Offsets::None => (dummies[4], dummies[5], 1, 1),
                };
                let pick = |v: &[BlockId], scope: Scope, slot: usize| {
                    if scope == Scope::Off {
                        dummies[slot]
                    } else {
                        v[scope.group(ci)]
                    }
                };
                let [bi, bd, bt, bf] = cam_blocks[ci];
                let blocks = vec![
                    bi,
                    bd,
                    bt,
                    bf,
                    pick(&bow_blocks, cfg.bow, 1),
                    pick(&cover_blocks, cfg.cover, 2),
                    pick(&angle_blocks, cfg.angle, 3),
                    off_a,
                    off_b,
                    pb,
                ];
                let r = JResidual {
                    board: [p3.x, p3.y],
                    observed: [p2.x, p2.y],
                    obs_norm: [(p2.x - w / 2.0) / (w / 2.0), (p2.y - h / 2.0) / (h / 2.0)],
                    half: [w / 2.0, h / 2.0],
                    field_at_obs: cfg.field_at_obs,
                    board_norm: [
                        (p3.x - frame.centre[0]) / frame.half,
                        (p3.y - frame.centre[1]) / frame.half,
                    ],
                    bow_terms: bow_terms.clone(),
                    field_terms: field_terms.clone(),
                    offsets: cfg.offsets,
                    cover: cfg.cover != Scope::Off,
                    angle: cfg.angle != Scope::Off,
                    dims: vec![
                        4,
                        5,
                        2,
                        (2 * field_terms.len()).max(1),
                        bow_terms.len().max(1),
                        1,
                        1,
                        da,
                        db,
                        7,
                    ],
                };
                p.add_residual(
                    r.clone(),
                    &blocks,
                    RobustLoss::Huber {
                        scale: crate::HUBER_PX,
                    },
                    Noise::Estimated("px".into()),
                )?;
                residuals.push((ci, vi, id, r, blocks));
            }
        }
        pose_blocks.push(pbs);
    }

    let mut opts = SolveOptions::default();
    opts.max_iters = max_iters;
    let sol = solve(&p, &opts)?;
    let val = |b: BlockId| sol.value(b).map(<[f64]>::to_vec).unwrap_or_default();

    let cams_out: Vec<CamModel> = cam_blocks
        .iter()
        .map(|&[bi, bd, bt, bf]| {
            let (i, d, t) = (val(bi), val(bd), val(bt));
            CamModel {
                intr: [i[0], i[1], i[2], i[3]],
                dist: [d[0], d[1], d[2], d[3], d[4]],
                tilt: [t[0], t[1]],
                field: if bf == dummy { Vec::new() } else { val(bf) },
            }
        })
        .collect();
    let target = TargetModel {
        bow: bow_blocks.iter().map(|&b| val(b)).collect(),
        cover: cover_blocks.iter().map(|&b| val(b)[0]).collect(),
        angle: angle_blocks.iter().map(|&b| val(b)[0]).collect(),
        points: point_blocks
            .iter()
            .map(|(&k, &b)| {
                let v = val(b);
                (k, [v[0], v[1]])
            })
            .collect(),
        cols: col_blocks.iter().map(|(&k, &b)| (k, val(b)[0])).collect(),
        rows: row_blocks.iter().map(|(&k, &b)| (k, val(b)[0])).collect(),
    };
    let poses: Vec<Vec<Iso3>> = pose_blocks
        .iter()
        .map(|v| {
            v.iter()
                .map(|&b| sol.pose(b).expect("a pose block of this problem"))
                .collect()
        })
        .collect();

    // Residuals at the solution, with the board-frame residual and the view
    // geometry.
    let mut points = Vec::with_capacity(residuals.len());
    for (ci, vi, id, r, blocks) in &residuals {
        let vals: Vec<DVector<f64>> = blocks.iter().map(|&b| DVector::from_vec(val(b))).collect();
        let (u, v) = r.project(&vals);
        let res = [r.observed[0] - u, r.observed[1] - v];
        // ∂(u, v)/∂(board x, y), central differences of 1 µm.
        let hstep = 1e-6;
        let column = |axis: usize| {
            let mut rp = r.clone();
            let mut rm = r.clone();
            rp.board[axis] += hstep;
            rm.board[axis] -= hstep;
            let (up, vp) = rp.project(&vals);
            let (um, vm) = rm.project(&vals);
            [(up - um) / (2.0 * hstep), (vp - vm) / (2.0 * hstep)]
        };
        let (cx, cy) = (column(0), column(1));
        let jac = [[cx[0], cy[0]], [cx[1], cy[1]]];
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        let board_res_mm = [
            (jac[1][1] * res[0] - jac[0][1] * res[1]) / det * 1e3,
            (-jac[1][0] * res[0] + jac[0][0] * res[1]) / det * 1e3,
        ];
        let pose = &poses[*ci][*vi];
        let cb = pose.inverse_transform_point(&Point3::origin());
        let d = [r.board[0] - cb.x, r.board[1] - cb.y, -cb.z];
        let din = d[0].hypot(d[1]);
        points.push(TPoint {
            cam: *ci,
            view: *id,
            board_mm: [r.board[0] * 1e3, r.board[1] * 1e3],
            observed_px: r.observed,
            residual_px: res,
            board_res_mm,
            theta: din.atan2(d[2].abs()),
            depth_mm: 1e3 * (pose * Point3::new(r.board[0], r.board[1], 0.0)).z,
            dir: if din > 0.0 {
                [d[0] / din, d[1] / din]
            } else {
                [0.0; 2]
            },
        });
    }

    let mut sigma = BTreeMap::new();
    let covariance = match &sol.uncertainty {
        Uncertainty::Available(cov) => {
            for (i, l) in cov.labels.iter().enumerate() {
                if l.contains("t_mm") || l.contains("a3_mm") || l.starts_with("intr0") {
                    sigma.insert(l.clone(), cov.covariance[i][i].sqrt());
                }
            }
            "available".to_string()
        }
        Uncertainty::Unavailable { reason } => format!("{reason:?}"),
    };
    Ok(JFit {
        cams: cams_out,
        target,
        poses,
        points,
        iters: sol.report.num_iters,
        cost: sol.report.final_cost,
        covariance,
        sigma,
    })
}

/// Peak-to-valley and RMS of the shape `w` over the board points, µm.
pub fn bow_extent(bow: &[f64], deg: usize, frame: Frame, board_mm: &[[f64; 2]]) -> (f64, f64) {
    if bow.is_empty() || deg < 2 {
        return (0.0, 0.0);
    }
    let terms = monomials(2, deg);
    let ws: Vec<f64> = board_mm
        .iter()
        .map(|b| {
            let xn = (b[0] * 1e-3 - frame.centre[0]) / frame.half;
            let yn = (b[1] * 1e-3 - frame.centre[1]) / frame.half;
            terms
                .iter()
                .zip(bow)
                .map(|(&(i, j), c)| c * xn.powi(i) * yn.powi(j))
                .sum::<f64>()
        })
        .collect();
    // Remove the best plane (a pose) before measuring.
    let n = ws.len() as f64;
    let mut ata = nalgebra::Matrix3::<f64>::zeros();
    let mut atb = nalgebra::Vector3::<f64>::zeros();
    for (b, w) in board_mm.iter().zip(&ws) {
        let r = nalgebra::Vector3::new(1.0, b[0], b[1]);
        ata += r * r.transpose();
        atb += r * *w;
    }
    let plane = ata.try_inverse().map(|m| m * atb).unwrap_or_default();
    let flat: Vec<f64> = board_mm
        .iter()
        .zip(&ws)
        .map(|(b, w)| w - plane[0] - plane[1] * b[0] - plane[2] * b[1])
        .collect();
    let max = flat.iter().copied().fold(f64::MIN, f64::max);
    let min = flat.iter().copied().fold(f64::MAX, f64::min);
    let rms = (flat.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    (1e3 * (max - min), 1e3 * rms)
}

/// RMS of per-point offsets after removing the best planar similarity
/// (the gauge), µm.
pub fn offsets_rms(points: &BTreeMap<Key, [f64; 2]>) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    // Model: o = t + s·x + r·J·x  with J the 90° rotation.
    let mut ata = nalgebra::Matrix4::<f64>::zeros();
    let mut atb = nalgebra::Vector4::<f64>::zeros();
    let rows = |k: &Key| {
        let (x, y) = (k.0 as f64 / 100.0, k.1 as f64 / 100.0);
        [
            nalgebra::Vector4::new(1.0, 0.0, x, -y),
            nalgebra::Vector4::new(0.0, 1.0, y, x),
        ]
    };
    for (k, o) in points {
        let r = rows(k);
        for c in 0..2 {
            ata += r[c] * r[c].transpose();
            atb += r[c] * o[c];
        }
    }
    let beta = ata.try_inverse().map(|m| m * atb).unwrap_or_default();
    let mut ss = 0.0;
    for (k, o) in points {
        let r = rows(k);
        for c in 0..2 {
            ss += (o[c] - r[c].dot(&beta)).powi(2);
        }
    }
    1e3 * (ss / points.len() as f64).sqrt()
}

/// RMS of a separable offset table after removing a line (the gauge), µm.
pub fn lines_rms(lines: &BTreeMap<i64, f64>) -> f64 {
    let n = lines.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (&k, &v) in lines {
        let x = k as f64;
        sx += x;
        sy += v;
        sxx += x * x;
        sxy += x * v;
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let icpt = (sy - slope * sx) / n;
    let ss: f64 = lines
        .iter()
        .map(|(&k, &v)| (v - icpt - slope * k as f64).powi(2))
        .sum();
    1e3 * (ss / n).sqrt()
}
