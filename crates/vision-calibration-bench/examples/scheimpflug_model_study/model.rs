//! An experimental tilted-camera model written as a user residual, so terms
//! the library does not have can be tried against the data: a distortion
//! centre apart from the principal point, the tilt applied before the
//! distortion, a depth-dependent `k1`, an out-of-plane board shape, a free
//! image-space field (an upper bound for any central-camera refinement), and
//! a principal point measured from the tilt's valley.
//!
//! The baseline configuration reproduces the library's Brown-Conrady +
//! Scheimpflug projection exactly (checked against it in `main`).

use std::collections::BTreeMap;

use anyhow::Result;
use nalgebra::{DVector, Matrix3, Point3, RealField, Vector3};
use vision_calibration_core::{Iso3, NoMeta, Uncertainty, View};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    BlockId, Manifold, Noise, Problem, Residual, SolveOptions, se3, solve,
};

use crate::PointResidual;

/// Distortion coefficient names, in block order.
pub const DIST_NAMES: [&str; 9] = ["k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4"];
/// Reference depth of the depth-dependent `k1`, metres.
const Z0_M: f64 = 0.2;
/// Half-size of the board's normalized coordinates, metres.
const BOARD_HALF_M: f64 = 0.06;

/// Which terms a fit has.
#[derive(Debug, Clone)]
pub struct GenCfg {
    pub name: &'static str,
    /// Free distortion coefficients (names from [`DIST_NAMES`]).
    pub free_dist: &'static [&'static str],
    /// A distortion centre offset from the principal point (normalized).
    pub dc: bool,
    /// The tilt homography applied to the undistorted point.
    pub tilt_first: bool,
    /// `k1 + kz·(z0/Z − 1)`.
    pub depth: bool,
    /// The tilt scaled by `1 + η·r_d²` over the field (a field-dependent
    /// exit-pupil distance).
    pub pupil: bool,
    /// Lowest degree of the free image-space field.
    pub field_min_deg: usize,
    /// Hold the tilt at its initial value.
    pub fix_tilt: bool,
    /// Degree of the out-of-plane board shape (terms of degree 2..=deg).
    pub board_deg: usize,
    /// Degree of the free image-space field (terms of degree 2..=deg).
    pub field_deg: usize,
    /// Store `c̃ = c − s·f·tan(tilt)` in place of the principal point; the
    /// signs `(s_x, s_y)` pair `cx` with `tilt_y` and `cy` with `tilt_x`.
    pub reparam: Option<(f64, f64)>,
}

impl GenCfg {
    pub const fn base(name: &'static str) -> Self {
        Self {
            name,
            free_dist: &["k1", "k2"],
            dc: false,
            tilt_first: false,
            depth: false,
            pupil: false,
            field_min_deg: 2,
            fix_tilt: false,
            board_deg: 0,
            field_deg: 0,
            reparam: None,
        }
    }
}

pub fn monomials(min: usize, deg: usize) -> Vec<(i32, i32)> {
    let mut m = Vec::new();
    for d in min as i32..=deg as i32 {
        for i in (0..=d).rev() {
            m.push((i, d - i));
        }
    }
    m
}

/// Model values, physical (the principal point is the real one).
#[derive(Debug, Clone)]
pub struct GenParams {
    pub intr: [f64; 4],
    pub dist: [f64; 9],
    pub tilt: [f64; 2],
    pub dc: [f64; 2],
    pub depth: f64,
    pub pupil: f64,
    pub board: Vec<f64>,
    pub field: Vec<f64>,
    /// Board centre (metres) the board shape is expanded about.
    pub board_centre: [f64; 2],
}

/// OpenCV's tilted-sensor homography `proj_z · Ry(τy) · Rx(τx)`.
pub fn tilt_h<T: RealField>(tx: T, ty: T) -> Matrix3<T> {
    let (sx, cx) = (tx.clone().sin(), tx.cos());
    let (sy, cy) = (ty.clone().sin(), ty.cos());
    let (z, o) = (T::zero(), T::one());
    let rx = Matrix3::new(
        o.clone(),
        z.clone(),
        z.clone(),
        z.clone(),
        cx.clone(),
        sx.clone(),
        z.clone(),
        -sx,
        cx,
    );
    let ry = Matrix3::new(
        cy.clone(),
        z.clone(),
        -sy.clone(),
        z.clone(),
        o.clone(),
        z.clone(),
        sy,
        z.clone(),
        cy,
    );
    let r = ry * rx;
    let r22 = r[(2, 2)].clone();
    let p = Matrix3::new(
        r22.clone(),
        z.clone(),
        -r[(0, 2)].clone(),
        z.clone(),
        r22,
        -r[(1, 2)].clone(),
        z.clone(),
        z,
        o,
    );
    p * r
}

pub fn apply_h<T: RealField>(h: &Matrix3<T>, x: T, y: T) -> (T, T) {
    let q = h * Vector3::new(x, y, T::one());
    (q.x.clone() / q.z.clone(), q.y.clone() / q.z.clone())
}

/// The projection of one board point; see [`GenCfg`].
#[derive(Clone)]
pub struct GenResidual {
    board: [f64; 2],
    observed: [f64; 2],
    /// Normalized observed position, for the field basis.
    obs_norm: [f64; 2],
    board_terms: Vec<(i32, i32)>,
    field_terms: Vec<(i32, i32)>,
    board_centre: [f64; 2],
    dc: bool,
    tilt_first: bool,
    depth: bool,
    pupil: bool,
    reparam: Option<(f64, f64)>,
}

/// Block order: intr(4), dist(9), tilt(2), dc(2), depth(1), board(nb|1),
/// field(2·nf|1), pose(7).
const N_BLOCKS: usize = 8;

impl GenResidual {
    fn dims(&self) -> Vec<usize> {
        vec![
            4,
            9,
            2,
            2,
            2,
            self.board_terms.len().max(1),
            (2 * self.field_terms.len()).max(1),
            7,
        ]
    }

    /// Projected pixel at the blocks' values.
    fn project<T: RealField>(&self, b: &[DVector<T>]) -> (T, T) {
        let c = |v: f64| T::from_f64(v).unwrap();
        let (fx, fy) = (b[0][0].clone(), b[0][1].clone());
        let (tx, ty) = (b[2][0].clone(), b[2][1].clone());
        let (mut cx, mut cy) = (b[0][2].clone(), b[0][3].clone());
        if let Some((sx, sy)) = self.reparam {
            cx += c(sx) * fx.clone() * ty.clone().tan();
            cy += c(sy) * fy.clone() * tx.clone().tan();
        }
        // Board point, with the out-of-plane shape.
        let (bx, by) = (self.board[0], self.board[1]);
        let mut w = T::zero();
        if !self.board_terms.is_empty() {
            let xn = (bx - self.board_centre[0]) / BOARD_HALF_M;
            let yn = (by - self.board_centre[1]) / BOARD_HALF_M;
            for (k, &(i, j)) in self.board_terms.iter().enumerate() {
                w += b[5][k].clone() * c(1e-3 * xn.powi(i) * yn.powi(j));
            }
        }
        let pose = se3(&b[7]);
        let pc = pose * Point3::new(c(bx), c(by), w);
        let (mut x, mut y) = (pc.x.clone() / pc.z.clone(), pc.y.clone() / pc.z.clone());
        let h = tilt_h(tx.clone(), ty.clone());
        if self.tilt_first {
            (x, y) = apply_h(&h, x, y);
        }
        let d = &b[1];
        let mut k1 = d[0].clone();
        if self.depth {
            k1 += b[4][0].clone() * (c(Z0_M) / pc.z.clone() - T::one());
        }
        let (ox, oy) = if self.dc {
            (b[3][0].clone(), b[3][1].clone())
        } else {
            (T::zero(), T::zero())
        };
        let (ux, uy) = (x - ox.clone(), y - oy.clone());
        let r2 = ux.clone() * ux.clone() + uy.clone() * uy.clone();
        let r4 = r2.clone() * r2.clone();
        let radial = T::one()
            + k1 * r2.clone()
            + d[1].clone() * r4.clone()
            + d[2].clone() * r4.clone() * r2.clone();
        let two = c(2.0);
        let xy = ux.clone() * uy.clone();
        let mut xd = ux.clone() * radial.clone()
            + two.clone() * d[3].clone() * xy.clone()
            + d[4].clone() * (r2.clone() + two.clone() * ux.clone() * ux)
            + d[5].clone() * r2.clone()
            + d[6].clone() * r4.clone()
            + ox;
        let mut yd = uy.clone() * radial
            + d[3].clone() * (r2.clone() + two.clone() * uy.clone() * uy)
            + two * d[4].clone() * xy
            + d[7].clone() * r2
            + d[8].clone() * r4
            + oy;
        if !self.tilt_first {
            let h = if self.pupil {
                let s = T::one()
                    + b[4][1].clone() * (xd.clone() * xd.clone() + yd.clone() * yd.clone());
                tilt_h(tx * s.clone(), ty * s)
            } else {
                h
            };
            (xd, yd) = apply_h(&h, xd, yd);
        }
        let mut u = fx * xd + cx;
        let mut v = fy * yd + cy;
        let nf = self.field_terms.len();
        for (k, &(i, j)) in self.field_terms.iter().enumerate() {
            let m = c(self.obs_norm[0].powi(i) * self.obs_norm[1].powi(j));
            u += b[6][k].clone() * m.clone();
            v += b[6][nf + k].clone() * m;
        }
        (u, v)
    }
}

impl Residual for GenResidual {
    fn dim(&self) -> usize {
        2
    }
    fn block_dims(&self) -> Vec<usize> {
        self.dims()
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let (u, v) = self.project(blocks);
        DVector::from_vec(vec![
            u - T::from_f64(self.observed[0]).unwrap(),
            v - T::from_f64(self.observed[1]).unwrap(),
        ])
    }
}

/// A fit of the experimental model.
#[derive(Debug, Clone)]
pub struct GenFit {
    pub params: GenParams,
    pub residuals: Vec<PointResidual>,
    pub iters: usize,
    pub cost: f64,
    pub covariance: String,
    pub sigma: BTreeMap<String, f64>,
    pub corr: BTreeMap<String, f64>,
}

/// Which model blocks a solve frees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Free {
    /// The model's free terms and every pose.
    All,
    /// Only the poses.
    PosesOnly,
    /// The model's free terms but the field, held, and every pose.
    AllButField,
}

/// Fit `cfg` to `views` (indices `ids`) from `init` and `poses`.
// A study helper: the arguments are the fit's independent inputs, and a
// struct to carry them would only be built at each call.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    cfg: &GenCfg,
    ids: &[usize],
    views: &[View<NoMeta>],
    init: &GenParams,
    poses: &[Iso3],
    image_size: [u32; 2],
    free: Free,
    max_iters: usize,
) -> Result<GenFit> {
    let board_terms = monomials(2, cfg.board_deg);
    let field_terms = monomials(cfg.field_min_deg, cfg.field_deg);
    let mut p = Problem::new();
    // The reparameterized principal point.
    let mut intr = init.intr;
    if let Some((sx, sy)) = cfg.reparam {
        intr[2] -= sx * intr[0] * init.tilt[1].tan();
        intr[3] -= sy * intr[1] * init.tilt[0].tan();
    }
    let b_intr = p.add_block("intr", &intr, Manifold::Euclidean)?;
    p.set_labels(b_intr, &["fx", "fy", "cx", "cy"])?;
    let b_dist = p.add_block("dist", &init.dist, Manifold::Euclidean)?;
    p.set_labels(b_dist, &DIST_NAMES)?;
    let b_tilt = p.add_block("tilt", &init.tilt, Manifold::Euclidean)?;
    p.set_labels(b_tilt, &["tilt_x", "tilt_y"])?;
    let b_dc = p.add_block("dc", &init.dc, Manifold::Euclidean)?;
    let b_depth = p.add_block("depth", &[init.depth, init.pupil], Manifold::Euclidean)?;
    let pad = |v: &[f64], n: usize| -> Vec<f64> {
        if n == 0 {
            vec![0.0]
        } else {
            let mut out = v.to_vec();
            out.resize(n, 0.0);
            out
        }
    };
    let b_board = p.add_block(
        "board",
        &pad(&init.board, board_terms.len()),
        Manifold::Euclidean,
    )?;
    let b_field = p.add_block(
        "field",
        &pad(&init.field, 2 * field_terms.len()),
        Manifold::Euclidean,
    )?;
    let model_blocks = [b_intr, b_dist, b_tilt, b_dc, b_depth, b_board, b_field];
    if free == Free::PosesOnly {
        for b in model_blocks {
            p.fix(b)?;
        }
    } else {
        let fixed: Vec<usize> = DIST_NAMES
            .iter()
            .enumerate()
            .filter(|(_, n)| !cfg.free_dist.contains(n))
            .map(|(i, _)| i)
            .collect();
        p.fix_components(b_dist, &fixed)?;
        if !cfg.dc {
            p.fix(b_dc)?;
        }
        let depth_fixed: Vec<usize> = [(0, cfg.depth), (1, cfg.pupil)]
            .iter()
            .filter(|(_, on)| !on)
            .map(|(i, _)| *i)
            .collect();
        p.fix_components(b_depth, &depth_fixed)?;
        if cfg.fix_tilt {
            p.fix(b_tilt)?;
        }
        if free == Free::AllButField {
            p.fix(b_field)?;
        }
        if board_terms.is_empty() {
            p.fix(b_board)?;
        }
        if field_terms.is_empty() {
            p.fix(b_field)?;
        }
    }
    let (w, h) = (f64::from(image_size[0]), f64::from(image_size[1]));
    let mut pose_blocks: Vec<BlockId> = Vec::new();
    let mut residuals: Vec<(usize, GenResidual)> = Vec::new();
    for ((&id, view), pose) in ids.iter().zip(views).zip(poses) {
        let pb = p.add_pose(format!("pose{id}"), pose)?;
        pose_blocks.push(pb);
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            let r = GenResidual {
                board: [p3.x, p3.y],
                observed: [p2.x, p2.y],
                obs_norm: [(p2.x - w / 2.0) / (w / 2.0), (p2.y - h / 2.0) / (h / 2.0)],
                board_terms: board_terms.clone(),
                field_terms: field_terms.clone(),
                board_centre: init.board_centre,
                dc: cfg.dc,
                tilt_first: cfg.tilt_first,
                depth: cfg.depth,
                pupil: cfg.pupil,
                reparam: cfg.reparam,
            };
            let mut blocks = model_blocks.to_vec();
            blocks.push(pb);
            debug_assert_eq!(blocks.len(), N_BLOCKS);
            p.add_residual(
                r.clone(),
                &blocks,
                RobustLoss::Huber {
                    scale: crate::HUBER_PX,
                },
                Noise::Estimated("px".into()),
            )?;
            residuals.push((id, r));
        }
    }
    let mut opts = SolveOptions::default();
    opts.max_iters = max_iters;
    let sol = solve(&p, &opts)?;
    let val = |b: BlockId| sol.value(b).map(<[f64]>::to_vec).unwrap_or_default();
    let mut intr_out = [0.0; 4];
    intr_out.copy_from_slice(&val(b_intr));
    let mut dist = [0.0; 9];
    dist.copy_from_slice(&val(b_dist));
    let tilt_v = val(b_tilt);
    let tilt = [tilt_v[0], tilt_v[1]];
    let mut phys = intr_out;
    if let Some((sx, sy)) = cfg.reparam {
        phys[2] += sx * phys[0] * tilt[1].tan();
        phys[3] += sy * phys[1] * tilt[0].tan();
    }
    let dc_v = val(b_dc);
    let params = GenParams {
        intr: phys,
        dist,
        tilt,
        dc: [dc_v[0], dc_v[1]],
        depth: val(b_depth)[0],
        pupil: val(b_depth)[1],
        board: if board_terms.is_empty() {
            Vec::new()
        } else {
            val(b_board)
        },
        field: if field_terms.is_empty() {
            Vec::new()
        } else {
            val(b_field)
        },
        board_centre: init.board_centre,
    };
    let poses_out: Vec<Iso3> = pose_blocks
        .iter()
        .map(|&b| sol.pose(b).expect("a pose block of this problem"))
        .collect();

    // Residuals at the solution.
    let model_vals: Vec<DVector<f64>> = model_blocks
        .iter()
        .map(|&b| DVector::from_vec(val(b)))
        .collect();
    let mut pts = Vec::with_capacity(residuals.len());
    let pose_of: BTreeMap<usize, usize> = ids.iter().enumerate().map(|(k, &id)| (id, k)).collect();
    for (id, r) in &residuals {
        let pose = &poses_out[pose_of[id]];
        let mut blocks = model_vals.clone();
        let q = pose.rotation.coords;
        let t = pose.translation.vector;
        blocks.push(DVector::from_vec(vec![q.x, q.y, q.z, q.w, t.x, t.y, t.z]));
        let (u, v) = r.project(&blocks);
        let pc = pose * Point3::new(r.board[0], r.board[1], 0.0);
        pts.push(PointResidual {
            view: *id,
            board_mm: [r.board[0] * 1e3, r.board[1] * 1e3],
            observed_px: r.observed,
            residual_px: [r.observed[0] - u, r.observed[1] - v],
            cam_mm: [pc.x * 1e3, pc.y * 1e3, pc.z * 1e3],
            blur_px: None,
        });
    }

    let mut sigma = BTreeMap::new();
    let mut corr = BTreeMap::new();
    let covariance = match &sol.uncertainty {
        Uncertainty::Available(cov) => {
            let idx = |name: &str| {
                cov.labels
                    .iter()
                    .position(|l| l.ends_with(&format!(".{name}")))
            };
            for (i, l) in cov.labels.iter().enumerate() {
                if !l.starts_with("pose") {
                    sigma.insert(l.clone(), cov.covariance[i][i].sqrt());
                }
            }
            for (a, b) in [
                ("cy", "tilt_x"),
                ("cx", "tilt_y"),
                ("fx", "k1"),
                ("fx", "tilt_x"),
            ] {
                if let (Some(i), Some(j)) = (idx(a), idx(b)) {
                    corr.insert(
                        format!("{a}/{b}"),
                        cov.covariance[i][j] / (cov.covariance[i][i] * cov.covariance[j][j]).sqrt(),
                    );
                }
            }
            "available".to_string()
        }
        Uncertainty::Unavailable { reason } => format!("{reason:?}"),
    };
    Ok(GenFit {
        params,
        residuals: pts,
        iters: sol.report.num_iters,
        cost: sol.report.final_cost,
        covariance,
        sigma,
        corr,
    })
}
