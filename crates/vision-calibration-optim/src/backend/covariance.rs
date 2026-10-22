//! The covariance of a solve's free parameters at its optimum.
//!
//! A backend supplies a [`RawLinearization`]: the residual and Jacobian of the
//! IR with every robust loss removed, at the solution, and the Jacobian
//! column of each IR block coordinate. From there [`estimate`] works on IR
//! terms alone, so both backends report the same covariance.
//!
//! The estimate is Huber's asymptotic covariance of an M-estimator, pooled
//! per residual family `f` (reprojection, laser plane, laser line):
//!
//! ```text
//! Σ = H⁻¹ G H⁻¹,   H = Σ_f κ_f A_f + A_p,   G = Σ_f s_f A_f + B_p
//! A_f = Σ_{i∈f} JᵢᵀJᵢ,   A_p = Σ_priors JᵢᵀJᵢ,   B_p = Σ_priors σᵢ² JᵢᵀJᵢ
//! κ_f = E_f[ψ′],   s_f = E_f[ψ²] · m / (m − n)
//! ```
//!
//! with `ψ = ρ′ r` the residual's influence and `ψ′ = tr(ρ′ I + 2ρ″ r rᵀ)`
//! its derivative, both averaged over the family's scalar residuals. `H` is
//! the curvature of the objective the solver minimized and `G` the scatter
//! of its gradient, so the sandwich holds for the weights the solve used,
//! with priors as known information. `σᵢ` is a prior's residual standard
//! deviation: 1 for a prior in its own unit, the pixel noise for a
//! robot-pose prior weighed against the reprojection residuals. With one family and no priors it is
//! `σ_f² A_f⁻¹`, `σ_f² = s_f / κ_f²`, and without a robust loss the classical
//! `Σ‖r‖² / (m − n) · (JᵀJ)⁻¹`. With a loss, the outliers' bounded influence
//! enters as it scatters the estimate: the IRLS form `σ̂² (JᵀWJ)⁻¹`
//! overstates `σ` about twofold under Huber with 5 % outliers, and a
//! per-residual sandwich, though calibrated in `σ`, is too noisy run to run.
//! `H` is normalized to a unit diagonal and factored with the sparse
//! Cholesky plan; a singular or near-singular `H` is diagnosed with a dense
//! eigendecomposition and reported, not inverted.
//!
//! Reprojection residuals whitened by stated point covariances
//! ([`crate::ir::ReprojSqrtInfo::Whitened`]) are one family with unit
//! covariance up to a common factor: nothing above changes, and the family's
//! `σ` is that factor, the dimensionless `σ̂₀` ([`FamilyNoise::whitened`]).
//!
//! Coordinates are those of the IR blocks: a Euclidean block's components,
//! the right-perturbation tangent `[δθ, δt]` of an SE3 block, and the ambient
//! components of an S² unit vector (its tangent covariance mapped through the
//! tangent basis).

use std::collections::HashMap;

use faer::Mat;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, SparseColMatRef};
use nalgebra::{DMatrix, DVector, SymmetricEigen, Vector3};

use crate::backend::normal_equations::NormalEquations;
use crate::backend::s2;
use crate::ir::{FactorKind, ManifoldKind, ProblemIR};
use vision_calibration_core::ResidualKind;

/// `H` is rank-deficient when its unit-diagonal form has an eigenvalue below
/// this fraction of the largest.
const RANK_TOLERANCE: f64 = 1e-12;
/// A null vector's coordinates of at least this fraction of its largest
/// component are reported as unobservable.
const NULL_VECTOR_SHARE: f64 = 0.1;
/// `σ̂ = MAD_TO_SIGMA · median |rᵢⱼ|`: the median absolute residual of a
/// zero-mean Gaussian is `0.6745 σ`.
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;
/// Bound tolerance relative to `max(1, |bound|)`.
const BOUND_TOLERANCE: f64 = 1e-10;

/// The residual and Jacobian of a loss-free IR at the solution.
pub(crate) struct RawLinearization {
    /// Residuals, IR residual blocks in order, each block's rows consecutive.
    pub residual: Mat<f64>,
    /// Jacobian over the backend's columns.
    pub jacobian: SparseColMat<usize, f64>,
    /// Per IR parameter block (by id), the Jacobian column of each tangent
    /// coordinate; `None` where the coordinate is fixed.
    pub columns: Vec<Vec<Option<usize>>>,
}

/// The residual kind (one noise level each) of a factor; `None` for a
/// prior.
fn residual_kind(factor: &FactorKind) -> Option<ResidualKind> {
    match factor {
        FactorKind::ReprojPoint { .. } => Some(ResidualKind::Reprojection),
        FactorKind::LaserPointToPlane { .. } => Some(ResidualKind::LaserPlane),
        FactorKind::LaserLineDistance { .. } => Some(ResidualKind::LaserLine),
        FactorKind::Se3TangentPrior { .. } => None,
    }
}

/// One estimated scalar: coordinate `index` of IR parameter block `block`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Coordinate {
    /// IR parameter block id.
    pub block: usize,
    /// Euclidean: the component. SE3: the tangent `[δθ, δt]`. S²: the
    /// ambient component.
    pub index: usize,
}

/// The noise level of one residual family, estimated from its residuals.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FamilyNoise {
    pub family: ResidualKind,
    /// Scalar residuals in the family.
    pub num_residuals: usize,
    /// `σ̂_f`, in the family's residual unit: the Gaussian-equivalent
    /// standard deviation of one scalar residual from the median absolute
    /// residual, so outliers do not inflate it.
    pub sigma: f64,
    /// Whether the family's residuals are whitened by stated covariances,
    /// which makes `sigma` the dimensionless `σ̂₀`.
    pub whitened: bool,
}

/// The covariance of a solve's free coordinates.
#[derive(Debug, Clone)]
pub(crate) struct CovarianceEstimate {
    /// The coordinates, in matrix order: IR block order, then coordinate.
    pub coordinates: Vec<Coordinate>,
    /// Their covariance.
    pub matrix: DMatrix<f64>,
    /// Scalar residuals, priors included.
    pub num_residuals: usize,
    /// Residuals minus free tangent coordinates.
    pub dof: usize,
}

/// Why a solve has no covariance.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UnavailableReason {
    /// The solve stopped on its iteration budget.
    NotConverged,
    /// These coordinates are (nearly) unobservable.
    RankDeficient { unobservable: Vec<Coordinate> },
    /// These coordinates sit on a bound.
    AtBound { coordinates: Vec<Coordinate> },
    /// No more observation residuals than free coordinates.
    NoRedundancy,
}

/// The uncertainty of a solve.
#[derive(Debug, Clone)]
pub(crate) enum ParamUncertainty {
    Available(Box<CovarianceEstimate>),
    Unavailable(UnavailableReason),
}

/// Tangent dimension of a block.
pub(crate) fn tangent_dim(manifold: ManifoldKind, dim: usize) -> usize {
    match manifold {
        ManifoldKind::Euclidean => dim,
        ManifoldKind::SE3 => 6,
        ManifoldKind::SO3 => 3,
        ManifoldKind::S2 => 2,
    }
}

/// The covariance of `ir`'s free coordinates at the solution `params`.
///
/// `raw` is `None` when every coordinate is fixed.
pub(crate) fn estimate(
    ir: &ProblemIR,
    params: &HashMap<String, DVector<f64>>,
    raw: Option<RawLinearization>,
    converged: bool,
) -> ParamUncertainty {
    if !converged {
        return ParamUncertainty::Unavailable(UnavailableReason::NotConverged);
    }
    let at_bound = coordinates_at_bounds(ir, params);
    if !at_bound.is_empty() {
        return ParamUncertainty::Unavailable(UnavailableReason::AtBound {
            coordinates: at_bound,
        });
    }
    let Some(raw) = raw else {
        let m = ir.residuals.iter().map(|r| r.residual_dim).sum();
        return ParamUncertainty::Available(Box::new(CovarianceEstimate {
            coordinates: Vec::new(),
            matrix: DMatrix::zeros(0, 0),
            num_residuals: m,
            dof: m,
        }));
    };

    // Free tangent coordinates in IR order, and the backend column of each.
    let free: Vec<(Coordinate, usize)> = raw
        .columns
        .iter()
        .enumerate()
        .flat_map(|(block, cols)| {
            cols.iter()
                .enumerate()
                .filter_map(move |(index, col)| col.map(|c| (Coordinate { block, index }, c)))
        })
        .collect();
    let n = free.len();

    let blocks = residual_blocks(ir, &raw.residual);
    let m: usize = blocks.iter().map(|b| b.rows.len()).sum();
    let m_obs: usize = blocks
        .iter()
        .filter(|b| b.family.is_some())
        .map(|b| b.rows.len())
        .sum();
    if m_obs > 0 && m <= n {
        return ParamUncertainty::Unavailable(UnavailableReason::NoRedundancy);
    }
    let dof_scale = if m_obs > 0 {
        m as f64 / (m - n) as f64
    } else {
        1.0
    };

    // Work in the backend's columns; columns that are not free coordinates
    // (fixed components a backend keeps, with zero Jacobian) are decoupled
    // and get a unit diagonal.
    let mut normal = NormalEquations::new(raw.jacobian.symbolic())
        .expect("the solve planned a normal matrix of this pattern");
    let moments = family_moments(&blocks, dof_scale);
    let mut h = weighted_normal(&mut normal, &raw.jacobian, &blocks, |b| moments.kappa(b));
    // `G ∝ H` when every block has the same `s / κ` (one family and no
    // priors, or priors whose residual variance is the family's): then
    // `Σ = (s / κ) H⁻¹` and `G` need not be formed.
    let ratios: Vec<f64> = blocks
        .iter()
        .map(|b| moments.s(b) / moments.kappa(b))
        .collect();
    let proportional = ratios
        .iter()
        .all(|r| (r - ratios[0]).abs() <= 1e-12 * ratios[0].abs());
    let g = (!proportional)
        .then(|| weighted_normal(&mut normal, &raw.jacobian, &blocks, |b| moments.s(b)));
    let ncols = raw.jacobian.ncols();
    let mut is_free = vec![false; ncols];
    for &(_, c) in &free {
        is_free[c] = true;
    }
    let pattern = normal.pattern();
    for (c, _) in is_free.iter().enumerate().filter(|(_, free)| !**free) {
        h[pattern.col_range(c).start] = 1.0;
    }

    let h_inverse = match covariance(&normal, &mut h, ncols) {
        Ok(inverse) => inverse,
        Err(null) => {
            let by_column: HashMap<usize, Coordinate> =
                free.iter().map(|&(c, col)| (col, c)).collect();
            return ParamUncertainty::Unavailable(UnavailableReason::RankDeficient {
                unobservable: ambient_coordinates(
                    ir,
                    null.iter().filter_map(|col| by_column.get(col).copied()),
                ),
            });
        }
    };
    let sigma = match g {
        None => h_inverse * ratios[0],
        Some(g) => {
            let g = dense_symmetric(&normal, &g, ncols);
            &h_inverse * g * &h_inverse
        }
    };
    let tangent_coords: Vec<Coordinate> = free.iter().map(|&(c, _)| c).collect();
    let sigma_tangent = DMatrix::from_fn(n, n, |a, b| sigma[(free[a].1, free[b].1)]);
    let (coordinates, matrix) = to_ambient(ir, params, &tangent_coords, &sigma_tangent);
    ParamUncertainty::Available(Box::new(CovarianceEstimate {
        coordinates,
        matrix,
        num_residuals: m,
        dof: m.saturating_sub(n),
    }))
}

/// The noise level of each residual family in `raw`'s residuals; empty
/// without a linearization (every coordinate fixed).
pub(crate) fn noise(ir: &ProblemIR, raw: Option<&RawLinearization>) -> Vec<FamilyNoise> {
    let whitened = ir.reprojection_whitened();
    raw.map(|raw| family_noise(&residual_blocks(ir, &raw.residual), &raw.residual, whitened))
        .unwrap_or_default()
}

/// Free coordinates within [`BOUND_TOLERANCE`] of a bound.
fn coordinates_at_bounds(
    ir: &ProblemIR,
    params: &HashMap<String, DVector<f64>>,
) -> Vec<Coordinate> {
    let mut out = Vec::new();
    for (block, param) in ir.params.iter().enumerate() {
        let (Some(bounds), Some(values)) = (&param.bounds, params.get(&param.name)) else {
            continue;
        };
        for bound in bounds {
            if param.fixed.contains(bound.idx) {
                continue;
            }
            let v = values[bound.idx];
            let near = |b: f64| (v - b).abs() <= BOUND_TOLERANCE * b.abs().max(1.0);
            if near(bound.lower) || near(bound.upper) {
                out.push(Coordinate {
                    block,
                    index: bound.idx,
                });
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// One residual block: its rows, family, squared norm and loss weights.
struct BlockTerms {
    rows: std::ops::Range<usize>,
    /// `None` for a prior.
    family: Option<ResidualKind>,
    /// A prior's residual variance, `residual_sigma²`; 1 for an
    /// observation, whose variance its family's moments carry.
    prior_variance: f64,
    /// `s = ‖rᵢ‖²`, and `ρ′`, `ρ″` there.
    s: f64,
    rho1: f64,
    rho2: f64,
}

/// Every residual block's terms. Rows are the IR's residual blocks in order.
fn residual_blocks(ir: &ProblemIR, residual: &Mat<f64>) -> Vec<BlockTerms> {
    let mut out = Vec::with_capacity(ir.residuals.len());
    let mut start = 0;
    for block in &ir.residuals {
        let rows = start..start + block.residual_dim;
        start = rows.end;
        let s: f64 = rows.clone().map(|r| residual[(r, 0)].powi(2)).sum();
        let prior_variance = match block.factor {
            FactorKind::Se3TangentPrior { residual_sigma, .. } => residual_sigma * residual_sigma,
            _ => 1.0,
        };
        out.push(BlockTerms {
            rows,
            family: residual_kind(&block.factor),
            prior_variance,
            s,
            rho1: block.loss.rho_prime(s),
            rho2: block.loss.rho_second(s),
        });
    }
    debug_assert_eq!(
        start,
        residual.nrows(),
        "one row per scalar residual, in IR order"
    );
    out
}

/// One residual family and its pooled moments.
struct Group {
    family: ResidualKind,
    /// `κ = E[ψ′]`.
    kappa: f64,
    /// `s = E[ψ²] · m / (m − n)`.
    s: f64,
}

struct Moments {
    groups: Vec<Group>,
}

impl Moments {
    fn group(&self, family: ResidualKind) -> &Group {
        self.groups
            .iter()
            .find(|g| g.family == family)
            .expect("every family is grouped")
    }

    /// The block's weight in `H`: its family's `κ`, 1 for a prior.
    fn kappa(&self, block: &BlockTerms) -> f64 {
        block.family.map_or(1.0, |f| self.group(f).kappa)
    }

    /// The block's weight in `G`: its family's `s`, a prior's residual
    /// variance.
    fn s(&self, block: &BlockTerms) -> f64 {
        block
            .family
            .map_or(block.prior_variance, |f| self.group(f).s)
    }
}

/// Huber's pooled moments per group, for i.i.d. residuals within a family:
/// `ψ = ρ′ r` and `ψ′` the trace of `ρ′ I + 2ρ″ r rᵀ` per scalar residual,
/// with `E[ψ²]` scaled by `m / (m − n)` for the degrees of freedom the fit
/// absorbs.
fn family_moments(blocks: &[BlockTerms], dof_scale: f64) -> Moments {
    // (family, Σψ², Σψ′, scalar residuals)
    let mut sums: Vec<(ResidualKind, f64, f64, usize)> = Vec::new();
    for block in blocks {
        let Some(family) = block.family else {
            continue;
        };
        let i = match sums.iter().position(|g| g.0 == family) {
            Some(i) => i,
            None => {
                sums.push((family, 0.0, 0.0, 0));
                sums.len() - 1
            }
        };
        let g = &mut sums[i];
        g.1 += block.rho1 * block.rho1 * block.s;
        g.2 += block.rows.len() as f64 * block.rho1 + 2.0 * block.rho2 * block.s;
        g.3 += block.rows.len();
    }
    let groups = sums
        .into_iter()
        .map(|(family, psi2, dpsi, count)| Group {
            family,
            kappa: (dpsi / count as f64).max(f64::MIN_POSITIVE),
            s: (dof_scale * psi2 / count as f64).max(f64::MIN_POSITIVE),
        })
        .collect();
    Moments { groups }
}

/// `Σ wᵢ JᵢᵀJᵢ` as a lower triangle on `normal`'s pattern, with the weight
/// `w` of each block.
fn weighted_normal(
    normal: &mut NormalEquations,
    jac: &SparseColMat<usize, f64>,
    blocks: &[BlockTerms],
    weight: impl Fn(&BlockTerms) -> f64,
) -> Vec<f64> {
    // `√w` per row, so the assembled `J̃ᵀJ̃` is `Σ wᵢ JᵢᵀJᵢ`.
    let mut row_scale = vec![0.0; jac.nrows()];
    for block in blocks {
        row_scale[block.rows.clone()].fill(weight(block).sqrt());
    }
    let symbolic = jac.symbolic();
    let mut values = jac.val().to_vec();
    for col in 0..jac.ncols() {
        for (pos, &row) in symbolic
            .col_range(col)
            .zip(symbolic.row_idx_of_col_raw(col))
        {
            values[pos] *= row_scale[row];
        }
    }
    let unit = vec![1.0; jac.ncols()];
    let zero = Mat::<f64>::zeros(jac.nrows(), 1);
    normal.assemble(SparseColMatRef::new(symbolic, &values), &zero, &unit);
    normal.hessian().to_vec()
}

/// `σ̂` per residual family present, from the median absolute scalar
/// residual.
fn family_noise(blocks: &[BlockTerms], residual: &Mat<f64>, whitened: bool) -> Vec<FamilyNoise> {
    let mut by_family: Vec<(ResidualKind, Vec<f64>)> = Vec::new();
    for block in blocks {
        let Some(family) = block.family else {
            continue;
        };
        let values = match by_family.iter_mut().find(|(f, _)| *f == family) {
            Some((_, values)) => values,
            None => {
                by_family.push((family, Vec::new()));
                &mut by_family.last_mut().expect("just pushed").1
            }
        };
        values.extend(block.rows.clone().map(|r| residual[(r, 0)].abs()));
    }
    by_family.sort_by_key(|(f, _)| *f);
    by_family
        .into_iter()
        .map(|(family, mut values)| {
            let mid = values.len() / 2;
            let (_, &mut upper, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
            let median = if values.len() % 2 == 0 {
                let lower = values[..mid].iter().copied().fold(f64::MIN, f64::max);
                0.5 * (lower + upper)
            } else {
                upper
            };
            FamilyNoise {
                family,
                num_residuals: values.len(),
                sigma: MAD_TO_SIGMA * median,
                whitened: whitened && family == ResidualKind::Reprojection,
            }
        })
        .collect()
}

/// `H⁻¹` for `H` given by its lower triangle, or the columns that make it
/// singular. Inverted in the unit-diagonal frame `D H D`.
fn covariance(
    normal: &NormalEquations,
    h: &mut [f64],
    n: usize,
) -> Result<DMatrix<f64>, Vec<usize>> {
    let pattern = normal.pattern();
    let d: Vec<f64> = (0..n)
        .map(|j| {
            let v = h[pattern.col_range(j).start];
            if v > 0.0 { 1.0 / v.sqrt() } else { 1.0 }
        })
        .collect();
    for j in 0..n {
        for (pos, &i) in pattern.col_range(j).zip(pattern.row_idx_of_col_raw(j)) {
            h[pos] *= d[i] * d[j];
        }
    }
    let inverse = invert(normal, h, n)?;
    Ok(DMatrix::from_fn(n, n, |i, j| d[i] * d[j] * inverse[(i, j)]))
}

/// `Ĥₙ⁻¹` for the unit-diagonal `Ĥₙ` given by its lower triangle, or the
/// tangent coordinates that make it singular.
fn invert(normal: &NormalEquations, lower: &[f64], n: usize) -> Result<DMatrix<f64>, Vec<usize>> {
    let inverse = normal.factor(lower).and_then(|llt| {
        let inv = llt.solve(&Mat::<f64>::identity(n, n));
        let well_posed = (0..n).all(|i| {
            let v = inv[(i, i)];
            v.is_finite() && v < 1.0 / RANK_TOLERANCE
        });
        well_posed.then(|| DMatrix::from_fn(n, n, |i, j| inv[(i, j)]))
    });
    match inverse {
        Some(inv) => Ok(inv),
        None => eigen_inverse(&dense_symmetric(normal, lower, n)),
    }
}

/// The dense symmetric matrix with lower triangle `lower`.
fn dense_symmetric(normal: &NormalEquations, lower: &[f64], n: usize) -> DMatrix<f64> {
    let pattern = normal.pattern();
    let mut dense = DMatrix::zeros(n, n);
    for j in 0..n {
        for (pos, &i) in pattern.col_range(j).zip(pattern.row_idx_of_col_raw(j)) {
            dense[(i, j)] = lower[pos];
            dense[(j, i)] = lower[pos];
        }
    }
    dense
}

/// The inverse through an eigendecomposition, or the coordinates carrying
/// the eigenvectors whose eigenvalues fall below [`RANK_TOLERANCE`].
fn eigen_inverse(h: &DMatrix<f64>) -> Result<DMatrix<f64>, Vec<usize>> {
    let eigen = SymmetricEigen::new(h.clone());
    let max = eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let mut null: Vec<usize> = Vec::new();
    for (k, &lambda) in eigen.eigenvalues.iter().enumerate() {
        if lambda.is_finite() && lambda > RANK_TOLERANCE * max {
            continue;
        }
        let v = eigen.eigenvectors.column(k);
        let largest = v.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        null.extend(
            v.iter()
                .enumerate()
                .filter(|(_, x)| x.abs() >= NULL_VECTOR_SHARE * largest)
                .map(|(i, _)| i),
        );
    }
    if !null.is_empty() {
        null.sort_unstable();
        null.dedup();
        return Err(null);
    }
    let inv_lambda = eigen.eigenvalues.map(|l| 1.0 / l);
    Ok(&eigen.eigenvectors * DMatrix::from_diagonal(&inv_lambda) * eigen.eigenvectors.transpose())
}

/// Tangent coordinates as reported: an S² block's tangent coordinates
/// become its three ambient components.
fn ambient_coordinates(
    ir: &ProblemIR,
    coords: impl Iterator<Item = Coordinate>,
) -> Vec<Coordinate> {
    let mut out: Vec<Coordinate> = coords
        .flat_map(|c| {
            if ir.params[c.block].manifold == ManifoldKind::S2 {
                (0..3)
                    .map(|index| Coordinate {
                        block: c.block,
                        index,
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Map the tangent covariance to the reported coordinates: S² blocks go
/// through their tangent basis `[b₁ b₂]` at the solution.
fn to_ambient(
    ir: &ProblemIR,
    params: &HashMap<String, DVector<f64>>,
    tangent: &[Coordinate],
    sigma: &DMatrix<f64>,
) -> (Vec<Coordinate>, DMatrix<f64>) {
    if tangent
        .iter()
        .all(|c| ir.params[c.block].manifold != ManifoldKind::S2)
    {
        return (tangent.to_vec(), sigma.clone());
    }
    // Each reported coordinate as a combination of tangent coordinates.
    let mut coords = Vec::new();
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut i = 0;
    while i < tangent.len() {
        let c = tangent[i];
        let block = &ir.params[c.block];
        if block.manifold != ManifoldKind::S2 {
            coords.push(c);
            rows.push(vec![(i, 1.0)]);
            i += 1;
            continue;
        }
        let x = &params[&block.name];
        let (b1, b2) = s2::tangent_basis(&Vector3::new(x[0], x[1], x[2]).normalize());
        for axis in 0..3 {
            coords.push(Coordinate {
                block: c.block,
                index: axis,
            });
            rows.push(vec![(i, b1[axis]), (i + 1, b2[axis])]);
        }
        i += 2;
    }
    let matrix = DMatrix::from_fn(rows.len(), rows.len(), |a, b| {
        rows[a]
            .iter()
            .flat_map(|&(p, cp)| rows[b].iter().map(move |&(q, cq)| cp * cq * sigma[(p, q)]))
            .sum()
    });
    (coords, matrix)
}
