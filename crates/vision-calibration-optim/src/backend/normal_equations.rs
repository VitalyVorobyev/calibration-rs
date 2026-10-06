//! The damped normal equations of a Levenberg–Marquardt step.
//!
//! With `J̃ = J·S` the Jacobian under the Jacobi column scaling `S`, the step
//! solves `(J̃ᵀJ̃ + λD)·δ = J̃ᵀ(−r)`, where `D` is the clamped diagonal of
//! `J̃ᵀJ̃`.
//!
//! The Jacobian's sparsity pattern is the same at every iterate, so the
//! pattern of `J̃ᵀJ̃`, the plan for filling it and the symbolic Cholesky
//! analysis are built once per solve. Only the lower triangle is stored:
//! it is all the factorization reads.
//!
//! The rows of a calibration Jacobian come in runs that touch one set of
//! columns: every residual of a view touches the same intrinsics and the
//! same pose. [`NormalEquations::assemble`] walks those runs (segments) and
//! accumulates each into a small dense block, instead of a general sparse
//! product. Every entry still sums its products in ascending row order, so
//! the result is bit-for-bit that of the general product.

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::{Mat, Side};
use nalgebra::DVector;

/// Bounds on the Marquardt diagonal `D`.
const MIN_DIAGONAL: f64 = 1e-6;
const MAX_DIAGONAL: f64 = 1e32;

/// Jacobi column scaling `1 / (1 + ‖J_c‖)`.
pub(super) fn jacobi_scaling(jac: SparseColMatRef<'_, usize, f64>) -> Vec<f64> {
    (0..jac.ncols())
        .map(|c| {
            let norm = jac.val_of_col(c).iter().map(|&v| v * v).sum::<f64>().sqrt();
            1.0 / (1.0 + norm)
        })
        .collect()
}

/// Consecutive Jacobian rows that touch the same columns, each column's
/// entries for these rows stored consecutively.
struct Segment {
    /// First row.
    row: usize,
    /// Number of rows.
    len: usize,
    /// Range of the segment's columns in [`NormalEquations::seg_cols`] and
    /// [`NormalEquations::seg_start`].
    cols: std::ops::Range<usize>,
    /// Start of the segment's lower-triangle slots in
    /// [`NormalEquations::seg_slots`], pairs `(a, b ≤ a)` row by row.
    slots: usize,
}

/// Normal equations over one Jacobian sparsity pattern.
pub(super) struct NormalEquations {
    /// The Jacobian pattern the plan was built for.
    jac_col_ptr: Vec<usize>,
    jac_col_nnz: Option<Vec<usize>>,
    jac_row_idx: Vec<usize>,
    segments: Vec<Segment>,
    /// Column of each (segment, local column).
    seg_cols: Vec<usize>,
    /// Position in the Jacobian's values of the segment's first row, per
    /// (segment, local column).
    seg_start: Vec<usize>,
    /// Position in `hessian` of each (segment, lower-triangle pair).
    seg_slots: Vec<usize>,
    /// Pattern of the lower triangle of `J̃ᵀJ̃`, diagonal included.
    pattern: SymbolicSparseColMat<usize>,
    /// Position of each diagonal entry in `hessian`.
    diag: Vec<usize>,
    llt: SymbolicLlt<usize>,
    /// `J̃ᵀJ̃` (lower triangle) at the current iterate.
    hessian: Vec<f64>,
    /// `J̃ᵀ(−r)` at the current iterate.
    gradient: Mat<f64>,
    /// `J̃ᵀJ̃ + λD` for the step attempt in flight.
    damped: Vec<f64>,
}

impl NormalEquations {
    /// Plan for Jacobians with the pattern `jac`.
    ///
    /// Returns `None` if the symbolic factorization cannot be allocated.
    pub(super) fn new(jac: SymbolicSparseColMatRef<'_, usize>) -> Option<Self> {
        let (nrows, ncols) = (jac.nrows(), jac.ncols());

        // Row-major copy of the pattern: each row's columns in ascending
        // order, with the entry's position in the column-major values.
        let mut row_ptr = vec![0usize; nrows + 1];
        for c in 0..ncols {
            for &r in jac.row_idx_of_col_raw(c) {
                row_ptr[r + 1] += 1;
            }
        }
        for r in 0..nrows {
            row_ptr[r + 1] += row_ptr[r];
        }
        let mut fill = row_ptr.clone();
        let mut row_col = vec![0usize; row_ptr[nrows]];
        let mut row_pos = vec![0usize; row_ptr[nrows]];
        for c in 0..ncols {
            for pos in jac.col_range(c) {
                let r = jac.row_idx()[pos];
                row_col[fill[r]] = c;
                row_pos[fill[r]] = pos;
                fill[r] += 1;
            }
        }

        // Maximal runs of rows with one column set and consecutive
        // positions in every column.
        let mut segments: Vec<Segment> = Vec::new();
        let mut seg_cols = Vec::new();
        let mut seg_start = Vec::new();
        for r in 0..nrows {
            let entries = row_ptr[r]..row_ptr[r + 1];
            if entries.is_empty() {
                continue;
            }
            if let Some(seg) = segments.last_mut() {
                let continues = seg.row + seg.len == r
                    && seg_cols[seg.cols.clone()] == row_col[entries.clone()]
                    && seg_start[seg.cols.clone()]
                        .iter()
                        .zip(&row_pos[entries.clone()])
                        .all(|(&start, &pos)| pos == start + seg.len);
                if continues {
                    seg.len += 1;
                    continue;
                }
            }
            let first = seg_cols.len();
            seg_cols.extend_from_slice(&row_col[entries.clone()]);
            seg_start.extend_from_slice(&row_pos[entries]);
            segments.push(Segment {
                row: r,
                len: 1,
                cols: first..seg_cols.len(),
                slots: 0,
            });
        }

        // Lower-triangle pattern: every pair of columns a segment couples,
        // plus the whole diagonal (the damping needs it).
        let mut col_rows: Vec<Vec<usize>> = (0..ncols).map(|c| vec![c]).collect();
        for seg in &segments {
            let cols = &seg_cols[seg.cols.clone()];
            for (a, &ca) in cols.iter().enumerate() {
                for &cb in &cols[..a] {
                    col_rows[cb].push(ca);
                }
            }
        }
        let mut col_ptr = Vec::with_capacity(ncols + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for rows in &mut col_rows {
            rows.sort_unstable();
            rows.dedup();
            row_idx.extend_from_slice(rows);
            col_ptr.push(row_idx.len());
        }
        let diag = col_ptr[..ncols].to_vec();

        let slot = |row: usize, col: usize| {
            let range = col_ptr[col]..col_ptr[col + 1];
            range.start
                + row_idx[range]
                    .binary_search(&row)
                    .expect("every coupled pair is in the pattern")
        };
        let mut seg_slots = Vec::new();
        for seg in &mut segments {
            seg.slots = seg_slots.len();
            let cols = &seg_cols[seg.cols.clone()];
            for (a, &ca) in cols.iter().enumerate() {
                for &cb in &cols[..=a] {
                    seg_slots.push(slot(ca, cb));
                }
            }
        }

        let nnz = row_idx.len();
        let pattern = SymbolicSparseColMat::new_checked(ncols, ncols, col_ptr, None, row_idx);
        let llt = SymbolicLlt::try_new(pattern.as_ref(), Side::Lower).ok()?;

        Some(Self {
            jac_col_ptr: jac.col_ptr().to_vec(),
            jac_col_nnz: jac.col_nnz().map(<[usize]>::to_vec),
            jac_row_idx: jac.row_idx().to_vec(),
            segments,
            seg_cols,
            seg_start,
            seg_slots,
            pattern,
            diag,
            llt,
            hessian: vec![0.0; nnz],
            gradient: Mat::zeros(ncols, 1),
            damped: vec![0.0; nnz],
        })
    }

    /// Whether this plan was built for the pattern `jac`.
    pub(super) fn fits(&self, jac: SymbolicSparseColMatRef<'_, usize>) -> bool {
        jac.col_ptr() == self.jac_col_ptr.as_slice()
            && jac.col_nnz() == self.jac_col_nnz.as_deref()
            && jac.row_idx() == self.jac_row_idx.as_slice()
    }

    /// Fill `J̃ᵀJ̃` and `J̃ᵀ(−r)` from `jac`, which must [`fit`](Self::fits),
    /// the residual `r` and the column scaling.
    pub(super) fn assemble(
        &mut self,
        jac: SparseColMatRef<'_, usize, f64>,
        residual: &Mat<f64>,
        scaling: &[f64],
    ) {
        debug_assert!(self.fits(jac.symbolic()));
        let values = jac.val();
        self.hessian.fill(0.0);
        self.gradient.fill(0.0);

        let max_cols = self
            .segments
            .iter()
            .map(|s| s.cols.len())
            .max()
            .unwrap_or(0);
        let mut row = vec![0.0; max_cols];
        let mut grad = vec![0.0; max_cols];
        let mut acc = vec![0.0; max_cols * (max_cols + 1) / 2];

        for seg in &self.segments {
            let cols = &self.seg_cols[seg.cols.clone()];
            let start = &self.seg_start[seg.cols.clone()];
            let k = cols.len();
            let pairs = k * (k + 1) / 2;
            let slots = &self.seg_slots[seg.slots..seg.slots + pairs];
            let (row, grad, acc) = (&mut row[..k], &mut grad[..k], &mut acc[..pairs]);

            for (a, &s) in acc.iter_mut().zip(slots) {
                *a = self.hessian[s];
            }
            for (g, &c) in grad.iter_mut().zip(cols) {
                *g = self.gradient[(c, 0)];
            }
            for t in 0..seg.len {
                let neg_r = -residual[(seg.row + t, 0)];
                for a in 0..k {
                    row[a] = values[start[a] + t] * scaling[cols[a]];
                }
                let mut p = 0;
                for a in 0..k {
                    let xa = row[a];
                    for (h, &xb) in acc[p..=p + a].iter_mut().zip(&row[..=a]) {
                        *h += xa * xb;
                    }
                    p += a + 1;
                    grad[a] += neg_r * xa;
                }
            }
            for (&a, &s) in acc.iter().zip(slots) {
                self.hessian[s] = a;
            }
            for (&g, &c) in grad.iter().zip(cols) {
                self.gradient[(c, 0)] = g;
            }
        }
    }

    /// The step `δ` solving `(J̃ᵀJ̃ + λD)·δ = J̃ᵀ(−r)`, in scaled coordinates.
    ///
    /// Returns `None` when the damped matrix is not positive definite.
    pub(super) fn solve_damped(&mut self, damping: f64) -> Option<Mat<f64>> {
        self.damped.copy_from_slice(&self.hessian);
        for &d in &self.diag {
            self.damped[d] += damping * self.hessian[d].clamp(MIN_DIAGONAL, MAX_DIAGONAL);
        }
        let matrix = SparseColMatRef::new(self.pattern.as_ref(), &self.damped);
        let llt = Llt::try_new_with_symbolic(self.llt.clone(), matrix, Side::Lower).ok()?;
        Some(llt.solve(&self.gradient))
    }

    /// `δᵀ(2·J̃ᵀ(−r) − J̃ᵀJ̃·δ)`: the decrease of the undamped local model
    /// `‖r + J̃δ‖²` along `δ`.
    pub(super) fn model_decrease(&self, step: &Mat<f64>) -> f64 {
        // `J̃ᵀJ̃·δ` from the lower triangle, each output summed in the column
        // order of a product with the full matrix.
        let mut product = Mat::<f64>::zeros(step.nrows(), 1);
        for j in 0..self.pattern.ncols() {
            for pos in self.pattern.col_range(j) {
                let i = self.pattern.row_idx()[pos];
                let h = self.hessian[pos];
                product[(i, 0)] += h * step[(j, 0)];
                if i != j {
                    product[(j, 0)] += h * step[(i, 0)];
                }
            }
        }
        let change = step.transpose() * (2.0 * &self.gradient - &product);
        change[(0, 0)]
    }
}

/// Undo the scaling: `dx = S·δ`.
pub(super) fn unscale(step: &Mat<f64>, scaling: &[f64]) -> DVector<f64> {
    DVector::from_fn(scaling.len(), |i, _| scaling[i] * step[(i, 0)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use std::ops::Mul;

    /// Deterministic values in `[-1, 1)`.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    /// A Jacobian shaped like a planar calibration: 4 shared columns, then
    /// 6 per view, 2 rows per point. Views 1 and 2 interleave their points,
    /// row 7 is empty and the last rows touch only the shared columns, so
    /// segments break in every way they can.
    fn calibration_like_jacobian(rng: &mut Lcg) -> SparseColMat<usize, f64> {
        let views = 4;
        let ncols = 4 + 6 * views;
        let mut triplets = Vec::new();
        let mut row = 0;
        let mut point = |view: usize, row: &mut usize, triplets: &mut Vec<_>| {
            for _ in 0..2 {
                for c in (0..4).chain(4 + 6 * view..10 + 6 * view) {
                    triplets.push(Triplet::new(*row, c, rng.next()));
                }
                *row += 1;
            }
        };
        for _ in 0..3 {
            point(0, &mut row, &mut triplets);
        }
        row += 1; // an empty row
        for _ in 0..3 {
            point(1, &mut row, &mut triplets);
            point(2, &mut row, &mut triplets);
        }
        for _ in 0..5 {
            point(3, &mut row, &mut triplets);
        }
        for c in 0..4 {
            triplets.push(Triplet::new(row, c, rng.next()));
        }
        row += 1;
        SparseColMat::try_new_from_triplets(row, ncols, &triplets).unwrap()
    }

    fn bits(m: &Mat<f64>) -> Vec<u64> {
        (0..m.nrows()).map(|i| m[(i, 0)].to_bits()).collect()
    }

    /// The same quantities through general sparse products of the full
    /// matrix.
    fn reference(
        jac: &SparseColMat<usize, f64>,
        residual: &Mat<f64>,
        scaling: &[f64],
        damping: f64,
    ) -> (SparseColMat<usize, f64>, Mat<f64>, Mat<f64>, f64) {
        let n = jac.ncols();
        let s: Vec<_> = (0..n).map(|c| Triplet::new(c, c, scaling[c])).collect();
        let s = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &s).unwrap();
        let scaled = jac * &s;
        let jtj = scaled
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(scaled.as_ref());
        let jtr = scaled.as_ref().transpose().mul(-residual);
        let mut damped = jtj.clone();
        for i in 0..n {
            damped[(i, i)] += damping * jtj[(i, i)].clamp(MIN_DIAGONAL, MAX_DIAGONAL);
        }
        let symbolic = SymbolicLlt::try_new(damped.symbolic(), Side::Lower).unwrap();
        let step = Llt::try_new_with_symbolic(symbolic, damped.as_ref(), Side::Lower)
            .unwrap()
            .solve(&jtr);
        let decrease = step.transpose().mul(2.0 * &jtr - &jtj * &step)[(0, 0)];
        (jtj, jtr, step, decrease)
    }

    #[test]
    fn matches_the_general_sparse_product_bit_for_bit() {
        let mut rng = Lcg(7);
        let jac = calibration_like_jacobian(&mut rng);
        let residual = Mat::from_fn(jac.nrows(), 1, |_, _| rng.next());
        let scaling = jacobi_scaling(jac.as_ref());
        let damping = 1e-3;

        let mut normal = NormalEquations::new(jac.symbolic()).unwrap();
        assert!(normal.fits(jac.symbolic()));
        normal.assemble(jac.as_ref(), &residual, &scaling);
        let (jtj, jtr, step, decrease) = reference(&jac, &residual, &scaling, damping);

        for j in 0..jac.ncols() {
            for pos in normal.pattern.col_range(j) {
                let i = normal.pattern.row_idx()[pos];
                let expected = jtj.get(i, j).copied().unwrap_or(0.0);
                assert_eq!(
                    normal.hessian[pos].to_bits(),
                    expected.to_bits(),
                    "({i}, {j})"
                );
            }
        }
        assert_eq!(bits(&normal.gradient), bits(&jtr));
        let ours = normal.solve_damped(damping).unwrap();
        assert_eq!(bits(&ours), bits(&step));
        assert_eq!(normal.model_decrease(&ours).to_bits(), decrease.to_bits());
    }

    #[test]
    fn segments_group_rows_that_share_columns() {
        let mut rng = Lcg(11);
        let jac = calibration_like_jacobian(&mut rng);
        let normal = NormalEquations::new(jac.symbolic()).unwrap();
        // View 0 (6 rows), views 1/2 alternating (6 runs of 2 rows),
        // view 3 (10 rows), the shared-only row.
        let lens: Vec<usize> = normal.segments.iter().map(|s| s.len).collect();
        assert_eq!(lens, [6, 2, 2, 2, 2, 2, 2, 10, 1]);
    }

    #[test]
    fn a_different_pattern_does_not_fit() {
        let mut rng = Lcg(3);
        let jac = calibration_like_jacobian(&mut rng);
        let normal = NormalEquations::new(jac.symbolic()).unwrap();
        let other = SparseColMat::<usize, f64>::try_new_from_triplets(
            jac.nrows(),
            jac.ncols(),
            &[Triplet::new(0, 0, 1.0)],
        )
        .unwrap();
        assert!(!normal.fits(other.symbolic()));
    }

    #[test]
    fn an_untouched_column_still_gets_a_damped_diagonal() {
        // Column 1 has no entries: the step leaves it at zero instead of
        // failing the factorization.
        let jac = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            2,
            &[Triplet::new(0, 0, 2.0), Triplet::new(1, 0, 1.0)],
        )
        .unwrap();
        let residual = Mat::from_fn(2, 1, |i, _| [1.0, -1.0][i]);
        let scaling = jacobi_scaling(jac.as_ref());
        let mut normal = NormalEquations::new(jac.symbolic()).unwrap();
        normal.assemble(jac.as_ref(), &residual, &scaling);
        let step = normal.solve_damped(1e-4).unwrap();
        assert!(step[(0, 0)].is_finite());
        assert_eq!(step[(1, 0)], 0.0);
    }
}
