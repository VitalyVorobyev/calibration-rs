//! Per-view Jacobian assembly using `num-dual`.
//!
//! This module computes the global unweighted Jacobian by applying autodiff
//! to each view separately. Each view depends only on:
//! - the shared intrinsics/distortion block, and
//! - that view's 6-DoF pose block (axis-angle + translation).
//!
//! Using a small local parameter vector (`K = 10 + 6 = 16`) keeps AD fast and
//! avoids differentiating the full global parameter vector.
//! The resulting per-view Jacobians are scattered into the global matrix.
//!
//! Robust IRLS weights are handled elsewhere and are never differentiated.

use crate::planar_intrinsics::{
    residuals_view_unweighted, PlanarIntrinsicsProblem, INTRINSICS_DIM, LOCAL_DIM, POSE_DIM,
};
use nalgebra::{Const, DMatrix, DVector, Dyn, OMatrix, SVector};
use num_dual::{jacobian, DualSVec64};
use std::ops::Range;

/// Parameter layout for planar intrinsics optimization.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParamLayout {
    pub intrinsics_dim: usize,
    pub pose_dim: usize,
}

impl ParamLayout {
    pub const fn new() -> Self {
        Self {
            intrinsics_dim: INTRINSICS_DIM,
            pose_dim: POSE_DIM,
        }
    }

    pub fn local_dim(self) -> usize {
        self.intrinsics_dim + self.pose_dim
    }

    pub fn global_intrinsics_range(self) -> Range<usize> {
        0..self.intrinsics_dim
    }

    pub fn global_pose_offset(self, view_idx: usize) -> usize {
        self.intrinsics_dim + self.pose_dim * view_idx
    }
}

fn build_local_params(
    x: &DVector<f64>,
    view_idx: usize,
    layout: ParamLayout,
) -> SVector<f64, LOCAL_DIM> {
    let mut local = SVector::<f64, LOCAL_DIM>::zeros();
    let intr_range = layout.global_intrinsics_range();
    for (i, idx) in intr_range.clone().enumerate() {
        local[i] = x[idx];
    }
    let pose_offset = layout.global_pose_offset(view_idx);
    for k in 0..layout.pose_dim {
        local[layout.intrinsics_dim + k] = x[pose_offset + k];
    }
    local
}

fn scatter_jacobian(
    j_global: &mut DMatrix<f64>,
    j_view: &OMatrix<f64, Dyn, Const<LOCAL_DIM>>,
    row_offset: usize,
    view_idx: usize,
    layout: ParamLayout,
) {
    let intr_range = layout.global_intrinsics_range();
    let pose_offset = layout.global_pose_offset(view_idx);

    for r in 0..j_view.nrows() {
        for (local_col, global_col) in intr_range.clone().enumerate() {
            j_global[(row_offset + r, global_col)] = j_view[(r, local_col)];
        }
        for k in 0..layout.pose_dim {
            let local_col = layout.intrinsics_dim + k;
            let global_col = pose_offset + k;
            j_global[(row_offset + r, global_col)] = j_view[(r, local_col)];
        }
    }
}

/// Compute the global unweighted Jacobian using per-view autodiff.
pub(crate) fn jacobian_unweighted_ad(
    problem: &PlanarIntrinsicsProblem,
    x: &DVector<f64>,
) -> DMatrix<f64> {
    let layout = ParamLayout::new();
    debug_assert_eq!(layout.local_dim(), LOCAL_DIM);
    debug_assert_eq!(
        x.len(),
        layout.intrinsics_dim + layout.pose_dim * problem.num_views()
    );

    let m = problem.residual_dim();
    let n = x.len();
    let mut j_global = DMatrix::zeros(m, n);

    let mut row_offset = 0;
    for (view_idx, view) in problem.views.iter().enumerate() {
        let p0_local = build_local_params(x, view_idx, layout);
        let (r_view, j_view) = jacobian(
            |p: SVector<DualSVec64<LOCAL_DIM>, LOCAL_DIM>| {
                let p_slice = p.as_slice();
                let intr = &p_slice[..layout.intrinsics_dim];
                let pose = &p_slice[layout.intrinsics_dim..];
                residuals_view_unweighted(intr, pose, view)
            },
            &p0_local,
        );

        debug_assert_eq!(r_view.len(), view.points_3d.len() * 2);
        scatter_jacobian(&mut j_global, &j_view, row_offset, view_idx, layout);
        row_offset += r_view.len();
    }

    debug_assert_eq!(row_offset, m);
    j_global
}
