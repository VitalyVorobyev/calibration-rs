//! IR → factrs graph and values.
//!
//! factrs residuals take at most six variables, while an IR factor can take
//! nine parameter blocks. Blocks that always travel together fuse into one
//! factrs variable:
//!
//! - a camera's `(intrinsics, distortion, sensor)` blocks → `VectorVar<N>`;
//! - a laser plane's `(plane_normal, plane_distance)` blocks → [`PlaneVar`].
//!
//! The widest factor (laser + rig hand-eye + robot correction) then has six
//! variables. Every factor must use a block in the same fused group: a
//! camera's blocks are the same triple everywhere.
//!
//! Fixed components keep their variable; [`Compiled::fixed_columns`] lists
//! their Jacobian columns, which the engine zeroes so the step leaves them
//! exactly unchanged. Bounds are clamped after each step, like the
//! tiny-solver backend.

use std::collections::HashMap;

use factrs::containers::{Graph, Key, Values, ValuesOrder};
use factrs::linalg::{Vector, Vector3};
use factrs::variables::{SE3, VectorVar};
use nalgebra::DVector;

use super::residuals::build_factor;
use super::variables::{PlaneVar, VECTOR_DIMS, se3_from_ir, se3_to_ir, with_vector_dim};
use crate::Error;
use crate::ir::{FactorKind, ManifoldKind, ParamBlock, ProblemIR};

/// A factrs variable and the IR blocks fused into it.
#[derive(Debug, Clone)]
struct Var {
    kind: VarKind,
    /// Tangent components the IR holds fixed.
    fixed: Vec<usize>,
    /// Box bounds on vector components: `(component, lower, upper)`.
    bounds: Vec<(usize, f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarKind {
    /// `VectorVar<dim>`: a fused camera or a robot-pose correction.
    Vector {
        dim: usize,
    },
    Se3,
    Plane,
}

/// Where an IR block lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Components `offset .. offset + dim` of a vector variable.
    Vector {
        var: usize,
        offset: usize,
        dim: usize,
    },
    Se3 {
        var: usize,
    },
    PlaneNormal {
        var: usize,
    },
    PlaneDistance {
        var: usize,
    },
}

impl Slot {
    fn var(self) -> usize {
        match self {
            Slot::Vector { var, .. }
            | Slot::Se3 { var }
            | Slot::PlaneNormal { var }
            | Slot::PlaneDistance { var } => var,
        }
    }
}

/// A compiled problem.
pub(super) struct Compiled {
    pub(super) graph: Graph,
    pub(super) values: Values,
    vars: Vec<Var>,
    /// Indexed by IR parameter id.
    slots: Vec<Slot>,
}

/// Assigns factrs variables to IR blocks.
#[derive(Default)]
struct Layout {
    vars: Vec<Var>,
    slots: Vec<Option<Slot>>,
    /// Fused groups seen so far, keyed by their IR parameter ids.
    groups: HashMap<Vec<usize>, usize>,
}

impl Layout {
    fn new(n_params: usize) -> Self {
        Self {
            slots: vec![None; n_params],
            ..Self::default()
        }
    }

    fn push_var(&mut self, kind: VarKind) -> usize {
        self.vars.push(Var {
            kind,
            fixed: Vec::new(),
            bounds: Vec::new(),
        });
        self.vars.len() - 1
    }

    /// The variable fusing `ids` (in order) with the given slots, created on
    /// first use. Fails if a block is already part of a different variable.
    fn group(
        &mut self,
        ids: &[usize],
        kind: VarKind,
        slot_of: impl Fn(usize, usize) -> Slot,
        ir: &ProblemIR,
    ) -> Result<usize, Error> {
        if let Some(&var) = self.groups.get(ids) {
            return Ok(var);
        }
        if let Some(&id) = ids.iter().find(|&&id| self.slots[id].is_some()) {
            return Err(Error::invalid_input(format!(
                "factrs backend: parameter {} is used in two different groups \
                 (a camera's blocks, or a plane's normal and distance, must \
                 travel together in every factor)",
                ir.params[id].name
            )));
        }
        let var = self.push_var(kind);
        for (i, &id) in ids.iter().enumerate() {
            self.slots[id] = Some(slot_of(var, i));
        }
        self.groups.insert(ids.to_vec(), var);
        Ok(var)
    }

    /// The variable of a standalone block, created on first use.
    fn single(&mut self, block: &ParamBlock) -> Result<usize, Error> {
        let id = block.id.0;
        if let Some(slot) = self.slots[id] {
            return match (slot, block.manifold) {
                (Slot::Se3 { var }, ManifoldKind::SE3) => Ok(var),
                (
                    Slot::Vector {
                        var,
                        offset: 0,
                        dim,
                    },
                    ManifoldKind::Euclidean,
                ) if dim == block.dim && self.vars[var].kind == (VarKind::Vector { dim }) => {
                    Ok(var)
                }
                _ => Err(Error::invalid_input(format!(
                    "factrs backend: parameter {} is used both alone and in a group",
                    block.name
                ))),
            };
        }
        let (kind, slot): (VarKind, fn(usize, usize) -> Slot) = match block.manifold {
            ManifoldKind::SE3 => (VarKind::Se3, |var, _| Slot::Se3 { var }),
            ManifoldKind::Euclidean if VECTOR_DIMS.contains(&block.dim) => {
                (VarKind::Vector { dim: block.dim }, |var, dim| {
                    Slot::Vector {
                        var,
                        offset: 0,
                        dim,
                    }
                })
            }
            _ => {
                return Err(Error::invalid_input(format!(
                    "factrs backend: unsupported standalone parameter {} ({:?}, dim {})",
                    block.name, block.manifold, block.dim
                )));
            }
        };
        let var = self.push_var(kind);
        self.slots[id] = Some(slot(var, block.dim));
        Ok(var)
    }
}

/// Compile `ir` at `initial` into a factrs graph and values.
pub(super) fn compile(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
) -> Result<Compiled, Error> {
    ir.validate()?;
    let init: Vec<&DVector<f64>> = ir
        .params
        .iter()
        .map(|p| {
            let v = initial.get(&p.name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "initial values missing parameter {} (id {:?})",
                    p.name, p.id
                ))
            })?;
            if v.len() != p.dim {
                return Err(Error::invalid_input(format!(
                    "initial dimension mismatch for {}: expected {}, got {}",
                    p.name,
                    p.dim,
                    v.len()
                )));
            }
            Ok(v)
        })
        .collect::<Result<_, _>>()?;

    // Variables, in first-use order over the residuals.
    let mut layout = Layout::new(ir.params.len());
    let mut factor_vars: Vec<Vec<usize>> = Vec::with_capacity(ir.residuals.len());
    for residual in &ir.residuals {
        let ids: Vec<usize> = residual.params.iter().map(|p| p.0).collect();
        let slots = residual.factor.param_layout();
        let n_cam = match &residual.factor {
            FactorKind::ReprojPoint { model, .. }
            | FactorKind::LaserPointToPlane { model, .. }
            | FactorKind::LaserLineDistance { model, .. } => model.num_cam_blocks(),
            FactorKind::Se3TangentPrior { .. } => 0,
        };
        let mut vars = Vec::with_capacity(6);
        if n_cam > 0 {
            let cam_ids = &ids[..n_cam];
            let dims: Vec<usize> = cam_ids.iter().map(|&id| ir.params[id].dim).collect();
            let total: usize = dims.iter().sum();
            let offsets: Vec<usize> = dims
                .iter()
                .scan(0, |acc, &d| {
                    let o = *acc;
                    *acc += d;
                    Some(o)
                })
                .collect();
            vars.push(layout.group(
                cam_ids,
                VarKind::Vector { dim: total },
                |var, i| Slot::Vector {
                    var,
                    offset: offsets[i],
                    dim: dims[i],
                },
                ir,
            )?);
        }
        let mut j = n_cam;
        while j < ids.len() {
            if slots[j].manifold == ManifoldKind::S2 {
                // A plane normal is always followed by its distance.
                vars.push(layout.group(
                    &ids[j..j + 2],
                    VarKind::Plane,
                    |var, i| {
                        if i == 0 {
                            Slot::PlaneNormal { var }
                        } else {
                            Slot::PlaneDistance { var }
                        }
                    },
                    ir,
                )?);
                j += 2;
            } else {
                vars.push(layout.single(&ir.params[ids[j]])?);
                j += 1;
            }
        }
        let mut distinct = vars.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != vars.len() {
            return Err(Error::invalid_input(
                "factrs backend: a factor uses the same variable twice",
            ));
        }
        factor_vars.push(vars);
    }
    for block in &ir.params {
        if layout.slots[block.id.0].is_none() {
            layout.single(block)?;
        }
    }
    let slots: Vec<Slot> = layout.slots.into_iter().map(Option::unwrap).collect();
    let mut vars = layout.vars;

    // Fixed components and bounds.
    for block in &ir.params {
        let slot = slots[block.id.0];
        let var = &mut vars[slot.var()];
        let bounds = block.bounds.as_deref().unwrap_or_default();
        match slot {
            Slot::Vector { offset, .. } => {
                var.fixed.extend(block.fixed.iter().map(|i| offset + i));
                var.bounds
                    .extend(bounds.iter().map(|b| (offset + b.idx, b.lower, b.upper)));
                continue;
            }
            Slot::Se3 { .. } | Slot::PlaneNormal { .. } => {
                if !block.fixed.is_empty() && !block.fixed.is_all_fixed(block.dim) {
                    return Err(Error::invalid_input(format!(
                        "factrs backend cannot partially fix {:?} manifold {}",
                        block.manifold, block.name
                    )));
                }
                if !block.fixed.is_empty() {
                    let tangent = if matches!(slot, Slot::Se3 { .. }) {
                        0..6
                    } else {
                        0..2
                    };
                    var.fixed.extend(tangent);
                }
            }
            Slot::PlaneDistance { .. } => {
                if !block.fixed.is_empty() {
                    var.fixed.push(2);
                }
            }
        }
        if !bounds.is_empty() {
            return Err(Error::invalid_input(format!(
                "factrs backend supports bounds on vector blocks only, not on {}",
                block.name
            )));
        }
    }

    // Values, keyed by variable index.
    let mut values = Values::new();
    for (index, var) in vars.iter().enumerate() {
        let key = Key(index as u64);
        let blocks: Vec<(usize, &DVector<f64>)> = slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.var() == index)
            .map(|(id, _)| (id, init[id]))
            .collect();
        match var.kind {
            VarKind::Vector { dim } => {
                let mut data = vec![0.0; dim];
                for (id, v) in &blocks {
                    if let Slot::Vector { offset, .. } = slots[*id] {
                        data[offset..offset + v.len()].copy_from_slice(v.as_slice());
                    }
                }
                with_vector_dim!(dim, N => {
                    values.insert_unchecked(key, VectorVar::<N>(Vector::<N>::from_column_slice(&data)));
                }, _ => unreachable!("vector dims are checked when the variable is created"));
            }
            VarKind::Se3 => {
                values.insert_unchecked(key, se3_from_ir(blocks[0].1));
            }
            VarKind::Plane => {
                let pick = |want: fn(Slot) -> bool| {
                    blocks
                        .iter()
                        .find(|(id, _)| want(slots[*id]))
                        .map(|(_, v)| *v)
                        .expect("a plane has a normal and a distance")
                };
                let normal = pick(|s| matches!(s, Slot::PlaneNormal { .. }));
                let distance = pick(|s| matches!(s, Slot::PlaneDistance { .. }));
                values.insert_unchecked(key, PlaneVar::from_ir(normal, distance[0]));
            }
        }
    }

    // Factors.
    let mut graph = Graph::with_capacity(ir.residuals.len());
    for (residual, vars) in ir.residuals.iter().zip(&factor_vars) {
        let keys: Vec<Key> = vars.iter().map(|&v| Key(v as u64)).collect();
        graph.add_factor(build_factor(&residual.factor, &keys, residual.loss));
    }

    Ok(Compiled {
        graph,
        values,
        vars,
        slots,
    })
}

impl Compiled {
    /// Global Jacobian columns of the fixed components under `order`.
    pub(super) fn fixed_columns(&self, order: &ValuesOrder) -> Vec<usize> {
        let mut cols: Vec<usize> = self
            .vars
            .iter()
            .enumerate()
            .flat_map(|(index, var)| {
                let start = order
                    .get(Key(index as u64))
                    .expect("every variable is ordered")
                    .idx;
                var.fixed.iter().map(move |c| start + c)
            })
            .collect();
        cols.sort_unstable();
        cols.dedup();
        cols
    }

    /// Clamp bounded components to their bounds.
    pub(super) fn clamp_bounds(&self, values: &mut Values) {
        for (index, var) in self.vars.iter().enumerate() {
            if var.bounds.is_empty() {
                continue;
            }
            let VarKind::Vector { dim } = var.kind else {
                unreachable!("bounds are only accepted on vector blocks");
            };
            with_vector_dim!(dim, N => {
                let v: &mut VectorVar<N> = values
                    .get_unchecked_mut(Key(index as u64))
                    .expect("compiled variable");
                for &(c, lower, upper) in &var.bounds {
                    v.0[c] = v.0[c].max(lower).min(upper);
                }
            }, _ => unreachable!("vector dims are checked when the variable is created"));
        }
    }

    /// `‖x‖` over the IR's ambient parameters (SE3 as quaternion and
    /// translation), summed in variable order.
    pub(super) fn ambient_norm(&self, values: &Values) -> f64 {
        self.vars
            .iter()
            .enumerate()
            .map(|(index, var)| {
                let key = Key(index as u64);
                match var.kind {
                    VarKind::Vector { dim } => with_vector_dim!(dim, N => {
                        let v: &VectorVar<N> = values.get_unchecked(key).expect("compiled variable");
                        v.0.norm_squared()
                    }, _ => unreachable!("vector dims are checked when the variable is created")),
                    VarKind::Se3 => {
                        let p: &SE3 = values.get_unchecked(key).expect("compiled variable");
                        se3_to_ir(p).norm_squared()
                    }
                    VarKind::Plane => {
                        let p: &PlaneVar = values.get_unchecked(key).expect("compiled variable");
                        p.normal.norm_squared() + p.distance * p.distance
                    }
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    /// The IR parameter blocks at `values`.
    pub(super) fn read_back(
        &self,
        ir: &ProblemIR,
        values: &Values,
    ) -> HashMap<String, DVector<f64>> {
        ir.params
            .iter()
            .map(|block| {
                let slot = self.slots[block.id.0];
                let key = Key(slot.var() as u64);
                let v = match slot {
                    Slot::Vector { var, offset, dim } => {
                        let VarKind::Vector { dim: n } = self.vars[var].kind else {
                            unreachable!("vector slot in a vector variable");
                        };
                        with_vector_dim!(n, N => {
                            let v: &VectorVar<N> = values.get_unchecked(key).expect("compiled variable");
                            DVector::from_column_slice(&v.0.as_slice()[offset..offset + dim])
                        }, _ => unreachable!("vector dims are checked when the variable is created"))
                    }
                    Slot::Se3 { .. } => {
                        let p: &SE3 = values.get_unchecked(key).expect("compiled variable");
                        se3_to_ir(p)
                    }
                    Slot::PlaneNormal { .. } => {
                        let p: &PlaneVar = values.get_unchecked(key).expect("compiled variable");
                        let n: Vector3 = p.normal;
                        DVector::from_column_slice(n.as_slice())
                    }
                    Slot::PlaneDistance { .. } => {
                        let p: &PlaneVar = values.get_unchecked(key).expect("compiled variable");
                        DVector::from_element(1, p.distance)
                    }
                };
                (block.name.clone(), v)
            })
            .collect()
    }
}
