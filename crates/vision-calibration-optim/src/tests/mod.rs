//! White-box tests that build a [`ProblemIR`](crate::ir::ProblemIR) by hand
//! and drive it through the backend.
//!
//! They live inside the crate rather than under `tests/` because the IR and
//! the parameter-packing helpers are private: a caller assembles problems
//! through the `optimize_*` entry points, not by naming factor kinds and
//! parameter slots. Testing the lowering still requires reaching those, so
//! these are unit tests.

mod ir_distortion_models;
mod ir_scheimpflug;
