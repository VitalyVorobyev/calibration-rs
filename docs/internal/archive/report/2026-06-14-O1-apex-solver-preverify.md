# Track O pre-verify: apex-solver 1.3 — ON HOLD (architectural mismatch)

**Date:** 2026-06-14
**Decision:** Do **not** implement `ApexSolverBackend` against apex-solver 1.3.
The pre-verify gate (ROADMAP Track O / backlog O1) fails on the load-bearing
requirement. No backend code written; no branch created.

## What was checked

apex-solver 1.3.0 (latest, published 2026-05-07). Public API per docs.rs.

## Findings

1. **No autodiff — f64-only, caller-supplied Jacobians (blocker).**
   The core `Factor` trait is:
   ```rust
   pub trait Factor: Send + Sync {
       fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool)
           -> (DVector<f64>, Option<DMatrix<f64>>);
       fn get_dimension(&self) -> usize;
   }
   ```
   The caller must supply the Jacobian (analytic or numeric). There is no
   generic scalar, no dual numbers, no built-in autodiff or
   numerical-differentiation.

   Our optimization IR (ADR 0008) and the M0 factor generification
   (ADR 0020) are built entirely on **autodiff-generic** factors
   (`fn residual<T: RealField>()`), monomorphized per `CameraModelDesc`, with
   the tiny-solver backend providing autodiff. To use apex-solver we would have
   to hand-derive (or finite-difference) Jacobians for all four factor families
   — `ReprojPoint` across every camera-model × `ReprojChain` combination,
   `LaserPointToPlane`, `LaserLineDistance`, `Se3TangentPrior` — which defeats
   the entire point of M0 and is error-prone at exactly the geometry the project
   most needs to trust.

2. **No S2 / unit-vector manifold (secondary blocker).** apex-solver exposes
   `SE3/SE2/SO3/SO2/Rn` Lie groups but no sphere manifold. We use a custom S2
   manifold for laser-plane normals (`UnitVector3Manifold`). Fallback (R3 +
   renormalize) changes the problem geometry.

3. **No documented robust losses.** We require Huber / Cauchy / Arctan
   (`RobustLoss` is per-residual in the IR). apex-solver's docs show no loss /
   robustifier hook on the factor or graph API.

4. **Quaternion component order undocumented** for its `SE3` — the planned
   round-trip vs our `[qx,qy,qz,qw,tx,ty,tz]` could not be confirmed (moot given
   #1).

## Recommendation (for the user)

The roadmap premise — "apex-solver gets a second real `OptimBackend`" — assumed
a Ceres-like API. apex-solver 1.3 is instead a hand-Jacobian factor-graph
library, fundamentally mismatched to our autodiff-first IR. Options:

- **Drop apex-solver** from Track O. If a second backend is still wanted for the
  abstraction's sake, pick an **autodiff-capable** Rust optimizer (e.g. a
  `factrs`/`num-dual`-based stack) or keep tiny-solver as the sole backend.
- A **numeric-difference bridge backend** is technically possible (we already
  evaluate `reproj_residual_model_generic` at f64), but it would be slower, less
  accurate, and require re-deriving manifold tangent Jacobians by hand — low
  value, real correctness risk. Not recommended as an unsupervised task.
- **O3 (drop the `BackendKind::Ceres` stub)** is independent of apex-solver and
  still worth doing — it removes dead code (`backend/mod.rs:119`). It is a small,
  safe cleanup that can land on its own.

Track O is parked pending the user's call on the backend choice.
