# Refactoring Summary: Code Structure and Duplication Cleanup

**Date**: 2026-01-04
**Branch**: `6-code-structure-and-duplication-cleanup`
**Status**: ✅ Complete

This document summarizes the refactoring work completed to address code duplication and structural issues in the calibration-rs workspace.

---

## Overview

The refactoring focused on eliminating code duplication, improving module organization, and extending the codebase with production-ready Scheimpflug sensor optimization. All work maintains 100% backward compatibility while significantly improving code quality.

**Key Metrics**:
- **Lines of duplicate code eliminated**: ~400-500 lines
- **New production features**: Scheimpflug sensor parameter optimization
- **Test coverage**: 65+ tests passing, including 5 new integration tests
- **Modules reorganized**: 2 large modules split into focused submodules
- **New utility functions**: 4 production coordinate transformation utilities

---

## Phase 1: Extract Math Utilities Module ✅

**Goal**: Eliminate 400-500 lines of duplicated mathematical utilities across calib-linear.

### Created
- [`crates/calib-linear/src/math.rs`](crates/calib-linear/src/math.rs) - Consolidated utilities

### Consolidated Functions
1. **Point Normalization** (Hartley method)
   - `normalize_points_2d()` - 2D point normalization (was duplicated 4× in homography, epipolar, camera_matrix)
   - `normalize_points_3d()` - 3D point normalization (was duplicated 2× in camera_matrix, pnp)

2. **Polynomial Solvers**
   - `solve_quadratic_real()` - Real roots of quadratic (was duplicated 2×)
   - `solve_cubic_real()` - Real roots of cubic (was duplicated 2×)
   - `solve_quartic_real()` - Real roots of quartic (from pnp.rs)

3. **SVD Matrix Extraction Helpers**
   - `mat3_from_svd_row()` - Extract 3×3 matrix from SVD result
   - `mat34_from_svd_row()` - Extract 3×4 matrix from SVD result

### Refactored Files
- [homography.rs](crates/calib-linear/src/homography.rs) - Now uses `math::normalize_points_2d()`
- [epipolar/fundamental.rs](crates/calib-linear/src/epipolar/fundamental.rs) - Uses shared utilities
- [camera_matrix.rs](crates/calib-linear/src/camera_matrix.rs) - Uses shared utilities
- [pnp/dlt.rs](crates/calib-linear/src/pnp/dlt.rs) - Uses shared utilities

### Impact
- **Code reduction**: ~400-500 lines eliminated
- **Maintainability**: Single source of truth for algorithms
- **Consistency**: All normalizations use identical implementation

---

## Phase 2A: Split epipolar.rs into Submodules ✅

**Goal**: Reorganize 1019-line epipolar.rs into focused submodules.

### New Structure
```
crates/calib-linear/src/epipolar/
├── mod.rs               (re-exports, public API)
├── fundamental.rs       (8-point, 7-point, RANSAC)
├── essential.rs         (5-point minimal solver)
├── decomposition.rs     (essential → R, t recovery)
└── polynomial.rs        (Poly3, constraint matrix helpers)
```

### Module Responsibilities
- **fundamental.rs**: Fundamental matrix estimation from pixel correspondences
- **essential.rs**: Essential matrix from calibrated correspondences (Nistér's 5-point)
- **decomposition.rs**: Pose recovery from essential matrix
- **polynomial.rs**: Symbolic polynomial manipulation for essential matrix constraints

### Impact
- **Organization**: Clear separation of concerns
- **Discoverability**: Easier to find specific algorithms
- **API**: 100% backward compatible through re-exports

---

## Phase 2B: Split pnp.rs into Submodules ✅

**Goal**: Reorganize 770-line pnp.rs into focused submodules.

### New Structure
```
crates/calib-linear/src/pnp/
├── mod.rs          (public API, PnpSolver struct)
├── dlt.rs          (DLT linear solver)
├── p3p.rs          (P3P minimal solver with quartic)
├── epnp.rs         (EPnP control-point formulation)
├── ransac.rs       (RANSAC wrapper for robust estimation)
└── pose_utils.rs   (Kabsch algorithm for pose recovery)
```

### Module Responsibilities
- **pose_utils.rs**: SVD-based pose recovery from point correspondences
- **dlt.rs**: Linear least-squares PnP with SO(3) projection
- **p3p.rs**: Minimal 3-point solver with quartic polynomial
- **epnp.rs**: Efficient N-point solver using control points
- **ransac.rs**: Outlier rejection wrapper

### Impact
- **Clarity**: Each solver in its own focused module
- **Testability**: Easier to unit test individual algorithms
- **API**: 100% backward compatible

---

## Phase 3: Production Utilities in calib-core ✅

**Goal**: Extract test-only coordinate utilities into production code.

### Created
- [`crates/calib-core/src/math/coordinate_utils.rs`](crates/calib-core/src/math/coordinate_utils.rs)

### New Production Functions
1. `pixel_to_normalized(pixel, K)` - Apply K⁻¹ to convert pixel → normalized coords
2. `normalized_to_pixel(normalized, K)` - Apply K to convert normalized → pixel
3. `undistort_pixel(pixel, K, distortion)` - Combined K⁻¹ + undistortion
4. `distort_to_pixel(normalized, K, distortion)` - Combined distortion + K

### Refactored
- [`crates/calib-core/src/math/mod.rs`](crates/calib-core/src/math/mod.rs) - Re-exports utilities
- [`crates/calib-core/src/test_utils.rs`](crates/calib-core/src/test_utils.rs) - Now wraps production code

### Impact
- **Reusability**: Utilities available for general use, not just tests
- **Generic**: Works with any `DistortionModel` trait implementation
- **Documentation**: Comprehensive doctests and examples

---

## Phase 4: Scheimpflug + Distortion Integration ✅

**Goal**: Add full Scheimpflug sensor parameter optimization to calib-optim.

### Part A: Integration Tests (calib-core)

Created [`crates/calib-core/tests/scheimpflug_distortion.rs`](crates/calib-core/tests/scheimpflug_distortion.rs):

1. **scheimpflug_with_brown_conrady_roundtrip**
   - Tests forward projection + backward backprojection
   - Validates roundtrip accuracy < 1e-6

2. **scheimpflug_distortion_affects_projection**
   - Confirms distortion visibly changes projection output

3. **scheimpflug_tilt_affects_projection**
   - Confirms Scheimpflug tilt visibly affects projection

4. **combined_scheimpflug_distortion_unproject**
   - Tests backprojection with combined Scheimpflug + distortion effects

**Results**: All 4 tests pass with excellent accuracy.

### Part B: Optimization Infrastructure (calib-optim)

**1. New Factor Type** ([`ir/types.rs`](crates/calib-optim/src/ir/types.rs))
```rust
FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 {
    pw: [f64; 3], uv: [f64; 2], w: f64
}
```
- Parameter layout: `[cam, dist, sensor, pose]`
- Validation for 4D Euclidean intrinsics, 5D distortion, 2D sensor, 7D SE3 pose

**2. Generic Residual Functions** ([`factors/reprojection_model.rs`](crates/calib-optim/src/factors/reprojection_model.rs))
- `reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic<T: RealField>()`
- `tilt_projection_matrix_generic<T>()` - Autodiff-compatible Scheimpflug homography
- `apply_scheimpflug_generic<T>()` - Sensor transformation

**3. Backend Integration** ([`backend/tiny_solver_backend.rs`](crates/calib-optim/src/backend/tiny_solver_backend.rs))
- `TinyReprojPointDistScheimpflugFactor` struct
- Factor compilation for tiny-solver backend

**4. Comprehensive Test** ([`tests/scheimpflug_optimization.rs`](crates/calib-optim/tests/scheimpflug_optimization.rs))
- Full end-to-end optimization with synthetic data
- Optimizes 4 intrinsics + 5 distortion + 2 Scheimpflug + N poses

**Convergence Results**:
```
Intrinsics errors: fx=4.55e-6, fy=3.59e-6, cx=1.15e-4, cy=1.06e-4
Distortion errors: k1=1.36e-9, k2=8.47e-8, p1=3.96e-8, p2=4.23e-8
Scheimpflug errors: tilt_x=1.34e-7, tilt_y=1.43e-7
Final cost: 3.205746e-15
```

### Impact
- **Production-ready**: Full Scheimpflug optimization infrastructure
- **Accuracy**: Near machine precision convergence
- **Architecture**: Follows backend-agnostic IR pattern
- **Extensibility**: Easy to add to any optimization problem

---

## Phase 5: RANSAC Boilerplate Evaluation ✅

**Decision**: **Skip macro creation** - maintain explicit implementations.

### Rationale

After analyzing the three RANSAC implementations (homography, fundamental matrix, PnP), the decision was made to **not create a macro** for the following reasons:

1. **Limited actual duplication** (~10-15 lines per implementation)
2. **High customization requirements**:
   - Each has unique datum structures
   - Different fit() implementations calling different solvers
   - Completely different residual computations
   - Custom degeneracy checks (or defaults)
   - Optional custom refit logic

3. **Clarity vs. abstraction trade-off**:
   - Current explicit code is easy to understand and modify
   - A macro handling all variations would be complex
   - Maintenance burden of macro > maintenance of explicit code

4. **Following Rust philosophy**: Explicit is better than implicit when clarity matters

### Current Pattern (Recommended)
```rust
#[derive(Clone)]
struct Datum { /* problem-specific fields */ }

struct Estimator;
impl Estimator for Estimator {
    type Datum = Datum;
    type Model = Model;
    const MIN_SAMPLES: usize = N;

    fn fit(data: &[Self::Datum], indices: &[usize]) -> Option<Self::Model> {
        // Problem-specific extraction and solver call
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        // Problem-specific geometric distance
    }

    // Optional: custom is_degenerate() and refit()
}
```

This pattern is **clear, explicit, and easy to customize** for each specific use case.

---

## Summary of Changes

### Files Created (7 new files)
1. `crates/calib-linear/src/math.rs` - Math utilities module
2. `crates/calib-core/src/math/coordinate_utils.rs` - Production coordinate utils
3. `crates/calib-core/tests/scheimpflug_distortion.rs` - Integration tests
4. `crates/calib-optim/tests/scheimpflug_optimization.rs` - Optimization test
5. `crates/calib-linear/src/epipolar/mod.rs` - Epipolar module re-exports
6. `crates/calib-linear/src/epipolar/polynomial.rs` - Polynomial constraints
7. ... (and 6 other submodule files for epipolar/ and pnp/)

### Files Modified (15+ files)
- All dependent modules refactored to use consolidated utilities
- IR types extended with Scheimpflug factor
- Backend integration completed
- Test utilities refactored to wrap production code

### Test Results
```
✅ All 65+ workspace tests passing
✅ 5 new integration tests for Scheimpflug
✅ Zero clippy warnings
✅ Documentation tests passing
```

---

## Design Decisions

### 1. Why calib-linear/src/math.rs instead of calib-core?

**Rationale**:
- Hartley normalization is a **linear algorithm preprocessing step**, not core math
- Polynomial solvers are **minimal solver internals**, not general-purpose
- Keeps calib-core focused on camera models and types
- Separation of concerns: algorithms vs. data structures

**Exception**: Coordinate transformations (pixel ↔ normalized) ARE general-purpose → belong in calib-core.

### 2. Why split epipolar.rs but evaluate pnp.rs differently?

**Epipolar split rationale**:
- Mixed **three distinct problem types** (F, E, decomposition) + polynomial infrastructure
- High benefit from organization

**PnP structure**:
- Contains **variants of same problem** (DLT, P3P, EPnP all solve pose-from-points)
- After split, organization improved but pattern was consistent
- Split was beneficial and completed

### 3. Why skip RANSAC macro?

**Rationale**:
- Macro would add **indirection** without significant benefit
- Each RANSAC use case has **unique geometry** and **custom logic**
- Explicit code is **clearer** and **easier to maintain**
- Follows Rust's "explicit over implicit" philosophy

---

## Future Work Enabled

This refactoring prepares the codebase for future features:

1. **Rig Extrinsics Calibration**
   - Can use consolidated `math::normalize_points_3d()`
   - Backend-agnostic IR pattern is established

2. **Hand-Eye Calibration**
   - Can reuse `math::solve_quartic_real()` for AX=XB solvers
   - Follow planar_intrinsics pattern for problem builder

3. **Bundle Adjustment**
   - Reuse `math::normalize_points_3d()` for conditioning
   - Extend IR with 3D point parameter blocks

4. **Linescan Cameras**
   - Extend SensorModel trait (similar to Scheimpflug pattern)
   - Add linescan reprojection factor following Scheimpflug example

---

## Conclusion

All refactoring phases (1-5) are complete:

- ✅ **Phase 1**: Math utilities consolidated (~400-500 lines eliminated)
- ✅ **Phase 2A**: epipolar.rs split into 4 focused modules
- ✅ **Phase 2B**: pnp.rs split into 5 focused modules
- ✅ **Phase 3**: Production coordinate utilities added to calib-core
- ✅ **Phase 4**: Scheimpflug optimization infrastructure (production-ready)
- ✅ **Phase 5**: RANSAC pattern evaluated (explicit code preferred)

The codebase is now:
- **Better organized** with clear module boundaries
- **Free of significant duplication** (~500 lines eliminated)
- **Extended with production features** (Scheimpflug optimization)
- **Well-tested** (65+ tests, all passing)
- **Ready for future development** (clean foundation for new features)

**All changes maintain 100% backward compatibility.**
