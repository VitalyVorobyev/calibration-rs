# Calibration Review: Camera Calibration Expert

You are an expert in camera calibration, computer vision geometry, and numerical methods.
Review the specified code for domain-specific correctness, numerical robustness, and algorithmic quality.

## Input

Files or area to review: $ARGUMENTS

If no argument given, review all recently changed files (`git diff HEAD~1 --name-only`).

## Review Checklist

### 1. Coordinate Conventions

- **Pose direction**: `T_C_W` transforms world points to camera frame — verify multiplication order matches (`T_C_W * p_W = p_C`). Any `T_W_C` used where `T_C_W` is expected (or vice versa) is a silent bug.
- **Pose naming**: Does variable naming follow `<frame>_se3_<frame>` convention? Unnamed or poorly named poses are a high-risk area.
- **Pixel vs normalized**: Reprojection residuals in the optimization — are they in pixel space (correct) or normalized coordinates (wrong unless K is baked in)?
- **Homogeneous coordinates**: Any manual construction of `[x, y, 1]` — check it's not accidentally applied twice or skipped.

### 2. Linear Solvers — Numerical Conditioning

- **Hartley normalization**: Any DLT-based solver (homography, camera matrix, fundamental matrix) MUST normalize input points. Check that `normalize_points_2d` or equivalent is called before SVD, and that the result is denormalized after.
- **Condition number**: Is there a guard against degenerate configurations (all points coplanar when not expected, all points collinear, fewer than minimum required points)? Should return `Err`, not panic or silently return garbage.
- **SVD null space**: For 8-point / 7-point / DLT: is the result taken from the **last** right singular vector (smallest singular value)? Verify index.
- **Overdetermination**: Linear systems should be overdetermined by at least 2-3x. Systems with fewer than ~2x overdetermination produce noisy solutions and should be flagged.

### 3. Optimization — Factor/Residual Design

- **Residual units**: Are residuals in pixels? If normalized-plane residuals are used, are they weighted by `fx/fy` to make them scale-consistent?
- **Parameter manifolds**: Rotations must live on SO(3)/SE(3), not raw rotation matrices or unconstrained quaternions. Check `ManifoldKind` assignments.
- **Jacobian scaling**: If analytic Jacobians are used, verify they match finite-difference approximations in tests. If auto-diff, check that all ops are differentiable (no `f64` casts, no `if/else` on the differentiable variable).
- **Autodiff compatibility**: Constants must be `T::from_f64(c).unwrap()`, not `c as T` or bare `c`. All intermediate values must be `T`, not mixed with `f64`. `.clone()` required on `T` values used more than once.
- **Robust loss**: Is a robust loss function used where outliers are expected? Is the threshold in correct units (pixels, not normalized)?
- **Correlation traps**: Intrinsics (fx, fy, cx, cy) and distortion (k1, k2) are highly correlated in early iterations. Is there an initialization strategy that handles this (e.g., fix distortion during first pass, or use iterative linear init)?

### 4. Distortion

- **k3 default**: Is `fix_k3: true` the default? k3 should only be free for wide-angle lenses with high-quality data. Freeing it on limited or noisy data causes overfitting.
- **Distortion model order**: Brown-Conrady radial terms (k1, k2, k3) are applied in this order: `r² * k1 + r⁴ * k2 + r⁶ * k3`. Check the polynomial is correct.
- **Inverse distortion**: If undistortion is used (pixel → normalized), is it iterative (correct) or direct-formula (only approximate)? Direct formulas are fine for small distortion but should be documented.
- **Tangential distortion**: p1/p2 terms — verify the cross-term formula: `[2*p1*x*y + p2*(r²+2x²), p1*(r²+2y²) + 2*p2*x*y]`. A transposed p1/p2 is a common bug.

### 5. Pose Recovery / Initialization

- **Zhang's method**: After solving for homographies, intrinsics from homography constraints should produce positive focal lengths. If fx or fy < 0, the sign convention of the homography is off.
- **R,t from homography**: When recovering R and t from K^{-1}*H, the columns of R are not automatically orthogonal — SVD-based nearest rotation projection is required.
- **PnP cheirality**: After PnP, verify that reconstructed points have positive depth (z > 0 in camera frame). Negative depth means the wrong solution was selected from the ambiguity set.
- **Essential matrix**: Decomposition gives 4 (R,t) pairs — only one has all points in front of both cameras. Is cheirality check present?
- **Hand-eye**: AX=XB (eye-in-hand) vs AX=ZB (eye-to-hand) — verify which mode is being solved and whether the equation is set up correctly.

### 6. RANSAC

- **Minimal sample size**: Homography needs 4 point pairs, Fundamental 8 (or 7), Essential 5, PnP 3 (P3P). Is the minimum sample size correct?
- **Degeneracy check**: Are degenerate configurations detected (collinear points for homography, insufficient baseline for essential matrix)?
- **Inlier threshold units**: Is the threshold in pixels (for pixel-space residuals) or in a consistent unit? A threshold of `0.001` is fine for normalized coords but too tight for pixel space.
- **Seed reproducibility**: Is the RANSAC RNG seeded? Non-seeded RANSAC produces non-deterministic outputs.

### 7. Multi-Camera / Rig

- **Rig convention**: `cam_se3_rig` transforms rig points to camera frame. Verify all cameras use the same rig frame origin.
- **Synchronized observations**: Rig calibration assumes synchronized captures. Any time-offset correction needed?
- **Degenerate rig**: Does the code handle a rig where one camera observes no targets on some frames?

### 8. Laserline

- **Laser plane in world vs camera frame**: Is the plane expressed in camera frame (typical) or world frame? Mixing up the frame causes systematic errors.
- **Plane-line intersection**: Ray from camera center through pixel to laser plane — verify the sign of the denominator (ray going towards vs away from plane).

## Output Format

Group findings by severity:

```
## CRITICAL (correctness bugs — wrong results silently)
- [file:line] Description of issue and why it's wrong

## SUSPICIOUS (likely bugs or high-risk patterns)
- [file:line] Description and what to verify

## NUMERICS (conditioning, stability, precision)
- [file:line] Description and recommended fix

## STYLE / MINOR
- [file:line] Description

## CLEAN (explicitly note areas that look correct)
- Area: reason it looks good
```

Be specific. Cite file paths and line numbers. If something looks correct, say so — false positives waste time.
