# Hand-Eye Calibration (Tsai-Lenz)

Hand-eye calibration estimates the rigid transform between a camera and a robot gripper (or between a camera and a robot base). It is essential for any application where a camera is mounted on a robot arm and the robot needs to localize objects in its own coordinate frame.

## The AX = XB Problem

### Setup

Consider a camera rigidly mounted on a robot gripper (**eye-in-hand** configuration). The system involves four coordinate frames:

- **Base** (B): The robot's fixed base frame
- **Gripper** (G): The robot's end-effector frame (known from robot kinematics)
- **Camera** (C): The camera frame (observations come from here)
- **Target** (T): The calibration board frame (fixed in the world)

The unknown is $X = T_{G,C}$ (gripper-to-camera transform).

### The Equation

Given two observations $(i, j)$, we can compute:

- **Robot motion**: $A_{ij} = T_{G_j,G_i} = T_{B,G_j}^{-1} T_{B,G_i}$ (relative gripper motion, known from robot kinematics)
- **Camera motion**: $B_{ij} = T_{C_j,C_i} = T_{T,C_j}^{-1} T_{T,C_i}$ (relative camera motion, from calibration board observations)

Since the camera is rigidly attached to the gripper:

$$A_{ij} X = X B_{ij}$$

This is the classic $AX = XB$ equation. We need to find $X$ that satisfies this for all motion pairs.

### Eye-to-Hand Variant

When the camera is fixed and the target is on the gripper (**eye-to-hand**), the equation becomes:

$$A_{ij} X = X B_{ij}$$

where now $A$ is the relative gripper motion and $B$ is the relative target-in-camera motion, and $X = T_{C,B}$ (camera-to-base transform).

## The Tsai-Lenz Method

### Overview

Tsai and Lenz (1989) decomposed $AX = XB$ into separate rotation and translation subproblems:

1. **Rotation**: Solve $R_A R_X = R_X R_B$ for $R_X$
2. **Translation**: Given $R_X$, solve $(R_A - I) \mathbf{t}_X = R_X \mathbf{t}_B - \mathbf{t}_A$ for $\mathbf{t}_X$

### Step 1: All-Pairs Motion Computation

From $M$ observations, construct all $\binom{M}{2}$ motion pairs. For each pair $(i, j)$:

$$A_{ij} = T_{B,G_i}^{-1} \cdot T_{B,G_j}$$
$$B_{ij} = T_{T,C_i}^{-1} \cdot T_{T,C_j}$$

### Step 2: Filtering

Discard pairs with insufficient rotation (degenerate for the rotation subproblem):

- Reject pairs where $\|\text{angle}(R_A)\| < \theta_{\min}$ (default: 10°)
- Optionally reject pairs where rotation axes of $A$ and $B$ are near-parallel (ill-conditioned)

**Rotation diversity is critical**: If all robot motions are rotations around the same axis, the hand-eye rotation around that axis is undetermined. Use poses with diverse rotation axes (roll, pitch, and yaw).

### Step 3: Rotation Estimation

The rotation constraint $R_A R_X = R_X R_B$ is solved using quaternion algebra.

Convert $R_A$ and $R_B$ to quaternions $\mathbf{q}_A$ and $\mathbf{q}_B$. The constraint becomes:

$$\mathbf{q}_A \otimes \mathbf{q}_X = \mathbf{q}_X \otimes \mathbf{q}_B$$

Using the left and right quaternion multiplication matrices $L(\mathbf{q}_A)$ and $R(\mathbf{q}_B)$:

$$(L(\mathbf{q}_A) - R(\mathbf{q}_B)) \mathbf{q}_X = 0$$

Stacking all $N$ motion pairs gives a $4N \times 4$ system:

$$M \mathbf{q}_X = 0, \quad M = \begin{bmatrix} L(\mathbf{q}_{A_1}) - R(\mathbf{q}_{B_1}) \\ \vdots \\ L(\mathbf{q}_{A_N}) - R(\mathbf{q}_{B_N}) \end{bmatrix}$$

Solve via SVD: $\mathbf{q}_X$ is the right singular vector of $M$ corresponding to the smallest singular value. Normalize to unit quaternion.

### Step 4: Translation Estimation

Given $R_X$, the translation constraint for each motion pair is:

$$(R_A - I) \mathbf{t}_X = R_X \mathbf{t}_B - \mathbf{t}_A$$

This is a $3N \times 3$ overdetermined linear system $C \mathbf{t}_X = \mathbf{d}$, solved via least squares:

$$\mathbf{t}_X = (C^T C)^{-1} C^T \mathbf{d}$$

with optional ridge regularization for numerical stability.

## Target-in-Base Estimation

After finding $X = T_{G,C}$, the target pose in the base frame $T_{B,T}$ can be estimated. For each observation $i$:

$$T_{B,T} = T_{B,G_i} \cdot X \cdot T_{C_i,T}$$

The estimates from different views are averaged (quaternion averaging for rotation, arithmetic mean for translation).

Alternatively, $T_{B,T}$ is included in the non-linear optimization.

## Practical Requirements

- **Minimum 3 views** with diverse rotations (in practice, 5-10 views recommended)
- **Rotation diversity**: Motions should span multiple rotation axes. Pure translations provide no rotation constraint. Pure Z-axis rotations leave the Z-component of the hand-eye rotation undetermined.
- **Accuracy**: The linear Tsai-Lenz method typically gives 5-20% translation accuracy and 2-10° rotation accuracy. This initializes the non-linear joint optimization.

## API

```rust
let X = estimate_handeye_dlt(
    &base_se3_gripper,      // &[Iso3]: robot poses (base to gripper)
    &target_se3_camera,     // &[Iso3]: camera-to-target poses (inverted)
    min_angle_deg,          // f64: minimum rotation angle filter
)?;
```

Returns $X$ as an `Iso3` transform.

> **OpenCV equivalence**: `cv::calibrateHandEye` with method `CALIB_HAND_EYE_TSAI`.

## References

- Tsai, R.Y. & Lenz, R.K. (1989). "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358.
