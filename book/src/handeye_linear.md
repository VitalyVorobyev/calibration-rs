# Hand-Eye Initialization

Hand-eye calibration estimates the rigid transform between a camera and a robot gripper (or between a camera and a robot base). It is essential for any application where a camera is mounted on a robot arm and the robot needs to localize objects in its own coordinate frame. The workflows start the hand-eye bundle adjustment from a closed-form estimate; four methods are available.

## The Equations

### Frames

- **Base** (B): the robot's fixed base frame
- **Gripper** (G): the robot's end-effector frame (known from robot kinematics)
- **Camera** (C): the camera frame (observations come from here)
- **Target** (T): the calibration board frame

Each view $i$ pairs a robot pose $T_{B,G_i}$ (`base_se3_gripper`) with the board pose the camera observes, $T_{C_i,T}$ (`cam_se3_target`).

### Robot-world form: $A_i X = Z B_i$

In **eye-in-hand** (camera on the gripper, board fixed) the chain from base to camera closes through the board:

$$T_{B,G_i} \, T_{G,C} = T_{B,T} \, T_{T,C_i}$$

In **eye-to-hand** (board on the gripper, camera fixed) it closes through the camera:

$$T_{B,G_i} \, T_{G,T} = T_{B,C} \, T_{C,T_i}$$

Both are $A_i X = Z B_i$ with $A_i = T_{B,G_i}$:

| mode | $B_i$ | $X$ | $Z$ |
|---|---|---|---|
| eye-in-hand | $T_{T,C_i}$ | `gripper_se3_cam` | `base_se3_target` |
| eye-to-hand | $T_{C,T_i}$ | `gripper_se3_target` | `base_se3_cam` (returned as `cam_se3_base`) |

For a rig the observer is the rig frame instead of a camera.

### Motion form: $A X = X B$

Two views $i, j$ eliminate $Z$: with $A = A_i^{-1} A_j$ and $B = B_i^{-1} B_j$,

$$A X = X B$$

which splits into a rotation and a translation part:

$$R_A R_X = R_X R_B, \qquad (R_A - I)\,\mathbf{t}_X = R_X \mathbf{t}_B - \mathbf{t}_A$$

## The Methods

`handeye_init.method` selects the solver (`HandeyeMethod`, serialized `tsai_lenz`, `park_martin`, `daniilidis`, `shah`). The first three use all view pairs; Shah's method uses the views directly.

- **Tsai-Lenz** (default). The rotation from the quaternion form of $R_A R_X = R_X R_B$: each pair gives $(L(\mathbf{q}_A) - R(\mathbf{q}_B))\,\mathbf{q}_X = 0$ with the left and right quaternion-multiplication matrices; the stacked $4N \times 4$ system's null vector is $\mathbf{q}_X$. Then $\mathbf{t}_X$ from the stacked translation equations by least squares.
- **Park-Martin**. With the rotation vectors $\boldsymbol\alpha = \log R_A$, $\boldsymbol\beta = \log R_B$, the rotation constraint reads $\boldsymbol\alpha = R_X \boldsymbol\beta$; $R_X$ is the rotation closest to $\sum \boldsymbol\alpha \boldsymbol\beta^T$ (orthogonal Procrustes). It weighs a pair by its rotation angle squared, Tsai-Lenz by $\sin^2(\theta/2)$. The translation as for Tsai-Lenz.
- **Daniilidis**. Rotation and translation together, from dual quaternions $\check{q} = \mathbf{q} + \varepsilon \tfrac{1}{2}\mathbf{t}\mathbf{q}$: each pair gives six linear equations in the eight entries of $(\mathbf{q}_X, \mathbf{q}'_X)$, whose two-dimensional solution space is fixed by $|\mathbf{q}_X| = 1$ and $\mathbf{q}_X \cdot \mathbf{q}'_X = 0$.
- **Shah**. $X$ and $Z$ together from $A_i X = Z B_i$, without motion pairs. The rotations satisfy $R_{A_i} R_X = R_Z R_{B_i}$, which is linear in the entries of $R_X$ and $R_Z$ (a Kronecker-product system); the translations then follow from $R_{A_i}\mathbf{t}_X - \mathbf{t}_Z = R_Z \mathbf{t}_{B_i} - \mathbf{t}_{A_i}$ by least squares.

The pair methods close $Z$ through the first view, $Z = A_0 X B_0^{-1}$. Shah's method estimates it over all views, so the fixed target pose (eye-in-hand) or camera pose (eye-to-hand) the bundle adjustment starts from no longer depends on one view's noise.

### Motion requirements

- **Minimum rotation per pair**: a pair whose rotation, in the robot or in the camera, is below `min_motion_angle_deg` (default 5°) is dropped by the pair methods. Small rotations amplify noise into the translation.
- **Two rotation axes**: the pairs that remain must rotate the robot about at least two non-parallel axes. With every rotation about one axis, the rotation of $X$ about that axis and its translation along it are undetermined, for every method (Shah's included). The initialization then fails with a typed error naming the cause, instead of returning an arbitrary transform.
- **Pure translations** carry no rotation information; they are filtered out like any pair below the angle threshold.

## Rotation-Poor Motion

How much do the methods differ when the robot barely rotates? A synthetic study, run on every method from the same data:

- an eye-in-hand camera (fixed, known intrinsics; 9×7 board of 30 mm squares) and a random hand-eye transform (up to ~30°, 10 cm);
- ten robot stations whose viewing direction is spread by up to ±2°, ±5°, ±15° or ±45° about each axis around the board, at ~0.6 m; optionally four more stations that only translate;
- **low** noise: 0.1 px pixel noise, exact robot poses; **high** noise: 0.5 px, robot poses off by up to 0.05° and 0.3 mm per axis;
- `min_motion_angle_deg` lowered to 1° so the ±2° cells have pairs to use;
- the hand-eye bundle adjustment from each method's estimate, compared with the same refinement started from the truth. "Basin" is the share of trials whose refinement ends within half a standard deviation of that reference.

Median errors of the initial estimate over 100 trials per row (stations without the translation-only extras): hand-eye rotation (°) / hand-eye translation (mm) / fixed target translation (mm). Tsai-Lenz and Park-Martin agree to 0.001° and 0.01 mm in every row, so they share a column.

| spread | noise | Tsai-Lenz, Park-Martin | Daniilidis | Shah |
|---|---|---|---|---|
| ±2° | low | 0.45 / 7.2 / 4.8 | **0.24** / **5.3** / 5.0 | 0.45 / 7.3 / 4.9 |
| ±2° | high | 2.37 / **34.7** / **26.8** | **1.30** / 39.4 / 36.2 | 2.35 / 35.8 / **26.8** |
| ±5° | low | **0.17** / 2.4 / **1.6** | 0.19 / **2.2** / 2.1 | **0.17** / 2.3 / **1.6** |
| ±5° | high | **0.89** / 12.2 / **7.8** | 0.95 / **11.4** / 9.4 | **0.89** / 12.3 / 7.9 |
| ±15° | low | **0.041** / **0.56** / 0.39 | 0.063 / 0.62 / 0.45 | **0.041** / 0.58 / **0.38** |
| ±15° | high | **0.22** / **2.9** / 2.1 | 0.33 / 3.3 / 2.4 | **0.22** / 3.0 / **2.0** |
| ±45° | low | **0.008** / **0.09** / 0.07 | 0.009 / 0.10 / 0.07 | **0.008** / 0.10 / **0.06** |
| ±45° | high | 0.052 / 0.50 / 0.59 | 0.053 / 0.51 / 0.56 | 0.052 / 0.50 / **0.39** |

**Basin.** Every refinement, from every method's estimate in every trial, ended at the same minimum as the refinement from the truth: within 0.82 standard deviations of it at worst — the solver's stopping tolerance, the same for every method — with either solver backend and also with the robot poses refined. Even at ±2° and high noise, where the initial rotation is off by 1–2.5° and the translation by 3–4 cm, the bundle adjustment converges.

**Reading.**

- No method is better everywhere. Daniilidis halves the rotation error when the rotations are tiny (±2°), at the price of a larger translation error under noise; from ±5° on it is level or slightly worse. Shah's hand-eye matches Tsai-Lenz's; its fixed target, estimated over all views rather than closed through the first, is better under noise when the motion is rich (0.39 vs 0.59 mm at ±45°).
- Four translation-only stations improve every method by roughly 5–30 % (Daniilidis's rotation at ±2°, high noise: 0.94°): each adds pairs against the rotated stations.
- At the default `min_motion_angle_deg` of 5°, 91 of 100 ±2° captures have too few pairs and every method returns the typed error; lower the threshold to use rotation-poor data.
- Since the basin does not depend on the method, the default stays Tsai-Lenz. Choose Shah when the initial target (or camera) pose matters on its own, Daniilidis when the robot rotates by only a few degrees and the rotation matters more than the translation.

## API

The workflows run the initialization in `step_handeye_init`; the method comes from the config, or from the step's options for one call:

```rust
use vision_calibration::common::HandeyeInitOptions;
use vision_calibration::init::HandeyeMethod;

let mut config = SingleCamHandeyeConfig::default();
config.handeye_init.method = HandeyeMethod::Shah;
session.set_config(config)?;
// ... set the input, run the intrinsics steps ...
let mut opts = HandeyeInitOptions::default();
opts.method = Some(HandeyeMethod::Daniilidis); // this call only
step_handeye_init(&mut session, Some(opts))?;
```

The closed-form solvers are also available on their own, to start a problem of your own:

```rust
use vision_calibration::init::{HandeyeMethod, estimate_eye_in_hand};

let estimate = estimate_eye_in_hand(
    HandeyeMethod::TsaiLenz,
    &base_se3_gripper,  // &[Iso3]: robot poses
    &cam_se3_target, // &[Iso3]: observed board poses
    5.0,                // minimum rotation of a pair, degrees
)?;
let x = estimate.gripper_se3_cam;
let z = estimate.base_se3_target;
```

`estimate_eye_to_hand` takes the same arguments and returns `gripper_se3_target` and `cam_se3_base`.

> **OpenCV equivalence**: `cv::calibrateHandEye` with `CALIB_HAND_EYE_TSAI`, `CALIB_HAND_EYE_PARK` and `CALIB_HAND_EYE_DANIILIDIS`; `cv::calibrateRobotWorldHandEye` with `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`.

## References

- Tsai, R.Y. & Lenz, R.K. (1989). "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358.
- Park, F.C. & Martin, B.J. (1994). "Robot Sensor Calibration: Solving AX=XB on the Euclidean Group." *IEEE Transactions on Robotics and Automation*, 10(5), 717-721.
- Daniilidis, K. (1999). "Hand-Eye Calibration Using Dual Quaternions." *The International Journal of Robotics Research*, 18(3), 286-298.
- Shah, M. (2013). "Solving the Robot-World/Hand-Eye Calibration Problem Using the Kronecker Product." *Journal of Mechanisms and Robotics*, 5(3), 031007.
