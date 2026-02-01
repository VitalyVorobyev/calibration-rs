# References and Further Reading

> **[COLLAB]** Please help curate and prioritize this reference list, and add any additional references.

## Foundational Textbooks

- **Hartley, R.I. & Zisserman, A.** (2004). *Multiple View Geometry in Computer Vision*. 2nd edition. Cambridge University Press. — The definitive reference for projective geometry, fundamental/essential matrices, homography, triangulation, and bundle adjustment.

- **Ma, Y., Soatto, S., Kosecka, J., & Sastry, S.S.** (2004). *An Invitation to 3-D Vision*. Springer. — Rigorous treatment of Lie groups and geometric vision with emphasis on SE(3) and optimization on manifolds.

## Camera Calibration

- **Zhang, Z.** (2000). "A Flexible New Technique for Camera Calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334. — Zhang's planar calibration method: intrinsics from homographies via the image of the absolute conic.

- **Brown, D.C.** (1966). "Decentering Distortion of Lenses." *Photometric Engineering*, 32(3), 444-462. — The original Brown-Conrady distortion model with radial and tangential components.

## Minimal Solvers

- **Nister, D.** (2004). "An Efficient Solution to the Five-Point Relative Pose Problem." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(6), 756-770. — The 5-point essential matrix solver used in stereo and visual odometry.

- **Kneip, L., Scaramuzza, D., & Siegwart, R.** (2011). "A Novel Parametrization of the Perspective-Three-Point Problem for a Direct Computation of Absolute Camera Position and Orientation." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. — The P3P solver used for camera pose estimation.

## Hand-Eye Calibration

- **Tsai, R.Y. & Lenz, R.K.** (1989). "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358. — The Tsai-Lenz method for solving AX=XB.

## Robust Estimation

- **Fischler, M.A. & Bolles, R.C.** (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." *Communications of the ACM*, 24(6), 381-395. — The original RANSAC paper.

- **Hartley, R.I.** (1997). "In Defense of the Eight-Point Algorithm." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(6), 580-593. — Demonstrates that data normalization makes the 8-point algorithm competitive with iterative methods.

## Non-Linear Optimization

- **Levenberg, K.** (1944). "A Method for the Solution of Certain Non-Linear Problems in Least Squares." *Quarterly of Applied Mathematics*, 2(2), 164-168.

- **Marquardt, D.W.** (1963). "An Algorithm for Least-Squares Estimation of Nonlinear Parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.

- **Triggs, B., McLauchlan, P.F., Hartley, R.I., & Fitzgibbon, A.W.** (2000). "Bundle Adjustment — A Modern Synthesis." *International Workshop on Vision Algorithms*. Springer. — Comprehensive survey of bundle adjustment techniques.

## Lie Groups in Vision

- **Sola, J., Deray, J., & Atchuthan, D.** (2018). "A Micro Lie Theory for State Estimation in Robotics." *arXiv:1812.01537*. — Accessible introduction to Lie groups for robotics and vision, covering SO(3), SE(3), and their tangent spaces.

<!-- [COLLAB]: Add any additional references relevant to the implementations -->
