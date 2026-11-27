# References

Every work the book cites, once. Chapters link here by author and year.

## Textbooks

- <a id="hartley-zisserman-2004"></a>**Hartley, R.I. & Zisserman, A.** (2004). *Multiple View Geometry in Computer Vision*. 2nd edition. Cambridge University Press. — Projective geometry, the fundamental and essential matrices, homographies, triangulation and bundle adjustment.

- <a id="ma-2004"></a>**Ma, Y., Soatto, S., Kosecka, J., & Sastry, S.S.** (2004). *An Invitation to 3-D Vision*. Springer. — Lie groups and geometric vision, with emphasis on SE(3) and optimization on manifolds.

## Camera Models and Calibration

- <a id="zhang-2000"></a>**Zhang, Z.** (2000). "A Flexible New Technique for Camera Calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334. — Intrinsics from the homographies of a planar target, through the image of the absolute conic.

- <a id="brown-1966"></a>**Brown, D.C.** (1966). "Decentering Distortion of Lenses." *Photogrammetric Engineering*, 32(3), 444-462. — The Brown-Conrady distortion model with radial and tangential terms.

- <a id="fitzgibbon-2001"></a>**Fitzgibbon, A.W.** (2001). "Simultaneous Linear Estimation of Multiple View Geometry and Lens Distortion." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. — The one-parameter division model of radial distortion.

- <a id="kannala-brandt-2006"></a>**Kannala, J. & Brandt, S.S.** (2006). "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 28(8), 1335-1340. — The fisheye model: an odd polynomial in the angle from the optical axis.

- <a id="hagemann-2022"></a>**Hagemann, A., Knorr, M., Janssen, H., & Stiller, C.** (2022). "Inferring Bias and Uncertainty in Camera Calibration." *International Journal of Computer Vision*, 130(1), 17-32. — Systematic and random calibration error separated; two calibrations compared in image space after a compensating rotation of the viewing rays.

- <a id="steger-2017"></a>**Steger, C.** (2017). "A Comprehensive and Versatile Camera Model for Cameras with Tilt Lenses." *International Journal of Computer Vision*, 123(2), 121-159. — The tilt-lens camera model with an image-side distance, its telecentric variants and its degenerate cases.

- <a id="kumar-ahuja-2014"></a>**Kumar, A. & Ahuja, N.** (2014). "Non-frontal Camera Calibration using Focal Stack Imagery." *International Conference on Pattern Recognition (ICPR)*, 202-207. — A sensor tilt cannot be calibrated geometrically under ideal perspective imaging; image blur determines it.

## Calibration Targets

- <a id="garrido-jurado-2014"></a>**Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F.J., & Marín-Jiménez, M.J.** (2014). "Automatic Generation and Detection of Highly Reliable Fiducial Markers under Occlusion." *Pattern Recognition*, 47(6), 2280-2292. — The ArUco markers a ChArUco board carries.

- <a id="stelldinger-2024"></a>**Stelldinger, P., Schönherr, N., & Biermann, J.** (2024). "PuzzleBoard: A New Camera Calibration Pattern with Position Encoding." *DAGM German Conference on Pattern Recognition (GCPR)*. — The self-identifying chessboard.

## Minimal Solvers and Two-View Geometry

- <a id="hartley-1997"></a>**Hartley, R.I.** (1997). "In Defense of the Eight-Point Algorithm." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(6), 580-593. — Data normalization makes the linear 8-point algorithm competitive with iterative methods.

- <a id="nister-2004"></a>**Nister, D.** (2004). "An Efficient Solution to the Five-Point Relative Pose Problem." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(6), 756-770. — The 5-point essential matrix solver.

- <a id="kneip-2011"></a>**Kneip, L., Scaramuzza, D., & Siegwart, R.** (2011). "A Novel Parametrization of the Perspective-Three-Point Problem for a Direct Computation of Absolute Camera Position and Orientation." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. — The P3P solver.

- <a id="lepetit-2009"></a>**Lepetit, V., Moreno-Noguer, F., & Fua, P.** (2009). "EPnP: An Accurate O(n) Solution to the PnP Problem." *International Journal of Computer Vision*, 81(2), 155-166. — Pose from four or more points through four control points.

## Dense Stereo

- <a id="hirschmuller-2008"></a>**Hirschmüller, H.** (2008). "Stereo Processing by Semiglobal Matching and Mutual Information." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2), 328-341. — Semi-global matching: the matching cost aggregated along image paths with two smoothness penalties.

## Hand-Eye Calibration

- <a id="tsai-lenz-1989"></a>**Tsai, R.Y. & Lenz, R.K.** (1989). "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358. — The Tsai-Lenz solution of AX=XB.

- <a id="park-martin-1994"></a>**Park, F.C. & Martin, B.J.** (1994). "Robot Sensor Calibration: Solving AX=XB on the Euclidean Group." *IEEE Transactions on Robotics and Automation*, 10(5), 717-721. — Rotation from the Lie-algebra (rotation-vector) alignment.

- <a id="daniilidis-1999"></a>**Daniilidis, K.** (1999). "Hand-Eye Calibration Using Dual Quaternions." *The International Journal of Robotics Research*, 18(3), 286-298. — Rotation and translation together.

- <a id="shah-2013"></a>**Shah, M.** (2013). "Solving the Robot-World/Hand-Eye Calibration Problem Using the Kronecker Product." *Journal of Mechanisms and Robotics*, 5(3), 031007. — AX=ZB from absolute poses.

- <a id="tabb-2017"></a>**Tabb, A. & Ahmad Yousef, K.M.** (2017). "Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods." *Machine Vision and Applications*, 28(5-6), 569-590. — Source of a public hand-eye dataset of the regression suite.

- <a id="ali-2019"></a>**Ali, I., Suominen, O., Gotchev, A., & Ruiz Morales, E.** (2019). "Methods for Simultaneous Robot-World-Hand–Eye Calibration: A Comparative Study." *Sensors*, 19(12), 2837. — Source of a public hand-eye dataset of the regression suite.

## Robust Estimation

- <a id="fischler-bolles-1981"></a>**Fischler, M.A. & Bolles, R.C.** (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." *Communications of the ACM*, 24(6), 381-395. — RANSAC.

- <a id="huber-1964"></a>**Huber, P.J.** (1964). "Robust Estimation of a Location Parameter." *The Annals of Mathematical Statistics*, 35(1), 73-101. — The Huber loss.

## Nonparametric Regression and Inference

- <a id="fan-gijbels-1996"></a>**Fan, J. & Gijbels, I.** (1996). *Local Polynomial Modelling and Its Applications*. Monographs on Statistics and Applied Probability 66. Chapman & Hall. — Local polynomial regression, its boundary behaviour and bandwidth selection.

- <a id="cameron-miller-2015"></a>**Cameron, A.C. & Miller, D.L.** (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources*, 50(2), 317-372. — Standard errors when observations are correlated within clusters, including the few-cluster case.

## Non-Linear Optimization

- <a id="levenberg-1944"></a>**Levenberg, K.** (1944). "A Method for the Solution of Certain Non-Linear Problems in Least Squares." *Quarterly of Applied Mathematics*, 2(2), 164-168.

- <a id="marquardt-1963"></a>**Marquardt, D.W.** (1963). "An Algorithm for Least-Squares Estimation of Nonlinear Parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.

- <a id="triggs-2000"></a>**Triggs, B., McLauchlan, P.F., Hartley, R.I., & Fitzgibbon, A.W.** (2000). "Bundle Adjustment — A Modern Synthesis." *International Workshop on Vision Algorithms*. Springer. — Survey of bundle adjustment, including the second-order correction for robust losses.

## Lie Groups

- <a id="sola-2018"></a>**Sola, J., Deray, J., & Atchuthan, D.** (2018). "A Micro Lie Theory for State Estimation in Robotics." *arXiv:1812.01537*. — Introduction to SO(3), SE(3) and their tangent spaces for estimation.
