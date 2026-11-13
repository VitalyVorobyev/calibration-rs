# Measurements with Uncertainty

`mvg::metrology` measures with error bars. Given points and the joint covariance of their coordinates, it returns distances, angles, plane fits and point-to-plane distances, each with its standard deviation. The points come from [`triangulate`](mvg_pose.md) or [`bundle_adjust`](mvg_bundle_adjust.md), which return the covariance of the points they refined, the calibration's uncertainty included.

## Usage

```rust,no_run
use vision_calibration::control::RunControl;
use vision_calibration::core::CalibratedRig;
use vision_calibration::mvg::metrology::{PointSet, angle, distance, fit_plane, point_to_plane};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::Captures;
# fn main() -> anyhow::Result<()> {
# let rig: CalibratedRig = unimplemented!("the calibrated rig");
# let tracks: Vec<Vec<RigObservation>> = unimplemented!("the corners of a board, one track per corner");
# let all_corners: Vec<usize> = (0..49).collect();

// rig:    CalibratedRig;  tracks: the corners of a board, one track per corner
let triangulation = triangulate(&rig, Captures::Rig, &tracks, &RunControl::default())?;
let set = PointSet::from_triangulation(&triangulation)?;

let length = distance(&set, 0, 7)?;        // Scalar { value, sigma }, metres
let corner = angle(&set, 3, 0, 24)?;       // the angle at point 0, radians
let board = fit_plane(&set, &all_corners)?; // normal, distance, covariance, rms
let height = point_to_plane(&set, &board.indices, 49)?;

println!("{:.4} m ± {:.4} m", length.value, length.sigma);
# Ok(())
# }
```

```python
from vision_calibration import mvg

found = mvg.triangulate(result, np.eye(4), tracks)
points = mvg.PointSet.from_triangulation(found)

length = mvg.distance(points, 0, 7)                 # Scalar(value, sigma), metres
corner = mvg.angle(points, 3, 0, 24)                # the angle at point 0, radians
board = mvg.fit_plane(points, all_corners)          # normal, distance, covariance, rms
height = mvg.point_to_plane(points, board, 49)

print(f"{length.value:.4f} m ± {length.sigma:.4f} m")
```

`PointSet::new(points, covariance)` takes any points with a 3N × 3N covariance (rows ordered `x, y, z` per point), and `PointSet::independent` takes one 3 × 3 covariance per point when the errors are not correlated. `from_triangulation` and `from_bundle_adjustment` read the covariance out of the result and fail with an `InvalidInput` naming the point (and the reason, when the whole uncertainty is unavailable) if a point has none. In Python, `mvg.PointSet(points, covariance)` takes an `(N, 3)` array and a `(3N, 3N)` one, and `from_triangulation`, `from_laser_triangulation` and `from_bundle_adjustment` raise `ValueError` in the same cases.

## Why the Joint Covariance

Every point measured with one calibration shares that calibration's error: a focal length that is 0.1 % too long scales every depth, a baseline error shifts them all. The errors of two points are correlated, and a distance between them feels the shared part differently from the independent part. A σ built from each point's own covariance alone ignores that and is wrong in either direction.

All the points of a measurement must therefore come from one call that produced their joint covariance: two separate `triangulate` calls lose the cross terms.

The check: eight points with a covariance made of shared terms (a common translation, a scale, a stretch along the depth, a shear) plus independent noise, 20 000 draws. The scatter of each measurement over its stated σ:

| measurement | joint covariance | cross terms dropped |
|---|---|---|
| distance (short) | 0.994 | 0.52 |
| distance (long) | 0.992 | 0.74 |
| angle | 0.992 | 0.44 |
| point-to-plane, outside the plane | 0.990 | 0.64 |
| point-to-plane, a point of the plane | 1.002 | 0.44 |
| plane distance `d` | 1.000 | 0.91 |
| plane normal, two components | 0.995, 1.000 | 0.67, 0.46 |

With the joint covariance the stated σ is right to the 1 % that the draws resolve. Without the cross terms it is off by up to a factor of two.

On a calibrated stereo rig the same holds end to end: over 300 calibrations from freshly noisy board views spread over the image, with a held-out board triangulated each time, the scatter of a 7.5 cm length over its stated σ is 1.00, of a 17.5 cm length 1.02, of the right angle at a board corner 0.97 and of the board plane's distance 0.89. The mean error over σ stays within 0.2 for each. The σ built from the observations alone matches the lengths and the angle (the board's own geometry is dominated by the pixel noise) but understates the plane's distance 2.2-fold: that is the calibration's share.

## Propagation

A measurement $f(x)$ over the coordinates $x$ of its points has

$$\sigma^2 = J\,\Sigma\,J^\top,\qquad J = \frac{\partial f}{\partial x},$$

with $\Sigma$ the block of the joint covariance for the points involved. $J$ comes from dual numbers: the measurement is written once and differentiated exactly, with no finite-difference step. This is first-order propagation: it holds while the points' errors are small against what they measure. For a length the next term is a bias of about $\sigma_\perp^2 / 2d$ ($\sigma_\perp$ the error across the segment, $d$ its length), which is why a short distance between noisy points is biased high even when its σ is right.

`angle` is the angle at the vertex, in $(0, \pi)$; three collinear points are an error, since the angle is not differentiable there.

## Plane Fits

`fit_plane` fits $n \cdot x = d$ by total least squares (the centroid and the smallest eigenvector of the scatter matrix), with $n$ a unit vector oriented so that $d \ge 0$. It needs at least three points that are not collinear; fewer, repeated or out-of-range indices are errors before any work.

Its covariance holds the points as consider parameters. The fit is the stationary point $g(\theta, p) = 0$ of the sum of squared orthogonal distances over $\theta = (\alpha, \beta, d)$, where $n(\alpha, \beta)$ is the normalized $n_0 + \alpha t_1 + \beta t_2$ for a tangent basis $t_1, t_2$ of the fitted normal. The implicit-function theorem gives how the plane moves with the points,

$$\frac{\partial \theta}{\partial p} = -\Big(\frac{\partial g}{\partial \theta}\Big)^{-1}\frac{\partial g}{\partial p},\qquad \Sigma_\theta = \frac{\partial \theta}{\partial p}\,\Sigma_p\,\frac{\partial \theta}{\partial p}^\top .$$

`PlaneFit::covariance` is $\Sigma_\theta$ in the units rad², rad·m and m², with the basis in `tangent_basis`. It comes from the points' covariance, not from how well the points lie on the plane: a noisy but planar scene does not widen it, and `rms` (the orthogonal distance of the points to the plane) is a planarity diagnostic beside it.

## Point to Plane

`point_to_plane(set, plane, p)` is the signed distance $n\cdot p - d$ of a point to the plane fitted through `plane`, positive on the side the normal points to. The plane and the point are correlated, so the gradient has two parts,

$$J = \frac{\partial s}{\partial p}\Big|_{\text{direct}} + \frac{\partial s}{\partial \theta}\,\frac{\partial \theta}{\partial p},$$

and the σ uses every involved point's covariance with the cross terms. `p` may be one of the plane's points (a flatness measure of a corner against the board), and the result accounts for that.
