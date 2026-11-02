# Fisheye Intrinsics from Bearings

Zhang's method needs the board-to-pixel map to be a homography, which a
fisheye lens breaks. `estimate_equidistant_intrinsics` initializes a
Kannala-Brandt camera (equidistant projection, `KannalaBrandt4`
distortion) by scoring candidate cameras on their pixel error:

1. For a candidate focal length $f$, principal point $\mathbf{c}$ and
   coefficients $k_1, k_2$, every pixel back-projects to a bearing.
2. Each view's pose follows from its bearings
   ([pose from bearings](planar_pose.md#pose-from-bearings)).
3. The board reprojects through the candidate; the sum of squared pixel
   errors is its cost.

The search runs in three stages:

- **Focal sweep.** Forty log-spaced focal lengths, without distortion,
  with $\mathbf{c}$ at the centre of the observations. They span fields
  of view from a few degrees to beyond 270°, so neither the image size nor
  a nominal focal length is needed.
- **Simplex over $(f, c_x, c_y)$**, still without distortion.
- **Radial seed, then all five.** At the incidence angles $\theta$ of the
  current poses, the observed radii follow
  $r = f\theta + f k_1 \theta^3 + f k_2 \theta^5$, linear in
  $(f, f k_1, f k_2)$. Its least-squares solution seeds a simplex over
  $(f, c_x, c_y, k_1, k_2)$.

Pixels are square ($f_x = f_y$), and $k_3$, $k_4$ start at zero; the
non-linear refinement frees them. A coefficient the caller holds is not
fitted and stays at zero (`EquidistantInitOptions`).

```rust
use vision_calibration_linear::equidistant_init::{
    EquidistantInitOptions, estimate_equidistant_intrinsics,
};

let init = estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::default())?;
// init.intrinsics, init.distortion, init.poses, init.rms_error
```

On exact views of a lens seeing up to 115° off-axis, with coefficients
compressing the edge by about 5 %, the result lands within 1 % in focal
length and 1 px in principal point.
