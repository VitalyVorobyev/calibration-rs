# Sensor Models and Scheimpflug Tilt

In standard cameras, the sensor plane is perpendicular to the optical axis. Some industrial cameras — particularly laser triangulation profilers — deliberately tilt the sensor to achieve a larger depth of field along the laser plane (the Scheimpflug condition). The sensor model stage accounts for this tilt.

## The `SensorModel` Trait

```rust,ignore
pub trait SensorModel<S: RealField> {
    fn normalized_to_sensor(&self, n: &Point2<S>) -> Point2<S>;
    fn sensor_to_normalized(&self, s: &Point2<S>) -> Point2<S>;
}
```

The sensor model sits between distortion and intrinsics in the pipeline:

$$\mathbf{p} = K(\underbrace{S(D(\Pi(\mathbf{P}_c)))}_{\text{sensor coords}})$$

## IdentitySensor

For standard cameras with untilted sensors:

$$S(\mathbf{n}) = \mathbf{n}, \quad S^{-1}(\mathbf{s}) = \mathbf{s}$$

This is the default and most common case.

## Scheimpflug Tilt Model

The Scheimpflug condition states that when the lens plane, image plane, and object plane intersect along a common line, the entire object plane is in sharp focus. Laser profilers exploit this by tilting the sensor to keep the laser line in focus.

### Tilt Projection Matrix

The tilt is parameterized by two rotation angles:

- $\tau_x$ — tilt around the horizontal (X) axis
- $\tau_y$ — tilt around the vertical (Y) axis

The tilt homography is constructed as:

$$R_x = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\tau_x & -\sin\tau_x \\ 0 & \sin\tau_x & \cos\tau_x \end{bmatrix}, \quad R_y = \begin{bmatrix} \cos\tau_y & 0 & \sin\tau_y \\ 0 & 1 & 0 \\ -\sin\tau_y & 0 & \cos\tau_y \end{bmatrix}$$

$$R = R_y \cdot R_x$$

The projection onto the tilted sensor plane normalizes by the third row of $R$ applied to the point in homogeneous coordinates:

$$H_{\text{tilt}} = \text{proj}_z \cdot R$$

where $\text{proj}_z$ normalizes by the $Z$-component (perspective division through the tilted plane). In practice, the first two rows of $R$ divided by the third row give a $2 \times 2$ rational transform in the image plane.

> **OpenCV equivalence**: This matches OpenCV's `computeTiltProjectionMatrix` used in the 14-parameter rational model (`CALIB_TILTED_MODEL`).

### `ScheimpflugParams`

```rust,ignore
pub struct ScheimpflugParams {
    pub tilt_x: f64,   // tau_x in radians
    pub tilt_y: f64,   // tau_y in radians
}
```

The `compile()` method generates a `HomographySensor` containing the $3 \times 3$ homography and its precomputed inverse. `ScheimpflugParams::default()` returns zero tilt (identity sensor).

### `HomographySensor`

The general homography sensor applies an arbitrary $3 \times 3$ projective transform:

$$\mathbf{s} = \text{dehomogenize}(H \cdot [\mathbf{n}, 1]^T)$$

$$\mathbf{n} = \text{dehomogenize}(H^{-1} \cdot [\mathbf{s}, 1]^T)$$

This is used as the runtime representation compiled from `ScheimpflugParams`.

## When to Use Scheimpflug

- **Laser triangulation profilers**: The laser plane is at an angle to the camera axis. Tilting the sensor brings the entire laser line into focus.
- **Machine vision with tilted object planes**: When the depth of field is insufficient to keep the entire object in focus.

For standard cameras (no tilt), use `SensorKind::Identity` (the default).

## What Determines the Tilt

The tilt matrix is a product of an upper-triangular matrix and a rotation,
$H(\tau) = P_z(\tau)\,R(\tau)$. For a camera without lens distortion this
gives

$$
K\,H(\tau)\,(R_v X + t_v) \;=\; K'\,(R\,R_v X + R\,t_v),
\qquad K' = K\,P_z(\tau),
$$

and $K'$ is an ordinary zero-skew pinhole matrix. A tilted camera without
distortion is therefore exactly a plain pinhole camera whose poses are all
rotated by $R(\tau)$: no set of images can tell the two apart. The same
holds in a rig, where each camera's extrinsic rotation absorbs $R(\tau)$,
and in a hand-eye calibration, where the hand-eye rotation does.

What makes the tilt observable is something tied to the lens axis:

- **Lens distortion.** It is symmetric about the lens axis, and the tilt
  moves the sensor relative to that axis. The tilt's uncertainty then scales
  inversely with the distortion at the image corners and with the field of
  view; it does not depend on how large the tilt is.
- **A known mounting.** A tilt known from the mechanical design can be held
  fixed, as can a camera orientation known independently.

Consequences for a calibration:

- With a weakly distorting lens the tilt is poorly determined, and its
  correlation with the principal point is close to one. The projection is
  still accurate; only the split between tilt, principal point and pose is
  not. Fix the tilt at its design value in that case.
- To first order the tilt acts like the tangential distortion terms
  ($p_1 \approx -k_1 \tan\tau_x$, $p_2 \approx k_1 \tan\tau_y$). The two
  cannot be estimated together, which is why the tangential terms are held
  while a tilt is estimated. A lens that really is decentred then shows up
  as a small tilt.
- Views that cover the whole image determine the tilt several times better
  than views that fill only its centre or one side. The depth range and the
  board's own tilt do not change the tilt's uncertainty, though the focal
  length still needs tilted views.

The quality report's `parameter_conditioning` check flags a tilt whose σ
these trade-offs inflate and says which one is at work ([Judging a
Result](quality.md#the-checks)).

### What the Estimated Tilt Is

The tilt model sends a ray to the tilted sensor at the angle it had in
front of the lens. A real lens changes that angle: a chief ray that enters
at slope $\tan\theta$ leaves towards the sensor at slope
$\tan\theta / m_p$, where $m_p$ is the lens's pupil magnification (the
exit pupil's diameter over the entrance pupil's). For a sensor physically
tilted by $\tau_s$ behind such a lens, the calibration estimates

$$
\tau = \arctan\!\left(\frac{\tan\tau_s}{m_p}\right),
$$

the tilt that makes the lens distortion centred on the lens axis.

```text
 m_p    physical tilt for an estimated tilt of 5.0 deg
 0.7    3.5 deg
 1.0    5.0 deg
 1.5    7.5 deg
 2.0    9.9 deg
```

So the estimated tilt is the mechanical sensor tilt only when $m_p = 1$.
The projection is not affected: the model predicts image points equally
well for any $m_p$, and for a tilt about a pixel axis the focal lengths
take up the difference (the ratio $f_y / f_x$ moves by about
$1 - \cos\tau_s / \cos\tau$). Only the reading of $\tau$ as a
mechanical angle needs $m_p$, from the lens's data sheet.

Both statements of this section are published. That the tilt cannot be
estimated from point correspondences without lens distortion is in [Kumar & Ahuja (2014)](references.md#kumar-ahuja-2014) and, in general form, in
[Steger (2017)](references.md#steger-2017), who shows that every
projective camera is some tilt camera. Steger's model has a separate
image-side distance for the tilted sensor; holding it equal to the
principal distance, as this model and OpenCV's do, gives the relation
above.

## Optimization of Tilt Parameters

In the non-linear optimization stage, Scheimpflug parameters $(\tau_x, \tau_y)$ are treated as additional optimization variables. The tilt homography is generic over the scalar type, so it is differentiated by automatic differentiation, enabling joint optimization of tilt with intrinsics, distortion, and poses.
