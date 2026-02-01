# Sensor Models and Scheimpflug Tilt

In standard cameras, the sensor plane is perpendicular to the optical axis. Some industrial cameras — particularly laser triangulation profilers — deliberately tilt the sensor to achieve a larger depth of field along the laser plane (the Scheimpflug condition). The sensor model stage accounts for this tilt.

## The `SensorModel` Trait

```rust
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

```rust
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

For standard cameras (no tilt), use `IdentitySensor` (the default).

## Optimization of Tilt Parameters

In the non-linear optimization stage, Scheimpflug parameters $(\tau_x, \tau_y)$ are treated as additional optimization variables. The `tilt_projection_matrix_generic<T>()` function in the factor module computes the tilt homography using dual numbers for automatic differentiation, enabling joint optimization of tilt with intrinsics, distortion, and poses.
