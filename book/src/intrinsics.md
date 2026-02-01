# Intrinsics Matrix

The intrinsics matrix $K$ maps from the sensor coordinate system (normalized or sensor-transformed coordinates) to pixel coordinates. It encodes the camera's internal geometric properties: focal length, principal point, and optional skew.

## Definition

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where:

- $f_x, f_y$ — focal lengths in pixels (horizontal and vertical). These combine the physical focal length with the pixel pitch: $f_x = f / \Delta_x$ where $f$ is the focal length in mm and $\Delta_x$ is the pixel width in mm.
- $c_x, c_y$ — principal point in pixels. The projection of the optical axis onto the image plane. Typically near the image center but not exactly at it.
- $s$ — skew coefficient. Non-zero only if pixel axes are not perpendicular. Effectively zero for all modern sensors; calibration-rs defaults to `zero_skew: true` in most configurations.

## Forward and Inverse Transform

**Sensor to pixel** (forward):

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot s_x + s \cdot s_y + c_x \\ f_y \cdot s_y + c_y \end{bmatrix}$$

where $[s_x, s_y]^T$ are sensor coordinates (output of the distortion + sensor stages).

**Pixel to sensor** (inverse):

$$s_y = \frac{v - c_y}{f_y}, \quad s_x = \frac{u - c_x - s \cdot s_y}{f_x}$$

## The `IntrinsicsModel` Trait

```rust
pub trait IntrinsicsModel<S: RealField> {
    fn sensor_to_pixel(&self, sensor: &Point2<S>) -> Point2<S>;
    fn pixel_to_sensor(&self, pixel: &Point2<S>) -> Point2<S>;
}
```

The `FxFyCxCySkew<S>` struct implements this trait:

```rust
pub struct FxFyCxCySkew<S: RealField> {
    pub fx: S, pub fy: S,
    pub cx: S, pub cy: S,
    pub skew: S,
}
```

## Coordinate Utilities

The `coordinate_utils` module provides convenience functions that combine intrinsics with distortion:

- `pixel_to_normalized(pixel, K)` — apply $K^{-1}$ to get normalized coordinates on the $Z = 1$ plane
- `normalized_to_pixel(normalized, K)` — apply $K$ to get pixel coordinates
- `undistort_pixel(pixel, K, distortion)` — pixel → normalized → undistort → return normalized
- `distort_to_pixel(normalized, K, distortion)` — distort → apply $K$ → return pixel

These functions are used extensively in the linear initialization algorithms, which need to convert between pixel and normalized coordinate spaces.

## Aspect Ratio

For most cameras, $f_x \neq f_y$ because pixels are not perfectly square. The ratio $f_x / f_y$ reflects the pixel aspect ratio. Typical industrial cameras have aspect ratios very close to 1.0 (within 1-3%).

> **OpenCV equivalence**: `cv::initCameraMatrix2D` provides initial intrinsics estimates. The $K$ matrix format is identical to OpenCV's `cameraMatrix`.
