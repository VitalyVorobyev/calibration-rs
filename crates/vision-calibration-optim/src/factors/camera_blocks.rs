//! The camera blocks at the head of a factor's parameter list, evaluated
//! through the core camera models.
//!
//! A factor's camera is data (a [`CameraModelSpec`]); its parameter blocks
//! are `[intrinsics, projection?, distortion?, sensor?, field?]`, the optional
//! blocks present exactly when their kind has parameters. Evaluation dispatches per
//! slot through the kind's methods in `vision_calibration_core`, so this
//! module holds no camera math and a new model needs no change here.

use nalgebra::{DVector, Point2, RealField, Vector3};
use vision_calibration_core::{CameraModelSpec, Ray};

/// A factor's camera: its spec and its parameter blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraBlocks<'a, T: RealField> {
    spec: CameraModelSpec,
    intrinsics: &'a [T],
    projection: &'a [T],
    distortion: &'a [T],
    sensor: &'a [T],
    field: &'a [T],
}

impl<'a, T: RealField> CameraBlocks<'a, T> {
    /// A camera from its blocks; a slot without parameters takes `&[]`.
    pub(crate) fn new(
        spec: CameraModelSpec,
        intrinsics: &'a [T],
        projection: &'a [T],
        distortion: &'a [T],
        sensor: &'a [T],
        field: &'a [T],
    ) -> Self {
        debug_assert!(intrinsics.len() >= 4, "intrinsics must have 4 params");
        debug_assert_eq!(projection.len(), spec.projection.dim());
        debug_assert_eq!(distortion.len(), spec.distortion.dim());
        debug_assert_eq!(sensor.len(), spec.sensor.dim());
        debug_assert_eq!(field.len(), spec.field.dim());
        Self {
            spec,
            intrinsics,
            projection,
            distortion,
            sensor,
            field,
        }
    }

    /// Split an IR-ordered block list into the camera implied by `spec` and
    /// the remaining (chain) blocks.
    pub(crate) fn split(
        spec: CameraModelSpec,
        params: &'a [DVector<T>],
    ) -> (Self, &'a [DVector<T>]) {
        let mut rest = &params[1..];
        let mut take = |dim: usize| -> &'a [T] {
            if dim == 0 {
                return &[];
            }
            let (head, tail) = rest.split_first().expect("camera block missing");
            rest = tail;
            head.as_slice()
        };
        let projection = take(spec.projection.dim());
        let distortion = take(spec.distortion.dim());
        let sensor = take(spec.sensor.dim());
        let field = take(spec.field.dim());
        let camera = Self::new(
            spec,
            params[0].as_slice(),
            projection,
            distortion,
            sensor,
            field,
        );
        (camera, rest)
    }

    /// The intrinsics block `[fx, fy, cx, cy]`.
    pub(crate) fn intrinsics(&self) -> &'a [T] {
        self.intrinsics
    }

    /// Pixel of a camera-frame point.
    pub(crate) fn project(&self, p_c: &Vector3<T>) -> (T, T) {
        let n = self.spec.projection.project(self.projection, p_c);
        let d = self.spec.distortion.distort(self.distortion, &n);
        let [x, y]: [T; 2] = self.spec.sensor.to_sensor(self.sensor, &d).coords.into();
        let k = self.intrinsics;
        let u = k[0].clone() * x + k[2].clone();
        let v = k[1].clone() * y + k[3].clone();
        if self.spec.field.is_none() {
            return (u, v);
        }
        let [u, v]: [T; 2] = self
            .spec
            .field
            .displace(self.field, &Point2::new(u, v))
            .coords
            .into();
        (u, v)
    }

    /// Undistorted normalized coordinates of a pixel.
    pub(crate) fn pixel_to_normalized(&self, pixel: [f64; 2]) -> Point2<T> {
        let k = self.intrinsics;
        let u = T::from_f64(pixel[0]).unwrap();
        let v = T::from_f64(pixel[1]).unwrap();
        let (u, v) = if self.spec.field.is_none() {
            (u, v)
        } else {
            let p = self.spec.field.undisplace(self.field, &Point2::new(u, v));
            (p.x.clone(), p.y.clone())
        };
        let s = Point2::new(
            (u - k[2].clone()) / k[0].clone(),
            (v - k[3].clone()) / k[1].clone(),
        );
        let d = self.spec.sensor.to_normalized(self.sensor, &s);
        self.spec.distortion.undistort(self.distortion, &d)
    }

    /// The camera ray through a pixel.
    pub(crate) fn pixel_ray(&self, pixel: [f64; 2]) -> Ray<T> {
        let n = self.pixel_to_normalized(pixel);
        self.spec.projection.unproject(self.projection, &n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use num_dual::Dual64;
    use vision_calibration_core::{DistortionKind, SensorKind};

    /// At zero distortion the undistortion's real part converges before the
    /// first step, but its derivative with respect to the coefficients must
    /// still be the exact one: `∂x_u/∂k1 = −x·r²` (first order in `k`).
    #[test]
    fn undistort_derivative_at_zero_distortion() {
        let (x, y) = (0.3_f64, -0.2_f64);
        let r2 = x * x + y * y;
        let intr = [1.0, 1.0, 0.0, 0.0].map(Dual64::from);
        let mut d = DVector::from_element(5, Dual64::from(0.0));
        d[0] = Dual64::new(0.0, 1.0);
        let spec = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        let camera = CameraBlocks::new(spec, &intr, &[], d.as_slice(), &[], &[]);
        let n = camera.pixel_to_normalized([x, y]);
        assert!((n.x.re - x).abs() < 1e-15 && (n.y.re - y).abs() < 1e-15);
        assert!((n.x.eps + x * r2).abs() < 1e-12, "∂x_u/∂k1 = {}", n.x.eps);
        assert!((n.y.eps + y * r2).abs() < 1e-12, "∂y_u/∂k1 = {}", n.y.eps);
    }

    /// A camera with every slot filled, a residual field included, and its
    /// blocks concatenated: `[intrinsics, distortion, sensor, field]`.
    fn field_camera() -> (CameraModelSpec, Vec<f64>, [usize; 4]) {
        use vision_calibration_core::FieldKind;
        let field = FieldKind::BSpline {
            cells: [4, 3],
            image_size: [800, 600],
        };
        let spec = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug)
            .with_field(field);
        let mut v = vec![
            820.0, 815.0, 400.0, 300.0, -0.15, 0.03, 0.0, 0.001, -0.0005, 0.02, -0.01,
        ];
        let n = field.dim();
        v.extend((0..n).map(|k| 1.5 * (0.37 * k as f64).sin()));
        (spec, v, [4, 5, 2, n])
    }

    fn blocks_of<'a, T: RealField>(
        spec: CameraModelSpec,
        v: &'a [T],
        dims: [usize; 4],
    ) -> CameraBlocks<'a, T> {
        let (intr, rest) = v.split_at(dims[0]);
        let (dist, rest) = rest.split_at(dims[1]);
        let (sensor, field) = rest.split_at(dims[2]);
        CameraBlocks::new(spec, intr, &[], dist, sensor, field)
    }

    /// Forward-mode derivatives of the projection and of the pixel's
    /// back-projection, through a residual field, against central
    /// differences, over every camera parameter the field included.
    #[test]
    fn field_jacobians_match_finite_differences() {
        use num_dual::{DualDVec64, jacobian};
        let (spec, v, dims) = field_camera();
        let n = v.len();
        let x = DVector::from_vec(v.clone());
        let p = Vector3::new(0.12, -0.08, 1.0);
        let pixel = [612.3, 87.9];
        let project = |v: &[f64]| {
            let (u, w) = blocks_of(spec, v, dims).project(&p);
            [u, w]
        };
        let normalized = |v: &[f64]| {
            let q = blocks_of(spec, v, dims).pixel_to_normalized(pixel);
            [q.x, q.y]
        };
        let (_, jp) = jacobian(
            |v: DVector<DualDVec64>| {
                let p = p.map(DualDVec64::from_re);
                let (u, w) = blocks_of(spec, v.as_slice(), dims).project(&p);
                nalgebra::SVector::<DualDVec64, 2>::new(u, w)
            },
            &x,
        );
        let (_, jn) = jacobian(
            |v: DVector<DualDVec64>| {
                let q = blocks_of(spec, v.as_slice(), dims).pixel_to_normalized(pixel);
                nalgebra::SVector::<DualDVec64, 2>::new(q.x.clone(), q.y.clone())
            },
            &x,
        );
        for k in 0..n {
            let h = 1e-6 * v[k].abs().max(1.0);
            let (mut plus, mut minus) = (v.clone(), v.clone());
            plus[k] += h;
            minus[k] -= h;
            for r in 0..2 {
                let fd = (project(&plus)[r] - project(&minus)[r]) / (2.0 * h);
                assert!(
                    (jp[(r, k)] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                    "∂u{r}/∂θ{k}: {} vs {fd}",
                    jp[(r, k)]
                );
                let fd = (normalized(&plus)[r] - normalized(&minus)[r]) / (2.0 * h);
                assert!(
                    (jn[(r, k)] - fd).abs() < 1e-7 * fd.abs().max(1e-3),
                    "∂n{r}/∂θ{k}: {} vs {fd}",
                    jn[(r, k)]
                );
            }
        }
        // The field moves the pixel: its block has non-zero columns.
        let field_cols = dims[0] + dims[1] + dims[2]..n;
        assert!(field_cols.clone().any(|k| jp[(0, k)].abs() > 1e-3));
        assert!(field_cols.into_iter().any(|k| jn[(0, k)].abs() > 1e-6));
    }
}
