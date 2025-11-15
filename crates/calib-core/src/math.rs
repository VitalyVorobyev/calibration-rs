use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point2, Point3, Vector2, Vector3,
};

pub type Real = f64;

pub type Vec2 = Vector2<Real>;
pub type Vec3 = Vector3<Real>;
pub type Pt2 = Point2<Real>;
pub type Pt3 = Point3<Real>;
pub type Mat3 = Matrix3<Real>;
pub type Mat4 = Matrix4<Real>;
pub type Iso3 = Isometry3<Real>;

pub fn to_homogeneous(p: &Pt2) -> Vec3 {
    Vec3::new(p.x, p.y, 1.0)
}

pub fn from_homogeneous(v: &Vec3) -> Pt2 {
    Pt2::new(v.x / v.z, v.y / v.z)
}
