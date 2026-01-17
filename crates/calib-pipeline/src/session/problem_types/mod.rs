//! Problem type implementations for the calibration session framework.

pub mod handeye_single;
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye;

pub use handeye_single::*;
pub use planar_intrinsics::*;
pub use rig_extrinsics::*;
pub use rig_handeye::*;
