//! Problem type implementations for the calibration session framework.

pub mod handeye_single;
pub mod linescan;
pub mod rig_extrinsics;
pub mod rig_handeye;

pub use handeye_single::*;
pub use linescan::*;
pub use rig_extrinsics::*;
pub use rig_handeye::*;
