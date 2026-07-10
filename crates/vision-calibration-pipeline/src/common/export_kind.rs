//! Export-type discriminator (R7).
//!
//! Every pipeline `*Export` carries a `kind: ExportKind` tag as its first
//! field so consumers narrow on a single value instead of probing which
//! required fields are present. The tag serializes snake_case
//! (`"planar_intrinsics"`, …) — the same vocabulary as the problem-module
//! names — and is **required** on deserialize: a missing tag is an error, not
//! a silent default. The 0.7.0 breaking window regenerates every committed
//! export so nothing relies on the pre-tag shape.

use serde::{Deserialize, Serialize};

/// Identifies which calibration `*Export` a JSON payload holds.
///
/// One variant per pipeline problem type, serialized as the snake_case module
/// name in the `kind` field of every `*Export`. Consumers (the desktop app's
/// `detectExportKind`, downstream tooling) read this single tag rather than
/// sniffing field presence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ExportKind {
    /// [`PlanarIntrinsicsExport`](crate::planar_intrinsics::PlanarIntrinsicsExport).
    PlanarIntrinsics,
    /// [`ScheimpflugIntrinsicsExport`](crate::scheimpflug_intrinsics::ScheimpflugIntrinsicsExport).
    ScheimpflugIntrinsics,
    /// [`SingleCamHandeyeExport`](crate::single_cam_handeye::SingleCamHandeyeExport).
    SingleCamHandeye,
    /// [`LaserlineDeviceExport`](crate::laserline_device::LaserlineDeviceExport).
    LaserlineDevice,
    /// [`RigExtrinsicsExport`](crate::rig_extrinsics::RigExtrinsicsExport).
    RigExtrinsics,
    /// [`RigHandeyeExport`](crate::rig_handeye::RigHandeyeExport).
    RigHandeye,
    /// [`RigLaserlineDeviceExport`](crate::rig_laserline_device::RigLaserlineDeviceExport).
    RigLaserlineDevice,
    /// [`RigHandeyeLaserlineExport`](crate::rig_handeye_laserline::RigHandeyeLaserlineExport).
    RigHandeyeLaserline,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ExportKind::PlanarIntrinsics).unwrap(),
            "\"planar_intrinsics\""
        );
        assert_eq!(
            serde_json::to_string(&ExportKind::RigHandeyeLaserline).unwrap(),
            "\"rig_handeye_laserline\""
        );
    }

    #[test]
    fn roundtrips_every_variant() {
        for kind in [
            ExportKind::PlanarIntrinsics,
            ExportKind::ScheimpflugIntrinsics,
            ExportKind::SingleCamHandeye,
            ExportKind::LaserlineDevice,
            ExportKind::RigExtrinsics,
            ExportKind::RigHandeye,
            ExportKind::RigLaserlineDevice,
            ExportKind::RigHandeyeLaserline,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: ExportKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }
    }
}
