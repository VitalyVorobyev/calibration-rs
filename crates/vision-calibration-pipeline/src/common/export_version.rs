//! The format version every `*Export` carries.

use serde::Deserialize;
use serde::de::{DeserializeOwned, Deserializer, Error as _};

use crate::Error;

/// An export type with a versioned wire format.
///
/// Every `*Export` writes [`SCHEMA_VERSION`](Self::SCHEMA_VERSION) as its
/// `schema_version` field and refuses to read any other value. The version
/// changes when a field is renamed, removed or reordered, or a value changes
/// meaning; an added optional field keeps it.
pub trait VersionedExport {
    /// The format version this library writes and reads.
    const SCHEMA_VERSION: u32;
}

/// Reads an export's `schema_version` field, failing when it is not
/// `E::SCHEMA_VERSION`. Used as `deserialize_with` on the field itself, so
/// every serde path checks it.
pub(crate) fn read_schema_version<'de, D, E>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
    E: VersionedExport,
{
    let got = u32::deserialize(deserializer)?;
    if got == E::SCHEMA_VERSION {
        Ok(got)
    } else {
        Err(D::Error::custom(format!(
            "unsupported export schema_version {got} (this library reads {})",
            E::SCHEMA_VERSION
        )))
    }
}

/// Deserializes the export the JSON text `json` holds, checking its
/// `schema_version` before any other field is interpreted.
///
/// # Errors
///
/// [`Error::UnsupportedExportVersion`] when `schema_version` is a different
/// version than the one this library writes; [`Error::Serde`] when `json` is
/// not JSON, the version is missing or the export does not deserialize.
pub fn read_export<E>(json: &str) -> Result<E, Error>
where
    E: VersionedExport + DeserializeOwned,
{
    let value = serde_json::from_str(json).map_err(Error::serde)?;
    read_export_value(value)
}

/// [`read_export`] over a parsed document.
pub(crate) fn read_export_value<E>(value: serde_json::Value) -> Result<E, Error>
where
    E: VersionedExport + DeserializeOwned,
{
    if let Some(got) = value.get("schema_version").and_then(|v| v.as_u64())
        && got != u64::from(E::SCHEMA_VERSION)
    {
        return Err(Error::UnsupportedExportVersion {
            got,
            expected: E::SCHEMA_VERSION,
        });
    }
    serde_json::from_value(value).map_err(Error::serde)
}

/// Checks the `schema_version` of the export the JSON text `json` holds
/// against the version this library writes for the export's `kind`, without
/// reading anything else.
///
/// For a reader that keeps the export as JSON, such as the desktop app's
/// import.
///
/// # Errors
///
/// [`Error::UnsupportedExportVersion`] when the versions differ;
/// [`Error::Serde`] when `json` is not JSON or `kind` or `schema_version`
/// is missing or unknown.
pub fn check_export_version(json: &str) -> Result<(), Error> {
    use crate::common::ProblemKind;
    use crate::laserline_device::LaserlineDeviceExport;
    use crate::planar_intrinsics::PlanarIntrinsicsExport;
    use crate::rig_extrinsics::RigExtrinsicsExport;
    use crate::rig_handeye::RigHandeyeExport;
    use crate::rig_handeye_laserline::RigHandeyeLaserlineExport;
    use crate::rig_laserline_device::RigLaserlineDeviceExport;
    use crate::single_cam_handeye::SingleCamHandeyeExport;

    let value: serde_json::Value = serde_json::from_str(json).map_err(Error::serde)?;
    let field = |name: &str| {
        value
            .get(name)
            .cloned()
            .ok_or_else(|| Error::Serde(format!("missing field `{name}`")))
    };
    let kind: ProblemKind = serde_json::from_value(field("kind")?).map_err(Error::serde)?;
    let got: u64 = serde_json::from_value(field("schema_version")?).map_err(Error::serde)?;
    let expected = match kind {
        ProblemKind::PlanarIntrinsics => PlanarIntrinsicsExport::SCHEMA_VERSION,
        ProblemKind::SingleCamHandeye => SingleCamHandeyeExport::SCHEMA_VERSION,
        ProblemKind::LaserlineDevice => LaserlineDeviceExport::SCHEMA_VERSION,
        ProblemKind::RigExtrinsics => RigExtrinsicsExport::SCHEMA_VERSION,
        ProblemKind::RigHandeye => RigHandeyeExport::SCHEMA_VERSION,
        ProblemKind::RigLaserlineDevice => RigLaserlineDeviceExport::SCHEMA_VERSION,
        ProblemKind::RigHandeyeLaserline => RigHandeyeLaserlineExport::SCHEMA_VERSION,
        other => {
            return Err(Error::invalid_input(format!(
                "export kind {other:?} is not supported by this version"
            )));
        }
    };
    if got == u64::from(expected) {
        Ok(())
    } else {
        Err(Error::UnsupportedExportVersion { got, expected })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_foreign_version_is_a_typed_error_naming_both() {
        let export = json!({ "kind": "planar_intrinsics", "schema_version": 7 });
        let err = check_export_version(&export.to_string()).unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnsupportedExportVersion {
                    got: 7,
                    expected: 1
                }
            ),
            "{err}"
        );
        let message = err.to_string();
        assert!(message.contains('7') && message.contains('1'), "{message}");
    }

    #[test]
    fn the_current_version_passes_and_a_missing_one_is_an_error() {
        let current = json!({ "kind": "rig_handeye", "schema_version": 1 });
        check_export_version(&current.to_string()).unwrap();
        let missing = json!({ "kind": "rig_handeye" });
        assert!(matches!(
            check_export_version(&missing.to_string()),
            Err(Error::Serde(_))
        ));
    }

    #[test]
    fn read_export_checks_the_version_before_the_other_fields() {
        use crate::planar_intrinsics::PlanarIntrinsicsExport;
        // The rest of the payload is not an export at all: the version is
        // what is reported.
        let err = read_export::<PlanarIntrinsicsExport>(
            &json!({ "kind": "planar_intrinsics", "schema_version": 2, "camera": 3 }).to_string(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnsupportedExportVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "{err}"
        );
        // The serde path agrees.
        let err = serde_json::from_value::<PlanarIntrinsicsExport>(
            json!({ "kind": "planar_intrinsics", "schema_version": 2 }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("schema_version 2"), "{err}");
    }
}
