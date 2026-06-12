//! Robot-pose file loading: CSV / JSON / JSONL → canonical
//! `T_B_G` (gripper-in-base) isometries.
//!
//! The manifest's [`PoseConvention`] declares what each row encodes
//! (transform direction, rotation parameterisation, translation
//! units); this module normalizes everything to the pipeline's
//! convention: `RobotPoseMeta.base_se3_gripper` = `T_B_G` in metres.
//!
//! For the world-fixed conventions (`TWorldTcp` / `TTcpWorld`) the
//! world frame plays the role of the robot base — hand-eye math only
//! needs a consistent parent frame across views, not the frame's name.

use std::path::Path;

use nalgebra::{Quaternion, Rotation3, UnitQuaternion, Vector3};
use serde_json::Value;

use vision_calibration_core::Iso3;
use vision_calibration_dataset::{
    PoseConvention, RobotPoseFormat, RobotPoseSource, RotationFormat, TransformConvention,
    TranslationUnits,
};

use super::RunError;

/// One pose parsed from the file, normalized to the pipeline
/// convention (`T_B_G`, metres).
#[derive(Debug, Clone)]
pub(crate) struct ParsedPose {
    /// Value of the `pose_id` column, when mapped. Used by
    /// `shared_filename_token` pairing; `by_index` ignores it.
    pub id: Option<String>,
    /// Gripper pose in the robot base frame.
    pub base_se3_gripper: Iso3,
}

/// Load and normalize every pose row from `source`.
pub(crate) fn load_robot_poses(
    source: &RobotPoseSource,
    convention: &PoseConvention,
    base_dir: &Path,
) -> Result<Vec<ParsedPose>, RunError> {
    let path = if source.path.is_absolute() {
        source.path.clone()
    } else {
        base_dir.join(&source.path)
    };
    let text = std::fs::read_to_string(&path)?;

    let rows: Vec<RawRow> = match source.format {
        RobotPoseFormat::Csv => parse_csv_rows(&text, &path)?,
        RobotPoseFormat::Json => {
            let values: Vec<Value> =
                serde_json::from_str(&text).map_err(|e| RunError::PoseParse {
                    path: path.clone(),
                    row: 0,
                    message: format!("not a JSON array of objects: {e}"),
                })?;
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| object_to_row(v, i, &path))
                .collect::<Result<_, _>>()?
        }
        RobotPoseFormat::Jsonl => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| {
                let v: Value = serde_json::from_str(line).map_err(|e| RunError::PoseParse {
                    path: path.clone(),
                    row: i,
                    message: format!("invalid JSON object: {e}"),
                })?;
                object_to_row(v, i, &path)
            })
            .collect::<Result<_, _>>()?,
    };

    rows.iter()
        .enumerate()
        .map(|(i, row)| parse_pose_row(row, i, source, convention, &path))
        .collect()
}

/// One raw row: column name → string value. CSV and JSON funnel into
/// the same shape so the field extraction below is format-agnostic.
struct RawRow {
    fields: std::collections::HashMap<String, String>,
}

fn parse_csv_rows(text: &str, path: &Path) -> Result<Vec<RawRow>, RunError> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| RunError::PoseParse {
        path: path.to_path_buf(),
        row: 0,
        message: "empty pose file".into(),
    })?;
    // Hand-rolled split: pose files are plain numeric tables. Quoted
    // CSV would need a real parser dependency; reject it explicitly
    // rather than mis-parse.
    if header.contains('"') {
        return Err(RunError::PoseParse {
            path: path.to_path_buf(),
            row: 0,
            message: "quoted CSV is not supported; use plain comma-separated columns".into(),
        });
    }
    let columns: Vec<String> = header.split(',').map(|c| c.trim().to_string()).collect();

    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.contains('"') {
            return Err(RunError::PoseParse {
                path: path.to_path_buf(),
                row: i,
                message: "quoted CSV is not supported; use plain comma-separated columns".into(),
            });
        }
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if cells.len() != columns.len() {
            return Err(RunError::PoseParse {
                path: path.to_path_buf(),
                row: i,
                message: format!(
                    "row has {} cells but the header has {} columns",
                    cells.len(),
                    columns.len()
                ),
            });
        }
        let fields = columns
            .iter()
            .cloned()
            .zip(cells.iter().map(|c| c.to_string()))
            .collect();
        rows.push(RawRow { fields });
    }
    Ok(rows)
}

fn object_to_row(value: Value, row: usize, path: &Path) -> Result<RawRow, RunError> {
    let Value::Object(map) = value else {
        return Err(RunError::PoseParse {
            path: path.to_path_buf(),
            row,
            message: "expected a JSON object per pose".into(),
        });
    };
    let mut fields = std::collections::HashMap::new();
    for (k, v) in map {
        let s = match v {
            Value::String(s) => s,
            Value::Number(n) => n.to_string(),
            other => {
                return Err(RunError::PoseParse {
                    path: path.to_path_buf(),
                    row,
                    message: format!("field {k:?} must be a number or string, got {other}"),
                });
            }
        };
        fields.insert(k, s);
    }
    Ok(RawRow { fields })
}

fn parse_pose_row(
    row: &RawRow,
    index: usize,
    source: &RobotPoseSource,
    convention: &PoseConvention,
    path: &Path,
) -> Result<ParsedPose, RunError> {
    let columns = &source.columns;
    let get = |name: &str| -> Result<&str, RunError> {
        row.fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| RunError::PoseParse {
                path: path.to_path_buf(),
                row: index,
                message: format!("mapped column {name:?} not found in this row"),
            })
    };
    let get_f64 = |name: &str| -> Result<f64, RunError> {
        let raw = get(name)?;
        raw.parse::<f64>().map_err(|_| RunError::PoseParse {
            path: path.to_path_buf(),
            row: index,
            message: format!("column {name:?}: {raw:?} is not a number"),
        })
    };

    let id = match &columns.pose_id {
        Some(col) => Some(get(col)?.to_string()),
        None => None,
    };

    let rotation_values: Vec<f64> = columns
        .rotation
        .iter()
        .map(|c| get_f64(c))
        .collect::<Result<_, _>>()?;
    let rotation =
        rotation_from_values(convention.rotation_format, &rotation_values).map_err(|message| {
            RunError::PoseParse {
                path: path.to_path_buf(),
                row: index,
                message,
            }
        })?;

    // Translation always comes from the tx/ty/tz columns — even for
    // `Matrix4x4RowMajor`, where the matrix's fourth column is
    // redundant with them. One deterministic rule beats guessing
    // which copy the user meant.
    let scale = match convention.translation_units {
        TranslationUnits::M => 1.0,
        TranslationUnits::Mm => 1e-3,
    };
    let translation = Vector3::new(
        get_f64(&columns.tx)? * scale,
        get_f64(&columns.ty)? * scale,
        get_f64(&columns.tz)? * scale,
    );

    let pose = Iso3::from_parts(translation.into(), rotation);
    let base_se3_gripper = match convention.transform {
        // World-fixed conventions: the world frame plays the role of
        // the base (see module docs).
        TransformConvention::TBaseTcp | TransformConvention::TWorldTcp => pose,
        TransformConvention::TTcpBase | TransformConvention::TTcpWorld => pose.inverse(),
    };

    Ok(ParsedPose {
        id,
        base_se3_gripper,
    })
}

/// Build a rotation from the file's parameterisation.
///
/// Euler conventions are *intrinsic*, composed left-to-right in the
/// order the name states: `EulerXyz` = `Rx(a) · Ry(b) · Rz(c)`,
/// `EulerZyx` = `Rz(a) · Ry(b) · Rx(c)`. Note nalgebra's
/// `from_euler_angles(r, p, y)` builds `Rz(y) · Ry(p) · Rx(r)` —
/// equivalent to our `EulerZyx` with reversed argument order — so we
/// compose from axis-angles explicitly to keep both orders honest.
fn rotation_from_values(
    format: RotationFormat,
    values: &[f64],
) -> Result<UnitQuaternion<f64>, String> {
    let expect_len = |n: usize| -> Result<(), String> {
        if values.len() == n {
            Ok(())
        } else {
            Err(format!(
                "rotation format {format:?} needs {n} values, got {}",
                values.len()
            ))
        }
    };
    let rx = |a: f64| UnitQuaternion::from_axis_angle(&Vector3::x_axis(), a);
    let ry = |a: f64| UnitQuaternion::from_axis_angle(&Vector3::y_axis(), a);
    let rz = |a: f64| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), a);

    match format {
        RotationFormat::QuatXyzw => {
            expect_len(4)?;
            Ok(UnitQuaternion::from_quaternion(Quaternion::new(
                values[3], values[0], values[1], values[2],
            )))
        }
        RotationFormat::QuatWxyz => {
            expect_len(4)?;
            Ok(UnitQuaternion::from_quaternion(Quaternion::new(
                values[0], values[1], values[2], values[3],
            )))
        }
        RotationFormat::EulerXyzDeg | RotationFormat::EulerXyzRad => {
            expect_len(3)?;
            let k = if matches!(format, RotationFormat::EulerXyzDeg) {
                std::f64::consts::PI / 180.0
            } else {
                1.0
            };
            Ok(rx(values[0] * k) * ry(values[1] * k) * rz(values[2] * k))
        }
        RotationFormat::EulerZyxDeg | RotationFormat::EulerZyxRad => {
            expect_len(3)?;
            let k = if matches!(format, RotationFormat::EulerZyxDeg) {
                std::f64::consts::PI / 180.0
            } else {
                1.0
            };
            Ok(rz(values[0] * k) * ry(values[1] * k) * rx(values[2] * k))
        }
        RotationFormat::AxisAngleRad => {
            expect_len(3)?;
            Ok(UnitQuaternion::from_scaled_axis(Vector3::new(
                values[0], values[1], values[2],
            )))
        }
        RotationFormat::Matrix4x4RowMajor => {
            expect_len(16)?;
            // Row-major upper-left 3×3; `from_matrix` re-orthonormalizes,
            // tolerating mildly noisy exports.
            let m = nalgebra::Matrix3::new(
                values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                values[9], values[10],
            );
            Ok(UnitQuaternion::from_rotation_matrix(
                &Rotation3::from_matrix(&m),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use vision_calibration_dataset::PoseColumnMap;

    fn convention(
        transform: TransformConvention,
        rotation_format: RotationFormat,
        translation_units: TranslationUnits,
    ) -> PoseConvention {
        PoseConvention {
            transform,
            rotation_format,
            translation_units,
        }
    }

    fn quat_columns(pose_id: Option<&str>) -> PoseColumnMap {
        PoseColumnMap {
            pose_id: pose_id.map(str::to_string),
            tx: "tx".into(),
            ty: "ty".into(),
            tz: "tz".into(),
            rotation: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
        }
    }

    fn write_temp(content: &str, ext: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("poses.{ext}"));
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(content.as_bytes()).expect("write");
        (dir, path)
    }

    fn source(path: &Path, format: RobotPoseFormat, columns: PoseColumnMap) -> RobotPoseSource {
        RobotPoseSource {
            path: path.to_path_buf(),
            format,
            columns,
        }
    }

    const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn csv_quat_xyzw_roundtrip() {
        // 90° about Z: q = (0, 0, sin 45°, cos 45°) in xyzw order.
        let csv = format!("tx,ty,tz,qx,qy,qz,qw\n0.1,0.2,0.3,0,0,{FRAC_1_SQRT_2},{FRAC_1_SQRT_2}");
        let (_dir, path) = write_temp(&csv, "csv");
        let src = source(&path, RobotPoseFormat::Csv, quat_columns(None));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 1);
        let p = &poses[0].base_se3_gripper;
        assert!((p.translation.vector - Vector3::new(0.1, 0.2, 0.3)).norm() < 1e-12);
        let expected = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 90f64.to_radians());
        assert!(p.rotation.angle_to(&expected) < 1e-9);
    }

    #[test]
    fn euler_orders_are_intrinsic_and_distinct() {
        // For a pure 90° Z rotation both orders agree...
        let z90_xyz = rotation_from_values(RotationFormat::EulerXyzDeg, &[0.0, 0.0, 90.0]).unwrap();
        let z90_zyx = rotation_from_values(RotationFormat::EulerZyxDeg, &[90.0, 0.0, 0.0]).unwrap();
        let expected = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 90f64.to_radians());
        assert!(z90_xyz.angle_to(&expected) < 1e-12);
        assert!(z90_zyx.angle_to(&expected) < 1e-12);

        // ...but a mixed rotation must depend on the order. ZYX takes
        // its arguments as (z, y, x), so feed the same per-axis angles
        // to both and require the results to differ.
        let xyz = rotation_from_values(RotationFormat::EulerXyzDeg, &[90.0, 45.0, 10.0]).unwrap();
        let zyx = rotation_from_values(RotationFormat::EulerZyxDeg, &[10.0, 45.0, 90.0]).unwrap();
        let manual_xyz = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 90f64.to_radians())
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45f64.to_radians())
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 10f64.to_radians());
        assert!(xyz.angle_to(&manual_xyz) < 1e-12, "intrinsic XYZ order");
        assert!(
            xyz.angle_to(&zyx) > 1e-3,
            "XYZ and ZYX must differ on a mixed rotation"
        );

        // ZYX matches nalgebra's roll-pitch-yaw with reversed args.
        let nalg = UnitQuaternion::from_euler_angles(
            90f64.to_radians(),
            45f64.to_radians(),
            10f64.to_radians(),
        );
        assert!(zyx.angle_to(&nalg) < 1e-12, "ZYX == nalgebra rpy");
    }

    #[test]
    fn axis_angle_and_matrix_agree() {
        let axis_angle = [0.3, -0.2, 0.5];
        let q = UnitQuaternion::from_scaled_axis(Vector3::new(
            axis_angle[0],
            axis_angle[1],
            axis_angle[2],
        ));
        let from_aa = rotation_from_values(RotationFormat::AxisAngleRad, &axis_angle).unwrap();
        assert!(from_aa.angle_to(&q) < 1e-12);

        let m = q.to_rotation_matrix();
        #[rustfmt::skip]
        let values = [
            m[(0, 0)], m[(0, 1)], m[(0, 2)], 99.0, // 4th column ignored
            m[(1, 0)], m[(1, 1)], m[(1, 2)], 99.0,
            m[(2, 0)], m[(2, 1)], m[(2, 2)], 99.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let from_m = rotation_from_values(RotationFormat::Matrix4x4RowMajor, &values).unwrap();
        assert!(from_m.angle_to(&q) < 1e-9);
    }

    #[test]
    fn tcp_base_is_inverted_and_mm_scaled() {
        let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7);
        let t_b_g = Iso3::from_parts(Vector3::new(0.4, -0.1, 1.2).into(), q);
        let t_g_b = t_b_g.inverse();
        // Encode T_tcp_base in millimetres; expect T_base_tcp in metres back.
        let csv = format!(
            "tx,ty,tz,qx,qy,qz,qw\n{},{},{},{},{},{},{}",
            t_g_b.translation.x * 1000.0,
            t_g_b.translation.y * 1000.0,
            t_g_b.translation.z * 1000.0,
            t_g_b.rotation.i,
            t_g_b.rotation.j,
            t_g_b.rotation.k,
            t_g_b.rotation.w,
        );
        let (_dir, path) = write_temp(&csv, "csv");
        let src = source(&path, RobotPoseFormat::Csv, quat_columns(None));
        let conv = convention(
            TransformConvention::TTcpBase,
            RotationFormat::QuatXyzw,
            TranslationUnits::Mm,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        let got = &poses[0].base_se3_gripper;
        assert!((got.translation.vector - t_b_g.translation.vector).norm() < 1e-9);
        assert!(got.rotation.angle_to(&t_b_g.rotation) < 1e-9);
    }

    #[test]
    fn jsonl_with_pose_id() {
        let jsonl = concat!(
            r#"{"view": "0007", "tx": 1, "ty": 2, "tz": 3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}"#,
            "\n",
            r#"{"view": "0010", "tx": 4, "ty": 5, "tz": 6, "qx": 0, "qy": 0, "qz": 0, "qw": 1}"#,
        );
        let (_dir, path) = write_temp(jsonl, "jsonl");
        let src = source(&path, RobotPoseFormat::Jsonl, quat_columns(Some("view")));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].id.as_deref(), Some("0007"));
        assert_eq!(poses[1].id.as_deref(), Some("0010"));
        assert!((poses[1].base_se3_gripper.translation.x - 4.0).abs() < 1e-12);
    }

    #[test]
    fn json_array_parses() {
        let json = r#"[{"tx": 1, "ty": 0, "tz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}]"#;
        let (_dir, path) = write_temp(json, "json");
        let src = source(&path, RobotPoseFormat::Json, quat_columns(None));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 1);
    }

    #[test]
    fn missing_column_is_actionable() {
        let csv = "tx,ty,tz,qx,qy,qz\n1,2,3,0,0,0"; // qw missing
        let (_dir, path) = write_temp(csv, "csv");
        let src = source(&path, RobotPoseFormat::Csv, quat_columns(None));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let err = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap_err();
        match err {
            RunError::PoseParse { message, .. } => {
                assert!(message.contains("qw"), "got: {message}")
            }
            other => panic!("expected PoseParse, got {other:?}"),
        }
    }

    #[test]
    fn malformed_number_is_actionable() {
        let csv = "tx,ty,tz,qx,qy,qz,qw\n1,2,three,0,0,0,1";
        let (_dir, path) = write_temp(csv, "csv");
        let src = source(&path, RobotPoseFormat::Csv, quat_columns(None));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let err = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap_err();
        match err {
            RunError::PoseParse { row, message, .. } => {
                assert_eq!(row, 0);
                assert!(message.contains("three"), "got: {message}");
            }
            other => panic!("expected PoseParse, got {other:?}"),
        }
    }

    #[test]
    fn quoted_csv_rejected() {
        let csv = "tx,ty,tz,qx,qy,qz,qw\n\"1\",2,3,0,0,0,1";
        let (_dir, path) = write_temp(csv, "csv");
        let src = source(&path, RobotPoseFormat::Csv, quat_columns(None));
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::QuatXyzw,
            TranslationUnits::M,
        );
        let err = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap_err();
        assert!(format!("{err}").contains("quoted CSV"));
    }
}
