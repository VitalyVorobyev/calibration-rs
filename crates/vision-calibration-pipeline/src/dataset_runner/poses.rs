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
    DatasetSpec, PoseConvention, PosePairing, RobotPoseFormat, RobotPoseSource, RotationFormat,
    TransformConvention, TranslationUnits,
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

    // The headerless matrix format has no column mapping — parse it
    // directly instead of funnelling through `RawRow`.
    if source.format == RobotPoseFormat::Rowmajor4x4 {
        return text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| parse_matrix_row(line, i, convention, &path))
            .collect();
    }

    // The matrix-field shape (one nested 4×4 array per row) doesn't
    // fit the string-valued `RawRow` funnel — handle it directly. The
    // validator pinned the format to json/jsonl and the rotation
    // format to matrix4x4_row_major.
    if let Some(field) = &source.matrix_field {
        return json_values(source.format, &text, &path)?
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let matrix = v.get(field).ok_or_else(|| RunError::PoseParse {
                    path: path.clone(),
                    row: i,
                    message: format!("matrix_field {field:?} not found in this row"),
                })?;
                let flat = flatten_matrix4x4(matrix).map_err(|message| RunError::PoseParse {
                    path: path.clone(),
                    row: i,
                    message,
                })?;
                pose_from_matrix_values(&flat, i, convention, &path)
            })
            .collect();
    }

    let rows: Vec<RawRow> = match source.format {
        RobotPoseFormat::Csv => parse_csv_rows(&text, &path)?,
        RobotPoseFormat::Json | RobotPoseFormat::Jsonl => json_values(source.format, &text, &path)?
            .into_iter()
            .enumerate()
            .map(|(i, v)| object_to_row(v, i, &path))
            .collect::<Result<_, _>>()?,
        RobotPoseFormat::Rowmajor4x4 => {
            unreachable!("handled by the early return above")
        }
    };

    rows.iter()
        .enumerate()
        .map(|(i, row)| parse_pose_row(row, i, source, convention, &path))
        .collect()
}

/// Parse a json (array of objects) or jsonl (object per line) file
/// into raw row values.
fn json_values(format: RobotPoseFormat, text: &str, path: &Path) -> Result<Vec<Value>, RunError> {
    match format {
        RobotPoseFormat::Json => {
            serde_json::from_str::<Vec<Value>>(text).map_err(|e| RunError::PoseParse {
                path: path.to_path_buf(),
                row: 0,
                message: format!("not a JSON array of objects: {e}"),
            })
        }
        RobotPoseFormat::Jsonl => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_str(line).map_err(|e| RunError::PoseParse {
                    path: path.to_path_buf(),
                    row: i,
                    message: format!("invalid JSON object: {e}"),
                })
            })
            .collect(),
        RobotPoseFormat::Csv | RobotPoseFormat::Rowmajor4x4 => {
            unreachable!("json_values is only called for json/jsonl formats")
        }
    }
}

/// Flatten a nested `[[f64; 4]; 4]` or flat `[f64; 16]` JSON array into
/// 16 row-major values.
fn flatten_matrix4x4(value: &Value) -> Result<Vec<f64>, String> {
    let Value::Array(outer) = value else {
        return Err(format!("expected a 4×4 or flat-16 array, got {value}"));
    };
    let mut flat = Vec::with_capacity(16);
    if outer.len() == 4 && outer.iter().all(|v| v.is_array()) {
        for row in outer {
            let Value::Array(cells) = row else {
                unreachable!("all-array check above");
            };
            if cells.len() != 4 {
                return Err(format!(
                    "nested matrix row has {} cells, expected 4",
                    cells.len()
                ));
            }
            for c in cells {
                flat.push(c.as_f64().ok_or_else(|| format!("{c} is not a number"))?);
            }
        }
    } else {
        for c in outer {
            flat.push(c.as_f64().ok_or_else(|| format!("{c} is not a number"))?);
        }
    }
    if flat.len() != 16 {
        return Err(format!("matrix has {} values, expected 16", flat.len()));
    }
    Ok(flat)
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
    let columns = source.columns.as_ref().expect(
        "validate guarantees a columns mapping for the tabular pose formats, \
         and Rowmajor4x4 never reaches this row parser",
    );
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

/// Match one robot pose to every kept view, identified by its pairing
/// token. Shared by the rig and single-camera hand-eye converters.
///
/// - `by_index`: poses pair with *paired* views (pre-drop): pose `i`
///   belongs to view token `i` even when that view was later dropped,
///   so the pose count must equal `total_views`.
/// - `shared_filename_token`: poses pair by their `pose_id` column
///   equalling the view token; requires a `pose_id` mapping and
///   unique ids.
///
/// Returns one `T_B_G` per token, in token order.
pub(crate) fn match_poses_to_tokens(
    tokens: &[String],
    poses: &[ParsedPose],
    spec: &DatasetSpec,
    total_views: usize,
) -> Result<Vec<Iso3>, RunError> {
    let pairing = spec
        .pose_pairing
        .as_ref()
        .expect("converters require pose_pairing before matching poses");

    match pairing {
        PosePairing::ByIndex => {
            if poses.len() != total_views {
                return Err(RunError::PoseCountMismatch {
                    poses: poses.len(),
                    views: total_views,
                });
            }
            tokens
                .iter()
                .map(|token| {
                    let index: usize = token
                        .parse()
                        .expect("by_index pairing tokens are stringified indices");
                    Ok(poses[index].base_se3_gripper)
                })
                .collect()
        }
        PosePairing::SharedFilenameToken { .. } => {
            if spec
                .robot_poses
                .as_ref()
                .and_then(|s| s.columns.as_ref())
                .is_none_or(|c| c.pose_id.is_none())
            {
                return Err(RunError::AskUser {
                    field: "robot_poses.columns.pose_id".into(),
                    prompt: "shared_filename_token pairing needs a pose_id column mapping \
                             so each pose row can be matched to its view token."
                        .into(),
                    suggestions: vec![],
                });
            }
            // Collecting straight into a HashMap would silently keep only
            // the last row for a repeated pose_id, attaching an arbitrary
            // pose to the matching view. A duplicate id is an ambiguous
            // manifest, so reject it up front like a missing token.
            let mut by_id: std::collections::HashMap<&str, &ParsedPose> =
                std::collections::HashMap::with_capacity(poses.len());
            for pose in poses {
                if let Some(id) = pose.id.as_deref()
                    && by_id.insert(id, pose).is_some()
                {
                    return Err(RunError::PairingTokenMismatch {
                        message: format!(
                            "duplicate pose_id {id:?} in the robot-pose table; each \
                             pose_id must be unique so views pair unambiguously"
                        ),
                    });
                }
            }
            tokens
                .iter()
                .map(|token| {
                    let pose = by_id.get(token.as_str()).ok_or_else(|| {
                        RunError::PairingTokenMismatch {
                            message: format!(
                                "no pose row with pose_id {token:?} (the cameras produced a \
                                 view with this token)"
                            ),
                        }
                    })?;
                    Ok(pose.base_se3_gripper)
                })
                .collect()
        }
    }
}

/// Parse one headerless `rowmajor4x4` line: 16 whitespace-separated
/// floats forming a row-major 4×4 homogeneous transform. Rotation is
/// re-orthonormalized (KUKA exports carry float noise); translation
/// comes from the matrix's fourth column, scaled by the declared
/// units. `id` is always `None` — a headerless file has no pose-id
/// column, so only `by_index` pairing can consume it.
fn parse_matrix_row(
    line: &str,
    index: usize,
    convention: &PoseConvention,
    path: &Path,
) -> Result<ParsedPose, RunError> {
    let values: Vec<f64> = line
        .split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().map_err(|_| RunError::PoseParse {
                path: path.to_path_buf(),
                row: index,
                message: format!("{tok:?} is not a number"),
            })
        })
        .collect::<Result<_, _>>()?;
    if values.len() != 16 {
        return Err(RunError::PoseParse {
            path: path.to_path_buf(),
            row: index,
            message: format!(
                "rowmajor4x4 expects 16 whitespace-separated values per line, got {}",
                values.len()
            ),
        });
    }
    pose_from_matrix_values(&values, index, convention, path)
}

/// Normalize 16 row-major matrix values into a `ParsedPose`. Shared by
/// the headerless `rowmajor4x4` format and the json `matrix_field`
/// shape; both pin `rotation_format` to `Matrix4x4RowMajor` in the
/// validator. Translation comes from the matrix's fourth column,
/// scaled by the declared units. `id` is always `None` — neither shape
/// carries a pose-id column, so only `by_index` pairing can consume
/// them.
fn pose_from_matrix_values(
    values: &[f64],
    index: usize,
    convention: &PoseConvention,
    path: &Path,
) -> Result<ParsedPose, RunError> {
    let rotation =
        rotation_from_values(RotationFormat::Matrix4x4RowMajor, values).map_err(|message| {
            RunError::PoseParse {
                path: path.to_path_buf(),
                row: index,
                message,
            }
        })?;
    let scale = match convention.translation_units {
        TranslationUnits::M => 1.0,
        TranslationUnits::Mm => 1e-3,
    };
    let translation = Vector3::new(values[3] * scale, values[7] * scale, values[11] * scale);

    let pose = Iso3::from_parts(translation.into(), rotation);
    let base_se3_gripper = match convention.transform {
        TransformConvention::TBaseTcp | TransformConvention::TWorldTcp => pose,
        TransformConvention::TTcpBase | TransformConvention::TTcpWorld => pose.inverse(),
    };

    Ok(ParsedPose {
        id: None,
        base_se3_gripper,
    })
}

/// Nearest proper rotation to a (possibly mildly noisy) 3×3 matrix via SVD
/// polar decomposition: `R = U·Vᵀ`, flipping the sign of `U`'s last column when
/// `det(U·Vᵀ) < 0` to enforce `det(R) = +1`.
///
/// This replaces nalgebra's identity-seeded iterative `Rotation3::from_matrix`,
/// which silently mis-converges on exact 180° rotations (returning the wrong
/// axis). For a clean rotation matrix the SVD recovers it exactly; for a mildly
/// non-orthonormal export it returns the closest rotation. The input is a 3×3,
/// so the SVD is trivially small and well-conditioned.
fn nearest_rotation(m: nalgebra::Matrix3<f64>) -> Rotation3<f64> {
    let svd = m.svd(true, true);
    let u = svd.u.expect("3×3 SVD always yields U");
    let v_t = svd.v_t.expect("3×3 SVD always yields Vᵀ");
    let r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u_fixed = u;
        let last = -u_fixed.column(2).into_owned();
        u_fixed.set_column(2, &last);
        Rotation3::from_matrix_unchecked(u_fixed * v_t)
    } else {
        Rotation3::from_matrix_unchecked(r)
    }
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
            // Row-major upper-left 3×3.
            let m = nalgebra::Matrix3::new(
                values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                values[9], values[10],
            );
            Ok(UnitQuaternion::from_rotation_matrix(&nearest_rotation(m)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use vision_calibration_dataset::PoseColumnMap;

    #[test]
    fn matrix4x4_rotation_roundtrips_exact_180_degree() {
        // A 180° rotation about the y-axis is an exact, valid rotation that the
        // iterative `Rotation3::from_matrix` (identity-seeded) mis-converges on.
        // Robot poses with such orientations are common (e.g. the rtv3d rigs),
        // so the loader must recover them exactly.
        let m = [
            -1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let q = rotation_from_values(RotationFormat::Matrix4x4RowMajor, &m).unwrap();
        let r = q.to_rotation_matrix().into_inner();
        let expect = nalgebra::Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let err = (r - expect).norm();
        assert!(err < 1e-9, "180° rotation must round-trip (err={err})");
    }

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
            columns: Some(columns),
            matrix_field: None,
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

    fn matrix_source(path: &Path) -> RobotPoseSource {
        RobotPoseSource {
            path: path.to_path_buf(),
            format: RobotPoseFormat::Rowmajor4x4,
            columns: None,
            matrix_field: None,
        }
    }

    #[test]
    fn rowmajor4x4_parses_kuka_style_lines() {
        // Two near-identity rows with KUKA-grade float noise in the
        // rotation block (mirrors data/kuka_1/RobotPosesVec.txt).
        let text = "1.000000000\t0.000012689\t0.000002112\t1.204997681\t\
                    -0.000012689\t1.000000000\t-0.000003979\t-0.808000305\t\
                    -0.000002112\t0.000003979\t1.000000000\t0.523998047\t\
                    0.000000000\t0.000000000\t0.000000000\t1.000000000\n\
                    \n\
                    1 0 0 0.5  0 1 0 -0.6  0 0 1 0.7  0 0 0 1\n";
        let (_dir, path) = write_temp(text, "txt");
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        );
        let poses = load_robot_poses(&matrix_source(&path), &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 2, "blank lines are skipped");
        let p0 = &poses[0].base_se3_gripper;
        assert!(
            (p0.translation.vector - Vector3::new(1.204997681, -0.808000305, 0.523998047)).norm()
                < 1e-12
        );
        // Noisy rotation re-orthonormalizes to near-identity.
        assert!(p0.rotation.angle() < 1e-4);
        assert!(poses.iter().all(|p| p.id.is_none()), "headerless ⇒ no ids");
        assert!((poses[1].base_se3_gripper.translation.y + 0.6).abs() < 1e-12);
    }

    #[test]
    fn rowmajor4x4_applies_units_and_inversion() {
        let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7);
        let t_b_g = Iso3::from_parts(Vector3::new(0.4, -0.1, 1.2).into(), q);
        let t_g_b = t_b_g.inverse();
        let m = t_g_b.to_homogeneous();
        // Row-major flatten, translation in millimetres.
        let mut row = String::new();
        for r in 0..4 {
            for c in 0..4 {
                let v = if c == 3 && r < 3 {
                    m[(r, c)] * 1000.0
                } else {
                    m[(r, c)]
                };
                row.push_str(&format!("{v:.9} "));
            }
        }
        let (_dir, path) = write_temp(&row, "txt");
        let conv = convention(
            TransformConvention::TTcpBase,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::Mm,
        );
        let poses = load_robot_poses(&matrix_source(&path), &conv, path.parent().unwrap()).unwrap();
        let got = &poses[0].base_se3_gripper;
        assert!((got.translation.vector - t_b_g.translation.vector).norm() < 1e-6);
        assert!(got.rotation.angle_to(&t_b_g.rotation) < 1e-6);
    }

    #[test]
    fn rowmajor4x4_wrong_value_count_is_actionable() {
        let (_dir, path) = write_temp("1 0 0 0 1 0 0 0 1\n", "txt");
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        );
        let err =
            load_robot_poses(&matrix_source(&path), &conv, path.parent().unwrap()).unwrap_err();
        match err {
            RunError::PoseParse { row, message, .. } => {
                assert_eq!(row, 0);
                assert!(
                    message.contains("16") && message.contains("9"),
                    "got: {message}"
                );
            }
            other => panic!("expected PoseParse, got {other:?}"),
        }
    }

    #[test]
    fn rowmajor4x4_non_numeric_token_is_actionable() {
        let (_dir, path) = write_temp("1 0 0 oops 0 1 0 0 0 0 1 0 0 0 0 1\n", "txt");
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        );
        let err =
            load_robot_poses(&matrix_source(&path), &conv, path.parent().unwrap()).unwrap_err();
        match err {
            RunError::PoseParse { message, .. } => {
                assert!(message.contains("oops"), "got: {message}");
            }
            other => panic!("expected PoseParse, got {other:?}"),
        }
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

    // ── matrix_field (ADR 0021) ─────────────────────────────────────────

    fn matrix_field_source(path: &Path, format: RobotPoseFormat) -> RobotPoseSource {
        RobotPoseSource {
            path: path.to_path_buf(),
            format,
            columns: None,
            matrix_field: Some("tcp2base".into()),
        }
    }

    #[test]
    fn matrix_field_nested_4x4_mm_roundtrip() {
        // rtv3d shape: nested 4×4 array, translation in millimetres,
        // tcp2base = T_B_G directly (TBaseTcp).
        let json = r#"[
            {"tcp2base": [[1,0,0,290.0],[0,1,0,15.0],[0,0,1,110.0],[0,0,0,1]],
             "target_image": "target_0.png", "type": "double_snap"}
        ]"#;
        let (_dir, path) = write_temp(json, "json");
        let src = matrix_field_source(&path, RobotPoseFormat::Json);
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::Mm,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 1);
        let p = &poses[0].base_se3_gripper;
        assert!((p.translation.vector - Vector3::new(0.29, 0.015, 0.11)).norm() < 1e-12);
        assert!(p.rotation.angle() < 1e-12);
        assert!(poses[0].id.is_none(), "matrix rows carry no pose_id");
    }

    #[test]
    fn matrix_field_flat_16_jsonl() {
        let jsonl = r#"{"tcp2base": [1,0,0,0.5, 0,1,0,0, 0,0,1,0, 0,0,0,1]}"#;
        let (_dir, path) = write_temp(jsonl, "jsonl");
        let src = matrix_field_source(&path, RobotPoseFormat::Jsonl);
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        );
        let poses = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap();
        assert_eq!(poses.len(), 1);
        assert!((poses[0].base_se3_gripper.translation.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn matrix_field_missing_or_malformed_is_actionable() {
        let json = r#"[{"pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]"#;
        let (_dir, path) = write_temp(json, "json");
        let src = matrix_field_source(&path, RobotPoseFormat::Json);
        let conv = convention(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        );
        let err = load_robot_poses(&src, &conv, path.parent().unwrap()).unwrap_err();
        match err {
            RunError::PoseParse { message, .. } => {
                assert!(message.contains("tcp2base"), "got: {message}");
            }
            other => panic!("expected PoseParse, got {other:?}"),
        }

        let json = r#"[{"tcp2base": [[1,0,0],[0,1,0],[0,0,1]]}]"#;
        let (_dir2, path2) = write_temp(json, "json");
        let src2 = matrix_field_source(&path2, RobotPoseFormat::Json);
        let err2 = load_robot_poses(&src2, &conv, path2.parent().unwrap()).unwrap_err();
        assert!(matches!(err2, RunError::PoseParse { .. }));
    }
}
