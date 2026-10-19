//! Reference results and the drift check against them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::manifest::Tolerance;

/// A stored calibration result: `results/<id>/calibration-rs.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceResult {
    /// Suite version the result was produced for.
    pub suite_version: String,
    /// Software that produced it, e.g. `"calibration-rs 0.9.0"`.
    pub producer: String,
    /// The problem's full export: every estimated parameter and the
    /// reprojection statistics.
    pub export: Value,
}

/// One value of a result that differs from its reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Drift {
    /// JSON path of the value, e.g. `/cameras/0/k/fx`.
    pub path: String,
    /// The reference value.
    pub expected: String,
    /// The new value.
    pub actual: String,
}

/// Every value of `actual` that differs from `expected`: numbers beyond
/// `tolerance`, anything else when not equal, and keys or array elements
/// present on one side only. Paths are JSON pointers; object keys are
/// visited in sorted order.
pub fn drifts(expected: &Value, actual: &Value, tolerance: Tolerance) -> Vec<Drift> {
    let mut out = Vec::new();
    walk(expected, actual, tolerance, &mut String::new(), &mut out);
    out
}

fn walk(expected: &Value, actual: &Value, tol: Tolerance, path: &mut String, out: &mut Vec<Drift>) {
    match (expected, actual) {
        (Value::Number(e), Value::Number(a)) => {
            if !matches!((e.as_f64(), a.as_f64()), (Some(e), Some(a)) if tol.accepts(e, a)) {
                out.push(drift(path, short(expected), short(actual)));
            }
        }
        (Value::Object(e), Value::Object(a)) => {
            for (key, e_value) in e {
                let len = path.len();
                path.push('/');
                path.push_str(key);
                match a.get(key) {
                    Some(a_value) => walk(e_value, a_value, tol, path, out),
                    None => out.push(drift(path, short(e_value), short(&Value::Null))),
                }
                path.truncate(len);
            }
            for (key, a_value) in a {
                if !e.contains_key(key) {
                    let key_path = format!("{path}/{key}");
                    out.push(drift(&key_path, short(&Value::Null), short(a_value)));
                }
            }
        }
        (Value::Array(e), Value::Array(a)) if e.len() == a.len() => {
            for (i, (e_value, a_value)) in e.iter().zip(a).enumerate() {
                let len = path.len();
                path.push('/');
                path.push_str(&i.to_string());
                walk(e_value, a_value, tol, path, out);
                path.truncate(len);
            }
        }
        (Value::Array(e), Value::Array(a)) => out.push(drift(
            path,
            format!("{} elements", e.len()),
            format!("{} elements", a.len()),
        )),
        (e, a) if e == a => {}
        (e, a) => out.push(drift(path, short(e), short(a))),
    }
}

fn drift(path: &str, expected: String, actual: String) -> Drift {
    Drift {
        path: if path.is_empty() {
            "/".into()
        } else {
            path.into()
        },
        expected,
        actual,
    }
}

/// A value for a drift report: numbers and scalars in full, containers by
/// their kind.
fn short(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("[{} elements]", items.len()),
        Value::Object(_) => "{…}".into(),
        Value::Null => "(absent)".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn identical_documents_have_no_drift() {
        let doc = json!({"k": {"fx": 800.0, "fy": 801.0}, "dist": [0.1, -0.2], "name": "a"});
        assert!(drifts(&doc, &doc, Tolerance::default()).is_empty());
    }

    #[test]
    fn numbers_drift_only_beyond_tolerance() {
        let reference = json!({"fx": 800.0, "dist": [0.1, 0.0]});
        let close = json!({"fx": 800.0001, "dist": [0.1, 1e-12]});
        assert!(drifts(&reference, &close, Tolerance::default()).is_empty());

        let far = json!({"fx": 800.1, "dist": [0.1, 0.0]});
        let found = drifts(&reference, &far, Tolerance::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, "/fx");
        assert_eq!(found[0].expected, "800.0");
    }

    #[test]
    fn structure_changes_are_drift() {
        let reference = json!({"a": 1, "views": [1, 2], "mode": "eye_in_hand"});
        let changed = json!({"b": 1, "views": [1], "mode": "eye_to_hand"});
        let paths: Vec<_> = drifts(&reference, &changed, Tolerance::default())
            .into_iter()
            .map(|d| d.path)
            .collect();
        assert_eq!(paths, ["/a", "/mode", "/views", "/b"]);
    }
}
