//! What the map, comparison and detection tests share: the committed
//! `planar-demo` fixture project copied into a temporary root (calibrated
//! when asked), the golden shapes of the replies, and a map plane's blob
//! checked against its reply.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::api::{Api, Conditions, Setup};
use crate::scope::RootScope;

pub fn slash(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../e2e/fixtures/library")
        .join(name)
}

pub fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A temporary root `lib` holding a copy of `planar-demo` (six rendered
/// views of a chessboard by a 320 × 240 camera) beside a folder `outside`
/// it, and the API over `lib` alone.
pub struct Demo {
    _tmp: tempfile::TempDir,
    pub api: Api,
    pub project: PathBuf,
    pub outside: PathBuf,
}

impl Demo {
    pub fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let root = base.join("lib");
        let project = root.join("planar-demo");
        copy_dir(&fixture("planar-demo"), &project);
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let api = Api::with_scope(
            RootScope::fixed([root]),
            Setup::for_tests(Some(base.join("cache"))),
        );
        Self {
            _tmp: tmp,
            api,
            project,
            outside,
        }
    }

    /// The demo with its first run calibrated (`run-001`).
    pub fn calibrated() -> Self {
        let demo = Self::new();
        let job = demo
            .api
            .invoke("calibrate", json!({ "dir": demo.dir() }))
            .unwrap();
        let state = finish(&demo.api, &job["id"]);
        assert_eq!(state["kind"], "done", "{state}");
        demo
    }

    pub fn dir(&self) -> String {
        slash(&self.project)
    }

    pub fn run(&self, id: &str) -> Value {
        json!({ "dir": self.dir(), "id": id })
    }
}

/// Wait for job `id` to end; its terminal state.
pub fn finish(api: &Api, id: &Value) -> Value {
    for _ in 0..2400 {
        let info = api.invoke("job_status", json!({ "id": id })).unwrap();
        if !matches!(info["state"]["kind"].as_str(), Some("queued" | "running")) {
            return info["state"].clone();
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    panic!("job {id} did not end");
}

/// The shape of a reply: every key path with the JSON type of its value,
/// array elements merged under `[]`, one per line, sorted. Numbers are
/// one type: a value that may be fractional or whole is still a number.
pub fn shape(value: &Value) -> String {
    fn walk(value: &Value, path: String, out: &mut std::collections::BTreeSet<String>) {
        let kind = match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        out.insert(format!("{path}: {kind}"));
        match value {
            Value::Array(items) => {
                for item in items {
                    walk(item, format!("{path}[]"), out);
                }
            }
            Value::Object(map) => {
                for (key, item) in map {
                    walk(item, format!("{path}.{key}"), out);
                }
            }
            _ => {}
        }
    }
    let mut out = std::collections::BTreeSet::new();
    walk(value, "$".into(), &mut out);
    out.into_iter().collect::<Vec<_>>().join("\n") + "\n"
}

/// Compare `value`'s shape with the committed `golden/<name>.txt`
/// (`STUDIO_BLESS=1` writes it).
pub fn golden(name: &str, value: &Value) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/api/commands/golden")
        .join(format!("{name}.txt"));
    let actual = shape(value);
    if std::env::var_os("STUDIO_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &actual).unwrap();
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} (bless with STUDIO_BLESS=1)", path.display()));
    assert_eq!(
        actual, expected,
        "the shape of {name} changed: regenerate with STUDIO_BLESS=1 if on purpose"
    );
}

/// A plane's blob, read through the `blob` resource.
pub fn blob(api: &Api, plane: &Value) -> Vec<u8> {
    api.resource(
        "blob",
        json!({ "id": plane["blob"]["id"] }),
        &Conditions::default(),
    )
    .unwrap()
    .read_body()
    .unwrap()
}

pub fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let at = 24 + 4 * index;
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// A plane's header matches the reply's grid and channel list.
pub fn check_plane(api: &Api, grid: &Value, plane: &Value) -> Vec<u8> {
    let bytes = blob(api, plane);
    assert_eq!(&bytes[..4], b"VAM1");
    assert_eq!(
        u64::from(u32_at(&bytes, 4)),
        grid["columns"].as_u64().unwrap()
    );
    assert_eq!(u64::from(u32_at(&bytes, 8)), grid["rows"].as_u64().unwrap());
    assert_eq!(
        u64::from(u32_at(&bytes, 12)),
        grid["stride"].as_u64().unwrap()
    );
    let channels = plane["channels"].as_array().unwrap().len();
    assert_eq!(u32_at(&bytes, 16) as usize, channels);
    let nodes = (grid["columns"].as_u64().unwrap() * grid["rows"].as_u64().unwrap()) as usize;
    assert_eq!(bytes.len(), 24 + nodes * channels * 4);
    assert_eq!(plane["blob"]["len"].as_u64().unwrap() as usize, bytes.len());
    bytes
}
