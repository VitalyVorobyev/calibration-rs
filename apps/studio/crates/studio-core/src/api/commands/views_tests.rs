//! `project_views` on a robot and laser dataset: each view's pose and laser
//! frames, what pairs with no view, and a pose file that does not parse
//! reported beside the views. No image is read: the files are empty.

use std::path::Path;

use serde_json::{Value, json};

use crate::api::{Api, Setup};
use crate::scope::RootScope;

fn slash(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// A one-camera rig hand-eye laser manifest over `data`: three target
/// images, two laser frames and `poses` robot poses, paired by index.
fn robot_dataset(data: &Path, poses: usize) -> Value {
    std::fs::create_dir_all(data).unwrap();
    let touch = |name: &str| std::fs::write(data.join(name), b"").unwrap();
    for name in ["t0.png", "t1.png", "t2.png", "l0.png", "l1.png"] {
        touch(name);
    }
    let identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n";
    std::fs::write(data.join("poses.txt"), identity.repeat(poses)).unwrap();
    json!({
        "version": 1,
        "topology": "rig_handeye_laserline",
        "cameras": [{
            "id": "cam0",
            "images": { "kind": "list", "paths": ["t0.png", "t1.png", "t2.png"] },
            "laser_images": { "kind": "list", "paths": ["l0.png", "l1.png"] }
        }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
        "robot_poses": { "path": "poses.txt", "format": "rowmajor4x4" },
        "pose_pairing": { "kind": "by_index" },
        "pose_convention": {
            "transform": "t_base_tcp",
            "rotation_format": "matrix4x4_row_major",
            "translation_units": "m"
        }
    })
}

/// The API over a root holding `p`, and `p`'s path.
fn library() -> (tempfile::TempDir, Api, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(tmp.path()).unwrap();
    let root = base.join("lib");
    let project = root.join("p");
    std::fs::create_dir_all(&project).unwrap();
    let api = Api::with_scope(
        RootScope::fixed([root]),
        Setup::for_tests(Some(base.join("cache"))),
    );
    (tmp, api, project)
}

#[test]
fn each_view_names_its_robot_pose_and_laser_frames() {
    let (_tmp, api, project) = library();
    let manifest = robot_dataset(&project.join("data"), 4);
    let spec = json!({ "version": 2, "dataset": { "root": "data", "manifest": manifest } });
    let dir = slash(&project);
    api.invoke("create_project", json!({ "dir": dir, "spec": spec }))
        .unwrap();
    let views = api.invoke("project_views", json!({ "dir": dir })).unwrap();
    assert_eq!(views["views"].as_array().unwrap().len(), 3);
    assert_eq!(views["views"][1]["pose"], json!({ "index": 1 }));
    assert_eq!(views["views"][1]["laser_frames"], json!(["l1.png"]));
    // The third view has a pose and no laser frame.
    assert_eq!(views["views"][2]["pose"], json!({ "index": 2 }));
    assert_eq!(views["views"][2]["laser_frames"], json!([null]));
    // The fourth pose pairs with no view; every laser frame pairs.
    assert_eq!(views["poses"], json!({ "unpaired": [{ "index": 3 }] }));
    assert_eq!(views["laser_frames"], json!({ "unpaired": [] }));

    // A pose file that does not parse leaves the views and the laser
    // frames, with a message naming the pose file's field.
    std::fs::write(project.join("data/poses.txt"), "not a pose\n").unwrap();
    let views = api.invoke("project_views", json!({ "dir": dir })).unwrap();
    assert_eq!(views["views"].as_array().unwrap().len(), 3);
    assert!(views.get("poses").is_none(), "{views}");
    assert!(views["views"][0].get("pose").is_none(), "{views}");
    assert_eq!(views["views"][0]["laser_frames"], json!(["l0.png"]));
    let messages = views["messages"].as_array().unwrap();
    assert!(
        messages.iter().any(|m| m["field"] == "robot_poses"),
        "{views}"
    );
}

#[test]
fn a_dataset_without_robot_or_laser_has_neither() {
    let (_tmp, api, project) = library();
    let mut manifest = robot_dataset(&project.join("data"), 3);
    manifest["topology"] = json!("planar_intrinsics");
    for field in ["robot_poses", "pose_pairing", "pose_convention"] {
        manifest.as_object_mut().unwrap().remove(field);
    }
    manifest["cameras"][0]
        .as_object_mut()
        .unwrap()
        .remove("laser_images");
    let spec = json!({ "version": 2, "dataset": { "root": "data", "manifest": manifest } });
    let dir = slash(&project);
    api.invoke("create_project", json!({ "dir": dir, "spec": spec }))
        .unwrap();
    let views = api.invoke("project_views", json!({ "dir": dir })).unwrap();
    assert_eq!(views["views"].as_array().unwrap().len(), 3);
    assert!(views.get("poses").is_none() && views.get("laser_frames").is_none());
    assert_eq!(
        views["views"][0],
        json!({ "token": "0", "images": ["t0.png"] })
    );
}
