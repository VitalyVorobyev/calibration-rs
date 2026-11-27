"""Calibration projects: create, open, save, run, import, notes, scan.

The runs use synthetic images rendered by ``_images``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import vision_calibration as vc
from vision_calibration import dataset_runner
from vision_calibration.project import (
    PROJECT_FILE,
    PROJECT_VERSION,
    Project,
    ProjectError,
    scan_library,
)

import _images


def planar_project(root: Path, **extra: Any) -> Project:
    """A project in ``root`` whose manifest, written in the project file,
    describes six rendered views in ``root/data``."""
    data = root / "data"
    data.mkdir(parents=True)
    manifest = _images.write_planar_dataset(data, views=6)
    spec: dict[str, Any] = {
        "version": PROJECT_VERSION,
        "dataset": {"root": "data", "manifest": manifest},
        **extra,
    }
    return Project.create(root, spec)


class ProjectFileTest(unittest.TestCase):
    def test_create_open_save_and_changed_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = Project.create(
                root, {"version": 2, "dataset": {"manifest": "dataset.toml"}, "name": "bench"}
            )
            self.assertTrue((root / PROJECT_FILE).is_file())
            project = Project.open(root)
            self.assertEqual(project.revision, created.revision)
            self.assertEqual(project.file_version, PROJECT_VERSION)
            self.assertEqual(project.spec["name"], "bench")
            self.assertEqual(project.data_root(), root)

            spec = project.spec
            spec["device"] = {"product": "bench rig", "id": "SN-1"}
            project.spec = spec
            project.save()
            self.assertEqual(Project.open(root).spec["device"]["id"], "SN-1")

            # Another writer changes the file: this save is refused.
            other = Project.open(root)
            other.spec = {**other.spec, "description": "elsewhere"}
            other.save()
            project.spec = {**project.spec, "name": "mine"}
            with self.assertRaises(ProjectError) as caught:
                project.save()
            self.assertEqual(caught.exception.kind, "changed_on_disk")
            project.reload()
            self.assertEqual(project.spec["description"], "elsewhere")

    def test_errors_carry_their_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ProjectError) as caught:
                Project.open(root)
            self.assertEqual(caught.exception.kind, "not_a_project")
            self.assertIsInstance(caught.exception, RuntimeError)
            Project.create(root, {"version": 2, "dataset": {"manifest": "d.toml"}})
            with self.assertRaises(ProjectError) as caught:
                Project.create(root, {"version": 2, "dataset": {"manifest": "d.toml"}})
            self.assertEqual(caught.exception.kind, "project_exists")
            (root / "calibration-project.json").write_text("{}")
            with self.assertRaises(ProjectError) as caught:
                Project.open(root)
            self.assertEqual(caught.exception.kind, "ambiguous_project")

    def test_views_list_robot_poses_and_what_pairs_with_no_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("img_1.png", "img_2.png"):
                (root / name).write_bytes(b"-")
            (root / "poses.csv").write_text(
                "id,x,y,z,qx,qy,qz,qw\n2,0,0,0,0,0,0,1\n9,0,0,0,0,0,0,1\n"
            )
            manifest = {
                "version": 1,
                "topology": "single_cam_handeye",
                "cameras": [{"id": "cam0", "images": {"kind": "glob", "pattern": "img_*.png"}}],
                "target": {"kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03},
                "robot_poses": {
                    "path": "poses.csv",
                    "format": "csv",
                    "columns": {
                        "pose_id": "id",
                        "tx": "x",
                        "ty": "y",
                        "tz": "z",
                        "rotation": ["qx", "qy", "qz", "qw"],
                    },
                },
                "pose_convention": {
                    "transform": "t_base_tcp",
                    "rotation_format": "quat_xyzw",
                    "translation_units": "m",
                },
                "pose_pairing": {
                    "kind": "shared_filename_token",
                    "regex": "_(?P<id>\\d+)\\.png$",
                    "group": "id",
                },
            }
            project = Project.create(root, {"version": 2, "dataset": {"manifest": manifest}})
            views = project.views()
            self.assertEqual(views["tokens"], ["1", "2"])
            self.assertEqual(
                views["robot_poses"],
                {"views": [None, {"index": 0, "id": "2"}], "unpaired": [{"index": 1, "id": "9"}]},
            )
            self.assertIsNone(views["laser_frames"])
            (root / "poses.csv").unlink()
            self.assertIn("error", project.views()["robot_poses"])

    def test_a_version_1_project_opens_and_migrates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cam"
            root.mkdir()
            (root / "calibration-project.json").write_text(
                json.dumps({"version": 1, "dataset": "."}, indent=2) + "\n"
            )
            manifest = _images.write_planar_dataset(root, views=2)
            (root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
            project = Project.open(root)
            self.assertEqual(project.file_version, 1)
            self.assertEqual(project.spec["name"], "cam")
            self.assertEqual(project.dataset()["topology"], "planar_intrinsics")
            project.save()
            self.assertEqual(project.file_version, 2)
            self.assertFalse((root / "dataset.json").exists())


class ProjectRunTest(unittest.TestCase):
    def test_runs_are_written_listed_noted_and_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = planar_project(root, config={"solver": {"max_iters": 60}})
            project.check()
            self.assertEqual(project.config()["solver"]["max_iters"], 60)
            self.assertEqual(len(project.views()["paths"]), 6)

            reports: list[dict[str, Any]] = []
            run = project.calibrate(label="first", report_html=True, progress=reports.append)
            self.assertEqual(run["id"], "run-001")
            self.assertTrue(reports)
            record = run["state"]
            self.assertEqual(record["kind"], "recorded")
            self.assertTrue(record["tool"].startswith("python "))
            self.assertIn(record["summary"]["verdict"], ("pass", "warn", "fail"))
            self.assertEqual(record["source"]["usable_views"], 6)
            self.assertTrue(record["source"]["digest"]["inputs"].startswith("sha256:"))
            run_dir = Path(run["dir"])
            for name in ("export.json", "quality.json", "run.json", "report.html", "notes.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertEqual(run["notes"]["label"], "first")

            again = project.calibrate(config_patch={"solver": {"max_iters": 50}})
            self.assertEqual(again["id"], "run-002")
            self.assertEqual(
                again["state"]["source"]["inputs"]["config_patch"], {"solver": {"max_iters": 50}}
            )
            self.assertEqual([r["id"] for r in project.runs()], ["run-001", "run-002"])

            project.set_run_notes("run-002", {"starred": True})
            self.assertTrue(project.run("run-002")["notes"]["starred"])
            project.delete_run("run-001")
            self.assertEqual([r["id"] for r in project.runs()], ["run-002"])
            with self.assertRaises(ProjectError) as caught:
                project.run("run-001")
            self.assertEqual(caught.exception.kind, "run_not_found")

    def test_equal_inputs_give_identical_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = planar_project(Path(tmp))
            a = project.calibrate()
            b = project.calibrate()
            self.assertEqual(a["state"]["source"]["digest"], b["state"]["source"]["digest"])
            for name in ("export.json", "quality.json"):
                self.assertEqual(
                    (Path(a["dir"]) / name).read_bytes(), (Path(b["dir"]) / name).read_bytes()
                )

    def test_a_stale_exclusion_is_an_error_and_an_exclusion_leaves_the_image_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = planar_project(Path(tmp))
            spec = project.spec
            spec["exclude"] = {"images": [{"camera": "cam0", "image": "missing.png"}]}
            project.spec = spec
            with self.assertRaises(ProjectError) as caught:
                project.check()
            self.assertEqual(caught.exception.kind, "exclusion")
            spec["exclude"] = {"images": [{"camera": "cam0", "image": "view_02.png"}]}
            project.spec = spec
            run = project.calibrate()
            left_out = run["state"]["source"]["left_out"]
            self.assertEqual(left_out, [{"camera": "cam0", "image": "view_02.png", "reason": "excluded"}])

    def test_import_and_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = planar_project(root / "lib" / "bench", device={"product": "rig", "id": "SN-7"})
            run = project.calibrate()
            imported = project.import_export(Path(run["dir"]) / "export.json", label="copy")
            self.assertEqual(imported["state"]["source"], {"kind": "imported", "file_name": "export.json"})
            self.assertEqual(imported["notes"]["label"], "copy")

            scan = scan_library([root / "lib", root / "missing"], max_depth=4)
            self.assertEqual([p["name"] for p in scan["projects"]], ["bench"])
            self.assertEqual(len(scan["failures"]), 1)
            self.assertEqual(len(scan["devices"]), 1)
            self.assertEqual(scan["devices"][0]["device"]["id"], "SN-7")
            self.assertEqual([r["run"]["id"] for r in scan["devices"][0]["runs"]], ["run-001", "run-002"])

    def test_input_files_lists_without_reading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=3)
            files = dataset_runner.input_files(spec, base_dir=root)
            self.assertEqual([f.name for f in files], ["view_00.png", "view_01.png", "view_02.png"])


class TopLevelTest(unittest.TestCase):
    def test_the_module_is_reachable_from_the_package(self) -> None:
        self.assertIs(vc.project.Project, Project)


if __name__ == "__main__":
    unittest.main()
