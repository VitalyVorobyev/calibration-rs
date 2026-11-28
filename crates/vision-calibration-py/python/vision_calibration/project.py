"""Calibration projects: one dataset, the changes to its workflow's
configuration, what to leave out of it, and every run calibrated from it.

A project is a folder holding ``calibration-project.toml``, its runs under
``runs/`` and its detection cache. The library alone reads and writes it, so
``vcal``, Python, Rust and the desktop app see the same runs::

    from vision_calibration.project import Project

    project = Project.open("my-camera")
    run = project.calibrate(label="after refocus")
    print(run["id"], run["state"]["summary"]["verdict"])

The payloads are the generated types of :mod:`vision_calibration.types`:
``ProjectSpec`` for the project file, ``RunInfo`` for a run (its ``state``
holds the ``RunRecord`` of ``run.json``), ``RunNotes``, and ``LibraryScan``
for :func:`scan_library`.

A project's ``seed`` (``ProjectSpec["seed"]``, a ``SeedSource``: ``{"run":
path}``, ``{"device_spec": path}`` or ``{"file": path}``) names the seed every
run starts from; a run's record keeps the seed it used (``RunSeed``)::

    spec = project.spec
    spec["seed"] = {"run": "../unit-0007/runs/run-003"}
    project.spec = spec
    project.save()
    run = project.calibrate()
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from . import _vision_calibration as _native
from .dataset_runner import LaserPixelExtractor
from .models import QualityThresholds
from .types import (
    ConfigPatch,
    DatasetSpec,
    ProgressCallback,
    ProjectSpec,
    RunComparison,
    RunInfo,
    RunNotes,
    SeedSource,
)

__all__ = [
    "DETECTIONS_DIR",
    "EXPORT_FILE",
    "PROJECT_FILE",
    "PROJECT_FILE_JSON",
    "PROJECT_VERSION",
    "Project",
    "ProjectError",
    "QUALITY_FILE",
    "REPORT_FILE",
    "RUNS_DIR",
    "RUN_NOTES_FILE",
    "RUN_RECORD_FILE",
    "RUN_RECORD_VERSION",
    "representative_run",
    "compare_recorded_runs",
    "scan_library",
]

PROJECT_VERSION = 2
"""The project file format version this library writes; version 1 (the
desktop app's earlier format) is read and written as version 2 on save."""
PROJECT_FILE = "calibration-project.toml"
"""The project file of a new project."""
PROJECT_FILE_JSON = "calibration-project.json"
"""The project file of a project written as JSON."""
RUNS_DIR = "runs"
"""The folder of a project's runs."""
DETECTIONS_DIR = ".detections"
"""The folder of a project's detection cache."""
RUN_RECORD_FILE = "run.json"
"""A run's record."""
RUN_RECORD_VERSION = 1
"""The format version of a run's record."""
RUN_NOTES_FILE = "notes.json"
"""A run's notes, the only file of a run that changes."""
EXPORT_FILE = "export.json"
"""A run's export."""
QUALITY_FILE = "quality.json"
"""A run's quality report."""
REPORT_FILE = "report.html"
"""A run's page, when asked for."""

ProjectError = _native.ProjectError
"""Raised when a project operation fails. ``kind`` names the failure:
``"not_a_project"``, ``"ambiguous_project"``, ``"project_exists"``,
``"unsupported_version"``, ``"changed_on_disk"``, ``"io"``, ``"parse"``,
``"invalid"``, ``"manifest"``, ``"config"``, ``"exclusion"``,
``"run_not_found"``, ``"import_mismatch"``, ``"seed_unresolved"``,
``"calibrate"`` (a seed that does not fit the dataset is a ``"calibrate"``
error) or ``"compare"``. ``field`` is the field it is about, as a dotted path,
or ``None``: a field of the project file for the project's own errors
(``"device.id"``, ``"config"``, ``"exclude.images[0]"``, ``"seed"``), a
manifest field for the dataset's (``"target"``, ``"cameras[cam0].images"``). A
subclass of :class:`RuntimeError`."""


def _tool() -> str:
    return f"python {_native.library_version()}"


class Project:
    """A calibration project folder.

    :meth:`open` reads one, :meth:`create` makes one; :attr:`spec` is the
    project file's content, which :meth:`save` writes back (refusing with
    ``ProjectError(kind="changed_on_disk")`` when the file changed since it
    was read).
    """

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Project:
        """Open the project in the folder ``path``, or of the project file
        ``path``. Nothing is written."""
        return cls(_native.Project.open(os.fspath(path)))

    @classmethod
    def create(cls, dir: str | os.PathLike[str], spec: ProjectSpec | Mapping[str, Any]) -> Project:
        """Make the folder ``dir`` (created when missing) a project of
        ``spec``: writes ``calibration-project.toml`` and ``runs/``."""
        return cls(_native.Project.create(os.fspath(dir), dict(spec)))

    def save(self) -> None:
        """Write :attr:`spec` to the project file, in one step."""
        self._native.save()

    def reload(self) -> None:
        """Read the project file again, dropping unsaved edits."""
        self._native.reload()

    @property
    def dir(self) -> Path:
        """The project folder, absolute."""
        return Path(cast(str, self._native.dir))

    @property
    def file(self) -> Path:
        """The project file."""
        return Path(cast(str, self._native.file))

    @property
    def file_version(self) -> int:
        """The version of the project file on disk: 1 for a project of the
        desktop app's earlier format until it is saved."""
        return cast(int, self._native.file_version)

    @property
    def revision(self) -> str:
        """The content revision of the project's files as last read or
        written, ``sha256:<hex>``."""
        return cast(str, self._native.revision)

    @property
    def spec(self) -> ProjectSpec:
        """The project file's content, with unsaved edits. Assign a changed
        copy to edit it."""
        return cast(ProjectSpec, self._native.spec)

    @spec.setter
    def spec(self, spec: ProjectSpec | Mapping[str, Any]) -> None:
        self._native.spec = dict(spec)

    def data_root(self) -> Path:
        """The folder the manifest's relative paths resolve against."""
        return Path(self._native.data_root())

    def dataset(self) -> DatasetSpec:
        """The dataset manifest: the inline one, or the referenced file."""
        return cast(DatasetSpec, self._native.dataset())

    def config(self) -> dict[str, Any]:
        """The configuration payload of the dataset's workflow: its defaults
        with the project's changes."""
        return cast("dict[str, Any]", self._native.config())

    def check(self) -> None:
        """Check everything a run checks before it reads an image, the
        project's seed included; raises :class:`ProjectError` on the first
        problem."""
        self._native.check()

    def input_digest(self) -> dict[str, str]:
        """The digests of the project's inputs as they stand, without reading
        an image: ``{"dataset", "exclusions", "config", "quality"}``, with
        ``"seed"`` when the project has one, each as a run records it in its
        ``InputDigest``. A run whose digest differs in one of them was made
        from other inputs than the project now states."""
        return cast("dict[str, str]", self._native.input_digest())

    def views(self) -> dict[str, Any]:
        """The views the images pair into, without reading an image:
        ``{"paths": [[path or None per camera] per view], "tokens": [...],
        "robot_poses": ..., "laser_frames": ...}``.

        ``robot_poses`` is ``None`` when the manifest declares no robot
        poses, else ``{"views": [row or None per view], "unpaired": [row,
        ...]}``, a row being ``{"index", "id"}``: the pose's position in the
        pose file and its ``pose_id``. ``laser_frames`` is ``None`` when no
        camera declares laser images, else ``{"paths": [[path or None per
        camera] per view], "unpaired": [{"camera", "path", "token"}, ...]}``.
        What pairs with no view is listed under ``unpaired``; a run refuses
        it. When the pose file or the laser frames cannot be listed, the
        entry is ``{"error": message, "field": field or None}`` instead. The robot pose file is read
        and the laser image folders listed."""
        return cast("dict[str, Any]", self._native.views())

    def detection_cache_dir(self) -> Path:
        """The project's detection cache, ``.detections``."""
        return Path(self._native.detection_cache_dir())

    def runs(self) -> list[RunInfo]:
        """The runs: numbered runs by number, then the others by name."""
        return cast("list[RunInfo]", self._native.runs())

    def run(self, run_id: str) -> RunInfo:
        """The run ``run_id``."""
        return cast(RunInfo, self._native.run(run_id))

    def compare_runs(
        self,
        run_ids: Sequence[str],
        *,
        compensation: str = "rotation",
        grid_columns: int = 64,
        thresholds: QualityThresholds | None = None,
    ) -> RunComparison:
        """The runs ``run_ids`` side by side against the first, as
        :func:`vision_calibration.compare_runs` compares them; each run's data
        relation to the first is read from the records (the content of the
        images each run used), and each difference carries the ``data``
        overlap and the ``exclusion_changes`` behind it.
        """
        options: dict[str, Any] = {"compensation": compensation, "grid_columns": grid_columns}
        if thresholds is not None:
            options["thresholds"] = thresholds.to_payload()
        return cast(RunComparison, self._native.compare_runs(list(run_ids), options))

    def calibrate(
        self,
        *,
        label: str | None = None,
        notes: str | None = None,
        starred: bool = False,
        config_patch: ConfigPatch | Mapping[str, Any] | None = None,
        redetect: bool = False,
        report_html: bool = False,
        laser_extractor: LaserPixelExtractor | None = None,
        progress: ProgressCallback | None = None,
        tool: str | None = None,
        seed: SeedSource | Mapping[str, Any] | None = None,
    ) -> RunInfo:
        """Calibrate the dataset and write the next run.

        The run starts from the project's seed, or from ``seed`` (a
        ``SeedSource``, ``{"run": path}``, ``{"device_spec": path}`` or
        ``{"file": path}``, relative to the project folder) in its place;
        the record keeps it, and a seed that cannot be made or does not fit
        raises before any image is read.
        ``config_patch`` changes the configuration for this run only, over
        the project's; the record keeps it. ``redetect`` detects every image
        again; ``report_html`` also writes ``report.html``. ``label``,
        ``notes`` and ``starred`` are written as the run's notes.
        ``laser_extractor`` is required by the laser workflows; ``progress``
        is called with each progress report, and raising from it (or Ctrl-C)
        stops the run, which then writes nothing. ``tool`` names the program
        in the record (default ``"python <version>"``).
        """
        patch = None if config_patch is None else dict(config_patch)
        source = None if seed is None else {
            key: os.fspath(value) for key, value in dict(seed).items()
        }
        return cast(
            RunInfo,
            self._native.calibrate(
                tool if tool is not None else _tool(),
                label,
                notes,
                starred,
                patch,
                redetect,
                report_html,
                laser_extractor,
                progress,
                source,
            ),
        )

    def import_export(
        self,
        export: str | os.PathLike[str],
        *,
        label: str | None = None,
        notes: str | None = None,
        starred: bool = False,
    ) -> RunInfo:
        """Add an export calibrated elsewhere as a run: an export file, a
        ``vcal run -o`` folder or a run folder. Its workflow and camera count
        must be the project's."""
        return cast(
            RunInfo, self._native.import_export(os.fspath(export), label, notes, starred)
        )

    def set_run_notes(self, run_id: str, notes: RunNotes | Mapping[str, Any]) -> None:
        """Replace the notes of run ``run_id``; empty notes remove them."""
        self._native.set_run_notes(run_id, dict(notes))

    def delete_run(self, run_id: str) -> None:
        """Delete run ``run_id`` with every file in it."""
        self._native.delete_run(run_id)

    def __repr__(self) -> str:
        return f"Project({str(self.dir)!r})"


def representative_run(device: Mapping[str, Any]) -> dict[str, Any] | None:
    """The run that stands for a device, one of the ``devices`` of
    :func:`scan_library` (``{"device", "runs"}``): its latest starred run,
    else its latest run calibrated in a project (not imported) whose verdict
    is ``"pass"``; ``None`` when it has neither. The run is the device's
    ``{"project", "run"}`` entry; its folder (``entry["run"]["dir"]``) makes
    a seed with :meth:`vision_calibration.DeviceSeed.from_run`."""
    return cast("dict[str, Any] | None", _native.project_representative_run(dict(device)))


def scan_library(
    roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...],
    max_depth: int = 8,
) -> dict[str, Any]:
    """The projects under the folders ``roots``.

    Returns the ``LibraryScan`` payload (``projects``, ``failures``,
    ``skipped``) with ``devices``: each device's runs across its projects,
    grouped by the project file's ``[device]`` ``product`` and ``id``.
    Symbolic links and folders starting with ``.`` are not followed, a
    folder or project that cannot be read is listed under ``failures``, and
    a project inside another project's folder is not read but listed under
    ``skipped`` with the project it is inside. ``~`` in a root is expanded.
    """
    paths = [os.path.expanduser(os.fspath(root)) for root in roots]
    return cast("dict[str, Any]", _native.project_scan_library(paths, max_depth))


def compare_recorded_runs(
    runs: Sequence[RunInfo],
    *,
    compensation: str = "rotation",
    grid_columns: int = 64,
    thresholds: QualityThresholds | None = None,
) -> RunComparison:
    """Recorded runs side by side against the first, as
    :meth:`Project.compare_runs` compares them, the runs of any projects
    (``RunInfo`` payloads from :meth:`Project.runs` or
    :func:`scan_library`): two calibrations of one device from separate
    captures are compared as independent.
    """
    options: dict[str, Any] = {"compensation": compensation, "grid_columns": grid_columns}
    if thresholds is not None:
        options["thresholds"] = thresholds.to_payload()
    return cast(RunComparison, _native.project_compare_recorded_runs(list(runs), options))
