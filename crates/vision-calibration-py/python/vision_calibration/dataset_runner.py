"""Manifest-driven runs: from a dataset manifest to a calibration result.

:func:`calibrate_dataset` reads a manifest, detects the target in every
image, builds the problem's input, calibrates and returns a
:class:`DatasetCalibration`: the export and what detection kept. The
``build_*_input`` functions stop after the input, for a run driven step by
step through a session. :func:`detect_target` runs a manifest's target
detector on one numpy image; :func:`detect_target_detail` and
:func:`detection_detail` return a detection with what an overlay needs to
judge it.

The runner does not guess: a manifest that leaves a convention open (how
images pair with robot poses, which way a pose transforms) raises
:class:`AskUserError` naming the field to set. Every other failure raises
:class:`RunError` with a ``kind``.

.. code-block:: python

    from vision_calibration.dataset_runner import calibrate_dataset

    run = calibrate_dataset("data/kuka_1/dataset.toml")
    print(f"{run.usable_views} of {run.total_views} views used")
    print(run.export.mean_reproj_error)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _vision_calibration as _native
from ._generated_types import (
    DatasetSpec,
    DetectorSpec,
    LaserExtractionSpec,
    TargetSpec,
)
from ._generated_types import LaserlineDeviceInput as _LaserlineDeviceInputPayload
from ._generated_types import PlanarIntrinsicsInput as _PlanarIntrinsicsInputPayload
from ._generated_types import RigExtrinsicsInput as _RigExtrinsicsInputPayload
from ._generated_types import RigHandeyeInput as _RigHandeyeInputPayload
from ._generated_types import RigHandeyeLaserlineInput as _RigHandeyeLaserlineInputPayload
from ._generated_types import RigLaserlineDeviceInput as _RigLaserlineDeviceInputPayload
from ._generated_types import SingleCamHandeyeInput as _SingleCamHandeyeInputPayload
from .detect import DetectionDetail, Features, FsDetectionCache, _detection_detail, _features
from .device_seed import DeviceSeed
from .models import (
    LaserlineDeviceConfig,
    LaserlineDeviceExport,
    PlanarIntrinsicsConfig,
    PlanarIntrinsicsExport,
    RigExtrinsicsConfig,
    RigExtrinsicsExport,
    RigHandeyeConfig,
    RigHandeyeExport,
    RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineExport,
    RigLaserlineDeviceConfig,
    RigLaserlineDeviceExport,
    SingleCamHandeyeConfig,
    SingleCamHandeyeExport,
)
from .types import ProgressCallback

__all__ = [
    "AskUserError",
    "CalibrationConfig",
    "CalibrationExport",
    "ConfigPatch",
    "DatasetCalibration",
    "HandeyeRunResult",
    "LaserPixelExtractor",
    "LaserlineRunResult",
    "PlanarRunResult",
    "RejectReason",
    "RejectedDetection",
    "RigHandeyeLaserlineRunResult",
    "RigLaserlineRunResult",
    "RigRunResult",
    "RunError",
    "build_laserline_device_input",
    "build_planar_input",
    "build_rig_extrinsics_input",
    "build_rig_handeye_input",
    "build_rig_handeye_laserline_input",
    "build_rig_laserline_device_input",
    "build_single_cam_handeye_input",
    "calibrate_dataset",
    "camera_image_sizes",
    "input_files",
    "detect_target",
    "detect_target_detail",
    "cached_detection",
    "detection_detail",
    "patched_config",
]

RunError = _native.RunError
"""A manifest run failed; ``kind`` names the failure: ``"validation"``,
``"empty_image_match"``, ``"image_size_mismatch"``, ``"roi_out_of_bounds"``,
``"pose_count_mismatch"``, ``"detection"``, ``"insufficient_usable_views"``,
``"no_whole_chessboard_views"``, ``"upstream_view_match"`` (a rig laser view
that is not exactly one view of its upstream calibration), .... ``field`` is
the manifest field the failure is about, as a dotted path (``"target"``,
``"pose_pairing.regex"``, ``"cameras[cam0].images"``, a camera named by its
id), or ``None`` when it is about the images, the detections or the run."""

AskUserError = _native.AskUserError
"""The manifest leaves a convention open and the runner will not guess.
``field`` names the manifest field to set, ``prompt`` asks for it and
``suggestions`` lists candidate values. A subclass of :class:`RunError`
(``kind == "ask_user"``)."""

#: A manifest: the path of a TOML or JSON file, or the ``DatasetSpec`` payload.
Manifest = str | os.PathLike[str] | DatasetSpec | Mapping[str, Any]

CalibrationConfig = (
    PlanarIntrinsicsConfig
    | SingleCamHandeyeConfig
    | RigExtrinsicsConfig
    | RigHandeyeConfig
    | LaserlineDeviceConfig
    | RigLaserlineDeviceConfig
    | RigHandeyeLaserlineConfig
    | Mapping[str, Any]
)

ConfigPatch = Mapping[str, Any]
"""Changes to a workflow's default configuration, keyed like its config
(``{"solver": {"max_iters": 20}}``): what the patch names replaces the
default, what it leaves out keeps it. Two mappings merge key by key; any
other value replaces what it lands on. :func:`patched_config` applies one."""

CalibrationExport = (
    PlanarIntrinsicsExport
    | SingleCamHandeyeExport
    | RigExtrinsicsExport
    | RigHandeyeExport
    | LaserlineDeviceExport
    | RigLaserlineDeviceExport
    | RigHandeyeLaserlineExport
)


class LaserPixelExtractor(Protocol):
    """Extracts the laser line from a laser image, for the laser topologies.

    ``name`` identifies the implementation: it is part of the detection
    cache key, so a different extractor never reads another's pixels.
    ``extract`` receives the image (cropped to the camera's ROI) as a
    ``(H, W)`` ``uint8`` grayscale array and the manifest's ``laser``
    section, and returns the laser line's subpixel points ``(N, 2)`` as
    ``(x, y)`` in that image. An exception it raises fails the run with
    ``kind == "laser_extraction"``.
    """

    @property
    def name(self) -> str: ...

    def extract(self, image: NDArray[np.uint8], spec: LaserExtractionSpec) -> ArrayLike: ...


class _LaserAdapter:
    """Hands the runner a user extractor whose output is a float64 ``(N, 2)``
    array."""

    __slots__ = ("_inner", "name")

    def __init__(self, inner: LaserPixelExtractor) -> None:
        self._inner = inner
        self.name = str(inner.name)

    def extract(self, image: NDArray[np.uint8], spec: LaserExtractionSpec) -> NDArray[np.float64]:
        points = np.asarray(self._inner.extract(image, spec), dtype=np.float64)
        return np.ascontiguousarray(points.reshape(-1, 2))


RejectReason = Literal[
    "too_few_features",
    "partial_chessboard",
    "turned_chessboard",
    "ambiguous_chessboard_turn",
    "excluded",
    "other",
]
"""Why the runner left a detection out: fewer features than the per-view
floor, or, for a plain chessboard where views share one target frame, a
board that does not reach every edge, a board seen turned by 90° where the
target is fixed through the views (hand-eye), or a rig camera whose turn of
the board cannot be matched with the view's other cameras; or the run was
told to leave the image out (``"excluded"``)."""


@dataclass(frozen=True, slots=True)
class RejectedDetection:
    """A target detection the runner left out of the calibration input."""

    camera: int
    """Index of the camera in the manifest's ``cameras``."""
    path: Path
    """The image the detection came from."""
    reason: RejectReason
    """Why it was left out."""


@dataclass(frozen=True, slots=True)
class PlanarRunResult:
    """A planar intrinsics input built from a manifest."""

    input: _PlanarIntrinsicsInputPayload
    """The input payload, ready for a session's ``set_input``."""
    view_paths: list[Path]
    """The image of each kept view, in input order."""
    usable_views: int
    """Images with enough features to keep."""
    total_views: int
    """Images tried."""
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


@dataclass(frozen=True, slots=True)
class HandeyeRunResult:
    """A single-camera hand-eye input built from a manifest."""

    input: _SingleCamHandeyeInputPayload
    view_paths: list[Path]
    usable_views: int
    total_views: int
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


@dataclass(frozen=True, slots=True)
class RigRunResult:
    """A rig extrinsics or rig hand-eye input built from a manifest."""

    input: _RigExtrinsicsInputPayload | _RigHandeyeInputPayload
    view_paths: list[list[Path | None]]
    """``[view][camera]`` image; ``None`` where the camera added no
    observation."""
    usable_views: int
    total_views: int
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


@dataclass(frozen=True, slots=True)
class LaserlineRunResult:
    """A laserline device input built from a manifest."""

    input: _LaserlineDeviceInputPayload
    view_paths: list[Path]
    laser_paths: list[Path]
    """The laser image of each kept view."""
    usable_views: int
    total_views: int
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


@dataclass(frozen=True, slots=True)
class RigLaserlineRunResult:
    """A rig laserline device input built from a manifest."""

    input: _RigLaserlineDeviceInputPayload
    view_paths: list[list[Path | None]]
    laser_paths: list[list[Path | None]]
    usable_views: int
    total_views: int
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


@dataclass(frozen=True, slots=True)
class RigHandeyeLaserlineRunResult:
    """A joint rig hand-eye laserline input built from a manifest."""

    input: _RigHandeyeLaserlineInputPayload
    view_paths: list[list[Path | None]]
    laser_paths: list[list[Path | None]]
    usable_views: int
    total_views: int
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


def _manifest_arg(manifest: Manifest) -> Any:
    if isinstance(manifest, (str, os.PathLike)):
        return os.fspath(manifest)
    return dict(manifest)


def _base_dir_arg(base_dir: str | os.PathLike[str] | None) -> str | None:
    return None if base_dir is None else os.fspath(base_dir)


def _cache_arg(cache: FsDetectionCache | None) -> Any:
    return None if cache is None else cache._native


def _paths(values: Any) -> list[Path]:
    return [Path(p) for p in values]


def _rejected(values: Any) -> list[RejectedDetection]:
    return [
        RejectedDetection(camera=int(r["camera"]), path=Path(r["path"]), reason=r["reason"])
        for r in values
    ]


def _rig_paths(values: Any) -> list[list[Path | None]]:
    return [[None if p is None else Path(p) for p in row] for row in values]


def build_planar_input(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> PlanarRunResult:
    """Detect a ``planar_intrinsics`` manifest's images into a planar input.

    ``base_dir`` resolves the manifest's relative paths; it defaults to the
    manifest file's directory and is required for a payload. Without a
    ``cache`` every image is detected; with one, cached detections are
    reused unless ``force_redetect``. ``progress``, when given, is called
    with each image read (a :data:`~vision_calibration.types.Progress`
    mapping of kind ``"detection"``, or ``"laser_extraction"`` for a laser
    image); raising from it, or Ctrl-C, stops the detection and the
    exception propagates.
    """
    raw = _native.build_planar_input(
        _manifest_arg(manifest),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return PlanarRunResult(
        input=raw["input"],
        view_paths=_paths(raw["view_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_single_cam_handeye_input(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> HandeyeRunResult:
    """Detect a ``single_cam_handeye`` manifest's images and pair them with
    its robot poses. Arguments as for :func:`build_planar_input`."""
    raw = _native.build_single_cam_handeye_input(
        _manifest_arg(manifest),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return HandeyeRunResult(
        input=raw["input"],
        view_paths=_paths(raw["view_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_rig_extrinsics_input(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> RigRunResult:
    """Detect a ``rig_extrinsics`` manifest's images into rig views.
    Arguments as for :func:`build_planar_input`."""
    raw = _native.build_rig_extrinsics_input(
        _manifest_arg(manifest),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return RigRunResult(
        input=raw["input"],
        view_paths=_rig_paths(raw["view_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_rig_handeye_input(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> RigRunResult:
    """Detect a ``rig_handeye`` manifest's images into rig views paired with
    its robot poses. Arguments as for :func:`build_planar_input`."""
    raw = _native.build_rig_handeye_input(
        _manifest_arg(manifest),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return RigRunResult(
        input=raw["input"],
        view_paths=_rig_paths(raw["view_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_laserline_device_input(
    manifest: Manifest,
    laser_extractor: LaserPixelExtractor,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> LaserlineRunResult:
    """Detect a ``laserline_device`` manifest's target images and extract its
    laser images with ``laser_extractor``. Other arguments as for
    :func:`build_planar_input`."""
    raw = _native.build_laserline_device_input(
        _manifest_arg(manifest),
        _LaserAdapter(laser_extractor),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return LaserlineRunResult(
        input=raw["input"],
        view_paths=_paths(raw["view_paths"]),
        laser_paths=_paths(raw["laser_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_rig_laserline_device_input(
    manifest: Manifest,
    laser_extractor: LaserPixelExtractor,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> RigLaserlineRunResult:
    """Build a ``rig_laserline_device`` input: the rig's target and laser
    images against the manifest's upstream rig calibration. Arguments as for
    :func:`build_laserline_device_input`."""
    raw = _native.build_rig_laserline_device_input(
        _manifest_arg(manifest),
        _LaserAdapter(laser_extractor),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return RigLaserlineRunResult(
        input=raw["input"],
        view_paths=_rig_paths(raw["view_paths"]),
        laser_paths=_rig_paths(raw["laser_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def build_rig_handeye_laserline_input(
    manifest: Manifest,
    laser_extractor: LaserPixelExtractor,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    progress: ProgressCallback | None = None,
) -> RigHandeyeLaserlineRunResult:
    """Build a ``rig_handeye_laserline`` input: the rig's target and laser
    images paired with its robot poses. Arguments as for
    :func:`build_laserline_device_input`."""
    raw = _native.build_rig_handeye_laserline_input(
        _manifest_arg(manifest),
        _LaserAdapter(laser_extractor),
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        progress,
    )
    return RigHandeyeLaserlineRunResult(
        input=raw["input"],
        view_paths=_rig_paths(raw["view_paths"]),
        laser_paths=_rig_paths(raw["laser_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


# Each manifest topology's config and export classes.
_PROBLEMS: dict[str, tuple[type[Any], type[Any]]] = {
    "planar_intrinsics": (PlanarIntrinsicsConfig, PlanarIntrinsicsExport),
    "single_cam_handeye": (SingleCamHandeyeConfig, SingleCamHandeyeExport),
    "rig_extrinsics": (RigExtrinsicsConfig, RigExtrinsicsExport),
    "rig_handeye": (RigHandeyeConfig, RigHandeyeExport),
    "laserline_device": (LaserlineDeviceConfig, LaserlineDeviceExport),
    "rig_laserline_device": (RigLaserlineDeviceConfig, RigLaserlineDeviceExport),
    "rig_handeye_laserline": (RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport),
}


@dataclass(frozen=True, slots=True)
class DatasetCalibration:
    """The result of :func:`calibrate_dataset`: the export and what detection
    kept."""

    export: CalibrationExport
    """The calibration's export, the manifest topology's export class."""
    image_sizes: list[tuple[int, int]]
    """Each camera's image size ``(width, height)`` in pixels, in camera
    order: the ``image_sizes`` that :func:`~vision_calibration.assess` takes."""
    usable_views: int
    """Views that entered the calibration."""
    total_views: int
    """Views the manifest's images formed before screening."""
    rejected: list[RejectedDetection] = field(default_factory=list)
    """Detections left out, with the reason for each."""


def calibrate_dataset(
    manifest: Manifest,
    config: CalibrationConfig | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    laser_extractor: LaserPixelExtractor | None = None,
    progress: ProgressCallback | None = None,
    seed: DeviceSeed | Mapping[str, Any] | None = None,
) -> DatasetCalibration:
    """Calibrate a dataset from its manifest.

    The manifest's ``topology`` selects the problem: its images are detected
    into the problem's input (the ``build_*_input`` of the topology), which is
    calibrated with ``config`` (the problem's config dataclass or payload;
    ``None`` for the defaults) and exported.

    With a ``seed``, every initialization of the workflow starts from it: the
    lenses (intrinsics, distortion and tilt), the mounts, the hand-eye and the
    laser planes of a previous calibration of the device or of one of its
    type, or a datasheet's values. The seed is a starting point: the
    calibration still estimates everything. The mounts and the hand-eye are
    re-expressed in the frame of the config's reference camera. A seeded
    Scheimpflug tilt runs planar intrinsics' refinement bounded around the
    seed, and a laserline device holds a seeded tilt unless the config's
    ``camera.fix.sensor`` lists the tilts to estimate.

    Parameters
    ----------
    manifest:
        Path of a TOML or JSON manifest, or the ``DatasetSpec`` payload.
    config:
        The problem's config, e.g. a
        :class:`~vision_calibration.models.SingleCamHandeyeConfig` for a
        ``single_cam_handeye`` manifest. A config dataclass of another
        topology than the manifest's raises :class:`ValueError`; a payload
        mapping is read as the manifest's own.
    base_dir:
        Directory the manifest's relative paths resolve against; defaults to
        the manifest file's directory, required for a payload.
    cache:
        Detection cache; ``None`` detects every image.
    force_redetect:
        Detect again even where the cache has an entry.
    laser_extractor:
        The laser-line extractor, required by the laser topologies.
    progress:
        Called with each progress report, a
        :data:`~vision_calibration.types.Progress` mapping: each image
        detected, each calibration step begun, each solver iteration.
        Raising from it stops the run and the exception propagates; Ctrl-C
        stops it with :class:`KeyboardInterrupt`.
    seed:
        A :class:`~vision_calibration.DeviceSeed`, or its JSON form, whose
        cameras are the manifest's (:meth:`~vision_calibration.DeviceSeed.select`
        picks some of a device's), each for images of the dataset's size and,
        for a calibrated lens, of the configured camera model
        (:meth:`~vision_calibration.DeviceSeed.nominal` keeps only the
        intrinsics and the tilt). ``rig_laserline_device`` takes only the
        laser planes; its cameras are the input's upstream calibration.

    Returns
    -------
    A :class:`DatasetCalibration`: the topology's export dataclass (e.g.
    :class:`~vision_calibration.models.SingleCamHandeyeExport`) in
    ``export``, the cameras' image sizes, the view counts and the rejected
    detections. The export's ``diagnostics.image_manifest`` names the image
    of every view and camera the calibration used; its ``root`` is ``"."``,
    standing for ``base_dir``, and each frame's path is relative to
    ``base_dir``.

    Raises
    ------
    AskUserError
        The manifest leaves a convention open.
    RunError
        The manifest is invalid, or detection or pairing failed.
    ValueError
        The config is invalid or for another topology, a laser topology has
        no ``laser_extractor``, the problem rejects the detected input, or the
        seed does not fit the dataset or the config (raised before any image
        is detected).
    RuntimeError
        The calibration itself failed.
    KeyboardInterrupt
        Ctrl-C stopped the run.
    """
    config_topology: str | None = None
    config_payload: Any = None
    if config is not None:
        config_topology = next(
            (name for name, (config_type, _) in _PROBLEMS.items() if isinstance(config, config_type)),
            None,
        )
        config_payload = config.to_payload() if hasattr(config, "to_payload") else dict(config)
    raw = _native.calibrate_dataset(
        _manifest_arg(manifest),
        config_payload,
        config_topology,
        _base_dir_arg(base_dir),
        _cache_arg(cache),
        force_redetect,
        None if laser_extractor is None else _LaserAdapter(laser_extractor),
        progress,
        None if seed is None else _seed_payload(seed),
    )
    export_payload = raw["export"]
    export_type = _PROBLEMS[export_payload["kind"]][1]
    return DatasetCalibration(
        export=export_type.from_payload(export_payload),
        image_sizes=[(int(w), int(h)) for w, h in raw["image_sizes"]],
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


def _seed_payload(seed: DeviceSeed | Mapping[str, Any]) -> dict[str, Any]:
    """The JSON form of a seed given as a :class:`DeviceSeed` or a mapping."""
    if isinstance(seed, DeviceSeed):
        return seed.to_payload()
    return DeviceSeed.from_payload(seed).to_payload()


def patched_config(topology: str, patch: ConfigPatch) -> CalibrationConfig:
    """The default config of ``topology`` with ``patch`` applied: the config
    ``vcal run -c`` makes from the same patch.

    Parameters
    ----------
    topology:
        The manifest topology whose config to patch, e.g.
        ``"planar_intrinsics"``.
    patch:
        The :data:`ConfigPatch`, a mapping of the config's fields.

    Returns
    -------
    The topology's config dataclass, e.g.
    :class:`~vision_calibration.models.PlanarIntrinsicsConfig`.

    Raises
    ------
    ValueError
        The topology is unknown, or the patched fields do not form a valid
        config (a field the workflow does not have, a value out of range).
    """
    payload = _native.patched_config(topology, dict(patch))
    config: CalibrationConfig = _PROBLEMS[topology][0].from_mapping(payload)
    return config


def camera_image_sizes(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> list[tuple[int, int]]:
    """Each camera's image size ``(width, height)`` in pixels, in camera order.

    A camera's size is its region's (``roi_xywh``) when it has one, else the
    size its target images share: the ``image_sizes`` that :func:`assess`
    and :func:`projection_uncertainty` take. Only the image headers are read.

    Raises
    ------
    RunError
        The manifest is invalid, a camera's images cannot be listed or read,
        or one camera's images differ in size (``kind`` is
        ``"image_size_mismatch"``).
    """
    sizes = _native.camera_image_sizes(_manifest_arg(manifest), _base_dir_arg(base_dir))
    return [(int(w), int(h)) for w, h in sizes]


def input_files(
    manifest: Manifest,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Every file a run of the manifest reads, each once and sorted: each
    camera's target and laser images, the robot pose file and the upstream
    calibration. The image folders are listed; no file is read, so a caller
    can check what a run would touch before it starts one.

    Raises
    ------
    RunError
        A camera's image pattern is not a valid glob.
    """
    files = _native.input_files(_manifest_arg(manifest), _base_dir_arg(base_dir))
    return [Path(f) for f in files]


def detect_target(
    image: ArrayLike,
    target: TargetSpec | Mapping[str, Any],
    detector: DetectorSpec | Mapping[str, Any] | None = None,
) -> Features:
    """Detect a calibration target in one grayscale image.

    Parameters
    ----------
    image:
        ``(H, W)`` ``uint8`` grayscale image.
    target:
        The target, as a manifest's ``target`` section, e.g.
        ``{"kind": "chessboard", "rows": 17, "cols": 28, "square_size_m": 0.02}``.
    detector:
        The manifest's optional ``detector`` section (corner-detector
        overrides).

    Returns
    -------
    The detected features; empty when the image shows no usable board. A
    chessboard or PuzzleBoard target with ``"measure_blur": True`` also
    reports each corner's measured blur (``blur_psf_px``, ``blur_cov``).

    Raises
    ------
    TypeError
        If ``image`` is not a 2-D ``uint8`` array.
    RunError
        If the target is impossible (``kind == "invalid_target_config"``),
        e.g. an unknown ChArUco dictionary.
    """
    return _features(
        _native.detect_target(
            _gray_image(image),
            dict(target),
            None if detector is None else dict(detector),
        )
    )


def _gray_image(image: ArrayLike) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise TypeError(
            "image must be a 2-D uint8 grayscale array, got "
            f"{array.ndim}-D {array.dtype}; convert colour images first"
        )
    return np.ascontiguousarray(array)


def detect_target_detail(
    image: ArrayLike,
    target: TargetSpec | Mapping[str, Any],
    detector: DetectorSpec | Mapping[str, Any] | None = None,
) -> DetectionDetail:
    """:func:`detect_target`, with what an overlay needs to judge the detection.

    Returns a :class:`~vision_calibration.detect.DetectionDetail`: the
    features :func:`detect_target` returns for the same image, the image's
    outcome, the target-specific overlay data of each feature and the
    candidates the detector left out, each with its reason. Arguments and
    errors as for :func:`detect_target`.
    """
    return _detection_detail(
        _native.detect_target_detail(
            _gray_image(image),
            dict(target),
            None if detector is None else dict(detector),
        )
    )


def detection_detail(
    manifest: Manifest,
    camera: int,
    image: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
) -> DetectionDetail:
    """The detail of a run's detection of one image.

    ``image`` is one of camera ``camera``'s target images (a relative path
    resolves against ``base_dir``, which defaults to the manifest file's
    directory). It is detected with the detector, region of interest and
    cache key a run of the manifest uses, so the features are those a
    :func:`calibrate_dataset` run with the same ``cache`` reads. With a
    ``cache``, a stored detail is reused when it matches the cached
    features; otherwise the image is detected and its detail stored beside
    the features (``force_redetect`` detects again and overwrites both). A
    calibration never reads the detail.

    Raises
    ------
    RunError
        The image is not one of the camera's (``kind`` is
        ``"image_not_in_dataset"``), its cached features differ from a fresh
        detection (``"detail_mismatch"``; refresh the cache), or it cannot be
        read or detected.
    """
    return _detection_detail(
        _native.detection_detail(
            _manifest_arg(manifest),
            camera,
            os.fspath(image),
            _base_dir_arg(base_dir),
            _cache_arg(cache),
            force_redetect,
        )
    )


def cached_detection(
    manifest: Manifest,
    camera: int,
    image: str | os.PathLike[str],
    cache: FsDetectionCache,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """What ``cache`` holds for one image, without detecting it.

    ``image`` is one of camera ``camera``'s target images (a relative path
    resolves against ``base_dir``, which defaults to the manifest file's
    directory). Returns ``None`` when a run of the manifest through
    ``cache`` would detect the image, else ``{"feature_count",
    "detail_outcome"}``: how many features the cached detection holds, and
    the outcome of the detail stored beside them (the
    :func:`detection_detail` reused without detecting), or ``None`` when
    there is none. The image file is read for its cache key (its bytes with
    the detector, its configuration and the camera's region of interest, as
    a run keys it); nothing is decoded, detected or written.

    Raises
    ------
    RunError
        The image is not one of the camera's (``kind`` is
        ``"image_not_in_dataset"``), or it or a cache entry cannot be read.
    """
    return cast(
        "dict[str, Any] | None",
        _native.cached_detection(
            _manifest_arg(manifest),
            camera,
            os.fspath(image),
            _cache_arg(cache),
            _base_dir_arg(base_dir),
        ),
    )
