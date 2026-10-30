"""Manifest-driven runs: from a dataset manifest to a calibration result.

:func:`calibrate_dataset` reads a manifest, detects the target in every
image, builds the problem's input, calibrates and returns the result. The
``build_*_input`` functions stop after the input, for a run driven step by
step through a session. :func:`detect_target` runs a manifest's target
detector on one numpy image.

The runner does not guess: a manifest that leaves a convention open (how
images pair with robot poses, which way a pose transforms) raises
:class:`AskUserError` naming the field to set. Every other failure raises
:class:`RunError` with a ``kind``.

.. code-block:: python

    from vision_calibration.dataset_runner import calibrate_dataset

    result = calibrate_dataset("data/kuka_1/dataset.toml")
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
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
from .detect import Features, FsDetectionCache
from .laserline_device import LaserlineDeviceSession
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
from .planar_intrinsics import PlanarIntrinsicsSession
from .rig_extrinsics import RigExtrinsicsSession
from .rig_handeye import RigHandeyeSession
from .rig_handeye_laserline import RigHandeyeLaserlineSession
from .rig_laserline_device import RigLaserlineDeviceSession
from .session import CalibrationSession
from .single_cam_handeye import SingleCamHandeyeSession

__all__ = [
    "AskUserError",
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
    "detect_target",
]

RunError = _native.RunError
"""A manifest run failed; ``kind`` names the failure: ``"validation"``,
``"empty_image_match"``, ``"pose_count_mismatch"``, ``"detection"``,
``"insufficient_usable_views"``, ``"no_whole_chessboard_views"``, ...."""

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

CalibrationResult = (
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


RejectReason = Literal["too_few_features", "partial_chessboard", "turned_chessboard", "other"]
"""Why the runner left a detection out: fewer features than the per-view
floor, or (where views share one target frame) a plain chessboard that does
not reach every edge of the board or is seen turned by 90°."""


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
) -> PlanarRunResult:
    """Detect a ``planar_intrinsics`` manifest's images into a planar input.

    ``base_dir`` resolves the manifest's relative paths; it defaults to the
    manifest file's directory and is required for a payload. Without a
    ``cache`` every image is detected; with one, cached detections are
    reused unless ``force_redetect``.
    """
    raw = _native.build_planar_input(
        _manifest_arg(manifest), _base_dir_arg(base_dir), _cache_arg(cache), force_redetect
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
) -> HandeyeRunResult:
    """Detect a ``single_cam_handeye`` manifest's images and pair them with
    its robot poses. Arguments as for :func:`build_planar_input`."""
    raw = _native.build_single_cam_handeye_input(
        _manifest_arg(manifest), _base_dir_arg(base_dir), _cache_arg(cache), force_redetect
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
) -> RigRunResult:
    """Detect a ``rig_extrinsics`` manifest's images into rig views.
    Arguments as for :func:`build_planar_input`."""
    raw = _native.build_rig_extrinsics_input(
        _manifest_arg(manifest), _base_dir_arg(base_dir), _cache_arg(cache), force_redetect
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
) -> RigRunResult:
    """Detect a ``rig_handeye`` manifest's images into rig views paired with
    its robot poses. Arguments as for :func:`build_planar_input`."""
    raw = _native.build_rig_handeye_input(
        _manifest_arg(manifest), _base_dir_arg(base_dir), _cache_arg(cache), force_redetect
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
    )
    return RigHandeyeLaserlineRunResult(
        input=raw["input"],
        view_paths=_rig_paths(raw["view_paths"]),
        laser_paths=_rig_paths(raw["laser_paths"]),
        usable_views=int(raw["usable_views"]),
        total_views=int(raw["total_views"]),
        rejected=_rejected(raw["rejected"]),
    )


# Each topology's input builder and session class.
_TOPOLOGIES: dict[str, tuple[Callable[..., Any], type[CalibrationSession[Any, Any, Any]], bool]] = {
    "planar_intrinsics": (build_planar_input, PlanarIntrinsicsSession, False),
    "single_cam_handeye": (build_single_cam_handeye_input, SingleCamHandeyeSession, False),
    "rig_extrinsics": (build_rig_extrinsics_input, RigExtrinsicsSession, False),
    "rig_handeye": (build_rig_handeye_input, RigHandeyeSession, False),
    "laserline_device": (build_laserline_device_input, LaserlineDeviceSession, True),
    "rig_laserline_device": (build_rig_laserline_device_input, RigLaserlineDeviceSession, True),
    "rig_handeye_laserline": (
        build_rig_handeye_laserline_input,
        RigHandeyeLaserlineSession,
        True,
    ),
}


def calibrate_dataset(
    manifest: Manifest,
    config: CalibrationConfig | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    cache: FsDetectionCache | None = None,
    force_redetect: bool = False,
    laser_extractor: LaserPixelExtractor | None = None,
) -> CalibrationResult:
    """Calibrate a dataset from its manifest.

    The manifest's ``topology`` selects the problem: its images are detected
    into the problem's input (the ``build_*_input`` of the topology), which is
    calibrated with ``config`` (the problem's config dataclass or payload;
    ``None`` for the defaults) and exported.

    Parameters
    ----------
    manifest:
        Path of a TOML or JSON manifest, or the ``DatasetSpec`` payload.
    config:
        The problem's config, e.g. a
        :class:`~vision_calibration.models.SingleCamHandeyeConfig` for a
        ``single_cam_handeye`` manifest.
    base_dir:
        Directory the manifest's relative paths resolve against; defaults to
        the manifest file's directory, required for a payload.
    cache:
        Detection cache; ``None`` detects every image.
    force_redetect:
        Detect again even where the cache has an entry.
    laser_extractor:
        The laser-line extractor, required by the laser topologies.

    Returns
    -------
    The topology's export dataclass, e.g.
    :class:`~vision_calibration.models.SingleCamHandeyeExport`.

    Raises
    ------
    AskUserError
        The manifest leaves a convention open.
    RunError
        The manifest is invalid, or detection or pairing failed.
    ValueError
        The config is invalid, or the problem rejects the detected input.
    RuntimeError
        The calibration itself failed.
    """
    spec = cast(
        Mapping[str, Any],
        _native.load_manifest(os.fspath(manifest))
        if isinstance(manifest, (str, os.PathLike))
        else manifest,
    )
    topology = str(spec.get("topology"))
    entry = _TOPOLOGIES.get(topology)
    if entry is None:
        raise ValueError(f"unknown manifest topology {topology!r}")
    build, session_type, needs_laser = entry
    if needs_laser:
        if laser_extractor is None:
            raise ValueError(f"a {topology} manifest needs a laser_extractor")
        run = build(
            manifest, laser_extractor, base_dir=base_dir, cache=cache, force_redetect=force_redetect
        )
    else:
        run = build(manifest, base_dir=base_dir, cache=cache, force_redetect=force_redetect)
    session = session_type()
    session.set_input(run.input)
    if config is not None:
        session.set_config(config)
    session.run_calibration()
    return cast(CalibrationResult, session.export())


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
    The detected features; empty when the image shows no usable board.

    Raises
    ------
    TypeError
        If ``image`` is not a 2-D ``uint8`` array.
    RunError
        If the target is impossible (``kind == "invalid_target_config"``),
        e.g. an unknown ChArUco dictionary.
    """
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise TypeError(
            "image must be a 2-D uint8 grayscale array, got "
            f"{array.ndim}-D {array.dtype}; convert colour images first"
        )
    image_xy, world_xyz, cov_px = _native.detect_target(
        np.ascontiguousarray(array),
        dict(target),
        None if detector is None else dict(detector),
    )
    return Features(image_xy=image_xy, world_xyz=world_xyz, cov_px=cov_px)
