"""Dataset manifests: describe a folder of images for a calibration run.

A manifest is a ``DatasetSpec`` (TOML or JSON on disk, the
:class:`~vision_calibration.types.DatasetSpec` payload in Python): the
cameras' images, the target, the robot poses and the topology.
:func:`sniff_folder` drafts one from a folder;
:func:`~vision_calibration.dataset_runner.calibrate_dataset` runs one.
"""

from __future__ import annotations

import os
from typing import cast

from . import _vision_calibration as _native
from ._generated_types import DatasetSpec

__all__ = ["SniffError", "sniff_folder"]

SniffError = _native.SniffError
"""Raised when a folder cannot be sniffed; ``kind`` is ``"not_a_directory"``,
``"no_cameras"`` or ``"io"``."""


def sniff_folder(root: str | os.PathLike[str]) -> DatasetSpec:
    """Draft a manifest from a folder of images.

    Cameras are grouped from the image files found under ``root``; what the
    folder cannot tell (board geometry, pose conventions) is filled with
    placeholders and listed under ``_unresolved``. Review those fields and
    clear the list before running the manifest.

    Raises
    ------
    SniffError
        If ``root`` is not a readable directory or holds no images.
    """
    return cast(DatasetSpec, _native.sniff_folder(os.fspath(root)))
