"""Sample datasets: real calibration images with a ready manifest.

:func:`catalogue` lists the samples; :func:`fetch` downloads one from its
publisher, at a pinned revision, verifies every file against its SHA-256 and
writes the sample's manifest next to the files::

    import vision_calibration as vc

    manifest = vc.samples.fetch("opencv_left")
    result = vc.calibrate_dataset(manifest)

Files already in the cache and matching their checksum are not downloaded
again. The cache is :func:`cache_dir` unless ``cache`` names another; the
``VISION_CALIBRATION_DATA`` environment variable moves the default. Each
sample states its licence and the citation its publisher asks for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import _vision_calibration as _native

__all__ = ["Sample", "SampleFetchError", "cache_dir", "catalogue", "fetch"]

SampleFetchError = _native.SampleFetchError
"""Raised when a sample cannot be downloaded or verified: the publisher could
not be reached, a file did not match its checksum, or the cache could not be
written. A subclass of :class:`OSError`."""


@dataclass(frozen=True)
class Sample:
    """One sample dataset of the :func:`catalogue`."""

    id: str
    """The name :func:`fetch` takes, also the sample's directory in the cache."""
    description: str
    """What the sample holds: the workflow, the views and the target."""
    licence: str
    """SPDX licence identifier; ``NOASSERTION`` when the publisher states none."""
    citation: str
    """The citation the publisher asks for."""
    source_url: str
    """The publisher's page for the data."""


def catalogue() -> list[Sample]:
    """The sample datasets, in catalogue order."""
    return [Sample(**entry) for entry in _native.sample_catalogue()]


def cache_dir() -> Path:
    """The directory :func:`fetch` uses when ``cache`` is not given.

    ``VISION_CALIBRATION_DATA`` when set, else ``vision-calibration`` in the
    platform's cache directory.

    Raises
    ------
    SampleFetchError
        If neither exists.
    """
    return Path(_native.sample_cache_dir())


def fetch(id: str, cache: str | os.PathLike[str] | None = None) -> Path:
    """Download the sample ``id`` into ``<cache>/<id>/`` and verify it.

    Returns the path of the sample's manifest, ready for
    :func:`~vision_calibration.dataset_runner.calibrate_dataset`.

    Raises
    ------
    ValueError
        If no sample has this id.
    SampleFetchError
        If the download or the verification fails.
    """
    native_cache = None if cache is None else os.fspath(cache)
    return Path(_native.fetch_sample(id, native_cache))
