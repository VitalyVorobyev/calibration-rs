"""Low-level serde payload contracts for :mod:`vision_calibration`.

The ``TypedDict``\\ s and aliases here model the raw JSON payloads the Rust
extension reads and returns. They are generated from the JSON Schemas of the
Rust types (``_generated_types.py``, written by ``cargo xtask emit-schemas``)
and carry the Rust names: ``PlanarIntrinsicsConfig``, ``CameraParams``,
``Iso3``, ``DistortionParams``, …. An enum with data is a union of one
``TypedDict`` per variant, named ``<Enum>_<Variant>`` (``RobustLoss_Huber``).

Most users should use the dataclasses from :mod:`vision_calibration.models`
instead.

Warning
-------
This module is a low-level interop surface. It is not the recommended
high-level API for new code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from . import _generated_types
from ._generated_types import *  # noqa: F403 -- the generated payload types
from ._generated_types import Progress

# What a long call's ``progress`` argument is called with, once per report:
# an image read, a step begun, a solver iteration finished. Its return value
# is ignored; raising stops the call.
ProgressCallback: TypeAlias = Callable[[Progress], object]

__all__ = ["ProgressCallback"]
__all__ += _generated_types.__all__
