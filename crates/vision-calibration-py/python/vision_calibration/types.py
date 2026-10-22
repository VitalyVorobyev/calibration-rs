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

from typing import Any, TypeAlias

from . import _generated_types
from ._generated_types import *  # noqa: F403 -- the generated payload types
from ._generated_types import (
    CameraParams,
    DistortionKind,
    DistortionParams,
    DistortionParams_BrownConrady5,
    DistortionParams_Division,
    DistortionParams_Rational,
    DistortionParams_ThinPrism,
    FxFyCxCy,
    Iso3,
    LaserPlane,
    ProjectionKind,
    RobustLoss_Arctan,
    RobustLoss_Cauchy,
    RobustLoss_Huber,
    ScheimpflugParams,
    SensorKind,
    UncertaintyUnavailable,
)

# Untyped JSON containers; no schema describes "any object".
JsonObject: TypeAlias = dict[str, Any]
JsonArray: TypeAlias = list[Any]

# A robot pose correction `[rx, ry, rz, tx, ty, tz]`, the element of an
# export's `robot_deltas`. The schema has it as an anonymous fixed-size array,
# so it has no generated name.
Se3Delta: TypeAlias = tuple[float, float, float, float, float, float] | list[float]

# Python spellings of generated types. Each names the same type as its
# right-hand side; new code uses the Rust name.
Transform: TypeAlias = Iso3
CameraModel: TypeAlias = CameraParams
IntrinsicsPayload: TypeAlias = FxFyCxCy
DistortionPayload: TypeAlias = DistortionParams
DistortionParamsPayload: TypeAlias = DistortionParams
ScheimpflugPayload: TypeAlias = ScheimpflugParams
LaserPlanePayload: TypeAlias = LaserPlane
DistortionModel: TypeAlias = DistortionKind
ProjectionModel: TypeAlias = ProjectionKind
SensorModel: TypeAlias = SensorKind
DistortionBrownConrady5: TypeAlias = DistortionParams_BrownConrady5
DistortionDivision: TypeAlias = DistortionParams_Division
DistortionRational: TypeAlias = DistortionParams_Rational
DistortionThinPrism: TypeAlias = DistortionParams_ThinPrism
RobustLossHuber: TypeAlias = RobustLoss_Huber
RobustLossCauchy: TypeAlias = RobustLoss_Cauchy
RobustLossArctan: TypeAlias = RobustLoss_Arctan
UncertaintyUnavailableReason: TypeAlias = UncertaintyUnavailable

__all__ = [
    "JsonObject",
    "JsonArray",
    "Se3Delta",
    "Transform",
    "CameraModel",
    "IntrinsicsPayload",
    "DistortionPayload",
    "DistortionParamsPayload",
    "ScheimpflugPayload",
    "LaserPlanePayload",
    "DistortionModel",
    "ProjectionModel",
    "SensorModel",
    "DistortionBrownConrady5",
    "DistortionDivision",
    "DistortionRational",
    "DistortionThinPrism",
    "RobustLossHuber",
    "RobustLossCauchy",
    "RobustLossArctan",
    "UncertaintyUnavailableReason",
]
__all__ += _generated_types.__all__
