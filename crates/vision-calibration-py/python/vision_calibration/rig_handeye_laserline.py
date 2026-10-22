"""Rig hand-eye laserline: a camera rig, its hand-eye transform and one laser
plane per camera, solved jointly.

:class:`RigHandeyeLaserlineSession` has no individual steps: its
``run_calibration`` chains the rig hand-eye pipeline, the plane fit and the
joint refinement.
"""

from __future__ import annotations

from . import _vision_calibration as _native
from .models import (
    RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput,
)
from .session import CalibrationSession

__all__ = ["RigHandeyeLaserlineSession"]


class RigHandeyeLaserlineSession(
    CalibrationSession[
        RigHandeyeLaserlineInput, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport
    ]
):
    """A joint rig hand-eye laserline calibration."""

    _native_type = _native.RigHandeyeLaserlineSession
    _config_class = RigHandeyeLaserlineConfig
    _export_class = RigHandeyeLaserlineExport

    __slots__ = ()
