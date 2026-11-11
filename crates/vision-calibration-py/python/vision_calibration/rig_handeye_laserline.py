"""Rig hand-eye laserline: a camera rig, its hand-eye transform and one laser
plane per camera, solved jointly.

:class:`RigHandeyeLaserlineSession` has no individual steps: its
``run_calibration`` chains the rig hand-eye pipeline, the plane fit and the
joint refinement. ``run_calibration_with_seed`` starts the joint solve from
an earlier rig hand-eye result and/or given laser planes
(:class:`JointManualInit`) instead of estimating them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _vision_calibration as _native
from .models import (
    LaserPlane,
    RigHandeyeExport,
    RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput,
)
from .session import CalibrationSession, _payload, _seed_payload

__all__ = ["JointManualInit", "RigHandeyeLaserlineSession"]


@dataclass(slots=True)
class JointManualInit:
    """Manual seeds of ``run_calibration_with_seed``: the warm start of the
    joint solve. A field left ``None`` is estimated as ``run_calibration``
    estimates it.

    ``rig_handeye`` is a rig hand-eye result of the same views (for example
    from a :class:`~vision_calibration.RigHandeyeSession` run on this input);
    the rig hand-eye stage is skipped and its cameras, extrinsics, hand-eye
    transform, target reference pose, target poses and robot-pose
    corrections are used as they are. It must have the input's cameras (in
    the configured models) and views and the configured hand-eye mode.

    ``laser_planes_cam`` is one laser plane per camera, in that camera's frame (a
    design value, say); the plane stage refines them instead of fitting its
    starting planes linearly.

    A seed that does not fit the input or the config raises ``ValueError``
    before any solve.
    """

    rig_handeye: RigHandeyeExport | None = None
    laser_planes_cam: list[LaserPlane] | None = None

    def to_payload(self) -> dict[str, Any]:
        rig_handeye: dict[str, Any] | None = None
        if self.rig_handeye is not None:
            rig_handeye = self.rig_handeye.export
            if not rig_handeye:
                raise ValueError(
                    "rig_handeye must be an export the library returned "
                    "(RigHandeyeExport.export is empty)"
                )
        return _seed_payload({"rig_handeye": rig_handeye, "laser_planes_cam": self.laser_planes_cam})


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

    def run_calibration_with_seed(self, seed: JointManualInit) -> None:
        """``run_calibration`` from manual seeds of its warm start."""
        self._native.run_calibration_with_seed(_payload(seed))
