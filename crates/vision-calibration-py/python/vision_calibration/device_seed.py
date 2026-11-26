"""Seeds: the device quantities a calibration starts from.

A :class:`DeviceSeed` holds, by camera id, what belongs to the hardware rather
than to a capture: each camera's lens, its mount in the rig, its laser plane
and the rig's hand-eye. It comes from a datasheet (a device spec,
:meth:`DeviceSeed.from_spec`) or from a previous calibration of the device or
of one of its type (:meth:`DeviceSeed.from_export`). A seed is an initial
value: nothing in it constrains the solve or acts as a prior, and it carries
no uncertainty.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from . import _vision_calibration as _native
from . import _generated_types as _types
from ._api import _export_payload


@dataclass(frozen=True, slots=True)
class DeviceSeed:
    """The device quantities a calibration starts from, by camera id.

    ``payload`` is the seed's JSON form (:class:`vision_calibration.types.DeviceSeed`):
    ``cameras`` (each with ``id``, ``image_size``, a ``lens`` that is either
    ``{"kind": "calibrated", "camera": ...}``, a whole calibrated camera, or
    ``{"kind": "nominal", "intrinsics": ..., "sensor": ...}``, a datasheet's
    values, an optional ``cam_se3_rig`` and an optional ``laser_plane``), an
    optional ``handeye`` and an optional ``description``. Units are pixels,
    radians and metres. Store a seed with ``json.dump(seed.to_payload(), f)``
    and read it back with :meth:`from_payload`.

    Build one with :meth:`from_export` or :meth:`from_spec`; every
    constructor validates it.
    """

    payload: _types.DeviceSeed

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DeviceSeed:
        """A seed from its JSON form, validated.

        Raises
        ------
        ValueError
            If the seed's version is not this library's, it has no camera, an
            id repeats, a value is not finite, a lens has a homography sensor,
            only some cameras have a rig mount, or a seed of several cameras
            has a hand-eye but no mounts.
        TypeError
            If the payload does not have the seed's shape.
        """
        return cls(cast(_types.DeviceSeed, _native.device_seed_validate(dict(payload))))

    @classmethod
    def from_export(
        cls,
        result: Any,
        camera_ids: Sequence[str],
        image_sizes: Sequence[tuple[int, int]],
    ) -> DeviceSeed:
        """The device quantities of a calibration result.

        Every camera is carried whole (its model, intrinsics, distortion and
        sensor), with the mounts of a rig of several cameras, the laser
        planes and the hand-eye. Nothing of the capture is carried (target
        poses, robot-pose corrections, residuals), nor the uncertainty. A
        single-camera result gives a seed without mounts: the camera's frame
        is its rig frame.

        Parameters
        ----------
        result:
            A calibration result (any problem type), or an export JSON loaded
            as a dict.
        camera_ids:
            Each camera's id, in the result's camera order: the ids of the
            dataset manifest the next calibration runs on.
        image_sizes:
            ``(width, height)`` of each camera's images, pixels, in camera
            order.

        Raises
        ------
        ValueError
            If the ids or sizes are not one per camera, an id repeats, or a
            camera has a homography sensor.
        TypeError
            If ``result`` is not a calibration result.
        """
        payload = _native.device_seed_from_calibration(
            _export_payload(result),
            list(camera_ids),
            [list(size) for size in image_sizes],
        )
        return cls(cast(_types.DeviceSeed, payload))

    @classmethod
    def from_spec(cls, spec: str | os.PathLike[str] | Mapping[str, Any]) -> DeviceSeed:
        """The datasheet seed of a device spec (``spec.json``): each camera's
        intrinsics (``fx = fy = focal_mm * 1000 / pixel_pitch_um``, the
        principal point at the image centre unless stated) and mount tilt,
        the camera mounts and the hand-eye mount, converted to pixels,
        radians and metres. The spec's nominal laser planes are not carried.

        Parameters
        ----------
        spec:
            The path of a device spec file, or its JSON loaded as a dict.

        Raises
        ------
        ValueError
            If the spec does not validate, its rig layout mounts only some of
            its cameras, or it has several cameras and a hand-eye mount but no
            camera mounts.
        """
        if isinstance(spec, Mapping):
            payload = dict(spec)
        else:
            with open(spec, encoding="utf-8") as file:
                payload = json.load(file)
        return cls(cast(_types.DeviceSeed, _native.device_seed_from_spec(payload)))

    def to_payload(self) -> dict[str, Any]:
        """The seed's JSON form."""
        return dict(self.payload)

    @property
    def camera_ids(self) -> list[str]:
        """The cameras' ids, in the seed's order."""
        return [camera["id"] for camera in self.payload["cameras"]]

    def nominal(self) -> DeviceSeed:
        """The same seed with every calibrated lens reduced to its intrinsics
        and its tilt: the way to start a calibration of another camera model
        from a result. Mounts, laser planes and the hand-eye are kept."""
        return DeviceSeed(cast(_types.DeviceSeed, _native.device_seed_nominal(self.to_payload())))

    def select(self, camera_ids: Sequence[str]) -> DeviceSeed:
        """The seed of the cameras ``camera_ids`` only, in that order, with
        the rig frame and the hand-eye unchanged.

        Raises
        ------
        ValueError
            If an id is not the seed's, repeats, or none is given.
        """
        payload = _native.device_seed_select(self.to_payload(), list(camera_ids))
        return DeviceSeed(cast(_types.DeviceSeed, payload))


__all__ = ["DeviceSeed"]
