"""Dense diagnostic maps over an image, as numpy planes.

:func:`~vision_calibration.distortion_map` gives a camera's lens and sensor
displacement and :func:`~vision_calibration.residual_map` the systematic part
of its reprojection residuals, both on the same grid: the centres of
``stride × stride`` pixel blocks, row-major from the top-left, pixel centres
at integer coordinates. Every plane is a ``float64`` array shaped
``(rows, columns)``, NaN where the map has no value; :meth:`ImageGrid.nodes`
gives the node positions.

Every displacement is ``actual − reference`` in pixels: a distortion layer is
where the camera puts a ray relative to the simpler model without that stage,
and a residual is ``observed − projected``. A distortion the model lacks shows
in the residual map with the sign and units it has in the distortion map.

.. code-block:: python

    dmap = vc.distortion_map(result.export["camera"], (1280, 960))
    print(np.nanmax(dmap.total.magnitude()), "px of distortion at most")
    rmap = vc.residual_map(result, (1280, 960))
    significant = rmap.significance > 3.03
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["DisplacementPlane", "DistortionMap", "ImageGrid", "ResidualMap"]


def _plane(value: Any) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(value, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class ImageGrid:
    """The grid a map is sampled on."""

    image_size: tuple[int, int]
    """``(width, height)`` of the image, pixels."""
    stride: int
    """The side of a block, pixels."""
    columns: int
    """Nodes per row, ``ceil(width / stride)``."""
    rows: int
    """Rows of nodes, ``ceil(height / stride)``."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ImageGrid":
        size = cast(list[Any], payload["image_size"])
        return cls(
            image_size=(int(size[0]), int(size[1])),
            stride=int(payload["stride"]),
            columns=int(payload["columns"]),
            rows=int(payload["rows"]),
        )

    def nodes(self) -> NDArray[np.float64]:
        """The node positions, pixels, shaped ``(rows, columns, 2)`` as
        ``(x, y)``: the centre of each block's in-image pixels (a trailing
        partial block is centred on the pixels it holds)."""

        def centres(n: int, size: int) -> NDArray[np.float64]:
            start = np.arange(n, dtype=np.float64) * self.stride
            end = np.minimum(start + self.stride, size)
            return cast(NDArray[np.float64], (start + end - 1.0) / 2.0)

        x = centres(self.columns, self.image_size[0])
        y = centres(self.rows, self.image_size[1])
        xx, yy = np.meshgrid(x, y)
        return cast(NDArray[np.float64], np.stack([xx, yy], axis=-1))


@dataclass(frozen=True, slots=True)
class DisplacementPlane:
    """A displacement field, pixels: ``(rows, columns)`` arrays, NaN where the
    map has no value."""

    dx: NDArray[np.float64]
    """The ``x`` component, pixels."""
    dy: NDArray[np.float64]
    """The ``y`` component, pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DisplacementPlane":
        return cls(dx=_plane(payload["dx"]), dy=_plane(payload["dy"]))

    def magnitude(self) -> NDArray[np.float64]:
        """The length of the displacement at each node, pixels (NaN where
        there is none)."""
        return cast(NDArray[np.float64], np.hypot(self.dx, self.dy))


@dataclass(frozen=True, slots=True)
class DistortionMap:
    """The lens and sensor displacement of a camera over its image, in three
    layers that sum exactly: ``total = distortion + sensor``.

    Each layer is ``actual − reference`` at the same ray, indexed by the pixel
    where the ray lands: ``distortion = K(D(n)) − K(n)`` (the lens as the
    camera would show it without a sensor tilt), ``sensor = K(S(D(n))) −
    K(D(n))`` (zero for an untilted sensor). Barrel distortion points inward
    at the corners. For an equidistant (fisheye) camera the reference is the
    ``r = f·θ`` image.

    Nodes whose ray is past a fold of the distortion, outside a fisheye
    polynomial's invertible range, or sent to infinity by the sensor are NaN.
    The distortion layer depends on how the calibration split the projection
    between ``K`` and the distortion coefficients: compare two calibrations'
    projections, not their layers. Far from the observed features the values
    extrapolate the fitted polynomial.
    """

    grid: ImageGrid
    total: DisplacementPlane
    """The whole displacement, ``K∘S∘D∘P − K∘P``."""
    distortion: DisplacementPlane
    """``K∘D∘P − K∘P``: the lens distortion."""
    sensor: DisplacementPlane
    """``K∘S∘D∘P − K∘D∘P``: what the sensor model adds."""
    valid_nodes: int
    """Nodes with a value."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DistortionMap":
        return cls(
            grid=ImageGrid.from_payload(payload),
            total=DisplacementPlane.from_payload(payload["total"]),
            distortion=DisplacementPlane.from_payload(payload["distortion"]),
            sensor=DisplacementPlane.from_payload(payload["sensor"]),
            valid_nodes=int(payload["valid_nodes"]),
        )


@dataclass(frozen=True, slots=True)
class ResidualMap:
    """The systematic part of one camera's reprojection residuals over its
    image: a Gaussian-kernel local-linear mean of ``observed − projected``
    (records placed at their projection), with a view-clustered standard
    error.

    A significant field is not by itself a reason to add a model term (it
    averages this capture's depths and poses), a flat map does not prove the
    model (a missing term the parameters can mimic is absorbed into them),
    and the map is not a correction to subtract. Residuals of a hand-eye or
    rig result include the pose chain's error; for a camera-model diagnosis,
    map a fit with a free pose per view. ``significance`` is pointwise and
    nodes within about two bandwidths are strongly correlated.
    """

    grid: ImageGrid
    camera: int
    """The camera mapped."""
    bandwidth_px: float
    """The Gaussian kernel's σ, pixels."""
    bandwidth_selection: Mapping[str, Any]
    """How the bandwidth was chosen: ``{"kind": "given"}``,
    ``{"kind": "cross_validated", "folds", "ladder_px", "scores_px2",
    "score_std_errors_px2"}`` or ``{"kind": "few_views"}``."""
    mean_dx: NDArray[np.float64]
    """The local mean of ``observed − projected``, ``x``, pixels; NaN where
    the support is too thin."""
    mean_dy: NDArray[np.float64]
    """The local mean, ``y``, pixels."""
    rms: NDArray[np.float64]
    """The kernel-weighted RMS of ``|observed − projected|``, pixels."""
    std_error: NDArray[np.float64]
    """The major-axis standard error of the mean (clustered by view),
    pixels; NaN with too few effective views. Conditional on the fitted
    camera and poses: over repeated, refitted captures the map scatters
    less than this, but a systematic shift of the fit with the noise is not
    covered."""
    significance: NDArray[np.float64]
    """The pointwise test of a zero mean as the ``χ`` of a two-degree-of-
    freedom test: ``χ > 3.03`` is ``p < 0.01`` at any number of views. The
    covariance is estimated from ``G`` effective views, so ``T² = μᵀ V⁻¹ μ``
    is referred to Hotelling's distribution with ``ν = G − 1``:
    ``p = (1 + T²/ν)^(−(ν−1)/2)``, ``χ = sqrt(−2 ln p)``. NaN with fewer than
    three effective views. Conditional on the fitted parameters, as
    ``std_error``."""
    effective_samples: NDArray[np.float64]
    """The effective number of samples behind each node (0 where none)."""
    effective_views: NDArray[np.float64]
    """The effective number of views behind each node."""
    used: int
    """The records mapped."""
    unprojected: int
    """The selected records without a projection."""
    outliers: list[int]
    """The records left out as outliers, as indices into the export's
    ``per_feature_residuals.target`` records."""
    views: int
    """The views among the records mapped."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualMap":
        return cls(
            grid=ImageGrid.from_payload(payload),
            camera=int(payload["camera"]),
            bandwidth_px=float(payload["bandwidth_px"]),
            bandwidth_selection=cast(Mapping[str, Any], payload["bandwidth_selection"]),
            mean_dx=_plane(payload["mean_dx"]),
            mean_dy=_plane(payload["mean_dy"]),
            rms=_plane(payload["rms"]),
            std_error=_plane(payload["std_error"]),
            significance=_plane(payload["significance"]),
            effective_samples=_plane(payload["effective_samples"]),
            effective_views=_plane(payload["effective_views"]),
            used=int(payload["used"]),
            unprojected=int(payload["unprojected"]),
            outliers=[int(i) for i in cast(list[Any], payload["outliers"])],
            views=int(payload["views"]),
        )

    def mean_magnitude(self) -> NDArray[np.float64]:
        """The length of the local mean at each node, pixels."""
        return cast(NDArray[np.float64], np.hypot(self.mean_dx, self.mean_dy))
