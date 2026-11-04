"""Multiple-view geometry: measurements from a calibration.

Every function takes a **calibration result** (what a ``run_*`` function or a
session returns, or an export JSON loaded as a dict) and works through the
calibrated rig it describes: the cameras, where each sits in the rig, the
laser planes of a laser calibration and the covariance of the estimated
parameters. Pixels go in; each pixel is unprojected through its camera's own
model, so distortion, fisheye and tilted sensors need no separate step.

Poses are ``4x4`` numpy matrices named ``<frame>_se3_<frame>``; the matrix
maps the right frame into the left. Every σ and covariance is a numpy array in
SI units (metres, radians) or pixels.

.. code-block:: python

    import numpy as np
    import vision_calibration as vc
    from vision_calibration import mvg

    noise = mvg.reprojection_noise(result, camera=0)   # the calibration's level
    tracks = [
        [mvg.RigObservation(0, 0, (412.1, 300.4), noise),
         mvg.RigObservation(0, 1, (377.9, 301.0), noise)],
    ]
    found = mvg.triangulate(result, np.eye(4)[None], tracks)
    points = mvg.PointSet.from_triangulation(found)
    length = mvg.distance(points, 0, 1)
    print(length.value, length.sigma)

Errors: an input the caller can fix (a camera index out of range, too few
matches, a calibration without a covariance where one is needed) raises
:class:`ValueError`; a degenerate geometry or a failed numerical step raises
:class:`RuntimeError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _vision_calibration as _native
from ._api import CalibrationResult
from ._generated_types import RobustLoss
from .models import Camera, ParameterUncertainty, PinholeIntrinsics, PointCovariance
from .types import ProgressCallback

__all__ = [
    "BlockMatchOptions",
    "BundleAdjustment",
    "EssentialEstimate",
    "HomographyEstimate",
    "LaserObservation",
    "LaserPoint",
    "LaserTriangulation",
    "ObjectPose",
    "PixelMatch",
    "PlaneFit",
    "PointSet",
    "PoseObservation",
    "RansacOptions",
    "RelativePose",
    "RigObservation",
    "RobustRelativePose",
    "Scalar",
    "SceneDiagnostics",
    "StereoRectification",
    "TriangulatedPoint",
    "Triangulation",
    "angle",
    "bundle_adjust",
    "distance",
    "estimate_essential",
    "estimate_homography",
    "estimate_pose",
    "fit_plane",
    "match_block",
    "point_to_plane",
    "recover_relative_pose",
    "recover_relative_pose_robust",
    "rectify",
    "reprojection_noise",
    "triangulate",
    "triangulate_laser",
]

Pixel = tuple[float, float]
Matrix4 = NDArray[np.float64]
"""A ``(4, 4)`` homogeneous pose matrix."""


# ─── Conversions ──────────────────────────────────────────────────────────────


def _export_of(result: CalibrationResult | Mapping[str, Any]) -> dict[str, Any]:
    export = result if isinstance(result, Mapping) else result.export
    if not export:
        raise ValueError("the result carries no export; it was not produced by this library")
    return dict(export)


def _pixel(value: ArrayLike, name: str) -> list[float]:
    pixel = np.asarray(value, dtype=np.float64)
    if pixel.shape != (2,):
        raise ValueError(f"{name} must be a (u, v) pixel, got shape {pixel.shape}")
    return [float(pixel[0]), float(pixel[1])]


def _point(value: ArrayLike, name: str) -> list[float]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"{name} must be an (x, y, z) point, got shape {point.shape}")
    return [float(point[0]), float(point[1]), float(point[2])]


def _poses(poses: ArrayLike, name: str) -> NDArray[np.float64]:
    """``poses`` as a contiguous ``(K, 4, 4)`` array (one ``(4, 4)`` is a stack of one)."""
    array = np.asarray(poses, dtype=np.float64)
    if array.shape == (4, 4):
        array = array[None]
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (K, 4, 4) or (4, 4), got {array.shape}")
    return np.ascontiguousarray(array)


def _rows(values: ArrayLike, cols: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != cols:
        raise ValueError(f"{name} must have shape (N, {cols}), got {array.shape}")
    return np.ascontiguousarray(array)


def _matrix(payload: Any, rows: int, cols: int) -> NDArray[np.float64]:
    """A serde matrix (column-major flat list) as a ``(rows, cols)`` array."""
    return np.asarray(payload, dtype=np.float64).reshape(cols, rows).T.copy()


def _pose_matrix(payload: Mapping[str, Any]) -> Matrix4:
    """A serde ``Iso3`` (``rotation`` quaternion ``xyzw``, ``translation``) as a ``4x4`` matrix."""
    x, y, z, w = (float(v) for v in payload["rotation"])
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = [float(v) for v in payload["translation"]]
    return out


def _poses_array(payload: Sequence[Mapping[str, Any]]) -> NDArray[np.float64]:
    if not payload:
        return np.zeros((0, 4, 4))
    return np.stack([_pose_matrix(p) for p in payload])


def _vector(payload: Any) -> NDArray[np.float64]:
    return np.asarray(payload, dtype=np.float64)


# ─── Observations and options ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RansacOptions:
    """RANSAC settings of the robust estimators.

    ``thresh`` is the inlier threshold in the unit the estimator states: σ of
    the stated pixel noise for :func:`estimate_essential`,
    :func:`recover_relative_pose_robust` and :func:`estimate_pose`
    (``3.0`` keeps about 99.7 % of the inliers whatever the focal length or
    camera model). The seed makes a run reproducible.
    """

    thresh: float
    """Inlier threshold; finite and positive."""
    max_iters: int = 1000
    """Most RANSAC iterations."""
    min_inliers: int = 12
    """Fewest inliers a model needs to be accepted."""
    confidence: float = 0.99
    """Desired confidence in ``[0, 1]`` of finding a good model."""
    seed: int = 1_234_567
    """Seed of the sampler."""
    refit_on_inliers: bool = True
    """Refit the model on all inliers before scoring it."""

    def to_payload(self) -> dict[str, Any]:
        return {
            "thresh": float(self.thresh),
            "max_iters": int(self.max_iters),
            "min_inliers": int(self.min_inliers),
            "confidence": float(self.confidence),
            "seed": int(self.seed),
            "refit_on_inliers": bool(self.refit_on_inliers),
        }


@dataclass(frozen=True, slots=True)
class PixelMatch:
    """A pair of matched pixels with their noise: ``pt1`` is a pixel of the
    first camera of the call and ``pt2`` of the second.

    The noise (:class:`~vision_calibration.models.PointCovariance`, px²)
    states each pixel's measurement noise; the functions weigh the epipolar
    constraint by it. :func:`reprojection_noise` gives a calibration's level.
    """

    pt1: Pixel
    noise1: PointCovariance
    pt2: Pixel
    noise2: PointCovariance

    def to_payload(self) -> dict[str, Any]:
        return {
            "pt1": _pixel(self.pt1, "pt1"),
            "noise1": self.noise1.to_payload(),
            "pt2": _pixel(self.pt2, "pt2"),
            "noise2": self.noise2.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class RigObservation:
    """One pixel of a point: seen by rig camera ``camera`` at capture
    ``capture`` (an index into the ``rig_se3_world`` poses of the call), with
    its stated noise."""

    capture: int
    camera: int
    pixel: Pixel
    noise: PointCovariance

    def to_payload(self) -> dict[str, Any]:
        return {
            "capture": int(self.capture),
            "camera": int(self.camera),
            "pixel": _pixel(self.pixel, "pixel"),
            "noise": self.noise.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """One known object point (``object``, in the object's frame, metres) seen
    by rig camera ``camera`` at ``pixel``, with its stated noise."""

    camera: int
    object: tuple[float, float, float]
    pixel: Pixel
    noise: PointCovariance

    def to_payload(self) -> dict[str, Any]:
        return {
            "camera": int(self.camera),
            "object": _point(self.object, "object"),
            "pixel": _pixel(self.pixel, "pixel"),
            "noise": self.noise.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class LaserObservation:
    """One pixel of the laser line seen by rig camera ``camera``; the camera
    needs a laser plane in the calibration."""

    camera: int
    pixel: Pixel
    noise: PointCovariance

    def to_payload(self) -> dict[str, Any]:
        return {
            "camera": int(self.camera),
            "pixel": _pixel(self.pixel, "pixel"),
            "noise": self.noise.to_payload(),
        }


def reprojection_noise(
    result: CalibrationResult | Mapping[str, Any], camera: int = 0
) -> PointCovariance:
    """The pixel noise the calibration states for ``camera``: its reprojection
    residual level, as an isotropic covariance.

    Raises
    ------
    ValueError
        If ``camera`` is out of range, the calibration has no covariance (state
        the noise with :meth:`PointCovariance.isotropic` instead) or its
        reprojections were weighted by per-point covariances, whose level is
        dimensionless.
    """
    payload = cast(Mapping[str, Any], _native.mvg_reprojection_noise(_export_of(result), camera))
    return PointCovariance.from_payload(payload)


# ─── Results ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TriangulatedPoint:
    """A triangulated point with its quality diagnostics."""

    point: NDArray[np.float64]
    """``(3,)`` position, metres, in the frame the poses define."""
    reprojection_error: float | None
    """RMS pixel distance of the observations to the point's projections;
    ``None`` when the point does not project in one of the cameras."""
    parallax_deg: float
    """Largest angle between two observing rays; small means poor geometry."""
    in_front: bool
    """Whether the point lies ahead along every observing bearing."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TriangulatedPoint:
        error = payload["reprojection_error"]
        return cls(
            point=_vector(payload["point"]),
            reprojection_error=None if error is None else float(error),
            parallax_deg=float(payload["parallax_deg"]),
            in_front=bool(payload["in_front"]),
        )


class _Joint:
    """The labelled joint covariance of a result, as one numpy array."""

    __slots__ = ("covariance", "index")

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.index: dict[str, int] = {}
        self.covariance: NDArray[np.float64] | None = None
        if payload["status"] == "available":
            self.index = {str(label): i for i, label in enumerate(payload["labels"])}
            self.covariance = np.asarray(payload["covariance"], dtype=np.float64)

    def block(self, rows: Sequence[str], cols: Sequence[str]) -> NDArray[np.float64] | None:
        """The block of rows ``rows`` and columns ``cols`` (labels), or ``None``
        when the uncertainty is unavailable or a label is not in it."""
        if self.covariance is None:
            return None
        try:
            r = [self.index[label] for label in rows]
            c = [self.index[label] for label in cols]
        except KeyError:
            return None
        return self.covariance[np.ix_(r, c)]


def _axes(i: int) -> list[str]:
    return [f"p{i}.{axis}" for axis in "xyz"]


class _PointCovariances:
    """``point_covariance`` and ``cross_covariance`` over a result's joint
    covariance with the labels ``p{i}.x``, ``p{i}.y``, ``p{i}.z``."""

    _joint: _Joint

    def point_covariance(self, i: int) -> NDArray[np.float64] | None:
        """Point ``i``'s ``(3, 3)`` covariance, m²; ``None`` when the
        uncertainty is unavailable or ``i`` has none."""
        return self._joint.block(_axes(i), _axes(i))

    def cross_covariance(self, i: int, j: int) -> NDArray[np.float64] | None:
        """The ``(3, 3)`` cross-covariance of points ``i`` and ``j`` (rows ``i``,
        columns ``j``), m²: the calibration's uncertainty is shared by every
        point of the call, so it correlates them. ``None`` as for
        :meth:`point_covariance`."""
        return self._joint.block(_axes(i), _axes(j))


@dataclass(frozen=True, slots=True)
class Triangulation(_PointCovariances):
    """The points of one :func:`triangulate` call and their joint uncertainty."""

    points: list[TriangulatedPoint]
    """One point per track, in the world frame, with its diagnostics."""
    uncertainty: ParameterUncertainty
    """The joint covariance of every refined point's coordinates, m², with the
    labels ``p{i}.x``, ``p{i}.y``, ``p{i}.z`` for track ``i``; or the reason it
    is unavailable."""
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Triangulation:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        return cls(
            points=[
                TriangulatedPoint.from_payload(cast(Mapping[str, Any], p))
                for p in payload["points"]
            ],
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            _joint=_Joint(uncertainty),
        )


@dataclass(frozen=True, slots=True)
class LaserPoint:
    """A 3D point of the laser line."""

    point: NDArray[np.float64]
    """``(3,)`` position in the rig frame, metres."""
    camera: int
    """The camera that saw it."""
    incidence_deg: float
    """Angle between the viewing ray and the laser plane: near 0 the ray grazes
    the plane and the depth is ill-determined; 90 is head-on."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LaserPoint:
        return cls(
            point=_vector(payload["point"]),
            camera=int(payload["camera"]),
            incidence_deg=float(payload["incidence_deg"]),
        )


@dataclass(frozen=True, slots=True)
class LaserTriangulation(_PointCovariances):
    """The points of one :func:`triangulate_laser` call and their joint
    uncertainty (pixels, camera, laser plane and extrinsic)."""

    points: list[LaserPoint]
    """One point per observation, in the rig frame."""
    uncertainty: ParameterUncertainty
    """The joint covariance of the points' coordinates, as in :class:`Triangulation`."""
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LaserTriangulation:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        return cls(
            points=[
                LaserPoint.from_payload(cast(Mapping[str, Any], p)) for p in payload["points"]
            ],
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            _joint=_Joint(uncertainty),
        )


@dataclass(frozen=True, slots=True)
class SceneDiagnostics:
    """Whether a two-view scene constrains the geometry."""

    median_parallax_deg: float
    is_pure_rotation: bool
    is_planar: bool
    baseline_ratio: float
    """The baseline over the median scene depth."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SceneDiagnostics:
        return cls(
            median_parallax_deg=float(payload["median_parallax_deg"]),
            is_pure_rotation=bool(payload["is_pure_rotation"]),
            is_planar=bool(payload["is_planar"]),
            baseline_ratio=float(payload["baseline_ratio"]),
        )


_ESSENTIAL_LABELS = [f"essential.e{r}{c}" for r in range(3) for c in range(3)]
_RELATIVE_POSE_LABELS = [
    "rotation.rx",
    "rotation.ry",
    "rotation.rz",
    "direction.x",
    "direction.y",
    "direction.z",
]


@dataclass(frozen=True, slots=True)
class EssentialEstimate:
    """A robustly estimated essential matrix, refined on its inliers."""

    essential: NDArray[np.float64]
    """``(3, 3)``: ``b2ᵀ E b1 = 0`` on the bearings of matched pixels, scaled
    to a unit translation (Frobenius norm √2)."""
    inliers: list[int]
    """Indices of the inlier matches: those the estimate is fitted to."""
    inlier_rms: float
    """RMS weighted Sampson error over the inliers, in σ: about 1 when the
    stated noise describes the pixels."""
    uncertainty: ParameterUncertainty
    """Covariance of the nine entries of ``essential`` (``essential.e00`` ...
    ``essential.e22``, row-major; rank 5) from the inliers' stated noise, or
    the reason it is unavailable."""
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EssentialEstimate:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        return cls(
            essential=_matrix(payload["essential"], 3, 3),
            inliers=[int(i) for i in payload["inliers"]],
            inlier_rms=float(payload["inlier_rms"]),
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            _joint=_Joint(uncertainty),
        )

    def essential_covariance(self) -> NDArray[np.float64] | None:
        """The ``(9, 9)`` covariance of the entries of ``essential`` in row-major
        order; ``None`` when the uncertainty is unavailable."""
        return self._joint.block(_ESSENTIAL_LABELS, _ESSENTIAL_LABELS)


@dataclass(frozen=True, slots=True)
class HomographyEstimate:
    """A robustly estimated homography between two views of a plane."""

    homography: NDArray[np.float64]
    """``(3, 3)`` matrix mapping first-view pixels to second-view pixels."""
    inliers: list[int]
    """Indices of the inlier correspondences."""
    inlier_rms: float
    """RMS symmetric transfer error over the inliers, pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HomographyEstimate:
        return cls(
            homography=_matrix(payload["homography"], 3, 3),
            inliers=[int(i) for i in payload["inliers"]],
            inlier_rms=float(payload["inlier_rms"]),
        )


@dataclass(frozen=True, slots=True)
class RelativePose:
    """The relative pose of two cameras from matched pixels."""

    second_se3_first: Matrix4
    """``(4, 4)`` pose of the first camera's frame in the second's; the
    translation has unit norm (the scale is not observable from two views)."""
    essential: NDArray[np.float64]
    """``(3, 3)`` essential matrix used."""
    points: list[TriangulatedPoint]
    """One point per match, in the first camera's frame at the unit baseline."""
    scene: SceneDiagnostics

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RelativePose:
        return cls(
            second_se3_first=_pose_matrix(payload["second_se3_first"]),
            essential=_matrix(payload["essential"], 3, 3),
            points=[
                TriangulatedPoint.from_payload(cast(Mapping[str, Any], p))
                for p in payload["points"]
            ],
            scene=SceneDiagnostics.from_payload(payload["scene"]),
        )


@dataclass(frozen=True, slots=True)
class RobustRelativePose:
    """The relative pose of two cameras from matched pixels, with outliers
    rejected by RANSAC and the pose refined on the inliers."""

    second_se3_first: Matrix4
    """``(4, 4)`` pose of the first camera's frame in the second's (unit-norm translation)."""
    essential: NDArray[np.float64]
    """``(3, 3)`` essential matrix used."""
    inliers: list[int]
    """Indices of the inlier matches: those the pose is fitted to."""
    inlier_rms: float
    """RMS weighted Sampson error over the inliers, in σ."""
    uncertainty: ParameterUncertainty
    """Covariance of the pose in the gauge ``|t| = 1``: ``rotation.rx``,
    ``.ry``, ``.rz`` (rad, right-perturbation tangent of the rotation) and
    ``direction.x``, ``.y``, ``.z`` (the unit translation's components; rank
    2, no variance along the translation); or the reason it is unavailable."""
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RobustRelativePose:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        return cls(
            second_se3_first=_pose_matrix(payload["second_se3_first"]),
            essential=_matrix(payload["essential"], 3, 3),
            inliers=[int(i) for i in payload["inliers"]],
            inlier_rms=float(payload["inlier_rms"]),
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            _joint=_Joint(uncertainty),
        )

    def pose_covariance(self) -> NDArray[np.float64] | None:
        """The ``(6, 6)`` covariance over ``[rx ry rz x y z]``: the rotation's
        right-perturbation tangent (rad) and the unit translation's
        components; ``None`` when the uncertainty is unavailable."""
        return self._joint.block(_RELATIVE_POSE_LABELS, _RELATIVE_POSE_LABELS)


@dataclass(frozen=True, slots=True)
class ObjectPose:
    """The pose of a known object and its uncertainty."""

    rig_se3_object: Matrix4
    """``(4, 4)`` pose of the object in the rig frame (object to rig)."""
    inliers: list[int]
    """Indices into the observations of those the pose uses."""
    uncertainty: ParameterUncertainty
    """Covariance over the right-perturbation tangent with the labels
    ``object.rx``, ``.ry``, ``.rz`` (rad) and ``object.tx``, ``.ty``, ``.tz``
    (m); or the reason it is unavailable."""
    reprojection_rms_px: float
    """RMS reprojection error over the inliers, pixels."""
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ObjectPose:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        return cls(
            rig_se3_object=_pose_matrix(payload["rig_se3_object"]),
            inliers=[int(i) for i in payload["inliers"]],
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            reprojection_rms_px=float(payload["reprojection_rms_px"]),
            _joint=_Joint(uncertainty),
        )

    def pose_covariance(self) -> NDArray[np.float64] | None:
        """The ``(6, 6)`` covariance over ``[rx ry rz tx ty tz]`` (rad, m);
        ``None`` when the uncertainty is unavailable."""
        axes = [f"object.{a}" for a in ("rx", "ry", "rz", "tx", "ty", "tz")]
        return self._joint.block(axes, axes)


@dataclass(frozen=True, slots=True)
class BundleAdjustment:
    """Capture poses and points refined together, with their joint covariance."""

    rig_se3_world: NDArray[np.float64]
    """``(K, 4, 4)`` refined capture poses; capture 0 is the fixed gauge."""
    points: NDArray[np.float64]
    """``(N, 3)`` refined points, metres."""
    cameras: list[Camera]
    """The rig's cameras after the solve (refined under ``calibration="prior"``)."""
    cam_se3_rig: NDArray[np.float64]
    """``(C, 4, 4)`` camera poses in the rig after the solve."""
    uncertainty: ParameterUncertainty
    """Joint covariance of the refined captures (``capture{k}.rx`` ... ``.tz``)
    and points (``p{i}.x`` ... ``.z``), in the stated gauge."""
    final_cost: float
    num_iters: int
    reprojection_rms_px: float
    _joint: _Joint = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BundleAdjustment:
        uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
        points = np.asarray(payload["points"], dtype=np.float64).reshape(-1, 3)
        return cls(
            rig_se3_world=_poses_array(payload["rig_se3_world"]),
            points=points,
            cameras=[Camera.from_payload(c) for c in payload["cameras"]],
            cam_se3_rig=_poses_array(payload["cam_se3_rig"]),
            uncertainty=ParameterUncertainty.from_payload(uncertainty),
            final_cost=float(payload["final_cost"]),
            num_iters=int(payload["num_iters"]),
            reprojection_rms_px=float(payload["reprojection_rms_px"]),
            _joint=_Joint(uncertainty),
        )

    def point_covariance(self, i: int) -> NDArray[np.float64] | None:
        """Point ``i``'s ``(3, 3)`` covariance, m²; ``None`` when unavailable."""
        return self._joint.block(_axes(i), _axes(i))

    def pose_covariance(self, k: int) -> NDArray[np.float64] | None:
        """Capture ``k``'s ``(6, 6)`` covariance over ``[rx ry rz tx ty tz]``
        (rad, m); ``None`` when unavailable (capture 0 is the gauge and has none)."""
        axes = [f"capture{k}.{a}" for a in ("rx", "ry", "rz", "tx", "ty", "tz")]
        return self._joint.block(axes, axes)


# ─── Two-view geometry ────────────────────────────────────────────────────────


def _matches(matches: Sequence[PixelMatch]) -> list[dict[str, Any]]:
    return [m.to_payload() for m in matches]


def estimate_essential(
    result: CalibrationResult | Mapping[str, Any],
    cameras: tuple[int, int],
    matches: Sequence[PixelMatch],
    ransac: RansacOptions,
    *,
    progress: ProgressCallback | None = None,
) -> EssentialEstimate:
    """Estimate the essential matrix of two rig cameras from matched pixels with
    RANSAC; ``ransac.thresh`` is in σ of the stated noise. The consensus set is
    refined to the least squares of the weighted Sampson error, and the result
    carries the covariance of ``E``.

    ``progress``, when given, is called with each progress report (a
    :data:`~vision_calibration.types.Progress` mapping) as the refinement
    iterates; raising from it stops the call and the exception propagates.
    Ctrl-C stops it too, with or without a callable.

    Raises
    ------
    ValueError
        For a camera index out of range, fewer than 5 matches or a pixel with no
        finite viewing ray.
    RuntimeError
        If no model reaches ``ransac.min_inliers``.
    """
    payload = _native.mvg_estimate_essential(
        _export_of(result), list(cameras), _matches(matches), ransac.to_payload(), progress
    )
    return EssentialEstimate.from_payload(cast(Mapping[str, Any], payload))


def estimate_homography(
    correspondences: ArrayLike, ransac: RansacOptions
) -> HomographyEstimate:
    """Estimate the homography between two views of a plane from plain pixel
    correspondences with RANSAC (4-point DLT, symmetric transfer error).

    ``correspondences`` is an ``(N, 2, 2)`` array, or a list of
    ``((x1, y1), (x2, y2))``; ``ransac.thresh`` is in pixels. It needs no
    calibration.

    Raises
    ------
    ValueError
        For fewer than 4 correspondences, a wrong shape or an invalid threshold.
    RuntimeError
        If no model reaches ``ransac.min_inliers``.
    """
    array = np.ascontiguousarray(np.asarray(correspondences, dtype=np.float64))
    if array.ndim != 3 or array.shape[1:] != (2, 2):
        raise ValueError(f"correspondences must have shape (N, 2, 2), got {array.shape}")
    payload = _native.mvg_estimate_homography(array, ransac.to_payload())
    return HomographyEstimate.from_payload(cast(Mapping[str, Any], payload))


def recover_relative_pose(
    result: CalibrationResult | Mapping[str, Any],
    cameras: tuple[int, int],
    matches: Sequence[PixelMatch],
) -> RelativePose:
    """The relative pose of two rig cameras from at least 5 matched pixels, all
    treated as inliers, with the matched points triangulated.

    Raises
    ------
    ValueError
        As :func:`estimate_essential`.
    RuntimeError
        For a degenerate configuration (no candidate pose with the points ahead
        of both cameras).
    """
    payload = _native.mvg_recover_relative_pose(
        _export_of(result), list(cameras), _matches(matches)
    )
    return RelativePose.from_payload(cast(Mapping[str, Any], payload))


def recover_relative_pose_robust(
    result: CalibrationResult | Mapping[str, Any],
    cameras: tuple[int, int],
    matches: Sequence[PixelMatch],
    ransac: RansacOptions,
    *,
    progress: ProgressCallback | None = None,
) -> RobustRelativePose:
    """The relative pose of two rig cameras with RANSAC over the matches, refined
    on the inliers, with its covariance in the gauge ``|t| = 1``.
    ``progress`` as for :func:`estimate_essential`.

    Raises
    ------
    ValueError
        As :func:`estimate_essential`.
    RuntimeError
        If no consensus is found or the inliers are degenerate.
    """
    payload = _native.mvg_recover_relative_pose_robust(
        _export_of(result), list(cameras), _matches(matches), ransac.to_payload(), progress
    )
    return RobustRelativePose.from_payload(cast(Mapping[str, Any], payload))


# ─── Measurement ──────────────────────────────────────────────────────────────


def triangulate(
    result: CalibrationResult | Mapping[str, Any],
    rig_se3_world: ArrayLike,
    tracks: Sequence[Sequence[RigObservation]],
    *,
    progress: ProgressCallback | None = None,
) -> Triangulation:
    """Triangulate each track (the observations of one point) in the world
    frame, refine the points and give their joint covariance.

    ``rig_se3_world[k]`` is the ``(4, 4)`` pose of the rig at capture ``k``
    (world to rig frame), taken as exact. For one stereo capture pass
    ``np.eye(4)``: the points come out in the rig frame. The covariance has the
    observations' noise and the calibration's share, so the points are
    correlated (:meth:`Triangulation.cross_covariance`); it is dense over the
    ``3N`` coordinates, so the cost grows as ``N³``.

    ``progress``, when given, is called with each progress report (a
    :data:`~vision_calibration.types.Progress` mapping) as the refinement
    iterates; raising from it stops the call and the exception propagates.
    Ctrl-C stops it too, with or without a callable.

    Raises
    ------
    ValueError
        For a track with fewer than 2 observations or an index out of range
        (the message names the track).
    RuntimeError
        For a degenerate (zero-parallax) track.
    """
    payload = _native.mvg_triangulate(
        _export_of(result),
        _poses(rig_se3_world, "rig_se3_world"),
        [[o.to_payload() for o in track] for track in tracks],
        progress,
    )
    return Triangulation.from_payload(cast(Mapping[str, Any], payload))


def estimate_pose(
    result: CalibrationResult | Mapping[str, Any],
    observations: Sequence[PoseObservation],
    *,
    ransac: RansacOptions | None = None,
    loss: RobustLoss | None = None,
    progress: ProgressCallback | None = None,
) -> ObjectPose:
    """The pose of a known object in the rig frame from its pixels in one or
    several rig cameras, with the pose's covariance.

    With ``ransac`` the start comes from P3P inside a RANSAC whose residual is
    the reprojection distance in σ of the stated noise (``thresh=3.0`` is the
    usual choice). ``loss`` is a robust loss of the refinement, with its scale
    in σ (see :func:`~vision_calibration.robust_huber`). ``progress`` is as
    for :func:`triangulate`.

    Raises
    ------
    ValueError
        For too few observations or a camera index out of range.
    RuntimeError
        If RANSAC finds no consensus or the configuration is degenerate.
    """
    payload = _native.mvg_estimate_pose(
        _export_of(result),
        [o.to_payload() for o in observations],
        None if ransac is None else ransac.to_payload(),
        loss,
        progress,
    )
    return ObjectPose.from_payload(cast(Mapping[str, Any], payload))


def bundle_adjust(
    result: CalibrationResult | Mapping[str, Any],
    rig_se3_world: ArrayLike,
    points: ArrayLike,
    tracks: Sequence[Sequence[RigObservation]],
    *,
    scale: str | tuple[str, tuple[int, int], float, float],
    calibration: str = "consider",
    loss: RobustLoss | None = None,
    max_iters: int = 100,
    progress: ProgressCallback | None = None,
) -> BundleAdjustment:
    """Refine capture poses and points together on the reprojection error.

    Capture 0 is fixed, so the world frame is capture 0's rig frame. The scale
    has no default: ``"rig"`` takes it from the rig's extrinsics (the baseline
    between two cameras that see the same capture); ``("distance", (i, j),
    metres, sigma)`` from a stated distance between points ``i`` and ``j`` and
    its σ. ``points[i]`` is the start of track ``i``.

    ``calibration`` is how the calibration's uncertainty enters: ``"consider"``
    keeps the cameras at their values and carries their covariance into the
    result's; ``"prior"`` refines them too, held by their own covariance.

    ``progress`` is as for :func:`triangulate`: it receives each solver
    iteration, and raising from it stops the adjustment.

    Raises
    ------
    ValueError
        For a scale that cannot be applied (a rig with one camera under
        ``"rig"``), an index out of range or ``"prior"`` with a calibration
        that has no covariance.
    """
    if isinstance(scale, str):
        if scale != "rig":
            raise ValueError(f'scale must be "rig" or ("distance", (i, j), metres, sigma), got {scale!r}')
        scale_payload: Any = "Rig"
    else:
        kind, pair, metres, sigma = scale
        if kind != "distance":
            raise ValueError(f'scale must be "rig" or ("distance", (i, j), metres, sigma), got {scale!r}')
        scale_payload = {
            "Distance": {
                "points": [int(pair[0]), int(pair[1])],
                "distance": float(metres),
                "sigma": float(sigma),
            }
        }
    payload = _native.mvg_bundle_adjust(
        _export_of(result),
        _poses(rig_se3_world, "rig_se3_world"),
        _rows(points, 3, "points"),
        [[o.to_payload() for o in track] for track in tracks],
        scale_payload,
        calibration,
        loss,
        max_iters,
        progress,
    )
    return BundleAdjustment.from_payload(cast(Mapping[str, Any], payload))


def triangulate_laser(
    result: CalibrationResult | Mapping[str, Any],
    observations: Sequence[LaserObservation],
) -> LaserTriangulation:
    """The 3D points of laser-line pixels on their cameras' laser planes, in the
    rig frame, with the covariance of the pixels, the cameras, the planes and
    the extrinsics.

    Raises
    ------
    ValueError
        For a camera index out of range or a camera with no laser plane in the
        calibration.
    RuntimeError
        If a viewing ray is parallel to the plane or meets it behind the camera.
    """
    payload = _native.mvg_triangulate_laser(
        _export_of(result), [o.to_payload() for o in observations]
    )
    return LaserTriangulation.from_payload(cast(Mapping[str, Any], payload))


# ─── Metrology ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Scalar:
    """A measured scalar and its standard deviation, in the measurement's unit."""

    value: float
    sigma: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Scalar:
        return cls(value=float(payload["value"]), sigma=float(payload["sigma"]))


@dataclass(frozen=True, slots=True)
class PlaneFit:
    """A plane ``n · x = d`` (unit ``n``, ``d`` in metres) fitted to points,
    with its covariance."""

    normal: NDArray[np.float64]
    """``(3,)`` unit normal, oriented so that ``distance >= 0``."""
    distance: float
    """The plane's distance from the origin along ``normal``, metres."""
    covariance: NDArray[np.float64]
    """``(3, 3)`` covariance of ``[α, β, d]``: the normal's rotation along the
    two ``tangent_basis`` vectors (rad) and the distance (m). It comes from the
    points' joint covariance, not from how well the points fit the plane."""
    tangent_basis: NDArray[np.float64]
    """``(2, 3)`` unit vectors perpendicular to ``normal`` that ``covariance``
    is expressed in."""
    rms: float
    """RMS orthogonal distance of the points to the plane, metres: a planarity
    diagnostic that is not part of ``covariance``."""
    indices: list[int]
    """The points the plane was fitted to, in the order given."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlaneFit:
        return cls(
            normal=_vector(payload["normal"]),
            distance=float(payload["distance"]),
            covariance=_matrix(payload["covariance"], 3, 3),
            tangent_basis=np.asarray(payload["tangent_basis"], dtype=np.float64).reshape(2, 3),
            rms=float(payload["rms"]),
            indices=[int(i) for i in payload["indices"]],
        )


@dataclass(frozen=True, slots=True)
class PointSet:
    """Points and the joint covariance of all their coordinates, the input of
    the measurements.

    Points measured with one calibration share its uncertainty, so their errors
    are correlated and a length between two of them is more (or less) certain
    than independent errors would say. Every σ uses the full joint covariance,
    so all the points of a measurement must come from one call that produced
    them: build the set from that call's result.
    """

    points: NDArray[np.float64]
    """``(N, 3)`` points, metres."""
    covariance: NDArray[np.float64]
    """``(3N, 3N)`` covariance of the coordinates ``x0 y0 z0 x1 ...``, m²."""

    def __post_init__(self) -> None:
        points = _rows(self.points, 3, "points")
        covariance = np.ascontiguousarray(np.asarray(self.covariance, dtype=np.float64))
        n = 3 * points.shape[0]
        if covariance.shape != (n, n):
            raise ValueError(
                f"{points.shape[0]} points need a ({n}, {n}) covariance, got {covariance.shape}"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "covariance", covariance)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @classmethod
    def _from_joint(
        cls, points: NDArray[np.float64], joint: _Joint, uncertainty: ParameterUncertainty
    ) -> PointSet:
        if joint.covariance is None:
            raise ValueError(
                f"point 0 has no covariance: the uncertainty is unavailable ({uncertainty.reason})"
            )
        n = points.shape[0]
        labels = [label for i in range(n) for label in _axes(i)]
        block = joint.block(labels, labels)
        if block is None:
            missing = next(label for label in labels if label not in joint.index)
            raise ValueError(f"point {missing.split('.')[0][1:]} has no covariance ({missing} is not in it)")
        return cls(points, block)

    @classmethod
    def from_triangulation(cls, triangulation: Triangulation) -> PointSet:
        """The set of a :func:`triangulate` result's points.

        Raises
        ------
        ValueError
            If the triangulation has no covariance.
        """
        points = np.array([p.point for p in triangulation.points]).reshape(-1, 3)
        return cls._from_joint(points, triangulation._joint, triangulation.uncertainty)

    @classmethod
    def from_laser_triangulation(cls, laser: LaserTriangulation) -> PointSet:
        """The set of a :func:`triangulate_laser` result's points."""
        points = np.array([p.point for p in laser.points]).reshape(-1, 3)
        return cls._from_joint(points, laser._joint, laser.uncertainty)

    @classmethod
    def from_bundle_adjustment(cls, adjustment: BundleAdjustment) -> PointSet:
        """The set of a :func:`bundle_adjust` result's points."""
        return cls._from_joint(adjustment.points, adjustment._joint, adjustment.uncertainty)


def distance(points: PointSet, a: int, b: int) -> Scalar:
    """The distance between points ``a`` and ``b``, metres, with its σ.

    Raises
    ------
    ValueError
        For an index out of range or two coincident indices.
    """
    return Scalar.from_payload(
        cast(Mapping[str, Any], _native.mvg_distance(points.points, points.covariance, a, b))
    )


def angle(points: PointSet, a: int, vertex: int, b: int) -> Scalar:
    """The angle at ``vertex`` between the segments to ``a`` and ``b``, radians,
    with its σ.

    Raises
    ------
    ValueError
        For an index out of range or a segment of zero length.
    """
    return Scalar.from_payload(
        cast(
            Mapping[str, Any],
            _native.mvg_angle(points.points, points.covariance, a, vertex, b),
        )
    )


def fit_plane(points: PointSet, indices: Sequence[int]) -> PlaneFit:
    """Fit a plane to the points ``indices`` (at least 3), with its covariance.

    Raises
    ------
    ValueError
        For fewer than 3 points, an index out of range or collinear points.
    """
    return PlaneFit.from_payload(
        cast(
            Mapping[str, Any],
            _native.mvg_fit_plane(points.points, points.covariance, [int(i) for i in indices]),
        )
    )


def point_to_plane(points: PointSet, plane: PlaneFit, p: int) -> Scalar:
    """The signed distance of point ``p`` from the plane fitted to
    ``plane.indices``, metres, with its σ. Both come from the same joint
    covariance, so a point that helped fit the plane is handled exactly.

    Raises
    ------
    ValueError
        For an index out of range or collinear plane points.
    """
    return Scalar.from_payload(
        cast(
            Mapping[str, Any],
            _native.mvg_point_to_plane(points.points, points.covariance, list(plane.indices), p),
        )
    )


# ─── Stereo ───────────────────────────────────────────────────────────────────


class StereoRectification:
    """The rectification of two rig cameras: virtual cameras with parallel
    optical axes, a common orientation and the same intrinsics, so that
    matching pixels share a row. Build it with :func:`rectify`.

    ``side`` is 0 for the left camera of the pair and 1 for the right one.
    """

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native: Any = native

    @property
    def cameras(self) -> tuple[int, int]:
        """The rig camera indices ``(left, right)``."""
        left, right = self._native.cameras()
        return int(left), int(right)

    @property
    def baseline(self) -> float:
        """The pair's baseline, metres."""
        return float(self._native.baseline())

    @property
    def k_rect(self) -> PinholeIntrinsics:
        """The intrinsics of the rectified cameras."""
        fx, fy, cx, cy = self._native.k_rect()
        return PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def rect_r_cam(self, side: int) -> NDArray[np.float64]:
        """The ``(3, 3)`` rotation from a side's camera frame to the rectified frame."""
        return cast(NDArray[np.float64], self._native.rect_r_cam(side))

    def rectified_pixel(self, side: int, pixel: ArrayLike) -> Pixel | None:
        """Where a raw pixel of a side lands in the rectified image; ``None``
        when its ray does not point ahead of the rectified camera."""
        return cast(Pixel | None, self._native.rectified_pixel(side, _pixel(pixel, "pixel")))

    def raw_pixel(self, side: int, pixel: ArrayLike) -> Pixel | None:
        """The raw pixel of a side that a rectified pixel comes from; ``None``
        when the camera does not project it."""
        return cast(Pixel | None, self._native.raw_pixel(side, _pixel(pixel, "pixel")))

    def rectify_image(
        self, side: int, image: ArrayLike, width: int, height: int
    ) -> NDArray[np.uint8]:
        """Resample a raw ``(H, W)`` uint8 image of a side onto the rectified
        grid of ``width`` by ``height`` pixels. Pixels whose source lies outside
        the raw image are 0. Returns a ``(height, width)`` uint8 array."""
        array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError(
                f"image must be a (H, W) uint8 array, got shape {array.shape} and dtype {array.dtype}"
            )
        return cast(
            NDArray[np.uint8],
            self._native.rectify_image(side, np.ascontiguousarray(array), width, height),
        )

    def disparity_to_depth(self) -> NDArray[np.float64]:
        """The ``(4, 4)`` matrix ``Q`` that maps ``(x, y, disparity, 1)`` of the
        rectified left image to homogeneous coordinates in the left camera's
        frame (the layout of OpenCV's ``reprojectImageTo3D``)."""
        return cast(NDArray[np.float64], self._native.disparity_to_depth())

    def point(self, left: ArrayLike, disparity: float) -> NDArray[np.float64] | None:
        """The ``(3,)`` point in the left camera's frame of a rectified left
        pixel and its disparity; ``None`` for a non-positive disparity."""
        found = self._native.point(_pixel(left, "left"), float(disparity))
        return None if found is None else _vector(found)

    def disparity_covariance(
        self, left: ArrayLike, disparity: float, sigma_d: float
    ) -> NDArray[np.float64] | None:
        """The ``(3, 3)`` covariance, m², of :meth:`point` from a disparity σ
        ``sigma_d`` (pixels); ``None`` when the point is undefined."""
        return cast(
            NDArray[np.float64] | None,
            self._native.disparity_covariance(
                _pixel(left, "left"), float(disparity), float(sigma_d)
            ),
        )

    def triangulate(
        self,
        result: CalibrationResult | Mapping[str, Any],
        matches: Sequence[tuple[ArrayLike, float]],
        pixel_noise: PointCovariance,
        *,
        progress: ProgressCallback | None = None,
    ) -> Triangulation:
        """Triangulate rectified matches ``(left_pixel, disparity)`` through the
        raw cameras, with the calibration's covariance. ``result`` must be the
        calibration this rectification was built from. ``progress`` is as for
        :func:`triangulate`.

        Raises
        ------
        ValueError
            If ``result`` is another calibration or a match has no raw pixel.
        """
        payload = self._native.triangulate(
            _export_of(result),
            [{"left": _pixel(left, "left"), "disparity": float(d)} for left, d in matches],
            pixel_noise.to_payload(),
            progress,
        )
        return Triangulation.from_payload(cast(Mapping[str, Any], payload))

    def __repr__(self) -> str:
        return f"StereoRectification(cameras={self.cameras}, baseline={self.baseline:.6g})"


def rectify(
    result: CalibrationResult | Mapping[str, Any],
    cameras: tuple[int, int] = (0, 1),
    *,
    k_rect: PinholeIntrinsics | None = None,
    scale: float = 1.0,
) -> StereoRectification:
    """Rectify two cameras of a calibration, for any camera model.

    The rectified cameras share ``k_rect``; by default the mean of the two
    cameras' intrinsics times ``scale``.

    Raises
    ------
    ValueError
        For two equal or out-of-range camera indices, or a non-positive ``scale``
        or focal length.
    RuntimeError
        If the optical centres coincide.
    """
    native = _native.mvg_rectify(
        _export_of(result),
        list(cameras),
        None if k_rect is None else k_rect.to_payload(),
        float(scale),
    )
    return StereoRectification(native)


@dataclass(frozen=True, slots=True)
class BlockMatchOptions:
    """Settings of :func:`match_block`."""

    min_disparity: int = 0
    """Smallest disparity searched, pixels."""
    num_disparities: int = 64
    """Number of disparity levels searched from ``min_disparity``."""
    block_size: int = 7
    """Side of the square matching window, pixels; odd."""
    min_correlation: float = 0.5
    """Reject a pixel whose best ZNCC is below this, in ``[-1, 1]``."""
    uniqueness_ratio: float = 0.03
    """Reject a pixel unless its best ZNCC beats the best non-adjacent
    alternative by this margin."""
    lr_consistency: bool = True
    """Left-right consistency check (occlusion and mismatch rejection)."""
    lr_max_diff: float = 1.0
    """Largest ``|d_left - d_right|`` the consistency check tolerates, pixels."""
    subpixel: bool = True
    """Parabolic sub-pixel refinement of the correlation peak."""
    semi_global: bool = False
    """Semi-global aggregation of the matching cost along 8 paths, which
    propagates disparity into low-texture regions; it bypasses
    ``min_correlation`` and ``uniqueness_ratio``."""
    sgm_p1: float = 0.1
    """Semi-global penalty for a one-pixel disparity step."""
    sgm_p2: float = 1.5
    """Semi-global penalty for a larger step; at least ``sgm_p1``."""

    def to_payload(self) -> dict[str, Any]:
        return {
            "min_disparity": int(self.min_disparity),
            "num_disparities": int(self.num_disparities),
            "block_size": int(self.block_size),
            "min_correlation": float(self.min_correlation),
            "uniqueness_ratio": float(self.uniqueness_ratio),
            "lr_consistency": bool(self.lr_consistency),
            "lr_max_diff": float(self.lr_max_diff),
            "subpixel": bool(self.subpixel),
            "semi_global": bool(self.semi_global),
            "sgm_p1": float(self.sgm_p1),
            "sgm_p2": float(self.sgm_p2),
        }


def match_block(
    left: ArrayLike, right: ArrayLike, options: BlockMatchOptions | None = None
) -> NDArray[np.float32]:
    """Dense disparity of a rectified pair of ``(H, W)`` uint8 images by
    block matching on zero-mean normalized cross-correlation.

    Returns a ``(H, W)`` float32 disparity map of the left image, in pixels,
    with ``NaN`` where no valid match was found.

    Raises
    ------
    ValueError
        If the images differ in size or are smaller than the window, or the
        options are out of range (an even ``block_size``, no disparities).
    """
    images = []
    for name, image in (("left", left), ("right", right)):
        array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError(
                f"{name} must be a (H, W) uint8 array, got shape {array.shape} and dtype {array.dtype}"
            )
        images.append(np.ascontiguousarray(array))
    payload = (options or BlockMatchOptions()).to_payload()
    return cast(NDArray[np.float32], _native.mvg_match_block(images[0], images[1], payload))
