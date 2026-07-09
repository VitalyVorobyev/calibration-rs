"""Shared synthetic fixtures for the vision_calibration Python test suite.

Not a test module (the ``test_*.py`` discovery pattern skips it). ``unittest
discover`` puts the tests directory on ``sys.path``, so sibling ``test_*.py``
modules can ``import _fixtures``.
"""

from __future__ import annotations

import math

import vision_calibration as vc


def euler_rot_xyz(ax: float, ay: float, az: float) -> list[list[float]]:
    """Rotation matrix ``Rz(az) · Ry(ay) · Rx(ax)`` from XYZ Euler angles."""
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def transform_point(R: list[list[float]], t: list[float], p: list[float]) -> list[float]:
    """Apply ``R @ p + t``."""
    return [sum(R[i][k] * p[k] for k in range(3)) + t[i] for i in range(3)]


def planar_calibration_dataset(n_views: int = 10) -> vc.PlanarDataset:
    """Deterministic planar-board dataset (fx=800, fy=780, cx=640, cy=360).

    An 8x6 grid at 0.04 m spacing viewed from ``n_views`` poses along a fixed
    rotation/translation ramp. Noise-free, so intrinsics solvers recover the
    ground truth to numerical precision.
    """
    board = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]
    views: list[vc.PlanarView] = []
    for i in range(n_views):
        R = euler_rot_xyz(0.02 * i, -0.18 + 0.04 * i, -0.20 + 0.06 * i)
        t = [-0.04 + 0.02 * i, 0.02 - 0.01 * i, 0.55 + 0.04 * i]
        p3, p2 = [], []
        for pt in board:
            pc = transform_point(R, t, list(pt))
            if pc[2] <= 0.0:
                continue
            p3.append(pt)
            p2.append((640.0 + 800.0 * pc[0] / pc[2], 360.0 + 780.0 * pc[1] / pc[2]))
        views.append(vc.PlanarView(observation=vc.Observation(points_3d=p3, points_2d=p2)))
    return vc.PlanarDataset(views=views)
