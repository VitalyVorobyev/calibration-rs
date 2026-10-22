"""Synthetic camera images for the manifest-run tests: a chessboard seen by a
pinhole camera, and the stripe a laser plane draws on it.

Not a test module (the ``test_*.py`` discovery pattern skips it). Images are
grayscale ``uint8`` arrays, written as PNG with the standard library only.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Camera: 640x480, no distortion.
WIDTH, HEIGHT = 640, 480
FX, FY, CX, CY = 600.0, 590.0, 320.0, 240.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

# Board: 8x6 squares of 30 mm, so 7x5 inner corners.
SQUARE = 0.03
SQUARES_X, SQUARES_Y = 8, 6
TARGET = {"kind": "chessboard", "rows": 5, "cols": 7, "square_size_m": SQUARE}

# Laser plane in the camera frame, n·p + d = 0: through (0, 0.02, 0.55),
# tilted so it crosses every board roughly along an image row.
_n = np.array([0.0, 0.95, -0.3])
LASER_NORMAL = _n / np.linalg.norm(_n)
LASER_D = -float(LASER_NORMAL @ np.array([0.0, 0.02, 0.55]))


def rotation(ax: float, ay: float, az: float) -> NDArray[np.float64]:
    """``Rz(az) · Ry(ay) · Rx(ax)``."""
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def board_poses(n: int) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """``n`` camera-from-board poses ``(R, t)`` with varied tilts, the board
    centred near the optical axis about 0.55 m away."""
    centre = np.array([SQUARES_X * SQUARE / 2 - SQUARE, SQUARES_Y * SQUARE / 2 - SQUARE, 0.0])
    poses = []
    for i in range(n):
        r = rotation(0.35 * math.sin(1.3 * i + 0.4), 0.35 * math.cos(1.1 * i), 0.15 * math.sin(0.7 * i))
        t = np.array([0.01 * math.sin(i), 0.01 * math.cos(2 * i), 0.52 + 0.015 * i]) - r @ centre
        poses.append((r, t))
    return poses


def _board_points(
    r: NDArray[np.float64], t: NDArray[np.float64], supersample: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """For each (sub)pixel, the board point ``(X, Y)`` its ray hits and the
    camera-frame point."""
    s = supersample
    us = (np.arange(WIDTH * s) + 0.5) / s - 0.5
    vs = (np.arange(HEIGHT * s) + 0.5) / s - 0.5
    u, v = np.meshgrid(us, vs)
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    normal = r[:, 2]
    depth = (normal @ t) / (rays @ normal)
    points = rays * depth[..., None]
    local = (points - t) @ r
    return local[..., 0], local[..., 1], points


def _downsample(image: NDArray[np.float64], s: int) -> NDArray[np.uint8]:
    pooled = image.reshape(HEIGHT, s, WIDTH, s).mean(axis=(1, 3))
    return np.clip(np.rint(pooled), 0, 255).astype(np.uint8)


def render_board(r: NDArray[np.float64], t: NDArray[np.float64], supersample: int = 4) -> NDArray[np.uint8]:
    """The board on a white background, anti-aliased by supersampling."""
    x, y, _ = _board_points(r, t, supersample)
    ix = np.floor(x / SQUARE) + 1
    iy = np.floor(y / SQUARE) + 1
    inside = (ix >= 0) & (ix < SQUARES_X) & (iy >= 0) & (iy < SQUARES_Y)
    black = inside & ((ix + iy) % 2 == 0)
    return _downsample(np.where(black, 20.0, 235.0), supersample)


def render_laser(r: NDArray[np.float64], t: NDArray[np.float64], supersample: int = 4) -> NDArray[np.uint8]:
    """The laser stripe on the board, on a dark background: a Gaussian of
    1 mm across the plane."""
    x, y, points = _board_points(r, t, supersample)
    on_board = (
        (x > -SQUARE) & (x < (SQUARES_X - 1) * SQUARE) & (y > -SQUARE) & (y < (SQUARES_Y - 1) * SQUARE)
    )
    # Rays far off the board meet its plane at huge depths; their stripe
    # value underflows to zero, which is what they should get.
    with np.errstate(over="ignore", invalid="ignore"):
        distance = points @ LASER_NORMAL + LASER_D
        stripe = 240.0 * np.exp(-0.5 * (distance / 0.001) ** 2)
    return _downsample(np.where(on_board, 10.0 + stripe, 10.0), supersample)


def write_png(path: Path, image: NDArray[np.uint8]) -> None:
    """Write a grayscale ``uint8`` image as an 8-bit PNG."""
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class ColumnPeakExtractor:
    """A laser extractor for the tests: per image column, the intensity
    centroid of the pixels above half the column's peak."""

    name = "test_column_peak"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image: NDArray[np.uint8], spec: dict) -> NDArray[np.float64]:
        self.calls += 1
        assert image.dtype == np.uint8 and image.ndim == 2
        assert spec["scan_axis"] == "cols"
        values = image.astype(np.float64) - 10.0
        rows = np.arange(image.shape[0], dtype=np.float64)
        points = []
        for col in range(image.shape[1]):
            column = values[:, col]
            peak = column.max()
            if peak < 100.0:
                continue
            weights = np.where(column > peak / 2, column, 0.0)
            points.append((float(col), float(weights @ rows / weights.sum())))
        return np.array(points, dtype=np.float64).reshape(-1, 2)


def write_laserline_dataset(root: Path, views: int = 6) -> dict:
    """Write a laserline-device dataset (target and laser images) under
    ``root`` and return its manifest payload."""
    for i, (r, t) in enumerate(board_poses(views)):
        write_png(root / f"target_{i:02d}.png", render_board(r, t))
        write_png(root / f"laser_{i:02d}.png", render_laser(r, t))
    return {
        "version": 1,
        "topology": "laserline_device",
        "cameras": [
            {
                "id": "cam0",
                "images": {"kind": "glob", "pattern": "target_*.png"},
                "laser_images": {"kind": "glob", "pattern": "laser_*.png"},
            }
        ],
        "target": TARGET,
        "pose_pairing": {"kind": "by_index"},
    }


def write_planar_dataset(root: Path, views: int = 6) -> dict:
    """Write a planar-intrinsics dataset under ``root`` and return its
    manifest payload."""
    for i, (r, t) in enumerate(board_poses(views)):
        write_png(root / f"view_{i:02d}.png", render_board(r, t))
    return {
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{"id": "cam0", "images": {"kind": "glob", "pattern": "view_*.png"}}],
        "target": TARGET,
    }
