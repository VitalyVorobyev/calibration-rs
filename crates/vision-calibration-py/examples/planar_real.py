"""Planar intrinsics calibration from real stereo dataset images.

Python counterpart of:
`crates/vision-calibration/examples/planar_real.rs`
"""

from __future__ import annotations

import re
from pathlib import Path

import calib_targets
import numpy as np
from PIL import Image
import vision_calibration as vc

from _common import repo_root

BOARD_ROWS = 7
BOARD_COLS = 11
SQUARE_SIZE_M = 0.03


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def image_indices(folder: Path, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.png$")
    out: list[int] = []
    for path in folder.glob("*.png"):
        m = pattern.match(path.name)
        if m is not None:
            out.append(int(m.group(1)))
    out.sort()
    return out


def detect_chessboard_observation(path: Path) -> vc.Observation | None:
    image = load_gray(path)
    detection = calib_targets.detect_chessboard(
        image,
        chess_cfg=calib_targets.ChessConfig(),
        params=calib_targets.ChessboardParams(
            expected_rows=BOARD_ROWS,
            expected_cols=BOARD_COLS,
        ),
    )
    if detection is None:
        return None

    points_3d: list[tuple[float, float, float]] = []
    points_2d: list[tuple[float, float]] = []
    for corner in detection.detection.corners:
        if corner.grid is None:
            continue
        points_3d.append((float(corner.grid.i) * SQUARE_SIZE_M, float(corner.grid.j) * SQUARE_SIZE_M, 0.0))
        points_2d.append((float(corner.position[0]), float(corner.position[1])))
    if len(points_2d) < 4:
        return None
    return vc.Observation(points_3d=points_3d, points_2d=points_2d)


def load_dataset() -> vc.PlanarDataset:
    left_dir = repo_root() / "data/stereo/imgs/leftcamera"
    if not left_dir.exists():
        raise FileNotFoundError(f"dataset not found: {left_dir}")

    views: list[vc.PlanarView] = []
    skipped = 0
    for idx in image_indices(left_dir, "Im_L_"):
        obs = detect_chessboard_observation(left_dir / f"Im_L_{idx}.png")
        if obs is None:
            skipped += 1
            continue
        views.append(vc.PlanarView(observation=obs))

    if len(views) < 3:
        raise RuntimeError(f"need at least 3 usable views, got {len(views)}")
    print(f"Loaded {len(views)} views ({skipped} skipped)")
    return vc.PlanarDataset(views=views)


def main() -> None:
    print("=== Planar Intrinsics Calibration (Real Images, Python) ===")
    dataset = load_dataset()
    result = vc.run_planar_intrinsics(dataset)
    k = result.camera["k"]
    d = result.camera["dist"]

    print(f"Final cost: {result.final_cost:.3e}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Intrinsics: fx={k['fx']:.3f}, fy={k['fy']:.3f}, cx={k['cx']:.3f}, cy={k['cy']:.3f}")
    print(f"Distortion: k1={d['k1']:.6f}, k2={d['k2']:.6f}, k3={d['k3']:.6f}, p1={d['p1']:.6f}, p2={d['p2']:.6f}")


if __name__ == "__main__":
    main()
