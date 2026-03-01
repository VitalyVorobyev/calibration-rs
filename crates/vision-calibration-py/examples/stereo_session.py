"""Stereo rig extrinsics calibration from real stereo images.

Python counterpart of:
`crates/vision-calibration/examples/stereo_session.rs`
"""

from __future__ import annotations

import re
from pathlib import Path

import calib_targets
import numpy as np
from PIL import Image
import vision_calibration as vc

from _common import baseline_m, repo_root

BOARD_ROWS = 7
BOARD_COLS = 11
SQUARE_SIZE_M = 0.03


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def list_pair_indices(left_dir: Path, right_dir: Path) -> list[int]:
    left_pat = re.compile(r"^Im_L_(\d+)\.png$")
    right_pat = re.compile(r"^Im_R_(\d+)\.png$")
    left = set()
    right = set()
    for path in left_dir.glob("*.png"):
        m = left_pat.match(path.name)
        if m:
            left.add(int(m.group(1)))
    for path in right_dir.glob("*.png"):
        m = right_pat.match(path.name)
        if m:
            right.add(int(m.group(1)))
    return sorted(left.intersection(right))


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


def load_dataset() -> vc.RigExtrinsicsDataset:
    base = repo_root() / "data/stereo/imgs"
    left_dir = base / "leftcamera"
    right_dir = base / "rightcamera"
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"stereo dataset not found: {base}")

    views: list[vc.RigExtrinsicsView] = []
    skipped = 0
    left_ok = 0
    right_ok = 0
    for idx in list_pair_indices(left_dir, right_dir):
        left = detect_chessboard_observation(left_dir / f"Im_L_{idx}.png")
        right = detect_chessboard_observation(right_dir / f"Im_R_{idx}.png")
        if left is None and right is None:
            skipped += 1
            continue
        left_ok += int(left is not None)
        right_ok += int(right is not None)
        views.append(vc.RigExtrinsicsView(cameras=[left, right]))

    if len(views) < 3:
        raise RuntimeError(f"need at least 3 usable views, got {len(views)}")
    print(f"Loaded {len(views)} views ({skipped} skipped), usable: left={left_ok}, right={right_ok}")
    return vc.RigExtrinsicsDataset(num_cameras=2, views=views)


def main() -> None:
    print("=== Stereo Rig Calibration Session (Real Images, Python) ===")
    dataset = load_dataset()
    result = vc.run_rig_extrinsics(dataset)
    base_mm = baseline_m(result.cam_se3_rig) * 1000.0

    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Per-camera reprojection: {result.per_cam_reproj_errors}")
    print(f"Estimated baseline: {base_mm:.3f} mm")
    for idx, cam in enumerate(result.cameras):
        k = cam["k"]
        d = cam["dist"]
        print(f"Camera {idx}: fx={k['fx']:.3f}, fy={k['fy']:.3f}, cx={k['cx']:.3f}, cy={k['cy']:.3f}")
        print(f"  Distortion: k1={d['k1']:.6f}, k2={d['k2']:.6f}, p1={d['p1']:.6f}, p2={d['p2']:.6f}")


if __name__ == "__main__":
    main()
