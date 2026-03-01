"""Stereo rig calibration from real ChArUco images.

Python counterpart of:
`crates/vision-calibration/examples/stereo_charuco_session.rs`
"""

from __future__ import annotations

from pathlib import Path

import calib_targets
import numpy as np
from PIL import Image
import vision_calibration as vc

from _common import baseline_m, repo_root

BOARD_ROWS = 22
BOARD_COLS = 22
BOARD_CELL_SIZE_M = 0.00135
BOARD_MARKER_SIZE_REL = 0.75
BOARD_DICTIONARY = "DICT_4X4_1000"


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def list_pair_suffixes(left_dir: Path, right_dir: Path) -> list[str]:
    left = {p.name[len("Cam1_") :] for p in left_dir.glob("Cam1_*.png")}
    right = {p.name[len("Cam2_") :] for p in right_dir.glob("Cam2_*.png")}
    return sorted(left.intersection(right))


def make_charuco_params() -> calib_targets.CharucoDetectorParams:
    board = calib_targets.CharucoBoardSpec(
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        cell_size=BOARD_CELL_SIZE_M,
        marker_size_rel=BOARD_MARKER_SIZE_REL,
        dictionary=BOARD_DICTIONARY,
        marker_layout=calib_targets.MarkerLayout.OPENCV_CHARUCO,
    )
    return calib_targets.CharucoDetectorParams(
        board=board,
        graph=calib_targets.GridGraphParams(max_spacing_pix=120.0),
    )


def detect_charuco_observation(
    path: Path,
    params: calib_targets.CharucoDetectorParams,
) -> vc.Observation | None:
    image = load_gray(path)
    try:
        detection = calib_targets.detect_charuco(
            image,
            chess_cfg=calib_targets.ChessConfig(),
            params=params,
        )
    except Exception:
        return None

    points_3d: list[tuple[float, float, float]] = []
    points_2d: list[tuple[float, float]] = []
    for corner in detection.detection.corners:
        target = corner.target_position
        if target is None:
            continue
        points_3d.append((float(target[0]), float(target[1]), 0.0))
        points_2d.append((float(corner.position[0]), float(corner.position[1])))
    if len(points_2d) < 4:
        return None
    return vc.Observation(points_3d=points_3d, points_2d=points_2d)


def load_dataset() -> vc.RigExtrinsicsDataset:
    data_dir = repo_root() / "data/stereo_charuco"
    left_dir = data_dir / "cam1"
    right_dir = data_dir / "cam2"
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"stereo_charuco dataset not found: {data_dir}")

    params = make_charuco_params()
    views: list[vc.RigExtrinsicsView] = []
    skipped = 0
    left_ok = 0
    right_ok = 0
    for suffix in list_pair_suffixes(left_dir, right_dir):
        left = detect_charuco_observation(left_dir / f"Cam1_{suffix}", params)
        right = detect_charuco_observation(right_dir / f"Cam2_{suffix}", params)
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
    print("=== Stereo Rig Calibration Session (ChArUco, Python) ===")
    dataset = load_dataset()
    result = vc.run_rig_extrinsics(dataset)

    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Per-camera reprojection: {result.per_cam_reproj_errors}")
    print(f"Estimated baseline: {baseline_m(result.cam_se3_rig) * 1000.0:.3f} mm")


if __name__ == "__main__":
    main()
