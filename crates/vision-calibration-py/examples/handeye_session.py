"""Single-camera hand-eye calibration with real KUKA dataset images.

Python counterpart of:
`crates/vision-calibration/examples/handeye_session.rs`
"""

from __future__ import annotations

from pathlib import Path

import calib_targets
import numpy as np
from PIL import Image
import vision_calibration as vc

from _common import quat_xyzw_from_rot, repo_root

BOARD_ROWS = 17
BOARD_COLS = 28


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def parse_square_size_m(path: Path) -> float:
    raw = path.read_text(encoding="utf-8").strip().lower()
    if raw.endswith("mm"):
        return float(raw[:-2].strip()) / 1000.0
    if raw.endswith("m"):
        return float(raw[:-1].strip())
    value = float(raw)
    return value / 1000.0 if value > 1.0 else value


def parse_robot_poses(path: Path) -> list[vc.Pose]:
    poses: list[vc.Pose] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 16:
            raise ValueError(f"expected 16 values per pose line, got {len(values)}")
        r = [
            [values[0], values[1], values[2]],
            [values[4], values[5], values[6]],
            [values[8], values[9], values[10]],
        ]
        t = (values[3], values[7], values[11])
        poses.append(vc.Pose(rotation_xyzw=quat_xyzw_from_rot(r), translation_xyz=t))
    return poses


def detect_observation(path: Path, square_size_m: float) -> vc.Observation | None:
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
        points_3d.append((float(corner.grid.i) * square_size_m, float(corner.grid.j) * square_size_m, 0.0))
        points_2d.append((float(corner.position[0]), float(corner.position[1])))
    if len(points_2d) < 4:
        return None
    return vc.Observation(points_3d=points_3d, points_2d=points_2d)


def load_dataset() -> vc.SingleCamHandeyeDataset:
    data_dir = repo_root() / "data/kuka_1"
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset not found: {data_dir}")

    square_size_m = parse_square_size_m(data_dir / "squaresize.txt")
    robot_poses = parse_robot_poses(data_dir / "RobotPosesVec.txt")
    views: list[vc.SingleCamHandeyeView] = []
    skipped = 0

    for idx, robot_pose in enumerate(robot_poses, start=1):
        img_path = data_dir / f"{idx:02}.png"
        if not img_path.exists():
            skipped += 1
            continue
        obs = detect_observation(img_path, square_size_m)
        if obs is None:
            skipped += 1
            continue
        views.append(vc.SingleCamHandeyeView(observation=obs, base_se3_gripper=robot_pose))

    if len(views) < 3:
        raise RuntimeError(f"need at least 3 usable views, got {len(views)}")
    print(f"Loaded {len(views)} views ({skipped} skipped)")
    return vc.SingleCamHandeyeDataset(views=views)


def main() -> None:
    print("=== Single-Camera Hand-Eye Session (Real Images, Python) ===")
    dataset = load_dataset()
    result = vc.run_single_cam_handeye(dataset)
    cam = result.camera["k"]

    print(f"Hand-eye mode: {result.handeye_mode}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Recovered camera fx={cam['fx']:.3f}, fy={cam['fy']:.3f}, cx={cam['cx']:.3f}, cy={cam['cy']:.3f}")
    if result.gripper_se3_camera is not None:
        t = result.gripper_se3_camera.translation_xyz
        print(f"Recovered gripper->camera translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")


if __name__ == "__main__":
    main()
