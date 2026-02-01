#! /usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

import calib_targets


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def parse_index(path: Path) -> int:
    match = re.search(r"_([0-9]+)\.png$", path.name)
    if not match:
        raise ValueError(f"could not parse index from {path}")
    return int(match.group(1))


def detect_corners(path: Path, params: dict, expected: int) -> list[list[float]]:
    image = load_gray(path)
    result = calib_targets.detect_chessboard(image, params=params)
    if result is None:
        raise RuntimeError(f"no chessboard detected in {path}")

    detection = result.get("detection", {})
    corners = detection.get("corners", [])
    if len(corners) < expected:
        raise RuntimeError(
            f"incomplete detection in {path}: {len(corners)} < {expected}"
        )

    out = []
    for corner in corners:
        grid = corner.get("grid", {})
        i = int(grid.get("i"))
        j = int(grid.get("j"))
        x, y = corner.get("position", [0.0, 0.0])
        out.append([i, j, float(x), float(y)])

    out.sort(key=lambda c: (c[0], c[1]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chessboard corners for linear tests.")
    parser.add_argument(
        "--stereo-root",
        type=Path,
        default=Path("stereo"),
        help="Path to stereo dataset root.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("stereo/out/parameters.npz"),
        help="Path to ground-truth parameters.npz.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("crates/vision-calibration-linear/tests/data/stereo_linear.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    data = np.load(args.params)
    board_size = data["BoardSize"].astype(int).tolist()
    cols, rows = int(board_size[0]), int(board_size[1])
    square_size = float(data["SquareSize"])

    params = {
        "cell_size": int(square_size),
        "min_corner_strength": 0.2,
        "min_corners": 10,
        "completeness_threshold": 0.7,
        "use_orientation_clustering": True,
        "expected_rows": rows,
        "expected_cols": cols,
        "orientation_clustering_params": {
            "num_bins": 90,
            "max_iters": 10,
            "peak_min_separation_deg": 10.0,
            "outlier_threshold_deg": 30.0,
            "min_peak_weight_fraction": 0.05,
            "use_weights": True,
        },
    }

    left_dir = args.stereo_root / "imgs" / "leftcamera"
    right_dir = args.stereo_root / "imgs" / "rightcamera"
    left_imgs = sorted(left_dir.glob("*.png"), key=parse_index)
    right_imgs = sorted(right_dir.glob("*.png"), key=parse_index)

    if len(left_imgs) != len(right_imgs):
        raise RuntimeError("left/right image counts do not match")

    expected_corners = rows * cols
    views = []

    for left_path, right_path in zip(left_imgs, right_imgs, strict=True):
        left_idx = parse_index(left_path)
        right_idx = parse_index(right_path)
        if left_idx != right_idx:
            raise RuntimeError(
                f"mismatched view indices: {left_path} vs {right_path}"
            )

        left_corners = detect_corners(left_path, params, expected_corners)
        right_corners = detect_corners(right_path, params, expected_corners)

        views.append(
            {
                "view_index": left_idx - 1,
                "left": {
                    "image": left_path.name,
                    "corners": left_corners,
                },
                "right": {
                    "image": right_path.name,
                    "corners": right_corners,
                },
            }
        )

    payload = {
        "board": {
            "cols": cols,
            "rows": rows,
            "square_size": square_size,
        },
        "detector_params": params,
        "intrinsics": {
            "left": data["L_Intrinsic"].tolist(),
            "right": data["R_Intrinsic"].tolist(),
        },
        "distortion": {
            "left": data["L_Distortion"].reshape(-1).tolist(),
            "right": data["R_Distortion"].reshape(-1).tolist(),
        },
        "extrinsics": {
            "left": data["L_Extrinsics"].tolist(),
            "right": data["R_Extrinsics"].tolist(),
        },
        "essential": data["Essential"].tolist(),
        "fundamental": data["Fundamental"].tolist(),
        "stereo_transform": data["Transformation"].tolist(),
        "views": views,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out} with {len(views)} views")


if __name__ == "__main__":
    main()
