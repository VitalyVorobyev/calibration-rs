#! /usr/bin/env python
import sys

import numpy as np
from PIL import Image

import calib_targets


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: detect_chessboard.py <image_path>")
        return
    
    params = {
        'cell_size': 30,
        'min_corner_strength': 0.2,
        'min_corners': 10,
        'completeness_threshold': 0.7,
        'use_orientation_clustering': True,
        'expected_rows': 7,
        'expected_cols': 11,
        "orientation_clustering_params": {
            "num_bins": 90,
            "max_iters": 10,
            "peak_min_separation_deg": 10.0,
            "outlier_threshold_deg": 30.0,
            "min_peak_weight_fraction": 0.05,
            "use_weights": True,
        },
    }

    image = load_gray(sys.argv[1])
    result = calib_targets.detect_chessboard(image, params=params)

    if result is None:
        print("No chessboard detected")
        return

    detection = result.get("detection", {})
    corners = detection.get("corners", [])
    print(f"corners: {len(corners)}")
    print(f"inliers: {len(result.get('inliers', []))}")

    print(corners[1])
    i = corners[1]['grid']['i']
    j = corners[1]['grid']['i']
    pos = corners[1]['position']
    print(i, j, pos)

    # ground truth
    data = np.load(f'stereo/out/parameters.npz')
    print(list(data.keys()))

    print(data['BoardSize'], data['SquareSize'])
    print(data['Transformation'])
    print(data['R_Extrinsics'])
    print(data['L_Extrinsics'])
    print(data['Essential'])
    print(data['Fundamental'])


if __name__ == "__main__":
    main()
