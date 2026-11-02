"""Calibrate a folder of your own images.

Python counterpart of:
`crates/vision-calibration/examples/calibrate_folder.rs`

The first run drafts `dataset.json` in the folder and stops: open it, fill in
what it lists under `_unresolved` (at least the board), delete that list and
run again. The second run detects the board, calibrates and prints each
camera's parameters with their uncertainty and a quality verdict.

    python calibrate_folder.py <folder> <width> <height>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import vision_calibration as vc


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: calibrate_folder.py <folder> <image width> <image height>")
    folder = Path(sys.argv[1])
    image_size = (int(sys.argv[2]), int(sys.argv[3]))

    # 1. Draft the manifest. The sniffer reads file and folder names only and
    #    lists what it cannot know, such as the board, under `_unresolved`.
    manifest = folder / "dataset.json"
    if not manifest.exists():
        draft = vc.sniff_folder(folder)
        manifest.write_text(json.dumps(draft, indent=2))
        print(f"wrote {manifest}")
        print(f"fill in {draft['_unresolved']}, delete `_unresolved`, run again")
        return

    # 2. Calibrate: detect the board in every image (cached for the next run),
    #    then run the workflow the manifest's topology names.
    cache = vc.FsDetectionCache(folder / ".detections")
    result = vc.calibrate_dataset(manifest, cache=cache)

    # 3. Each camera and the uncertainty of its parameters. With several
    #    cameras, a camera's covariance labels carry its index (`cam1.fx`).
    cameras = result.cameras if hasattr(result, "cameras") else [result.camera]
    u = result.uncertainty
    for i, camera in enumerate(cameras):
        k = camera.intrinsics
        print(f"camera {i}: fx={k.fx:.2f} fy={k.fy:.2f} cx={k.cx:.2f} cy={k.cy:.2f}")
        if u.available:
            for name in ("fx", "fy", "cx", "cy"):
                label = f"cam{i}.{name}" if len(cameras) > 1 else name
                print(f"  σ({name}) = {u.std_dev(label):.3f} px")
    if not u.available:
        print("no covariance:", u.reason)

    # 4. The verdict: pass, warn or fail, and why.
    report = vc.assess(result, image_sizes=[image_size] * len(cameras))
    print("verdict:", report.verdict)
    for check in report.checks:
        if check.status != "pass":
            print(f"  {check.kind} {check.status}: {check.message}")

    # 5. The export is the result to keep: JSON with a published schema.
    out = folder / "calibration.json"
    out.write_text(json.dumps(result.export, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
