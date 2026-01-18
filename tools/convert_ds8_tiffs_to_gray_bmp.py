#! /usr/bin/env python3
"""
Convert all TIFF images under a folder into 8-bit grayscale BMPs.

Defaults are tailored for the DS8 dataset:
  in:  data/DS8/images
  out: data/DS8/bmp_gray

The output preserves the relative directory structure under the input root.
Example:
  data/DS8/images/camera0/image0.tiff
    -> data/DS8/bmp_gray/camera0/image0.bmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TIFF_SUFFIXES = {".tif", ".tiff"}


def iter_tiffs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES:
            paths.append(p)
    paths.sort()
    return paths


def to_gray_8bit(img):
    # Fast-path for common 8-bit grayscale.
    if img.mode == "L":
        return img

    # If we have alpha, drop it before grayscale conversion.
    if img.mode in {"LA", "RGBA"}:
        img = img.convert("RGB")

    # Handle common 16-bit TIFF modes by downshifting to 8-bit.
    if img.mode.startswith("I;16") or img.mode == "I":
        return img.point(lambda x: x >> 8).convert("L")

    return img.convert("L")


def convert_one(src: Path, dst: Path) -> None:
    from PIL import Image

    with Image.open(src) as img:
        gray = to_gray_8bit(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        gray.save(dst, format="BMP")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TIFF images to 8-bit grayscale BMPs."
    )
    parser.add_argument(
        "--in",
        dest="in_root",
        type=Path,
        default=Path("data/DS8/images"),
        help="Input root to search recursively for .tif/.tiff files.",
    )
    parser.add_argument(
        "--out",
        dest="out_root",
        type=Path,
        default=Path("data/DS8/bmp_gray"),
        help="Output root directory (ignored with --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write .bmp next to each source image (keeps input tree unchanged).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .bmp outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        print(
            "Missing dependency: Pillow\n"
            "Install with: python3 -m pip install pillow\n",
            file=sys.stderr,
        )
        return 2

    in_root: Path = args.in_root
    if not in_root.exists():
        print(f"Input path does not exist: {in_root}", file=sys.stderr)
        return 2

    tiffs = iter_tiffs(in_root)
    if not tiffs:
        print(f"No TIFF files found under: {in_root}")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for src in tiffs:
        if args.in_place:
            dst = src.with_suffix(".bmp")
        else:
            rel = src.relative_to(in_root).with_suffix(".bmp")
            dst = args.out_root / rel

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            print(f"{src} -> {dst}")
            converted += 1
            continue

        try:
            convert_one(src, dst)
            converted += 1
        except Exception as exc:  # noqa: BLE001 - CLI tool; show context per-file.
            failed += 1
            print(f"Failed: {src} ({exc})", file=sys.stderr)

    print(f"Converted: {converted} | skipped: {skipped} | failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

