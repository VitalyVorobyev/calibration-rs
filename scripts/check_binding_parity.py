#!/usr/bin/env python3
"""Verify every facade calibration workflow has a Python binding.

Each ``vision_calibration`` facade workflow module (``planar_intrinsics``,
``rig_handeye_laserline``, …) re-exports a ``run_calibration`` entry point. The
PyO3 crate is expected to register a matching ``run_<module>`` ``#[pyfunction]``.
Without this check, adding a new workflow in Rust silently drifts from the
Python surface (exactly how ``run_rig_handeye_laserline`` was missed before R5).

The complementary stub-coverage guard lives in ``check_pyi_coverage.py``; this
script only checks facade↔binding parity.

Usage:
    python3 scripts/check_binding_parity.py           # show drift, warn only
    python3 scripts/check_binding_parity.py --check    # exit 1 on drift (CI mode)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FACADE_LIB = REPO_ROOT / "crates/vision-calibration/src/lib.rs"
BINDING_LIB = REPO_ROOT / "crates/vision-calibration-py/src/lib.rs"

# Reuse the single definition of the `wrap_pyfunction!` registration regex.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pyi_coverage import extract_rust_pyfunctions  # noqa: E402


def _balanced_brace_body(text: str, open_idx: int) -> str:
    """Return the substring inside the braces starting at ``open_idx`` (a ``{``)."""
    depth = 0
    for i in range(open_idx, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return text[open_idx + 1 :]


def extract_facade_workflows(lib_rs: Path) -> set[str]:
    """Facade modules that re-export a workflow ``run_calibration``.

    A workflow module ``pub mod NAME { … }`` re-exports ``run_calibration``
    (unaliased) from its matching ``vision_calibration_pipeline::NAME::`` path.
    The ``vision_calibration_pipeline::NAME::`` guard excludes the ``prelude``
    module, which re-exports ``run_calibration as run_planar_intrinsics`` from a
    *different* submodule.
    """
    text = lib_rs.read_text()
    workflows: set[str] = set()
    for match in re.finditer(r"\bpub mod (\w+)\s*\{", text):
        name = match.group(1)
        body = _balanced_brace_body(text, match.end() - 1)
        if f"vision_calibration_pipeline::{name}::" not in body:
            continue
        if re.search(r"\brun_calibration\b(?!\s+as)", body):
            workflows.add(name)
    return workflows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    args = parser.parse_args()

    workflows = extract_facade_workflows(FACADE_LIB)
    pyfunctions = extract_rust_pyfunctions(BINDING_LIB)

    missing = sorted(
        name for name in workflows if f"run_{name}" not in pyfunctions
    )

    if missing:
        print("Facade workflow modules with no Python binding:")
        for name in missing:
            print(f"  - {name} (expected pyfunction: run_{name})")
        if args.check:
            return 1
        return 0

    print(
        f"binding parity OK ({len(workflows)} facade workflows, "
        f"all bound as run_<workflow>)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
