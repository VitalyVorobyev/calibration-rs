#!/usr/bin/env python3
"""Verify every facade calibration workflow has a Python binding with Rust's names.

Each ``vision_calibration`` facade workflow module (``planar_intrinsics``,
``rig_handeye_laserline``, …) re-exports a ``run_calibration`` entry point. The
PyO3 crate is expected to register a matching ``run_<module>`` ``#[pyfunction]``.
Without this check, adding a new workflow in Rust silently drifts from the
Python surface.

The script also checks that the Python names are the Rust names:

- every workflow's ``<Name>Input``, ``<Name>Config`` and ``<Name>Export`` (the
  facade module's own names) is a Python class of the same name;
- every Python ``*Config``, ``*Input``, ``*Export``, ``*Estimate``,
  ``*Params`` or ``*Result`` class names a public Rust type, so no Python-only spelling
  survives a Rust rename.

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
PY_INIT = REPO_ROOT / "crates/vision-calibration-py/python/vision_calibration/__init__.py"
CRATES = REPO_ROOT / "crates"

# Python class-name suffixes that must name a Rust type.
RUST_NAMED_SUFFIXES = ("Config", "Input", "Export", "Estimate", "Params", "Result")

# Reuse the single definition of the `wrap_pyfunction!` registration regex
# and of the `__all__` parser.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pyi_coverage import extract_all, extract_rust_pyfunctions  # noqa: E402


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


def extract_facade_workflows(lib_rs: Path) -> dict[str, str]:
    """Facade modules that re-export a workflow ``run_calibration``, with their bodies.

    A workflow module ``pub mod NAME { … }`` re-exports ``run_calibration``
    (unaliased) from its matching ``vision_calibration_pipeline::NAME::`` path.
    The ``vision_calibration_pipeline::NAME::`` guard excludes the ``prelude``
    module, which re-exports ``run_calibration as run_planar_intrinsics`` from a
    *different* submodule.
    """
    text = lib_rs.read_text()
    workflows: dict[str, str] = {}
    for match in re.finditer(r"\bpub mod (\w+)\s*\{", text):
        name = match.group(1)
        body = _balanced_brace_body(text, match.end() - 1)
        if f"vision_calibration_pipeline::{name}::" not in body:
            continue
        if re.search(r"\brun_calibration\b(?!\s+as)", body):
            workflows[name] = body
    return workflows


def camel_case(module: str) -> str:
    """``rig_handeye_laserline`` -> ``RigHandeyeLaserline``."""
    return "".join(part.capitalize() for part in module.split("_"))


def rust_public_types() -> set[str]:
    """Every ``pub struct|enum|type`` name declared in a workspace crate's sources."""
    names: set[str] = set()
    pattern = re.compile(r"^\s*pub (?:struct|enum|type) (\w+)", re.M)
    for path in CRATES.glob("*/src/**/*.rs"):
        names.update(pattern.findall(path.read_text()))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    args = parser.parse_args()

    workflows = extract_facade_workflows(FACADE_LIB)
    pyfunctions = extract_rust_pyfunctions(BINDING_LIB)
    python_names = extract_all(PY_INIT)
    rust_types = rust_public_types()

    problems: list[str] = []
    for name in sorted(workflows):
        if f"run_{name}" not in pyfunctions:
            problems.append(f"{name}: no Python binding (expected pyfunction run_{name})")
        stem = camel_case(name)
        for suffix in ("Input", "Config", "Export"):
            rust_name = f"{stem}{suffix}"
            if not re.search(rf"\b{rust_name}\b", workflows[name]):
                problems.append(f"{name}: facade module does not export {rust_name}")
            elif rust_name not in python_names:
                problems.append(f"{name}: Python does not export {rust_name}")

    for py_name in sorted(python_names):
        if py_name.endswith(RUST_NAMED_SUFFIXES) and py_name not in rust_types:
            problems.append(f"Python {py_name} names no Rust type")

    if problems:
        print("Binding parity drift:")
        for problem in problems:
            print(f"  - {problem}")
        return 1 if args.check else 0

    print(
        f"binding parity OK ({len(workflows)} facade workflows bound as run_<workflow>, "
        f"Python names match Rust)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
