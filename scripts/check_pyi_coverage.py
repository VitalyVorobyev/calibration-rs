#!/usr/bin/env python3
"""Verify vision_calibration typing stubs cover every public Python symbol.

The .pyi file is hand-maintained. Without this check it silently drifts when a
new symbol is added to ``__all__`` or a new ``#[pyfunction]`` is registered.

The payload ``TypedDict`` types are not hand-maintained: ``_generated_types.py``
is rendered from the Rust types' JSON Schemas by ``cargo xtask emit-schemas``.
The check also fails when another module of the package declares a
``TypedDict``, which would be a second, drifting definition of a payload shape.

Usage:
    python3 scripts/check_pyi_coverage.py           # show drift, warn only
    python3 scripts/check_pyi_coverage.py --check   # exit 1 on drift (CI mode)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_PKG = REPO_ROOT / "crates/vision-calibration-py/python/vision_calibration"
RUST_LIB = REPO_ROOT / "crates/vision-calibration-py/src/lib.rs"


def extract_all(init_py: Path) -> set[str]:
    tree = ast.parse(init_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, ast.List):
                return {
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
    return set()


def extract_pyi_symbols(pyi: Path) -> set[str]:
    tree = ast.parse(pyi.read_text())
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    symbols.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.asname or alias.name)
    return symbols


def extract_rust_pyfunctions(lib_rs: Path) -> set[str]:
    text = lib_rs.read_text()
    names: set[str] = set()
    for m in re.finditer(r"m\.add_function\(wrap_pyfunction!\(\s*(\w+)\s*,\s*m\s*\)\?\)\?;", text):
        names.add(m.group(1))
    return names


GENERATED_TYPES = "_generated_types.py"


def hand_written_typed_dicts(pkg: Path) -> list[str]:
    """``TypedDict`` declarations outside the generated module, as ``file:line name``."""
    found: list[str] = []
    for path in sorted([*pkg.glob("*.py"), *pkg.glob("*.pyi")]):
        if path.name == GENERATED_TYPES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "TypedDict" for base in node.bases
            ):
                found.append(f"{path.name}:{node.lineno} {node.name}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TypedDict"
            ):
                found.append(f"{path.name}:{node.lineno} TypedDict(...)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    args = parser.parse_args()

    init_py = PY_PKG / "__init__.py"
    init_pyi = PY_PKG / "__init__.pyi"

    all_syms = extract_all(init_py)
    pyi_syms = extract_pyi_symbols(init_pyi)
    rust_syms = extract_rust_pyfunctions(RUST_LIB)

    missing_from_pyi = sorted(all_syms - pyi_syms)
    untyped_rust = sorted(rust_syms - pyi_syms)

    ok = True
    if missing_from_pyi:
        ok = False
        print("Symbols in __all__ but missing from __init__.pyi:")
        for name in missing_from_pyi:
            print(f"  - {name}")
    if untyped_rust:
        ok = False
        print("Rust #[pyfunction] registered in #[pymodule] but missing from __init__.pyi:")
        for name in untyped_rust:
            print(f"  - {name}")
    hand_written = hand_written_typed_dicts(PY_PKG)
    if hand_written:
        ok = False
        print(
            f"TypedDicts declared outside {GENERATED_TYPES} (payload shapes come from "
            "the Rust types; run `cargo xtask emit-schemas`):"
        )
        for entry in hand_written:
            print(f"  - {entry}")
    if ok:
        print(f"pyi coverage OK ({len(all_syms)} __all__ entries, {len(rust_syms)} pyfunctions)")
        return 0
    if args.check:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
