#!/usr/bin/env python3
"""Verify the Python package binds the facade, item by item.

``scripts/binding_parity.toml`` maps every public item of the facade
(``crates/vision-calibration/src/lib.rs``), keyed by its path under
``vision_calibration::``, to either

- the Python names that bind it (``[bound]``): names in the package's
  ``__all__``, dotted names into its modules (``detect.Features``,
  ``types.TargetSpec``) or a module class's member
  (``planar_intrinsics.PlanarIntrinsicsSession.step_init``); a
  ``builtins.`` name is a Python built-in exception, or
- the reason it stays Rust-only (``[rust_only]``).

The check fails when a facade item is in neither table, when an entry names
an item the facade no longer has, and when a bound Python name does not
exist. Glob re-exports (``pub use some::module::*``) are expanded from the
source module, so each item they bring in needs its own entry.

It also checks that the Python names are the Rust names:

- every workflow's ``<Name>Input``, ``<Name>Config`` and ``<Name>Export`` is a
  Python class of the same name;
- every Python ``*Config``, ``*Input``, ``*Export``, ``*Estimate``,
  ``*Params`` or ``*Result`` class in ``__all__`` names a public Rust type, so
  no Python-only spelling survives a Rust rename.

The complementary stub-coverage guard lives in ``check_pyi_coverage.py``.
Needs Python 3.11+ (``tomllib``).

Usage:
    python3 scripts/check_binding_parity.py           # show drift, warn only
    python3 scripts/check_binding_parity.py --check   # exit 1 on drift (CI mode)
    python3 scripts/check_binding_parity.py --list    # print the facade items
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FACADE_LIB = REPO_ROOT / "crates/vision-calibration/src/lib.rs"
PARITY_MAP = REPO_ROOT / "scripts/binding_parity.toml"
PY_PKG = REPO_ROOT / "crates/vision-calibration-py/python/vision_calibration"
CRATES = REPO_ROOT / "crates"

# Python class-name suffixes that must name a Rust type.
RUST_NAMED_SUFFIXES = ("Config", "Input", "Export", "Estimate", "Params", "Result")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pyi_coverage import extract_all  # noqa: E402


# ─── Rust side: the facade's items ────────────────────────────────────────────


def strip_comments(text: str) -> str:
    """Drop ``//`` line comments (doc comments included)."""
    return re.sub(r"//[^\n]*", "", text)


def balanced_body(text: str, open_idx: int) -> str:
    """The text inside the braces opening at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise ValueError("unbalanced braces")


def top_level(body: str) -> str:
    """``body`` with every nested ``{ … }`` block blanked out, except the
    braces of ``use`` lists."""
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(body):
        char = body[i]
        if char == "{":
            # A `use` list's braces are part of the statement.
            line_start = body.rfind(";", 0, i) + 1
            statement = body[line_start:i]
            if depth == 0 and re.search(r"\buse\b", statement):
                close = i + len(balanced_body(body, i)) + 1
                out.append(body[i : close + 1])
                i = close + 1
                continue
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0:
            out.append(char)
        i += 1
    return "".join(out)


def use_leaves(statement: str) -> list[tuple[str, str]]:
    """``(source_path, exported_name)`` for each leaf of a ``use`` tree."""
    statement = " ".join(statement.split())

    def expand(prefix: str, tree: str) -> list[tuple[str, str]]:
        tree = tree.strip()
        if not tree:
            return []
        if tree.endswith("}"):
            open_idx = tree.index("{")
            head = tree[:open_idx].rstrip(":").strip()
            inner = tree[open_idx + 1 : -1]
            base = f"{prefix}::{head}" if prefix and head else (prefix or head)
            leaves: list[tuple[str, str]] = []
            depth, start = 0, 0
            for i, char in enumerate(inner):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                elif char == "," and depth == 0:
                    leaves += expand(base, inner[start:i])
                    start = i + 1
            leaves += expand(base, inner[start:])
            return leaves
        alias = None
        if " as " in tree:
            tree, alias = (part.strip() for part in tree.split(" as "))
        path = f"{prefix}::{tree}" if prefix else tree
        name = alias or path.rsplit("::", 1)[-1]
        return [(path, name)]

    return expand("", statement)


def crate_root(crate: str) -> Path:
    return CRATES / crate.replace("_", "-") / "src/lib.rs"


def module_items(source_path: str) -> list[str]:
    """Public names of the module ``crate::a::b`` (for a glob re-export)."""
    crate, *segments = source_path.split("::")
    text = strip_comments(crate_root(crate).read_text())
    directory = crate_root(crate).parent
    for segment in segments:
        inline = re.search(rf"\bpub mod {segment}\s*\{{", text)
        if inline:
            text = balanced_body(text, inline.end() - 1)
            continue
        for candidate in (directory / f"{segment}.rs", directory / segment / "mod.rs"):
            if candidate.exists():
                text = strip_comments(candidate.read_text())
                directory = candidate.parent if candidate.name == "mod.rs" else directory / segment
                break
        else:
            raise FileNotFoundError(f"module {source_path} not found")
    body = top_level(text)
    names = re.findall(
        r"\bpub (?:fn|struct|enum|type|trait|const|static|mod) (\w+)", body
    )
    for statement in re.findall(r"\bpub use ([^;]+);", body):
        names += [name for _, name in use_leaves(statement) if name != "*"]
    return sorted(set(names))


def facade_items() -> dict[str, str]:
    """Every public item of the facade: ``module::name`` → its source path."""
    items: dict[str, str] = {}

    def visit(body: str, module: str) -> None:
        flat = top_level(body)
        for statement in re.findall(r"\bpub use ([^;]+);", flat):
            for source, name in use_leaves(statement):
                if name == "*":
                    for item in module_items(source.rsplit("::", 1)[0]):
                        items[f"{module}::{item}" if module else item] = source[:-1] + item
                else:
                    items[f"{module}::{name}" if module else name] = source
        for match in re.finditer(r"\bpub mod (\w+)\s*\{", body):
            # Only modules declared at this level; nested ones recurse.
            prefix = body[: match.start()]
            if prefix.count("{") != prefix.count("}"):
                continue
            inner = balanced_body(body, match.end() - 1)
            visit(inner, f"{module}::{match.group(1)}" if module else match.group(1))

    visit(strip_comments(FACADE_LIB.read_text()), "")
    return items


# ─── Python side: the package's names ─────────────────────────────────────────


def module_names(path: Path) -> set[str]:
    """Names a Python module defines or re-exports at top level."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*" and node.module:
                    names |= module_names(path.parent / f"{node.module.lstrip('.')}.py")
                else:
                    names.add(alias.asname or alias.name)
    return names


def class_members(path: Path, class_name: str) -> set[str]:
    """Methods and attributes declared in the body of a module's class."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    members.update(t.id for t in item.targets if isinstance(t, ast.Name))
            return members
    return set()


def python_name_exists(name: str, top: set[str]) -> bool:
    """Whether ``name`` exists: a name in ``__all__``, ``module.name``,
    ``module.Class.member`` (declared in that class's body) or a built-in."""
    if name.startswith("builtins."):
        return hasattr(builtins, name.split(".", 1)[1])
    parts = name.split(".")
    if len(parts) == 1:
        return name in top
    for split in range(len(parts) - 1, 0, -1):
        path = PY_PKG / ("/".join(parts[:split]) + ".py")
        if not path.exists():
            continue
        rest = parts[split:]
        if rest[0] not in module_names(path):
            return False
        if len(rest) == 1:
            return True
        return len(rest) == 2 and rest[1] in class_members(path, rest[0])
    return False


# ─── Checks ───────────────────────────────────────────────────────────────────


def camel_case(module: str) -> str:
    """``rig_handeye_laserline`` -> ``RigHandeyeLaserline``."""
    return "".join(part.capitalize() for part in module.split("_"))


def rust_public_types() -> set[str]:
    """Every ``pub struct|enum|type`` name declared in a workspace crate."""
    names: set[str] = set()
    pattern = re.compile(r"^\s*pub (?:struct|enum|type) (\w+)", re.M)
    for path in CRATES.glob("*/src/**/*.rs"):
        names.update(pattern.findall(path.read_text()))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    parser.add_argument("--list", action="store_true", help="print the facade items")
    args = parser.parse_args()

    items = facade_items()
    if args.list:
        for key in sorted(items):
            print(f"{key:60} {items[key]}")
        return 0

    parity = tomllib.loads(PARITY_MAP.read_text())
    bound: dict[str, str | list[str]] = parity.get("bound", {})
    rust_only: dict[str, str] = parity.get("rust_only", {})
    top = extract_all(PY_PKG / "__init__.py")

    problems: list[str] = []
    for key in sorted(items):
        if key in bound and key in rust_only:
            problems.append(f"{key}: both bound and rust_only")
        elif key not in bound and key not in rust_only:
            problems.append(f"{key}: unmapped (bind it in Python or mark it rust_only with a reason)")
    for key in sorted({*bound, *rust_only} - set(items)):
        problems.append(f"{key}: mapped, but the facade has no such item")
    for key, names in sorted(bound.items()):
        for name in [names] if isinstance(names, str) else names:
            if not python_name_exists(name, top):
                problems.append(f"{key}: bound to {name!r}, which the package does not define")
    for key, reason in sorted(rust_only.items()):
        if not reason.strip():
            problems.append(f"{key}: rust_only needs a reason")

    # Workflow types carry their Rust names.
    workflows = {key.split("::")[0] for key in items if key.endswith("::run_calibration")}
    for name in sorted(workflows):
        stem = camel_case(name)
        for suffix in ("Input", "Config", "Export"):
            rust_name = f"{stem}{suffix}"
            if f"{name}::{rust_name}" not in items:
                problems.append(f"{name}: facade module does not export {rust_name}")
            elif rust_name not in top:
                problems.append(f"{name}: Python does not export {rust_name}")
    rust_types = rust_public_types()
    for py_name in sorted(top):
        if py_name.endswith(RUST_NAMED_SUFFIXES) and py_name not in rust_types:
            problems.append(f"Python {py_name} names no Rust type")

    if problems:
        print("Binding parity drift:")
        for problem in problems:
            print(f"  - {problem}")
        return 1 if args.check else 0

    print(
        f"binding parity OK ({len(items)} facade items: {len(bound)} bound in Python, "
        f"{len(rust_only)} Rust-only; Python names match Rust)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
