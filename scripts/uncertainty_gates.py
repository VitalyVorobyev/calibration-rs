#!/usr/bin/env python3
"""Run the release-only gates: the parameter-uncertainty Monte-Carlo gates
and the other tests ``#[ignore]``d in the debug test run.

Builds every gate's test binary once (``cargo test --release --no-run``),
then runs the binaries directly, in parallel, each with ``--ignored``. A
line per gate is printed as it finishes and the full outputs follow, in
order; the exit status is non-zero if any gate fails.

    python3 scripts/uncertainty_gates.py               # every gate
    python3 scripts/uncertainty_gates.py focus laser   # gates whose line matches
    python3 scripts/uncertainty_gates.py --jobs 2      # at most two at a time
    python3 scripts/uncertainty_gates.py --list        # print the gates

Running the binaries directly (not one ``cargo test -p …`` per gate) keeps
the one build: cargo resolves features per invocation, and a single-package
invocation would rebuild what the joint build already compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (package, test target or "lib", test-name filters). Each gate runs that
# target's ignored tests, narrowed by the filters when there are any. Longest
# first (release times on a laptop, all gates at once): the pool starts them
# in this order, so the long ones do not start last.
GATES: list[tuple[str, str, tuple[str, ...]]] = [
    ("vision-calibration", "rig_extrinsics_scheimpflug", ()),  # 340 s
    ("vision-calibration", "second_order_error", ()),  # 250 s
    ("vision-calibration-optim", "uncertainty", ()),  # 200 s
    ("vision-calibration", "handeye_frame_uncertainty", ()),  # 190 s
    ("vision-calibration", "bundle_adjust_uncertainty", ()),  # 175 s
    ("vision-calibration-optim", "feature_covariance", ()),  # 175 s
    ("vision-calibration", "rig_laser_uncertainty", ()),  # 170 s
    ("vision-calibration-optim", "rig_laser_problem", ()),  # 160 s
    ("vision-calibration", "metrology_uncertainty", ()),  # 105 s
    ("vision-calibration", "pose_uncertainty", ()),  # 95 s
    ("vision-calibration", "dense_cloud_uncertainty", ()),  # 65 s
    ("vision-calibration", "triangulation_uncertainty", ()),  # 60 s
    ("vision-calibration", "stereo_uncertainty", ()),  # 55 s
    ("vision-calibration-optim", "robot_pose_priors", ("estimated_robot_sigma",)),  # 50 s
    ("vision-calibration-bench", "split_half_honesty", ()),  # 50 s
    ("vision-calibration", "compose_pipeline", ()),  # 40 s
    ("vision-calibration-pipeline", "lib", ("analysis::next_view",)),  # 40 s
    ("vision-calibration", "quality_tilt_conditioning", ()),  # 20 s
    ("vision-mvg", "triangulation_cov", ()),  # 20 s
    ("vision-calibration", "custom_residual", ()),  # 15 s
    ("vision-calibration-pipeline", "lib", ("analysis::cross_validation",)),  # 10 s
    ("vision-calibration-pipeline", "lib", ("analysis::focus",)),  # 4 s
]


def label(gate: tuple[str, str, tuple[str, ...]]) -> str:
    package, target, filters = gate
    where = "--lib" if target == "lib" else f"--test {target}"
    return " ".join([f"-p {package}", where, *filters])


def build(gates: list[tuple[str, str, tuple[str, ...]]]) -> dict[tuple[str, str], tuple[str, Path]]:
    """Build the gates' test binaries; map (package, target) to (executable, package dir)."""
    cmd = ["cargo", "test", "--release", "--locked", "--no-run", "--message-format=json-render-diagnostics"]
    for package in sorted({g[0] for g in gates}):
        cmd += ["-p", package]
    for target in sorted({g[1] for g in gates if g[1] != "lib"}):
        cmd += ["--test", target]
    if any(g[1] == "lib" for g in gates):
        cmd.append("--lib")
    print("+", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    if out.returncode != 0:
        sys.exit(out.returncode)
    found: dict[tuple[str, str], tuple[str, Path]] = {}
    for line in out.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("reason") != "compiler-artifact" or not msg.get("executable"):
            continue
        if not msg.get("profile", {}).get("test"):
            continue
        target = msg["target"]
        # Every workspace crate lives in `crates/<package name>/`.
        pkg_dir = Path(msg["manifest_path"]).parent
        kind = "lib" if "lib" in target["kind"] else target["name"]
        found[(pkg_dir.name, kind)] = (msg["executable"], pkg_dir)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", nargs="*", help="run only gates whose label contains one of these")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--list", action="store_true", help="print the gates and exit")
    args = parser.parse_args()

    gates = [g for g in GATES if not args.match or any(m in label(g) for m in args.match)]
    if args.list:
        print("\n".join(label(g) for g in gates))
        return 0
    if not gates:
        print(f"no gate matches {args.match}", file=sys.stderr)
        return 2

    binaries = build(gates)
    missing = [label(g) for g in gates if (g[0], g[1]) not in binaries]
    if missing:
        print("no test binary for: " + "; ".join(missing), file=sys.stderr)
        return 2

    def run(gate: tuple[str, str, tuple[str, ...]]) -> tuple[int, float, str]:
        exe, pkg_dir = binaries[(gate[0], gate[1])]
        env = dict(os.environ, CARGO_MANIFEST_DIR=str(pkg_dir))
        start = time.monotonic()
        # cargo runs a test binary from its package directory; so do we.
        done = subprocess.run(
            [exe, "--ignored", *gate[2]], cwd=pkg_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return done.returncode, time.monotonic() - start, done.stdout

    results: dict[int, tuple[int, float, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run, g): i for i, g in enumerate(gates)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            code, took, _ = results[i]
            print(f"{'PASS' if code == 0 else 'FAIL'} {took:6.1f} s  {label(gates[i])}", flush=True)

    in_ci = bool(os.environ.get("GITHUB_ACTIONS"))
    for i, gate in enumerate(gates):
        code, took, output = results[i]
        title = f"{'PASS' if code == 0 else 'FAIL'} {took:.1f} s  {label(gate)}"
        print(f"::group::{title}" if in_ci else f"\n===== {title}")
        print(output, end="")
        if in_ci:
            print("::endgroup::")

    failed = [label(g) for i, g in enumerate(gates) if results[i][0] != 0]
    print(f"\n{len(gates) - len(failed)}/{len(gates)} gates passed")
    for name in failed:
        print(f"FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
