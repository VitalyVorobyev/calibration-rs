"""UI smoke test: drive the frontend against the real rtv3d exports.

Catches viewer-contract crashes (uncaught exceptions / the router's
error boundary) that Rust-side tests can't see — e.g. an export kind
missing a field the TS code dereferences, like the
`data.mean_reproj_error.toFixed` crash on the laser export. Local-only:
needs the gitignored privatedata/rtv3d exports (produced by the
`rtv3d_laser_end_to_end` ignored test) and `pip install playwright`.

Run from `app/`:

    bun run dev &           # Vite at :1420
    python3 scripts/ui_smoke_rtv3d.py

The Vite dev server has no Tauri runtime, so image loading fails with a
handled error banner — expected and filtered.
"""

import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
LASER_EXPORT = f"{ROOT}/privatedata/rtv3d/rig_laserline_export.json"
HANDEYE_EXPORT = f"{ROOT}/privatedata/rtv3d/rig_handeye_export.json"

if not pathlib.Path(LASER_EXPORT).exists():
    print(f"skipping: {LASER_EXPORT} not present (run rtv3d_laser_end_to_end first)")
    sys.exit(0)

INJECT = """async (payload) => {
  const mod = await import('/src/store/index.ts');
  mod.useStore.getState().acceptLiveRunExport(payload.data, payload.dir);
  return mod.useStore.getState().kind;
}"""

failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(f"{name}: {detail}")


def crashed(page):
    """The router's error boundary renders this on any uncaught render error."""
    return page.locator("text=Unexpected Application Error").count() > 0


def run_workspace_suite(page, label, export_path, expect_kind, expect_laser):
    data = json.load(open(export_path))
    page.goto("http://localhost:1420/#/diagnose")
    page.wait_for_load_state("networkidle")
    kind = page.evaluate(INJECT, {"data": data, "dir": f"{ROOT}/privatedata/rtv3d"})
    check(f"{label}: store kind", kind == expect_kind, f"got {kind}")
    page.wait_for_timeout(800)

    # ── Diagnose ──────────────────────────────────────────────────────
    check(f"{label}: diagnose no crash", not crashed(page))
    chip = page.locator("text=mean reproj:")
    check(f"{label}: mean reproj chip", chip.count() > 0)
    laser_btn = page.get_by_role("button", name="Laser", exact=True)
    if expect_laser:
        check(f"{label}: Laser toggle present", laser_btn.count() == 1)
        laser_btn.click()
        page.wait_for_timeout(600)
        check(f"{label}: laser view no crash", not crashed(page))
        check(
            f"{label}: laser legend",
            page.locator("text=laser pts").count() > 0,
            "mm stats legend visible",
        )
        page.screenshot(path=f"/tmp/{label}_diagnose_laser.png")
        # Toggle back so the next steps see target view.
        page.get_by_role("button", name="Laser ✓").click()
    else:
        check(f"{label}: Laser toggle hidden", laser_btn.count() == 0)

    # ── 3D viewer ─────────────────────────────────────────────────────
    page.goto("http://localhost:1420/#/viewer3d")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2500)  # lazy chunk + R3F first frame
    check(f"{label}: 3D no crash", not crashed(page))
    check(
        f"{label}: 3D scene mounted (not the rig-export fallback)",
        page.locator("canvas").count() > 0,
        f"needs cameras+cam_se3_rig+rig_se3_target on {expect_kind}",
    )
    planes_btn = page.get_by_role("button", name="Laser planes ✓")
    if expect_laser:
        check(f"{label}: 3D laser-planes toggle on", planes_btn.count() == 1)
        page.screenshot(path=f"/tmp/{label}_viewer3d_planes.png")
        planes_btn.click()
        page.wait_for_timeout(400)
        check(f"{label}: 3D toggle off no crash", not crashed(page))
    else:
        check(f"{label}: 3D laser-planes toggle hidden", planes_btn.count() == 0)

    # ── Epipolar (must degrade gracefully, not crash) ────────────────
    page.goto("http://localhost:1420/#/epipolar")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(600)
    check(f"{label}: epipolar no crash", not crashed(page))


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader"])
    page = browser.new_page()
    page_errors = []
    page.on("pageerror", lambda e: page_errors.append(str(e)))

    run_workspace_suite(
        page, "laser", LASER_EXPORT, "rig_laserline_device", expect_laser=True
    )
    run_workspace_suite(
        page, "handeye", HANDEYE_EXPORT, "rig_handeye", expect_laser=False
    )

    # Tauri invoke failures are expected in a plain browser; anything
    # else uncaught is a real bug.
    real_errors = [
        e
        for e in page_errors
        if "invoke" not in e and "__TAURI" not in e and "Tauri" not in e
    ]
    check("no uncaught page errors", not real_errors, "; ".join(real_errors[:3]))
    browser.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
