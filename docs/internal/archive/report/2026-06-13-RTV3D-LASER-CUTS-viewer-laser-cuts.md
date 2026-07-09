# RTV3D-LASER-CUTS Viewer Laser Cuts

## Scope

Added active-pose laser/target intersections to the 3D viewer. For each laser
plane in `laser_planes_rig`, the viewer transforms the plane into the selected
target frame, intersects it with the target `z = 0` plane, clips the line to
the target board bbox built from `target_xyz_m`, and renders up to six colored
segments directly on the board.

The existing translucent full laser-plane quads remain available behind the
laser-plane toggle. The cut segments are shown by default when an export has
laser planes and an active target pose.

## Files Changed

- `app/src/workspaces/Viewer3DWorkspace/LaserTargetCuts.tsx`: plane transform,
  target-plane intersection, bbox clipping, and colored segment rendering.
- `app/src/workspaces/Viewer3DWorkspace/Scene.tsx`: active-pose cut overlay
  wiring.
- `app/src/workspaces/Viewer3DWorkspace/TargetBoard.tsx`: exported board bbox
  helper for reuse by the cut overlay.

## Validation Run

- PASS: `npm --prefix app run build`
- PASS: `cargo check --workspace --all-features`

There is no configured frontend unit-test runner in `app/package.json`; the
pure clipping helpers are exported for a future Vitest or Playwright test.

## Follow-Ups / Remaining Risks

- Add a small frontend math test once `B-INFRA` adds Vitest.
- Consider a viewer toggle for "laser cuts only" vs "cuts + planes" if the
  default overlay becomes visually dense on other datasets.
