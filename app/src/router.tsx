import { lazy, Suspense } from "react";
import { createHashRouter, Navigate } from "react-router-dom";
import { AppShell } from "./layouts/AppShell";
import { DiagnoseWorkspace } from "./workspaces/DiagnoseWorkspace";
import { EpipolarWorkspace } from "./workspaces/EpipolarWorkspace";
import { RunWorkspace } from "./workspaces/RunWorkspace";

// Three.js + R3F + drei are heavy (~900 KB gzipped). Lazy-load so the
// diagnose / epipolar / run paths don't pull them in.
const Viewer3DWorkspace = lazy(() =>
  import("./workspaces/Viewer3DWorkspace").then((m) => ({
    default: m.Viewer3DWorkspace,
  })),
);

function ViewerFallback() {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
      <p className="text-[13px] text-muted-foreground">Loading 3D scene…</p>
    </div>
  );
}

/** Hash router (not BrowserRouter): survives `tauri build`'s static-asset
 * paths cleanly. Routes intentionally have no params — workspace state
 * lives in the Zustand store and is shared across panels. */
export const router = createHashRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/diagnose" replace /> },
      { path: "diagnose", element: <DiagnoseWorkspace /> },
      {
        path: "viewer3d",
        element: (
          <Suspense fallback={<ViewerFallback />}>
            <Viewer3DWorkspace />
          </Suspense>
        ),
      },
      { path: "epipolar", element: <EpipolarWorkspace /> },
      { path: "run", element: <RunWorkspace /> },
    ],
  },
]);
