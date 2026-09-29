import { createHashRouter, Navigate } from "react-router";
import { AppShell } from "./layouts/AppShell";
import { DepthWorkspace } from "./workspaces/DepthWorkspace";
import { DiagnoseWorkspace } from "./workspaces/DiagnoseWorkspace";
import { EpipolarWorkspace } from "./workspaces/EpipolarWorkspace";
import { RunWorkspace } from "./workspaces/RunWorkspace";
import { LazyViewer3DWorkspace } from "./workspaces/Viewer3DWorkspace/LazyViewer3DWorkspace";

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
        element: <LazyViewer3DWorkspace />,
      },
      { path: "epipolar", element: <EpipolarWorkspace /> },
      { path: "depth", element: <DepthWorkspace /> },
      { path: "run", element: <RunWorkspace /> },
    ],
  },
]);
