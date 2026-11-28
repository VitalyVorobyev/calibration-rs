import { createHashRouter, type RouteObject } from "react-router";
import { RouteErrorFallback, ScreenErrorFallback } from "./components/ErrorBoundary";
import { CalibrateScreen } from "./screens/calibrate";
import { CompareScreen } from "./screens/compare";
import { DatasetScreen } from "./screens/dataset";
import { DetectScreen } from "./screens/detect";
import { DeviceScreen } from "./screens/device";
import { InspectScreen } from "./screens/inspect";
import { LibraryScreen } from "./screens/library";
import { Redirect, Unbuilt } from "./screens/Redirect";
import { Shell } from "./shell/Shell";
import { SCREENS } from "./shell/screens";

const UNBUILT = SCREENS.filter((s) => !s.built);

/** The addresses of the screens before the redesign, kept so a saved link
 * still opens: each goes to where that workspace lives now. */
const LEGACY: [string, string][] = [
  ["diagnose", "/inspect/frames"],
  ["viewer3d", "/inspect/scene"],
  ["epipolar", "/inspect/epipolar"],
  ["depth", "/inspect/depth"],
  ["run", "/calibrate"],
];

const screen: Pick<RouteObject, "errorElement"> = {
  errorElement: <ScreenErrorFallback />,
};

/** Hash router (not BrowserRouter): survives `tauri build`'s static-asset
 * paths cleanly. The address is `#/<screen>[/<tab>]?project=<dir>&run=<id>`
 * (`shell/screens.ts`); the project and run are the query, kept in step with
 * the open project by `useUrlSync`, so a link reopens the same place. */
export const router = createHashRouter([
  {
    path: "/",
    element: <Shell />,
    errorElement: <RouteErrorFallback />,
    children: [
      { index: true, element: <Redirect to="/library" /> },
      { path: "library", element: <LibraryScreen />, ...screen },
      { path: "dataset", element: <DatasetScreen />, ...screen },
      { path: "detect", element: <DetectScreen />, ...screen },
      { path: "calibrate", element: <CalibrateScreen />, ...screen },
      { path: "compare", element: <CompareScreen />, ...screen },
      { path: "device", element: <DeviceScreen />, ...screen },
      {
        path: "inspect",
        children: [
          { index: true, element: <Redirect to="/inspect/frames" /> },
          { path: ":tab", element: <InspectScreen />, ...screen },
        ],
      },
      ...UNBUILT.map((s) => ({
        path: s.path.slice(1),
        element: <Unbuilt id={s.id} />,
        ...screen,
      })),
      ...LEGACY.map(([from, to]) => ({
        path: from,
        element: <Redirect to={to} />,
      })),
      { path: "*", element: <Redirect to="/library" /> },
    ],
  },
]);
