import { Tabs } from "@vitavision/ui";
import { Navigate, useLocation, useNavigate } from "react-router";
import { useStore } from "../../store";
import { DepthWorkspace } from "../../workspaces/DepthWorkspace";
import { DiagnoseWorkspace } from "../../workspaces/DiagnoseWorkspace";
import { EpipolarWorkspace } from "../../workspaces/EpipolarWorkspace";
import { LazyViewer3DWorkspace } from "../../workspaces/Viewer3DWorkspace/LazyViewer3DWorkspace";
import { parsePlace, placePath, type InspectTab } from "../../shell/screens";
import { tabsFor } from "./tabs";

const LABELS: Record<InspectTab, string> = {
  scene: "Scene",
  frames: "Frames",
  epipolar: "Epipolar",
  depth: "Depth",
};

/** Inspect: judge one run. Its tabs are the viewers built before the
 * screens: `Scene` (the 3D viewer), `Frames` (the frame viewer with its
 * residuals), `Epipolar` and `Depth`. A tab the run cannot have (a
 * single-camera run has no epipolar geometry) is absent, not disabled; an
 * address naming one lands on Frames. The tab is in the address. */
export function InspectScreen() {
  const { pathname, search } = useLocation();
  const navigate = useNavigate();
  const place = parsePlace(pathname, search);
  const cameras = useStore((s) => (s.data === null ? null : s.cameraValues.length));
  const tabs = tabsFor(cameras);
  const active = place.tab ?? "frames";

  if (!tabs.includes(active)) {
    return <Navigate to={placePath({ ...place, tab: "frames" })} replace />;
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <Tabs
        label="Inspect"
        idPrefix="inspect"
        active={active}
        items={tabs.map((id) => ({ id, label: LABELS[id] }))}
        onSelect={(tab) => void navigate(placePath({ ...place, tab }))}
      />
      <div
        role="tabpanel"
        id={`inspect-panel-${active}`}
        aria-labelledby={`inspect-tab-${active}`}
        className="flex min-h-0 flex-1 flex-col"
      >
        {active === "scene" && <LazyViewer3DWorkspace />}
        {active === "frames" && <DiagnoseWorkspace />}
        {active === "epipolar" && <EpipolarWorkspace />}
        {active === "depth" && <DepthWorkspace />}
      </div>
    </div>
  );
}
