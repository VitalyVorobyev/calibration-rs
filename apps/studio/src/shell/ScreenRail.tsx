import { NavRail, NavRailItem } from "@vitavision/workbench";
import {
  Camera,
  Database,
  Diff,
  Library,
  Loader,
  Microscope,
  ScanSearch,
  SlidersHorizontal,
} from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";
import { Logo } from "../components/Logo";
import { useDatasetProblems } from "../project/problems";
import { useProject } from "../project/projectStore";
import { useJobs } from "../session/jobs";
import { isLive } from "../session/jobModel";
import {
  isEnabled,
  placePath,
  screenPlace,
  SCREENS,
  type Place,
  type ScreenId,
} from "./screens";

const ICONS: Record<ScreenId, ReactNode> = {
  library: <Library />,
  dataset: <Database />,
  detect: <ScanSearch />,
  calibrate: <SlidersHorizontal />,
  inspect: <Microscope />,
  compare: <Diff />,
  device: <Camera />,
};

/** Why a screen cannot be entered, said on hover. */
function reason(id: ScreenId, built: boolean): string {
  return built ? "Open a project first" : `${labelOf(id)} is coming in a later release`;
}

function labelOf(id: ScreenId): string {
  return SCREENS.find((s) => s.id === id)?.label ?? id;
}

/** The seven screens as a rail of router links, in workflow order, with
 * the app's mark above. The current one is `aria-current="page"`; one that
 * is not built, or needs a project that is not open, is disabled and says
 * why. The links keep the open project and run in the address. */
export function ScreenRail({ place }: { place: Place }) {
  const hasProject = useProject((s) => s.project !== null);
  const openDir = useProject((s) => s.project?.dir ?? null);
  // What the Dataset screen last judged, for the open project only.
  const datasetErrors = useDatasetProblems((s) => (s.dir === openDir ? s.errors : 0));
  const calibrating = useJobs((s) =>
    s.rows.some((r) => r.command === "calibrate" && isLive(r.state)),
  );
  return (
    <div className="flex h-full flex-col items-center gap-1 py-2">
      <span aria-hidden className="mb-2 mt-1 text-fg">
        <Logo size={22} />
      </span>
      <NavRail value={place.screen} labels="visible">
        {SCREENS.map((screen) => {
          const enabled = isEnabled(screen, hasProject);
          return (
            <NavRailItem
              key={screen.id}
              value={screen.id}
              label={screen.label}
              icon={ICONS[screen.id]}
              disabled={!enabled}
              badge={
                screen.id === "calibrate" && calibrating ? (
                  <Loader
                    aria-label="A calibration is running"
                    className="size-3 animate-spin motion-reduce:animate-none"
                  />
                ) : screen.id === "dataset" && datasetErrors > 0 ? (
                  <span
                    role="img"
                    aria-label={`${datasetErrors} ${datasetErrors === 1 ? "error" : "errors"} in the dataset`}
                    className="grid min-w-4 place-items-center rounded-full bg-defect px-1 font-mono text-[10px] leading-4 text-white"
                  >
                    {datasetErrors}
                  </span>
                ) : undefined
              }
              asChild
            >
              <Link
                to={placePath(screenPlace(screen.id, place))}
                {...(enabled ? {} : { title: reason(screen.id, screen.built) })}
              />
            </NavRailItem>
          );
        })}
      </NavRail>
    </div>
  );
}
