/** What the shell's component tests share: a mock back end that answers
 * `server_info`, a router with a placeholder body per screen, and the reset
 * of every store the shell fills. */
import { render } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { createMemoryRouter, RouterProvider } from "react-router";
import { setTransport } from "../api";
import type { ServerInfo } from "../api/generated/types";
import { createMockTransport, type MockOptions } from "../api/transport/mock";
import { useDraft } from "../project/draftStore";
import { useProject } from "../project/projectStore";
import { useJobs } from "../session/jobs";
import { queryCache } from "../session/query";
import { useSelection } from "../session/selection";
import { useSession } from "../session/session";
import { useUi } from "../session/ui";
import { Shell } from "../shell/Shell";
import { useStore } from "../store";
import { PLANAR_EXPORT_FIXTURE } from "./exportFixtures";
import { Body, LibraryBody } from "./shellKitParts";

export const SERVER: ServerInfo = {
  version: "0.1.0",
  shell: "server",
  roots: [{ path: "/data", samples: false }],
  roots_editable: false,
  samples: [],
};

export function backend(info: ServerInfo = SERVER, options: MockOptions = {}) {
  const mock = createMockTransport({
    ...options,
    commands: {
      server_info: () => info,
      recent_projects: () => [],
      list_jobs: () => [],
      run_export: () => ({ export: PLANAR_EXPORT_FIXTURE, image_root: null }),
      ...options.commands,
    },
  });
  setTransport(mock);
  return mock;
}

/** The shell over placeholder screens, at `initial`. */
export function renderShell(initial = "/library") {
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: <Shell />,
        children: [
          { path: "library", element: <LibraryBody /> },
          { path: "calibrate", element: <Body text="calibrate body" /> },
          { path: "inspect/:tab", element: <Body text="inspect body" /> },
        ],
      },
    ],
    { initialEntries: [initial] },
  );
  return render(
    <TooltipProvider>
      <RouterProvider router={router} />
    </TooltipProvider>,
  );
}

export function resetShellStores() {
  setTransport(null);
  queryCache.clear();
  useProject.setState({ project: null, recent: [], error: null });
  useDraft.getState().resetFrom(null);
  useSession.setState({ status: { kind: "connecting" } });
  useJobs.setState({
    rows: [],
    connection: "connecting",
    lastEventAt: null,
    trayOpen: false,
  });
  useUi.setState({ dialog: null });
  useSelection.getState().setSelection({ camera: null, view: null, feature: null });
  useStore.getState().resetExport();
}
