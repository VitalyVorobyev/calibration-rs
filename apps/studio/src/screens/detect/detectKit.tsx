/** What the Detect screen's component tests share: a mock back end with
 * the fixture overview, detail and detector form, the screen mounted over it,
 * and the resets between tests. */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { setTransport, StudioError } from "../../api";
import type {
  DetectProgress,
  DetectionDetail,
  DetectionOverview,
  ProjectView,
} from "../../api/generated/types";
import {
  createMockTransport,
  type MockJobControl,
  type MockTransport,
} from "../../api/transport/mock";
import { useProject } from "../../project/projectStore";
import { startJobWatch, useJobs } from "../../session/jobs";
import { queryCache } from "../../session/query";
import { useSelection } from "../../session/selection";
import { projectView } from "../../test/projectFixtures";
import { chessboardDetail, overview } from "./__fixtures__/details";
import { DetectScreen } from "./index";
import { useExclusionHistory } from "./useExclusions";

export const DIR = "/work/cell";

const FORM_SCHEMA = {
  type: "object",
  properties: {
    topology: { type: "string" },
    detector: { anyOf: [{ $ref: "#/$defs/DetectorSpec" }, { type: "null" }] },
  },
  required: ["topology"],
  $defs: {
    DetectorSpec: {
      type: "object",
      properties: {
        min_features_per_view: {
          type: "integer",
          minimum: 0,
          title: "Min features per view",
        },
      },
    },
  },
};

export interface Options {
  overview?: DetectionOverview;
  detail?: DetectionDetail;
  view?: ProjectView;
  /** Make save_project answer like a file that changed on disk, this many times. */
  stale?: number;
  detect?: (control: MockJobControl<DetectProgress>) => Promise<void>;
}

let stopWatch: (() => void) | null = null;

export function mount(options: Options = {}): MockTransport {
  let stale = options.stale ?? 0;
  const view = options.view ?? projectView(DIR);
  let current = view;
  const mock = createMockTransport({
    commands: {
      dataset_form: () => ({ schema: FORM_SCHEMA, default: {} }),
      detection_overview: () => options.overview ?? overview(),
      detection_detail: () => options.detail ?? chessboardDetail(),
      detect_preview: ({ detector }) => {
        const detail = chessboardDetail();
        // The preview finds one more corner than the saved detection.
        detail.features.push({ image_xy: [400, 300], world_xyz: [0.5, 0.5, 0] });
        if (detail.target.kind === "chessboard")
          detail.target.corners.push({ grid: [5, 5], score: 0.2 });
        void detector;
        return detail;
      },
      open_project: () => current,
      save_project: (args) => {
        if (stale > 0) {
          stale -= 1;
          current = { ...current, revision: `sha256:disk${stale}` };
          throw new StudioError({
            kind: "changed_on_disk",
            message: "The project file changed on disk.",
          });
        }
        current = { ...current, revision: "sha256:r2", spec: args.spec };
        return current;
      },
    },
    jobs: {
      detect_all: async (_args, control) => {
        await options.detect?.(control);
        return options.overview ?? overview();
      },
    },
  });
  setTransport(mock);
  useProject.setState({ project: view });
  stopWatch = startJobWatch(() => {});
  render(
    <TooltipProvider>
      <DetectScreen />
    </TooltipProvider>,
  );
  return mock;
}

export const saves = (mock: MockTransport) =>
  mock.calls.filter((c) => c.command === "save_project").map((c) => c.args) as {
    revision: string;
    spec: { exclude?: unknown; dataset: { manifest: { detector?: unknown } } };
  }[];

export async function matrix() {
  return screen.findByRole("grid", { name: "Detections by camera and view" });
}

export async function select(name: RegExp) {
  fireEvent.click(within(await matrix()).getByRole("gridcell", { name }));
  // The detail is read once the image is selected.
  await screen.findByRole("table", { name: "Features of this image" });
}

export function press(key: string, init: KeyboardEventInit = {}) {
  fireEvent.keyDown(window, { key, code: `Key${key.toUpperCase()}`, ...init });
}

export function resetDetect() {
  queryCache.clear();
  useExclusionHistory.getState().clear();
}

export function cleanupDetect() {
  cleanup();
  stopWatch?.();
  stopWatch = null;
  setTransport(null);
  useProject.setState({ project: null, error: null });
  useJobs.setState({ rows: [], connection: "connecting" });
  useSelection.getState().setSelection({ camera: null, view: null, feature: null });
}
