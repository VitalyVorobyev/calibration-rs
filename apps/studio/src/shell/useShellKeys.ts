/** Carry out the global keys (`keymap.ts`): switch screens, step the
 * view, camera and run, clear the selection, open the help and the
 * palette, save. One listener on the window; while a dialog is open only
 * `⌘K` (to close the palette) gets through, so a dialog's own keys are its
 * own. */
import { toast } from "@vitavision/ui";
import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router";
import { stepId } from "../lib/selection";
import { isDirty, useDraft } from "../project/draftStore";
import { useDatasetProblems } from "../project/problems";
import { runsNewestFirst } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useJobs } from "../session/jobs";
import { useSelection } from "../session/selection";
import { useUi } from "../session/ui";
import { useStore } from "../store";
import { isTyping, keyAction } from "./keymap";
import { isEnabled, parsePlace, placePath, screenById, screenPlace } from "./screens";

function dialogOpen(): boolean {
  return document.querySelector('[role="dialog"], [role="alertdialog"]') !== null;
}

export function useShellKeys(): void {
  const navigate = useNavigate();
  const { pathname, search } = useLocation();

  useEffect(() => {
    const place = parsePlace(pathname, search);
    const onKey = (e: KeyboardEvent) => {
      const action = keyAction(e, isTyping(e.target));
      if (action === null) return;
      if (dialogOpen() && action.kind !== "palette") return;

      switch (action.kind) {
        case "screen": {
          const screen = screenById(action.id);
          e.preventDefault();
          if (isEnabled(screen, useProject.getState().project !== null)) {
            void navigate(placePath(screenPlace(action.id, place)));
          }
          return;
        }
        case "view":
          e.preventDefault();
          useStore.getState().stepPose(action.delta);
          return;
        case "camera":
          e.preventDefault();
          useStore.getState().stepCamera(action.delta);
          return;
        case "run": {
          e.preventDefault();
          const { project } = useProject.getState();
          const shown = useStore.getState().run;
          if (!project) return;
          const ids = runsNewestFirst(project.runs).map((r) => r.id);
          const current = shown?.dir === project.dir ? shown.id : null;
          // Listed newest first: the next (newer) run is earlier in the list.
          const next = stepId(ids, current, -action.delta);
          if (next !== null && next !== current) void useProject.getState().openRun(next);
          return;
        }
        case "escape":
          if (!useSelection.getState().escape()) useJobs.getState().setTrayOpen(false);
          return;
        case "help":
          e.preventDefault();
          useUi.getState().toggle("shortcuts");
          return;
        case "palette":
          e.preventDefault();
          useUi.getState().toggle("palette");
          return;
        case "save":
          e.preventDefault();
          if (isDirty(useDraft.getState())) {
            // A dataset the library refuses is not saved by a key either.
            const problems = useDatasetProblems.getState();
            if (
              useDraft.getState().manifestDirty &&
              problems.errors > 0 &&
              problems.dir === useProject.getState().project?.dir
            ) {
              toast({
                title: "Fix the errors first",
                description: `The dataset has ${problems.errors} ${problems.errors === 1 ? "error" : "errors"}.`,
                tone: "error",
              });
              return;
            }
            void useDraft
              .getState()
              .save()
              .then((saved) => {
                if (saved) toast({ title: "Project saved", tone: "success" });
              });
          }
          return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, pathname, search]);
}
