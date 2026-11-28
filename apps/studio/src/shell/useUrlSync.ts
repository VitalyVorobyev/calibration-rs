/** The address and the open project and run say the same thing. A link
 * (`#/inspect/frames?project=…&run=run-003`) opens that project and run;
 * choosing a project or run any other way (the menu, the run selector,
 * a calibration that finished) writes it to the address, replacing the
 * history entry so the back button steps through screens, not through
 * every run.
 *
 * Each direction is an effect that depends only on its own source and acts
 * when the other side differs, so neither can chase the other. A project
 * that opens with no run in its address opens on its newest run that
 * reads, once. */
import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router";
import { defaultRun } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { placePath, parsePlace } from "./screens";
import { useSession } from "../session/session";

export function useUrlSync(): void {
  const { pathname, search } = useLocation();
  const navigate = useNavigate();
  const place = parsePlace(pathname, search);
  const ready = useSession((s) => s.status.kind === "ready");
  const project = useProject((s) => s.project);
  const dir = project?.dir ?? null;
  const shown = useStore((s) => s.run);
  const shownId = shown !== null && shown.dir === dir ? shown.id : null;
  /** The project being opened for the address, so the write-back does not
   * clear it while it opens. */
  const opening = useRef<string | null>(null);
  /** The project whose newest run was opened by default. */
  const defaulted = useRef<string | null>(null);

  /** Write the project and run to the address, keeping any other
   * parameter a screen keeps there (the Library's search, the Device
   * screen's device and tab). */
  const replace = (next: { project: string | null; run: string | null }) => {
    const params = new URLSearchParams(search);
    const written = new URLSearchParams(
      placePath({ ...place, ...next }).split("?")[1] ?? "",
    );
    for (const key of ["project", "run"]) {
      const value = written.get(key);
      if (value === null) params.delete(key);
      else params.set(key, value);
    }
    const query = params.toString();
    void navigate({ pathname, search: query ? `?${query}` : "" }, { replace: true });
  };

  // Address → project.
  useEffect(() => {
    if (!ready || place.project === null || place.project === dir) return;
    if (opening.current === place.project) return;
    opening.current = place.project;
    void useProject
      .getState()
      .open(place.project)
      .finally(() => {
        opening.current = null;
      });
    // Only a change of the address's project acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, place.project]);

  // Project → address.
  useEffect(() => {
    if (!ready || dir === place.project) return;
    if (place.project !== null && opening.current === place.project) return;
    replace({ project: dir, run: dir === null ? null : shownId });
    // Only a change of the open project acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, dir]);

  // Address → run, and the default run of a project opened without one.
  useEffect(() => {
    if (!project || place.project !== project.dir) return;
    if (place.run !== null) {
      if (place.run !== shownId && project.runs.some((r) => r.id === place.run)) {
        void useProject.getState().openRun(place.run);
      }
      return;
    }
    if (shownId === null && defaulted.current !== project.dir) {
      defaulted.current = project.dir;
      const run = defaultRun(project.runs);
      if (run) void useProject.getState().openRun(run.id);
    }
    // Only a change of the address's run, or a project newly open, acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [place.run, place.project, project?.dir]);

  // Run → address.
  useEffect(() => {
    if (!project || place.project !== project.dir) return;
    if (shownId === null || shownId === place.run) return;
    replace({ project: project.dir, run: shownId });
    // Only a change of the run shown acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [shownId]);
}
